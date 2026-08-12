#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
from cv_bridge import CvBridge
import numpy as np
import cv2
import rospy
from duckietown_msgs.msg import BoolStamped, AntiInstagramThresholds
from duckietown.dtros import DTROS, NodeType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int64MultiArray
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from dt_computer_vision.anti_instagram import AntiInstagram


class IntersectionTypeDetectorNode(DTROS):
    def __init__(self, node_name):
        super(IntersectionTypeDetectorNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True
        )

        self.node_name = node_name
        self.turn_type = -1
        rospy.loginfo(f"[{self.node_name}] Initializing.")
        self.bridge = CvBridge()
        self._veh        = rospy.get_param("~veh")
        self._img_size      = rospy.get_param("~img_size", None)
        self._top_cutoff    = rospy.get_param("~top_cutoff", None)
        self._bottom_cutoff = rospy.get_param("~bottom_cutoff", 50)
        self.verbose = DTParam("~verbose", param_type=ParamType.INT)

        # ------ Red HSV thresholds — configurable per vehicle -----
        self.lower_red1 = np.array(rospy.get_param("~red_hsv_low1",  [0,   70,  50]))
        self.upper_red1 = np.array(rospy.get_param("~red_hsv_high1", [10, 255, 255]))
        self.lower_red2 = np.array(rospy.get_param("~red_hsv_low2",  [170,  70,  50]))
        self.upper_red2 = np.array(rospy.get_param("~red_hsv_high2", [180, 255, 255]))

        self._traffic_mode = DTParam(f"/{self._veh}/behavior/traffic_mode", None)

        # ------ DBSCAN / classifier params — configurable per vehicle -----
        self.STEEPNESS_THRESHOLD = rospy.get_param("~steepness_threshold", 0.25)
        self.EPS                 = rospy.get_param("~dbscan_eps",          10)
        self.MIN_SAMPLES         = rospy.get_param("~dbscan_min_samples",  15)

        self.ai_thresholds_received = False
        self.anti_instagram_thresholds = dict()
        self.ai = AntiInstagram()

        self.last_call = time.time()

        self.pub_stop_sign = rospy.Publisher(
            "~stop_sign_intersection_detected", BoolStamped, queue_size=1, latch=True)
        self.pub_traffic_light = rospy.Publisher(
            "~traffic_light_intersection_detected", BoolStamped, queue_size=1, latch=True)
        self.pub_topic_avail_turns = rospy.Publisher(
            "~available_turns", Int64MultiArray, queue_size=1)
        self.pub_debug_image = rospy.Publisher(
            "~debug/clusters/compressed", CompressedImage, queue_size=1)

        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.image_cb,
            buff_size=10000000, queue_size=1)
        self.sub_thresholds = rospy.Subscriber(
            "~thresholds", AntiInstagramThresholds, self.thresholds_cb, queue_size=1)

        # Save one debug image per intersection into the robot's /data mount.
        # This node is fsm_controlled and only active in STOP_SIGN_INTERSECTION,
        # so image_cb runs *only* while at an intersection: a gap in callbacks
        # means the node was switched off in between, i.e. a new intersection.
        self.save_debug_images = rospy.get_param("~save_debug_images", True)
        self.save_dir          = rospy.get_param("~save_dir", "/data/intersection_debug")
        # image_cb is rate-limited to ~1 Hz, so frames within one intersection are
        # ~1 s apart; anything longer means we were switched off in between.
        self.new_episode_gap   = rospy.get_param("~new_episode_gap", 3.0)
        self._last_cb_time     = None
        self._saved_this_intersection = False
        self._intersection_count      = 0
        if self.save_debug_images:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                rospy.loginfo(f"[{self.node_name}] Saving one debug image per "
                              f"intersection to {self.save_dir}")
            except Exception as e:
                rospy.logwarn(f"[{self.node_name}] Cannot create {self.save_dir}: {e} "
                              f"— image saving disabled")
                self.save_debug_images = False

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.loginfo("Using CUDA GPU for line detection.")
            self.cuda_enabled = True
        else:
            self.loginfo("Using the CPU for line detection.")
            self.cuda_enabled = False

        rospy.loginfo(f"[{self.node_name}] Initialized.")

    def image_cb(self, image_msg):
        if (time.time() - self.last_call) < 1:
            return
        self.last_call = time.time()

        # A long gap since the last processed frame means this node was switched
        # off in between (we left the intersection) -> this is a new intersection.
        now = time.time()
        if (self._last_cb_time is None
                or (now - self._last_cb_time) > self.new_episode_gap):
            self._saved_this_intersection = False
            self._intersection_count += 1
        self._last_cb_time = now

        try:
            obtained_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return

        bgr_img = self.preprocess_image(obtained_image)
        _, width = bgr_img.shape[:2]

        hsv_img   = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask1     = cv2.inRange(hsv_img, self.lower_red1, self.upper_red1)
        mask2     = cv2.inRange(hsv_img, self.lower_red2, self.upper_red2)
        red_mask  = cv2.bitwise_or(mask1, mask2)

        pixel_coords = np.argwhere(red_mask > 0)

        if len(pixel_coords) == 0:
            if self.verbose.value >= 1:
                rospy.logwarn_throttle(2.0, "[IntersType] no red pixels in frame")
            return

        # Flip to (x, y) order
        pixel_coords = np.flip(pixel_coords, axis=1)

        db     = DBSCAN(eps=self.EPS, min_samples=self.MIN_SAMPLES).fit(pixel_coords)
        labels = db.labels_

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if self.verbose.value >= 1:
            noise_px = int(np.sum(labels == -1))
            rospy.loginfo_throttle(1.0,
                f"[IntersType] red_px={len(pixel_coords)}  clusters={num_clusters}  "
                f"noise_px={noise_px}  eps={self.EPS}  min_samples={self.MIN_SAMPLES}"
            )

        # ----- Build cluster summaries (slope-fit per cluster) -----
        cluster_summary = {}
        for k in set(labels):
            if k == -1:
                continue
            pts      = pixel_coords[labels == k]
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            if len(x_coords) > 5:
                slope, *_ = linregress(x_coords, y_coords)
                cluster_summary[k] = {
                    'centroid_x':    x_coords.mean(),
                    'centroid_y':    y_coords.mean(),
                    'is_horizontal': abs(slope) < self.STEEPNESS_THRESHOLD,
                    'slope':         slope,
                    'x_min':         float(x_coords.min()),
                    'x_max':         float(x_coords.max()),
                }

        # ----- Classification -----
        has_left = has_middle = has_right = False
        middle_cluster_id = None

        # Step 1: find horizontal (middle / straight) cluster
        for k, data in cluster_summary.items():
            if data['is_horizontal']:
                has_middle = True
                middle_cluster_id = k
                break

        # Step 2: classify remaining clusters as left / right
        if has_middle:
            middle_x = cluster_summary[middle_cluster_id]['centroid_x']
            for k, data in cluster_summary.items():
                if k == middle_cluster_id:
                    continue
                if data['centroid_x'] < middle_x:
                    has_left = True
                elif data['centroid_x'] > middle_x:
                    has_right = True
        else:
            # Fallback: no horizontal cluster — split by image centre
            if self.verbose.value >= 1:
                rospy.logwarn_throttle(2.0,
                    f"[IntersType] no horizontal cluster (steepness_threshold={self.STEEPNESS_THRESHOLD}) "
                    f"— using centre-split fallback"
                )
            for k, data in cluster_summary.items():
                if data['centroid_x'] < (width / 2.0):
                    has_left = True
                else:
                    has_right = True

        # Step 3: encode [0=L, 1=M, 2=R]
        available_turns = []
        if has_left:   available_turns.append(0)
        if has_middle: available_turns.append(1)
        if has_right:  available_turns.append(2)

        # ----- Verbose decision log -----
        if self.verbose.value >= 1:
            cluster_lines = []
            for k, data in cluster_summary.items():
                role = "MIDDLE(H)" if data['is_horizontal'] else "side"
                cluster_lines.append(
                    f"  cluster {k}: cx={data['centroid_x']:.1f}  "
                    f"slope={data['slope']:+.3f}  threshold={self.STEEPNESS_THRESHOLD}  -> {role}"
                )
            rospy.loginfo_throttle(1.0,
                f"[IntersType] turns={available_turns}  "
                f"L={has_left} M={has_middle} R={has_right}  "
                f"clusters_eval={len(cluster_summary)}\n" +
                "\n".join(cluster_lines)
            )
        else:
            rospy.loginfo_throttle(5.0, f"[IntersType] turns={available_turns}")

        avail_turns_msg      = Int64MultiArray()
        avail_turns_msg.data = available_turns
        self.pub_topic_avail_turns.publish(avail_turns_msg)

        # ----- Debug image -----
        cluster_colors = [
            (0, 0, 255), (0, 255, 0), 
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
        ]
        debug_img = (bgr_img.astype(np.float32) * 0.25).astype(np.uint8)
        for k in set(labels):
            if k == -1:
                continue
            color = cluster_colors[k % len(cluster_colors)]
            pts   = pixel_coords[labels == k]
            debug_img[pts[:, 1], pts[:, 0]] = color

        # Scale up 6× — raw image is only ~20 px tall after cropping
        SCALE     = 6
        h0, w0    = debug_img.shape[:2]
        debug_disp = cv2.resize(debug_img, (w0 * SCALE, h0 * SCALE),
                                interpolation=cv2.INTER_NEAREST)

        # Snapshot of the clustered stage before any annotation, for the
        # per-intersection pipeline image series.
        stage_clustered = debug_disp.copy()

        # Grey centre-split line (used by fallback)
        cx_line = (width * SCALE) // 2
        cv2.line(debug_disp, (cx_line, 0),
                 (cx_line, debug_disp.shape[0] - 1), (80, 80, 80), 1)

        # Fitted slope per cluster, drawn across the cluster's x-extent.
        # Orange on purpose: not one of cluster_colors, so the fit is always
        # distinguishable from the pixels it was fitted to.
        SLOPE_COLOR = (255, 0, 0)
        for data in cluster_summary.values():
            m = data['slope']
            if not np.isfinite(m):
                continue
            x1, x2 = data['x_min'], data['x_max']
            # regression line passes through the centroid
            y1 = m * (x1 - data['centroid_x']) + data['centroid_y']
            y2 = m * (x2 - data['centroid_x']) + data['centroid_y']
            # A near-vertical fit gives a huge slope; bound the endpoints so the
            # int coords stay sane. cv2.line clips to the image, and the visible
            # part keeps the correct direction.
            y1 = float(np.clip(y1, -1e4, 1e4))
            y2 = float(np.clip(y2, -1e4, 1e4))
            cv2.line(debug_disp,
                     (int(x1 * SCALE), int(y1 * SCALE)),
                     (int(x2 * SCALE), int(y2 * SCALE)),
                     SLOPE_COLOR, 2, cv2.LINE_AA)

        # Per-cluster annotation: role + slope
        for k, data in cluster_summary.items():
            px   = int(data['centroid_x'] * SCALE)
            py   = int(data['centroid_y'] * SCALE)
            role = ("H" if data['is_horizontal']
                    else ("L" if data['centroid_x'] < (width / 2.0) else "R"))
            cv2.putText(debug_disp, f"{role} s={data['slope']:+.2f}",
                        (max(2, px - 20), max(10, py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Footer: turn result
        turns_str = ",".join(
            ["L" if t == 0 else "M" if t == 1 else "R" for t in available_turns]
        ) or "none"
        cv2.putText(debug_disp, f"turns:{turns_str}",
                    (2, debug_disp.shape[0] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 0), 1, cv2.LINE_AA)

        # One image series per intersection: save the pipeline stages from the
        # first frame of this episode, then skip until the next intersection.
        if self.save_debug_images and not self._saved_this_intersection:
            self._save_intersection_stages(
                raw=obtained_image, cropped=bgr_img, red_mask=red_mask,
                clustered=stage_clustered, slopes=debug_disp,
                turns_str=turns_str, scale=SCALE)
            self._saved_this_intersection = True

        debug_msg        = self.bridge.cv2_to_compressed_imgmsg(debug_disp)
        debug_msg.header = image_msg.header
        self.pub_debug_image.publish(debug_msg)

    def _save_intersection_stages(self, raw, cropped, red_mask, clustered,
                                  slopes, turns_str, scale):
        """Save the detection pipeline stage-by-stage, one folder per
        intersection, so the evolution raw -> cropped -> mask -> clusters ->
        slopes can be shown side by side.
        """
        ts     = datetime.now().strftime("%Y%m%d-%H%M%S")
        turns  = turns_str.replace(",", "") or "none"
        folder = os.path.join(
            self.save_dir,
            f"int_{self._intersection_count:03d}_{ts}_turns-{turns}")

        def up(img):
            # The cropped stages are only ~20 px tall — upscale by the same
            # factor as the debug view so all stages are legible and comparable.
            # ascontiguousarray: bgr_img may be a np.fliplr view (LHT mode),
            # which cv2 refuses to write to.
            img  = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            return cv2.resize(img, (w * scale, h * scale),
                              interpolation=cv2.INTER_NEAREST)

        stages = [
            ("1_raw.jpg",       np.ascontiguousarray(raw)),  # full frame, uncropped
            ("2_cropped.jpg",   up(cropped)),
            ("3_red_mask.jpg",  up(red_mask)),                # single channel -> greyscale
            ("4_clustered.jpg", clustered),                   # already upscaled
            ("5_slopes.jpg",    slopes),                      # already upscaled
        ]
        try:
            os.makedirs(folder, exist_ok=True)
            for name, img in stages:
                path = os.path.join(folder, name)
                if not cv2.imwrite(path, img):
                    rospy.logwarn(f"[{self.node_name}] imwrite returned False for {path}")
            rospy.loginfo(f"[{self.node_name}] Saved {len(stages)} pipeline images to {folder}")
        except Exception as e:
            rospy.logwarn(f"[{self.node_name}] Could not save stages to {folder}: {e}")

    def preprocess_image(self, obtained_image):
        if self.ai_thresholds_received:
            obtained_image = self.ai.apply(
                image=obtained_image,
                lower_threshold=self.anti_instagram_thresholds["lower"],
                higher_threshold=self.anti_instagram_thresholds["higher"]
            )

        if self.cuda_enabled:
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(obtained_image)
        else:
            gpu_image = obtained_image

        height_original, width_original = gpu_image.shape[0:2]
        img_size = (self._img_size[1], self._img_size[0])
        if img_size[0] != width_original or img_size[1] != height_original:
            if self.cuda_enabled:
                gpu_image = cv2.cuda.resize(gpu_image, img_size,
                                            interpolation=cv2.INTER_NEAREST)
            else:
                gpu_image = cv2.resize(gpu_image, img_size,
                                       interpolation=cv2.INTER_NEAREST)

        bgr_img = gpu_image[self._top_cutoff:, :, :]

        if self._traffic_mode.value == "LHT":
            bgr_img = np.fliplr(bgr_img)

        height, _ = bgr_img.shape[:2]
        bgr_img = bgr_img[0: height - self._bottom_cutoff, :]

        return bgr_img

    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    node = IntersectionTypeDetectorNode(node_name="intersection_type_detector_node")
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()
