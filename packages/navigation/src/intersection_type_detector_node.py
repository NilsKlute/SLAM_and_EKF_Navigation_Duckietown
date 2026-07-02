#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from cv_bridge import CvBridge
import numpy as np
import cv2
import rospy
from duckietown_msgs.msg import AprilTagsWithInfos, FSMState, TurnIDandType, BoolStamped, AntiInstagramThresholds
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int16, Int64MultiArray  # Imports msg
from scipy.stats import linregress
import os
from sklearn.cluster import DBSCAN
from dt_computer_vision.anti_instagram import AntiInstagram


class IntersectionTypeDetectorNode(DTROS):
    def __init__(self, node_name):
        super(IntersectionTypeDetectorNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True
        )

        # Save the name of the node
        self.node_name = node_name
        self.turn_type = -1
        rospy.loginfo(f"[{self.node_name}] Initializing.")
        self.bridge = CvBridge()
        self._veh = rospy.get_param("~veh")
        self._img_size = rospy.get_param("~img_size", None)
        self._top_cutoff = rospy.get_param("~top_cutoff", None)
        self._colors = DTParam("~colors", None)

        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

        self._traffic_mode = DTParam(f"/{self._veh}/behavior/traffic_mode", None)

        self.ai_thresholds_received = False
        self.anti_instagram_thresholds = dict()
        self.ai = AntiInstagram()

        self.last_call = time.time()

        # Setup publishers
        self.pub_stop_sign = rospy.Publisher(
            "~stop_sign_intersection_detected",
            BoolStamped,
            queue_size=1,
            latch=True
        )

        self.pub_traffic_light = rospy.Publisher(
            "~traffic_light_intersection_detected",
            BoolStamped,
            queue_size=1,
            latch=True
        )

        # Setup subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )

        self.sub_thresholds = rospy.Subscriber(
            "~thresholds", AntiInstagramThresholds, self.thresholds_cb, queue_size=1
        )

        self.pub_topic_avail_turns = rospy.Publisher("~available_turns", Int64MultiArray, queue_size=1)

        self.pub_debug_image = rospy.Publisher("~debug/clusters/compressed", CompressedImage, queue_size=1)

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.loginfo("Using CUDA GPU for line detection.")
            self.cuda_enabled = True
        else:
            self.loginfo("Using the CPU for line detection.")
            self.cuda_enabled = False

        rospy.loginfo(f"[{self.node_name}] Initialzed.")


    def image_cb(self, image_msg):
            
            if (time.time() - self.last_call) < 1:
                return
            
            self.last_call = time.time()

            try:
                obtained_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            except ValueError as e:
                self.logerr(f"Could not decode image: {e}")
                return
            
            # Perform color correction
            if self.ai_thresholds_received:
                obtained_image = self.ai.apply(
                    image = obtained_image,
                    lower_threshold = self.anti_instagram_thresholds["lower"],
                    higher_threshold = self.anti_instagram_thresholds["higher"]
                )

            if self.cuda_enabled:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(obtained_image)
            else:
                gpu_image = obtained_image

            # Resize the gpu_image to the desired dimensions
            height_original, width_original = gpu_image.shape[0:2]
            img_size = (self._img_size[1], self._img_size[0])
            if img_size[0] != width_original or img_size[1] != height_original:
                if self.cuda_enabled:
                    gpu_image = cv2.cuda.resize(gpu_image, img_size, interpolation=cv2.INTER_NEAREST)
                else:
                    gpu_image = cv2.resize(gpu_image, img_size, interpolation=cv2.INTER_NEAREST)

            bgr_img = gpu_image[self._top_cutoff :, :, :]

            # mirror the bgr_img if left-hand traffic mode is set
            if self._traffic_mode.value == "LHT":
                bgr_img = np.fliplr(bgr_img)
            
            height, width = bgr_img.shape[:2]
            bgr_img = bgr_img[0 : height - 50, :]
                
            # Convert from BGR to RGB (for plotting) and BGR to HSV (for masking)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            
            # Create masks for both ends of the red spectrum and combine them
            mask1 = cv2.inRange(hsv_img, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv_img, self.lower_red2, self.upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply the mask to isolate red pixels on the original RGB image
            masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=red_mask)

            pixel_coords = np.argwhere(red_mask > 0)

            if len(pixel_coords) == 0:
                print("No red pixels found to cluster!")
            else:
                # Flip columns to get standard (x, y) coordinates for clustering/plotting
                pixel_coords = np.flip(pixel_coords, axis=1)

                # 2. Apply DBSCAN
                # eps: max distance between two samples to be considered in the same neighborhood
                # min_samples: minimum number of pixels to form a cluster core
                db = DBSCAN(eps=15, min_samples=15).fit(pixel_coords)
                labels = db.labels_

                # Number of clusters found (excluding noise labeled as -1)
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"Found {num_clusters} distinct red cluster(s).")

                

                # Plot each valid cluster with a unique color
                unique_labels = set(labels)
                for k in unique_labels:
                    if k == -1:
                        continue
                    class_member_mask = (labels == k)
                    cluster_pixels = pixel_coords[class_member_mask]
                    
                    # Calculate center of the cluster for tracking
                    centroid = cluster_pixels.mean(axis=0)
                    

                

                # --- CONFIGURATION PARAMETERS ---
                # A perfectly horizontal line has a slope of 0. 
                # Due to perspective warp, we allow a small threshold (e.g., 0.35).
                STEEPNESS_THRESHOLD = 0.25

                # We need the image width to handle fallback checks if no middle line is present
                # width = bgr_img.shape[1]

                # Variables to store our classified structural roles
                has_left = False
                has_middle = False
                has_right = False

                # Dictionary to hold the center X coordinate of identified components
                cluster_summary = {}

                unique_labels = set(labels)
                for k in unique_labels:
                    if k == -1:
                        continue  # Skip noise
                        
                    # Extract pixels belonging to this specific cluster
                    class_member_mask = (labels == k)
                    cluster_pixels = pixel_coords[class_member_mask]
                    
                    # Extract X and Y arrays for line fitting
                    x_coords = cluster_pixels[:, 0]
                    y_coords = cluster_pixels[:, 1]
                    
                    # Ensure we have enough points to calculate a line
                    if len(x_coords) > 5:
                        # Fit line: y = slope * x + intercept
                        slope, intercept, r_value, p_value, std_err = linregress(x_coords, y_coords)
                        centroid_x = x_coords.mean()
                        
                        # Check if it fits our horizontal condition
                        is_horizontal = abs(slope) < STEEPNESS_THRESHOLD
                        
                        cluster_summary[k] = {
                            'centroid_x': centroid_x,
                            'is_horizontal': is_horizontal,
                            'slope': slope
                        }

                # --- CLASSIFICATION LOGIC ---

                # Step 1: Search for the Middle Lane using the horizontal rule
                middle_cluster_id = None
                for k, data in cluster_summary.items():
                    if data['is_horizontal']:
                        has_middle = True
                        middle_cluster_id = k
                        break  # Found our anchor point

                # Step 2: Differentiate Left and Right relative to the Middle Lane anchor
                if has_middle:
                    middle_x = cluster_summary[middle_cluster_id]['centroid_x']
                    
                    for k, data in cluster_summary.items():
                        if k == middle_cluster_id:
                            continue
                        # If it sits to the left of the horizontal middle line
                        if data['centroid_x'] < middle_x:
                            has_left = True
                            print("its left")
                        # If it sits to the right of the horizontal middle line
                        elif data['centroid_x'] > middle_x:
                            has_right = True

                else:
                    # Fallback Option: If no horizontal line is found, it's a T-junction missing a straight option
                    # Split the screen down the middle to classify remaining lines as Left or Right
                    print("[Info] No middle horizontal stop line detected.")
                    for k, data in cluster_summary.items():
                        if data['centroid_x'] < (width / 2.0):
                            has_left = True
                        else:
                            has_right = True

                # Step 3: Encode available turns as [0, 1, 2] -> [Left, Middle, Right]
                available_turns = []
                if has_left: available_turns.append(0)
                if has_middle: available_turns.append(1)
                if has_right: available_turns.append(2)

                # --- Print Structured Output ---
                rospy.loginfo(f"--- Slope-Refined Decision Matrix ---")
                rospy.loginfo(f"Clusters Evaluated: {len(cluster_summary)}")
                for k, data in cluster_summary.items():
                    line_type = "Horizontal (Middle)" if data['is_horizontal'] else "Steep/Angled (Side)"
                    rospy.loginfo(f"  * Cluster {k}: Center X={data['centroid_x']:.1f}, Slope={data['slope']:.3f} -> {line_type}")
                rospy.loginfo(f"Encoded Output     : {available_turns}")

                avail_turns_msg = Int64MultiArray()
                avail_turns_msg.data = available_turns

                self.pub_topic_avail_turns.publish(avail_turns_msg)

                # Debug image: faint background with cluster pixels overlaid
                debug_img = (bgr_img.astype(np.float32) * 0.25).astype(np.uint8)
                cluster_colors = [
                    (0, 0, 255), (0, 255, 0), (255, 0, 0),
                    (0, 255, 255), (255, 0, 255), (255, 255, 0),
                ]
                for k in set(labels):
                    if k == -1:
                        continue
                    color = cluster_colors[k % len(cluster_colors)]
                    pts = pixel_coords[labels == k]
                    debug_img[pts[:, 1], pts[:, 0]] = color

                debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                debug_msg.header = image_msg.header
                self.pub_debug_image.publish(debug_msg)
            
    
    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    # Create the NodeName object
    node = IntersectionTypeDetectorNode(node_name="intersection_type_detector_node")

    # Setup proper shutdown behavior
    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    rospy.spin()
