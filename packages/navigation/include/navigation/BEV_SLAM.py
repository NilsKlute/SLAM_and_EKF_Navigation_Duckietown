#!/usr/bin/env python3
from typing import Tuple

import numpy as np
import cv2
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.ground_projection.types import GroundPoint
import os
import cv2
import yaml
from typing import Union
import time
import rospy
import numpy as np

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.camera.types import Rectifier
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.camera.homography import Homography, HomographyToolkit

from turbojpeg import TurboJPEG
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown.utils.image.ros import compressed_imgmsg_to_rgb, rgb_to_compressed_imgmsg

from dt_computer_vision.camera.types import Pixel  # adjust import if name differs in your version

# Get the steering gain (omega_max) from the calibration file
# It defines the maximum omega used to scale normalized steering command


#!/usr/bin/env python3

import time
import cv2
import numpy as np

from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.ground_projection.types import GroundPoint

from duckietown.utils.image.ros import compressed_imgmsg_to_rgb


class Bev_slam:

    def __init__(self, vehicle,rectifier:Rectifier,projector: GroundProjector):

        self.veh = vehicle
        self.projector = projector
        self.rectifier = rectifier

        # ---------- keyframes ----------
        self.keyframes = [] 

        # ---------- global map ----------
        self.global_map = None
        self.weight_map = None

        # pixels / meter
        self.resolution = 1000.0

        # map size in pixels
        self.canvas_size = 5000

        # horizon estimation
        try:
            far_away_point_ground = GroundPoint(
                x=10000000,
                y=0
            )

            normalized_vector = projector.ground2vector(
                far_away_point_ground
            )

            far_away_point_image = (
                projector.camera.vector2pixel(
                    normalized_vector
                )
            )

            self.horizon = (far_away_point_image.as_integers()[0])

        except Exception:
            self.horizon = 270

        print("BEV horizon =", self.horizon)
        self.pub_global_map = rospy.Publisher(
            f"/{self.veh}/bev_slam/global_map/compressed",
            CompressedImage,
            queue_size=1
        )
        self.pub_bev_debug = rospy.Publisher(
            f"/{self.veh}/bev_slam/bev_debug/compressed",
            CompressedImage,
            queue_size=1
        )

        self.jpeg = TurboJPEG()

    def save_keyframe(self, bev, q, P):

        self.keyframes.append({
            "timestamp": time.time(),
            "bev": bev.copy(),
            "q": q.copy(),
            "P": P.copy()
        })

    def initialize_canvas(self):

        if self.global_map is not None:
            return

        self.global_map = np.zeros(
            (
                self.canvas_size,
                self.canvas_size,
                3
            ),
            dtype=np.float32
        )

        self.weight_map = np.zeros((self.canvas_size,self.canvas_size),dtype=np.float32)

    def pose_to_canvas(self, q):

        x = q[0]
        y = q[1]

        cx = self.canvas_size // 2
        cy = self.canvas_size // 2

        px = cx + x * self.resolution
        py = cy - y * self.resolution

        return px, py

    def build_transform(self, bev, q):

        theta = q[2]

        h, w = bev.shape[:2]

        # robot roughly sits at bottom center
        robot_x = w / 2
        robot_y = h

        px, py = self.pose_to_canvas(q)

        R = cv2.getRotationMatrix2D(
            (robot_x, robot_y),
            np.degrees(theta),
            1.0
        )

        T = np.eye(3)

        T[:2, :] = R

        T[0, 2] += px - robot_x
        T[1, 2] += py - robot_y

        return T

    def warp_to_global(self, bev, q):

        T = self.build_transform(bev,q)

        warped = cv2.warpPerspective(
            bev,
            T,
            (
                self.canvas_size,
                self.canvas_size
            )
        )

        return warped

    def blend_into_map(self, warped, P):

        sigma = np.sqrt(
            P[0, 0] +
            P[1, 1]
        )

        weight = 1.0 / (sigma + 1e-3)

        mask = (
            warped.sum(axis=2) > 0
        ).astype(np.float32)

        self.global_map += (
            warped.astype(np.float32)
            * weight
        )

        self.weight_map += (
            mask * weight
        )

    def get_map(self):

        result = self.global_map.copy()

        valid = self.weight_map > 0

        result[valid] /= (self.weight_map[valid, None])

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result


    def create_bev2(self,image_rgb,
        forward_range=(0.00, 2),   # meters in front of the robot (near, far) - START SMALL
        lateral_range=(-1, 1),   # meters left(+)/right(-) of robot center
        out_size=(1000, 1000),         # (width, height) in pixels
    ):
        """
        Reproject a rectified-camera RGB image onto the ground plane (BEV),
        using inverse warping so we never sample ground points beyond the
        calibration horizon.

        Ground frame convention (Duckietown): x = forward, y = left.
        Adjust the signs below if your GroundPoint convention differs.
        """
        out_w, out_h = out_size
        x_near, x_far = forward_range
        y_right, y_left = lateral_range  # y_right is the more-negative bound

        # Build the per-output-pixel ground coordinates.
        # row 0 = far away (top of BEV image), row out_h-1 = near robot (bottom)
        # col 0 = left side, col out_w-1 = right side  (flip if it looks mirrored)
        xs = np.linspace(x_far, x_near, out_h)      # forward distance per row
        ys = np.linspace(y_left, y_right, out_w)    # lateral distance per col

        # Cache the maps the first time (or if size/range changes), since this
        # only depends on calibration + chosen extents, not on the image content.
        cache_key = (forward_range, lateral_range, out_size)
        if getattr(self, "_bev_map_cache_key", None) != cache_key:
            mapx = np.zeros((out_h, out_w), dtype=np.float32)
            mapy = np.zeros((out_h, out_w), dtype=np.float32)

            for row, x in enumerate(xs):
                for col, y in enumerate(ys):
                    try:
                        ground_pt = GroundPoint(x=float(x), y=float(y))
                        vec = self.projector.ground2vector(ground_pt)
                        pix = self.projector.camera.vector2pixel(vec)
                        mapx[row, col] = pix.x
                        mapy[row, col] = pix.y
                    except Exception:
                        # Point projects somewhere degenerate; mark invalid.
                        mapx[row, col] = -1
                        mapy[row, col] = -1

            self._bev_mapx = mapx
            self._bev_mapy = mapy
            self._bev_map_cache_key = cache_key

        # Rectify the incoming image first - GroundProjector assumes rectified pixels.
        rectified = self.rectifier.rectify(image_rgb)

        """valid = (self._bev_mapx >= 0) & (self._bev_mapy >= 0)
        print("valid fraction:", valid.mean())
        print("mapx range:", self._bev_mapx.min(), self._bev_mapx.max())
        print("mapy range:", self._bev_mapy.min(), self._bev_mapy.max())
        print("image shape:", rectified.shape)"""
        
        gp = GroundPoint(x=0.2, y=0.0)
        vec = self.projector.ground2vector(gp)
        pix = self.projector.camera.vector2pixel(vec)
        print(vec, pix)
        bev = cv2.remap(
            rectified,
            self._bev_mapx,
            self._bev_mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return bev

    

    # TODO: calibrate this against a known real-world distance in the WARP_DST
    # rectangle (e.g. measure the lane width/length the points correspond to)
    # so the BEV pixel grid lines up with self.resolution (px/meter) used
    # elsewhere in the map pipeline.
    METERS_PER_DST_PIXEL = None  # fill in once calibrated

    def create_bev3(self, image_rgb, out_size=(600, 480)):
        WARP_SRC = np.float32([
            [199, 296],
            [479, 298],
            [600, 476],
            [66, 476]
        ])
        WARP_DST = np.float32([
            [100, 0],
            [540, 0],
            [600, 480],
            [100, 480],
        ])
        WARP_MATRIX = cv2.getPerspectiveTransform(WARP_SRC, WARP_DST)
        """
        Reproject a rectified-camera RGB image onto the ground plane using a
        fixed 4-point perspective homography (WARP_MATRIX), instead of the
        per-pixel GroundProjector loop. Stable everywhere inside WARP_SRC's
        field of view; no near-horizon blow-up since it's one fixed 3x3
        matrix rather than an inverse ground projection.
        """
        rectified = self.rectifier.rectify(image_rgb)

        w, h = out_size
        bev = cv2.warpPerspective(
            rectified,
            WARP_MATRIX,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    # A square region in the ORIGINAL image (pick 4 corners that form a square
    # in pixel space — e.g. a region of interest you want to inspect/warp)
        
    def square_to_trapezoid(self,image: np.ndarray, out_size=(600, 480)) -> np.ndarray:
        SQUARE_SRC = np.float32([
            [200, 200],   # top-left
            [400, 200],   # top-right
            [400, 400],   # bottom-right
            [200, 400],   # bottom-left
        ])

        # The trapezoid it should map TO in the destination (BEV) image.
        # Narrower top / wider bottom (or vice versa) gives the trapezoid shape.
        TRAPEZOID_DST = np.float32([
            [150, 0],     # top-left   (narrower)
            [450, 0],     # top-right
            [350, 480],   # bottom-right (wider)
            [250, 480],   # bottom-left
        ])

        WARP_MATRIX = cv2.getPerspectiveTransform(SQUARE_SRC, TRAPEZOID_DST)
        WARP_MATRIX_INV = cv2.getPerspectiveTransform(TRAPEZOID_DST, SQUARE_SRC)
        
        w, h = out_size
        return cv2.warpPerspective(
            image,
            WARP_MATRIX,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    def create_bev(self, image_rgb, out_size=(600, 600)):
        rectified = self.rectifier.rectify(image_rgb)
        bev = self.square_to_trapezoid(rectified, out_size=out_size)
        bev = cv2.rotate(bev, cv2.ROTATE_90_CLOCKWISE)
        return bev

    def run(self, image_msg, ekf_q, ekf_P):
        self.initialize_canvas()

        image_rgb = compressed_imgmsg_to_rgb(image_msg)
        image_rgb = np.array(image_rgb)

        bev = self.create_bev(image_rgb)
        self.publish_bev_debug(bev)   # <-- debug output, isolated from the rest of the pipeline

        self.save_keyframe(bev, ekf_q, ekf_P)

        warped = self.warp_to_global(bev, ekf_q)
        self.blend_into_map(warped, ekf_P)
        self.publish_map()
        return self.get_map()

    def publish_map(self):
        #print("publish_map")
        if self.global_map is None:
            return

        img = self.get_map()

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = self.jpeg.encode(img)

        self.pub_global_map.publish(msg)

    def publish_bev_debug(self, bev):
        if bev is None or bev.size == 0:
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = self.jpeg.encode(bev)
        self.pub_bev_debug.publish(msg)