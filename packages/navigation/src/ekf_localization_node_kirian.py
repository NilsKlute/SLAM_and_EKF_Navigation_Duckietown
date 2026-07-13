#!/usr/bin/env python3

import cv2
import os
import rospy
import tempfile
import numpy as np
import tf
from contextlib import contextmanager
from multiprocessing import Lock
from typing import Optional

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.camera.types import Rectifier
from dt_apriltags import Detector
from turbojpeg import TurboJPEG
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance
from navigation.ekf import *
from navigation.odometry_utils import delta_phi, get_odometry
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown.utils.image.ros import compressed_imgmsg_to_rgb, rgb_to_compressed_imgmsg

from navigation.BEV_SLAM import *
from navigation.clustered_graph import (
    cluster_nodes, build_directed_edges, visualize_clustered_graph
)

import matplotlib
matplotlib.use("Agg")  # headless — required inside a ROS node
import matplotlib.pyplot as plt
from std_msgs.msg import Header
from navigation.tile_graph import *
#    TILE_SIZE, analyze_tile_traversals, classify_tiles, plot_street_graph


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle, Arc

def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi


@contextmanager
def capture_c_stdio():
    """Capture C-level stdout/stderr (fd 1 & 2) for the duration of the block.

    The dt_apriltags C library prints messages such as
    'Error, more than one new minimum found.' via printf directly to the OS
    file descriptors, which Python-level logging / rospy throttling cannot
    intercept. This redirects fd 1 and 2 into a temp buffer (so the raw spam
    never reaches the console) and, after the block, stores what was printed in
    the yielded one-element list so the caller can inspect it.
    """
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved_out, saved_err = os.dup(1), os.dup(2)
    captured = [""]
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        yield captured
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tmp.seek(0)
        captured[0] = tmp.read().decode("utf-8", "replace")
        tmp.close()


class EKFLocalizationNode(DTROS):
    """
    Publisher: ~/estimate_pose (:obj:`PoseWithCovarianceStamped`)

    Subscribers:
        ~/image/compressed (:obj:`CompressedImage`):
            compressed image
    """
    right_tick_prev: Optional[int]
    left_tick_prev: Optional[int]
    delta_phi_left: float
    delta_phi_right: float


    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(EKFLocalizationNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.loginfo("Initializing...")
        # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")
        self.no_predict = rospy.get_param("~no_predict", False)
        self.no_update = rospy.get_param("~no_update", False)
        self.sim = rospy.get_param("~test_sim", False)
        self.right_wheel_mutex = Lock()
        self.left_wheel_mutex = Lock()
        self.gt_pose = None
        self.latest_img = None

        # in __init__, near your other BEV_SLAM state:
        self.log_file_path = "/data/ekf_position_log.txt"
        self.log_file = open(self.log_file_path, "a")

        self.gt_log_file_path = "/data/gt_position_log.txt"
        self.gt_log_file = open(self.gt_log_file_path, "a")

        self.corrected_log_file_path = "/data/ekf_position_log_smoothed.txt"
        rospy.on_shutdown(lambda: self.log_file.close())
        rospy.on_shutdown(lambda: self.gt_log_file.close())

        # Init the parameters
        self.resetParameters()

        #BEV_SLAM
        self.camera_model = None
        self.mapx = None
        self.mapy = None
        self.jpeg = TurboJPEG()
        self.homography = None
        self.projector = None
        self.bev_slam = None

        # nominal R and L, you may change these if needed:

        self.R = 0.0318  # meters, default value of wheel radius
        self.baseline = 0.11  # meters, default value of baseline for DB21
        self.camera_model = None
        self.rectifier = None
        self.rect_camera_K = None
        self.jpeg = TurboJPEG()
        self.mapx = None
        self.mapy = None
        self.homography = None
        self.projector = None
        self.bev_slam = None

        q_0 = np.array([
            rospy.get_param("~x_0", 0.0),
            rospy.get_param("~y_0", 0.0),
            rospy.get_param("~theta_0", 0.0),
        ])

        P_0 = np.array([
            [ rospy.get_param("~P_0_xx", 0.0), 0.0, 0.0 ],
            [ 0.0, rospy.get_param("~P_0_yy", 0.0), 0.0 ],
            [ 0.0, 0.0, rospy.get_param("~P_0_tt", 0.0) ],
        ])


        Q = np.array([
            [ rospy.get_param("~Q_dX", 0.0), 0.0 ],
            [ 0.0, rospy.get_param("~Q_dT", 0.0) ],
        ])

        R = np.array([
            [ rospy.get_param("~R_rr", 0.0), 0.0 ],
            [ 0.0, rospy.get_param("~R_tt", 0.0) ],
        ])
        self.ekf = EKF(q_0, P_0, Q, R)

        map_file = rospy.get_param("~map", None)
        if map_file is None:
            rospy.logerr("No map provided")
            rospy.signal_shutdown("No map provided")

        self.map = {int(id): np.array(value["position"]) for id, value   in map_file.items()}

        self.apriltag_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25
        )

        self.gt_trajectory = []
        self.ekf_trajectory = []

        self.pub_trajectory_plot = rospy.Publisher(
            f"/{self.veh}/ekf_localization_node/trajectory_plot/compressed",
            CompressedImage, queue_size=1, latch=True)

        self.pub_street_graph_plot = rospy.Publisher(
            f"/{self.veh}/ekf_localization_node/street_graph_plot/compressed",
            CompressedImage, queue_size=1, latch=True)

        # Defining subscribers:
        rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            buff_size=10000000,
            queue_size=1,
        )
    

        # Wheel encoder subscriber:
        left_encoder_topic = f"/{self.veh}/left_wheel_encoder_driver_node/tick"
        rospy.Subscriber(left_encoder_topic,
                         WheelEncoderStamped,
                         self.cbLeftEncoder)

        right_encoder_topic = f"/{self.veh}/right_wheel_encoder_driver_node/tick"
        rospy.Subscriber(right_encoder_topic,
                         WheelEncoderStamped,
                         self.cbRightEncoder)

        self.sub_gt_pose = rospy.Subscriber(
            f"/{self.veh}/duckiematrix_interface_node/state",
            Odometry,
            self.cbGTPose,
            queue_size=1,
        )

        self.pub_detections = rospy.Publisher(
            f"/{self.veh}/detections/image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
            dt_help="Camera image with tag publishes superimposed",
            latch=True
        )

        self.pub_pose_covariance = rospy.Publisher(
            f"/{self.veh}/ekf_localization_node/pose",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
            latch=True
        )

        self.pub_landmark_markers = rospy.Publisher(
            f"/{self.veh}/map_markers",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.pub_clustered_graph_compare = rospy.Publisher(
            f"/{self.veh}/ekf_localization_node/clustered_graph_compare/compressed",
            CompressedImage, queue_size=1, latch=True)

        # Get the steering gain (omega_max) from the calibration file
        # It defines the maximum omega used to scale normalized steering command
        kinematics_calib = self.read_params_from_calibration_file()
        self.omega_max = kinematics_calib.get("omega_max", 2.0)


        # Need to sleep for a bit for the publisher to register with master
        rospy.sleep(0.5)
        self.publish_landmarks([])
        self.publish_pose()

        self.sub_camera_info = rospy.Subscriber(
            f"/{self.veh}/camera_node/camera_info",
            CameraInfo,
            self.cb_info,
            queue_size=1,
        )

        # we will do prediction at a fixed frequency rather than asynchronously
        # when the encoder data arrives
        rospy.Timer(rospy.Duration(1.0/10.0), self.doPredict)


    @staticmethod
    def _fig_to_compressed_imgmsg(fig):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        msg = CompressedImage()
        msg.header = Header(stamp=rospy.Time.now())
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_bgr)[1]).tobytes()
        return msg

    def _build_trajectory_comparison_figure(self, smooth_traj, ax=None):
        """
        Plot ground truth, EKF forward, and RTS-smoothed trajectory.
        
        Parameters
        ----------
        smooth_traj : np.ndarray
            Smoothed trajectory (N, 2).
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, a new figure and axis are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axis used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        if len(self.gt_trajectory) > 0:
            gt = np.array(self.gt_trajectory)
            ax.plot(gt[:, 0], gt[:, 1], color='green', linewidth=1.5, label='Ground truth')
        if len(self.ekf_trajectory) > 0:
            ekf = np.array(self.ekf_trajectory)
            ax.plot(ekf[:, 0], ekf[:, 1], color='red', linewidth=1.0,
                    linestyle='--', label='EKF (forward)')
        if smooth_traj.shape[0] > 0:
            ax.plot(smooth_traj[:, 0], smooth_traj[:, 1], color='blue',
                    linewidth=1.5, label='RTS-smoothed')

        ax.set_aspect('equal')
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("GT vs EKF vs RTS-smoothed trajectory")
        ax.legend(loc='best', fontsize=8)
        
        # Grid lines every 0.6m (tile size) to match tile maps
        TILE_SIZE = 0.6
        
        # Get data bounds from all trajectories
        all_data = []
        if len(self.gt_trajectory) > 0:
            all_data.append(np.array(self.gt_trajectory))
        if len(self.ekf_trajectory) > 0:
            all_data.append(np.array(self.ekf_trajectory))
        if smooth_traj.shape[0] > 0:
            all_data.append(smooth_traj)
        
        if all_data:
            all_data = np.vstack(all_data)
            
            # Calculate tile-aligned grid boundaries
            x_min = np.floor(all_data[:, 0].min() / TILE_SIZE) * TILE_SIZE
            x_max = np.ceil(all_data[:, 0].max() / TILE_SIZE) * TILE_SIZE
            y_min = np.floor(all_data[:, 1].min() / TILE_SIZE) * TILE_SIZE
            y_max = np.ceil(all_data[:, 1].max() / TILE_SIZE) * TILE_SIZE
            
            # Create ticks at every 0.6m
            x_ticks = np.arange(x_min, x_max + TILE_SIZE, TILE_SIZE)
            y_ticks = np.arange(y_min, y_max + TILE_SIZE, TILE_SIZE)
            
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)

        return fig, ax

    def publish_corrected_trajectory_and_map(self):
        smooth_traj = self.ekf.rts_smooth()
        if len(smooth_traj) < 2:
            return

        self.save_smoothed_trajectory(smooth_traj)

        # --------------------------------------------
        # 1. Extract probabilistic tile traversals
        # --------------------------------------------
        TILE_TYPES = [
            "empty", "N-S", "E-W", "NE", "ES", "SW", "WN",
            "NES", "ESW", "SWN", "WNE", "4-way"
        ]
        TILE_SIZE = 0.6

        tile_probabilities, uncertainty, vote_counts, total_visits = analyze_tile_traversals_tile_types(
            smooth_traj, tile_size=TILE_SIZE, tile_types=TILE_TYPES
        )

        observed_tiles = {pos: max(probs, key=probs.get) for pos, probs in tile_probabilities.items()}

        final_types, intersection_directions, updated_observed = propagate_constraints(
            vote_counts=vote_counts,
            total_visits=total_visits,
            observed_tiles=observed_tiles,
            max_iterations=25,
            damping=0.4,
            verbose=False,
        )

        bp_classification, bp_tile_counts = bp_to_street_graph_inputs(final_types)

        # --------------------------------------------
        # 2. Build the 2×2 combined figure
        # --------------------------------------------
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # --- Trajectory comparison (top‑left) ---
        ax_traj = fig.add_subplot(gs[0, 0])
        _, _ = self._build_trajectory_comparison_figure(smooth_traj, ax=ax_traj)

        # --- Tile probabilities (top‑right) ---
        ax_prob = fig.add_subplot(gs[0, 1])
        plot_tile_probabilities(tile_probabilities, uncertainty,
                                tile_size=TILE_SIZE, ax=ax_prob)

        # --- Globally consistent tile types (bottom‑left) ---
        ax_types = fig.add_subplot(gs[1, 0])
        plot_tile_types(final_types, updated_observed,
                        tile_size=TILE_SIZE, ax=ax_types)

        # --- Street graph (bottom‑right) ---
        ax_graph = fig.add_subplot(gs[1, 1])
        plot_street_graph(bp_classification, bp_tile_counts,
                        tile_size=TILE_SIZE, min_events=1, ax=ax_graph)

        # --------------------------------------------
        # 3. Publish the combined figure
        # --------------------------------------------
        self.pub_street_graph_plot.publish(self._fig_to_compressed_imgmsg(fig))
        plt.close(fig)

    #BEV_SLAM
    def read_params_from_calibration_file(self):
        """
        Reads the saved parameters from `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`
        or uses the default values if the file doesn't exist. Adjusts the ROS parameters for the
        node with the new values.
        """

        def readFile(fname):
            with open(fname, "r") as in_file:
                try:
                    return yaml.load(in_file, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    self.logfatal("YAML syntax error. File: %s fname. Exc: %s" % (fname, exc))
                    return None

        # Check file existence
        cali_file_folder = "/data/config/calibrations/kinematics/"
        fname = cali_file_folder + self.veh + ".yaml"
        # Use the default values from the config folder if a robot-specific file does not exist.
        if not os.path.isfile(fname):
            fname = cali_file_folder + "default.yaml"
            self.logwarn("Kinematic calibration %s not found! Using default instead." % fname)
            return readFile(fname)
        else:
            return readFile(fname)


    def cbGTPose(self, odom_msg):
        q = odom_msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        p = odom_msg.pose.pose.position
        self.gt_pose = [p.x, p.y, yaw]
        self.gt_log_file.write(f"{p.x:.4f},{p.y:.4f},{yaw:.4f}\n")
        self.gt_log_file.flush()

    def cbLeftEncoder(self, encoder_msg):
        """
        Wheel encoder callback
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """
        with self.left_wheel_mutex:
            # initializing ticks to stored absolute value
            if self.left_tick_prev is None:
                self.left_tick_prev = encoder_msg.data
                return

            left_ticks_curr = encoder_msg.data
            # running the DeltaPhi() function copied from the notebooks to calculate rotations
            delta_phi_left = delta_phi(
                left_ticks_curr, self.left_tick_prev, encoder_msg.resolution
            )
            if delta_phi_left == 0:
                return
            self.left_tick_prev = left_ticks_curr
            self.delta_phi_left += delta_phi_left

    def cbRightEncoder(self, encoder_msg):
        """
        Wheel encoder callback, the rotation of the wheel.
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """

        with self.right_wheel_mutex:
            if self.right_tick_prev is None:
                self.right_tick_prev = encoder_msg.data
                return

            right_ticks_curr = encoder_msg.data

            # calculate rotation of right wheel
            delta_phi_right = delta_phi(
                right_ticks_curr, self.right_tick_prev, encoder_msg.resolution
            )
            if delta_phi_right == 0:
                return
            self.right_tick_prev = right_ticks_curr
            self.delta_phi_right += delta_phi_right


    def doPredict(self, event=None):
        if self.delta_phi_right != 0 or self.delta_phi_left != 0:
            with self.left_wheel_mutex:
                with self.right_wheel_mutex:
                    dX, dT = get_odometry(
                        self.R,
                        self.baseline,
                        self.delta_phi_left,
                        self.delta_phi_right
                    )
                    self.ekf.predict(dX, dT)
                    self.delta_phi_left = 0
                    self.delta_phi_right = 0
        self.doUpdate()
        # Always publish the pose at 10 Hz, even when the vision update
        # is unavailable (no camera model / no image yet)
        self.publish_pose()


    def cb_info(self, msg):
        rospy.loginfo_throttle(2.0, "Camera info message received.")
        H, W = msg.height, msg.width

        d_arr = np.array(msg.D, dtype=float)
        if len(d_arr) != 5:
            self.logwarn_throttle(2.0, f"Camera D has {len(d_arr)} coefficients, expected 5 — padding with zeros")
            d_padded = np.zeros(5)
            d_padded[:min(len(d_arr), 5)] = d_arr[:5]
            d_arr = d_padded

        try:
            self.camera_model = CameraModel(
                width=W,
                height=H,
                K=np.reshape(msg.K, (3, 3)),
                D=d_arr,
                P=np.reshape(msg.P, (3, 4)),
            )
        except Exception as e:
            self.logerr_throttle(2.0, f"CameraModel construction failed: {e} — staying subscribed, will retry")
            return

        # Only stop listening once we have a valid camera model.
        try:
            self.sub_camera_info.shutdown()
        except BaseException:
            pass
        self.loginfo("Camera model built from camera_info.")

        self.rectifier = Rectifier(self.camera_model)
        self.rect_camera_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_model.K, self.camera_model.D, (W, H), 0.0
        )

        self.homography = self.load_extrinsics()
        if self.homography is not None and self.camera_model is not None:
            self.camera_model.H = self.homography
            self.projector = GroundProjector(self.camera_model)
            self.bev_slam = Bev_slam(self.veh, self.rectifier, self.projector)
        else:
            self.logwarn("No extrinsic calibration found — BEV_SLAM disabled, EKF continues")


    
    def cb_image(self, image_msg):
        self.latest_img = image_msg
        #print("cb_image...")
        if self.bev_slam is not None:
            BEV_map = self.bev_slam.run(image_msg,self.ekf.q.copy() , self.ekf.P.copy())

    def load_extrinsics(self) -> Union[Homography, None]:
        """
        Loads the homography matrix from the extrinsic calibration file.

        Returns:
            :obj:`Homography`: the loaded homography matrix

        """
        # load extrinsic calibration
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(
                f"Can't find calibration file: {cali_file}\n Using default calibration instead.",
                "warn",
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = "Found no calibration file ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        try:
            self.H: Homography = HomographyToolkit.load_from_disk(
                cali_file, return_date=False
            )  # type: ignore
            return self.H.reshape((3, 3))
        except Exception as e:
            msg = f"Error in parsing calibration file {cali_file}:\n{e}"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

    def doUpdate(self):
        if self.no_update:
            return False
        if self.camera_model is None:
            return False
        if self.latest_img is None:
            return
        
        # Decompress the image
        image_rgb = compressed_imgmsg_to_rgb(self.latest_img)

        rect_image = self.rectifier.rectify(image_rgb, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale for AprilTag detection
        rect_image_gray = cv2.cvtColor(rect_image, cv2.COLOR_RGB2GRAY)
        #image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        fx = self.camera_model.K[0, 0]; fy = self.camera_model.K[1, 1]
        cx = self.camera_model.K[0, 2]; cy = self.camera_model.K[1, 2]
        camera_params = [fx, fy, cx, cy]
        tag_size = 0.065

        # The C pose solver prints "Error, more than one new minimum found."
        # per tag at ~10 Hz. Capture that raw C output (rospy throttling can't
        # touch a printf) and, only if the ambiguity message actually appeared,
        # surface a single throttled warning instead.
        with capture_c_stdio() as c_out:
            detections = self.apriltag_detector.detect(
                rect_image_gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=tag_size
            )
        if "more than one new minimum" in c_out[0]:
            rospy.logwarn_throttle(
                5.0,
                f"[ekf_localization_node] AprilTag pose solver reported ambiguity"
            )
        # Schneller Sanity-Check direkt im doUpdate, VOR dem EKF-Zeug:
        #print(f"raw detections: {len(detections)}")
        #   for d in detections:
            #print(f"  id={d.tag_id}, margin={d.decision_margin:.1f}, in_map={d.tag_id in self.map}")

        tag_seen_and_matched = False
        for detection in detections:
            tag_id = detection.tag_id
            if tag_id not in self.map:
                continue
            tag_seen_and_matched = True

            tag_position = self.map[tag_id]
            tag_x, tag_y = tag_position[0], tag_position[1]

            if self.gt_pose is None:
                self.gt_pose = [0, 0, 0]
            dx = tag_x - self.gt_pose[0]
            dy = tag_y - self.gt_pose[1]
            sim_range_estimate = np.linalg.norm([dx, dy])
            sim_bearing = wrap_angle(np.arctan2(dy, dx) - self.gt_pose[2])

            t = detection.pose_t
            range_estimate = np.linalg.norm(t)
            bearing = wrap_angle(-np.arctan2(t[0, 0], t[2, 0]))

            if self.sim:
                range_estimate = sim_range_estimate
                bearing = sim_bearing

            self.ekf.update([range_estimate, bearing], [tag_x, tag_y])

        ids = [det.tag_id for det in detections]
        self.publish_landmarks(ids)
        self.publish_detections(rect_image_gray, detections, self.latest_img.header)

        self.ekf.finalize_step()

        x, y, theta = self.ekf.q
        self.log_file.write(f"{x:.4f},{y:.4f},{theta:.4f}\n")
        self.log_file.flush()

        self.ekf_trajectory.append([x, y, theta])
        if self.gt_pose is not None:
            self.gt_trajectory.append(list(self.gt_pose))

        if tag_seen_and_matched:
            self.publish_corrected_trajectory_and_map()

        return tag_seen_and_matched
    
    def publish_pose(self, header=None):

        pose_cov= PoseWithCovariance()
        pose_cov.pose.position.x = self.ekf.q[0]
        pose_cov.pose.position.y = self.ekf.q[1]
        pose_cov.pose.position.z = 0.0

        pose_cov.pose.orientation.x = 0.0
        pose_cov.pose.orientation.y = 0.0
        pose_cov.pose.orientation.z = np.sin(self.ekf.q[2] / 2)
        pose_cov.pose.orientation.w = np.cos(self.ekf.q[2] / 2)

        pose_cov.covariance = [
            self.ekf.P[0, 0], self.ekf.P[0, 1], 0.0, 0.0, 0.0, self.ekf.P[0, 2] ,
            self.ekf.P[1, 0], self.ekf.P[1, 1], 0.0, 0.0, 0.0, self.ekf.P[1, 2] ,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            self.ekf.P[2, 0], self.ekf.P[2, 1], 0.0, 0.0, 0.0, self.ekf.P[2, 2]
        ]
        odom_msg = Odometry()
        if header is None:
            odom_msg.header.stamp = rospy.Time.now()
        else:
            odom_msg.header = header
        odom_msg.header.frame_id = "map"
        odom_msg.pose = pose_cov

        self.pub_pose_covariance.publish(odom_msg)


    def resetParameters(self):

        self.log("Encoder data resetting")
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

    def publish_landmarks(self, detection_ids):
        # publishes the whole map as markers and colors the ones that were detected

        marker_array = MarkerArray()
        for i, (landmark_id, position) in enumerate(self.map.items()):


            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "landmarks"
            m.id = int(landmark_id)
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = position[0]
            m.pose.position.y = position[1]
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0

            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            if landmark_id in detection_ids:
                m.color.r = 0.0
                m.color.g = 0.0
                m.color.b = 1.0
                m.color.a = 1.0
            else:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 1.0


            marker_array.markers.append(m)
        self.pub_landmark_markers.publish(marker_array)

    def publish_detections(self, img, detections, header):

        # get a color buffer from the BW image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # draw each tag
        for detection in detections:
            for idx in range(len(detection.corners)):
                cv2.line(
                    img,
                    tuple(detection.corners[idx - 1, :].astype(int)),
                    tuple(detection.corners[idx, :].astype(int)),
                    (0, 255, 0),
                )
            # draw the tag ID
            cv2.putText(
                img,
                str(detection.tag_id),
                org=(detection.corners[0, 0].astype(int) + 10, detection.corners[0, 1].astype(int) + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
            )
        # pack image into a message
        img_msg = CompressedImage()
        img_msg.header.stamp = header.stamp
        img_msg.header.frame_id = header.frame_id
        img_msg.format = "jpeg"
        img_msg.data = self.jpeg.encode(img)
        # ---
        self.pub_detections.publish(img_msg)

    def save_smoothed_trajectory(self, smooth_traj):
        np.savetxt(
            self.corrected_log_file_path,
            smooth_traj,
            fmt="%.6f",
            delimiter=",",
        )


    def publish_clustered_graph_comparison(self, smooth_traj):
        if len(self.ekf_trajectory) < 2 or smooth_traj.shape[0] < 2:
            return

        try:
            kf_ekf = self._trajectory_to_keyframes(self.ekf_trajectory)
            mean_nodes_ekf = cluster_nodes(kf_ekf)
            edges_ekf = build_directed_edges(mean_nodes_ekf)
            fig_ekf, _ = visualize_clustered_graph(
                mean_nodes_ekf, edges_ekf, show=False,
                title="EKF (forward) clustered graph")

            kf_smooth = self._trajectory_to_keyframes(smooth_traj)
            mean_nodes_smooth = cluster_nodes(kf_smooth)
            edges_smooth = build_directed_edges(mean_nodes_smooth)
            fig_smooth, _ = visualize_clustered_graph(
                mean_nodes_smooth, edges_smooth, show=False,
                title="RTS-smoothed clustered graph")
        except Exception as e:
            rospy.logwarn(f"Clustered graph comparison skipped: {e}")
            return

        img_ekf = self._fig_to_bgr_array(fig_ekf)
        plt.close(fig_ekf)
        img_smooth = self._fig_to_bgr_array(fig_smooth)
        plt.close(fig_smooth)

        # Figures can render at slightly different pixel heights even at the
        # same figsize/dpi (legend wrapping, aspect-equal padding, etc.) —
        # normalize to the shorter one before hconcat, which requires equal heights.
        h = min(img_ekf.shape[0], img_smooth.shape[0])

        def resize_to_height(img, target_h):
            scale = target_h / img.shape[0]
            w = int(round(img.shape[1] * scale))
            return cv2.resize(img, (w, target_h))

        img_ekf_r = resize_to_height(img_ekf, h)
        img_smooth_r = resize_to_height(img_smooth, h)
        combined = cv2.hconcat([img_ekf_r, img_smooth_r])

        self.pub_clustered_graph_compare.publish(
        self._bgr_array_to_compressed_imgmsg(combined))

    @staticmethod
    def _fig_to_bgr_array(fig):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    @staticmethod
    def _bgr_array_to_compressed_imgmsg(img_bgr):
        msg = CompressedImage()
        msg.header = Header(stamp=rospy.Time.now())
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_bgr)[1]).tobytes()
        return msg

    def _fig_to_compressed_imgmsg(self, fig):
        return self._bgr_array_to_compressed_imgmsg(self._fig_to_bgr_array(fig))

    @staticmethod
    def _trajectory_to_keyframes(traj):
        """[x,y,theta] rows -> [id, x, y, theta] rows, matching read_nodes()'s format."""
        arr = np.array(traj, dtype=float)
        ids = np.arange(1, len(arr) + 1).reshape(-1, 1)
        return np.hstack([ids, arr])


if __name__ == "__main__":
    # Initialize the node
    encoder_localization_node = EKFLocalizationNode(node_name="ekf_localization_node")
    # Keep it spinning
    rospy.spin()
