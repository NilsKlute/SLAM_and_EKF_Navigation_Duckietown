#!/usr/bin/env python3
import math
import numpy as np
import rospy
from duckietown_msgs.msg import BoolStamped, \
    WheelEncoderStamped, \
    Twist2DStamped, \
    StopLineReading, \
    FSMState
from std_msgs.msg import Int16
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import message_filters
from tf import transformations as tr
from duckietown.dtros import DTROS, NodeType, TopicType
import geometry as g
import time

class UnicornIntersectionNode(DTROS):
    def __init__(self, node_name):
        super(UnicornIntersectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL,
            fsm_controlled=True)

        self.node_name = node_name
        self.internal_state = "READY"
        self.turn_type_received = False
        self.stop_line_pose_received = False

        self.setupParams()

        self.goal_poses = {
            "left": self.dictionary_pose_to_geometry(self.canonical_goal_pose_left),
            "right": self.dictionary_pose_to_geometry(self.canonical_goal_pose_right),
            "straight": self.dictionary_pose_to_geometry(self.canonical_goal_pose_straight)
        }

        self.reference_trajectory = []
        self.turn_type = -1
        self.stop_line_pose = Pose2D()
        self.fsm_state = None

        self.sub_turn_type = rospy.Subscriber("~turn_type", Int16, self.cbTurnType)
        self.sub_encoder_left = message_filters.Subscriber("~left_wheel_encoder_driver_node/tick", WheelEncoderStamped)
        self.sub_encoder_right = message_filters.Subscriber("~right_wheel_encoder_driver_node/tick", WheelEncoderStamped)
        self.sub_stop_line_reading = rospy.Subscriber("~stop_line_reading", StopLineReading, self.cbStopLineReading)
        self.sub_fsm_mode = rospy.Subscriber("~mode", FSMState, self.cbsetFSM)

        self.pub_int_done = rospy.Publisher("~intersection_done", BoolStamped, queue_size=1)
        self.pub_trans_done = rospy.Publisher("~transition_done", BoolStamped, queue_size=1)
        self.car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL)
        self.reference_trajectory_pub = rospy.Publisher(
            "~reference_trajectory",
            Odometry,
            queue_size=self.num_waypoints,
        )
        self.pub_intersection_go = rospy.Publisher("~intersection_go", BoolStamped, queue_size=1)

        self.hc_start_time = None

        self.ts_encoders = message_filters.ApproximateTimeSynchronizer(
            [self.sub_encoder_left, self.sub_encoder_right], 1, 1
        )
        self.ts_encoders.registerCallback(self.cb_ts_encoders_hardcoded)

        self.params_update = rospy.Timer(rospy.Duration.from_sec(1.0), self.updateParams)
        self.reset_odometry()
        self.log("Initialized unicorn intersection node")

    def cbStopLineReading(self, msg):
        if self.stop_line_pose_received:
            return

        if msg.at_stop_line:
            self.stop_line_pose = msg.stop_pose
            self.stop_line_pose_received = True
            rospy.loginfo(f"[unicorn_intersection_node] Received stop line pose: {self.stop_line_pose}")
            self.check_if_go()

    def check_if_go(self):
        if (self.stop_line_pose_received and self.turn_type_received and self.internal_state == "READY"):
            rospy.loginfo("[unicorn_intersection_node] We have what we need, calculating reference trajectory")
            self.reference_trajectory = self.calculate_goal_trajectory()
            rospy.loginfo(f"[unicorn_intersection_node] Reference trajectory calculated: {self.reference_trajectory}")
            self.reset_odometry()
            self.internal_state = "EXECUTING"

            go_msg = BoolStamped()
            go_msg.header.stamp = rospy.Time.now()
            go_msg.data = True
            self.pub_intersection_go.publish(go_msg)

        else:
            rospy.loginfo(f"[unicorn_intersection_node] We don't have what we need yet: "
                      f"stop_line received: {self.stop_line_pose_received} " 
                      f"turn_type_received: {self.turn_type_received} "
                      f"internal_state:{self.internal_state} ")

    def calculate_goal_trajectory(self):
        g_stop_pose = self.ros_pose_to_geometry(self.stop_line_pose)

        if self.turn_type == 0:
            canonical_goal_pose = self.goal_poses['left']
        elif self.turn_type == 1:
            canonical_goal_pose = self.goal_poses['straight']
        elif self.turn_type == 2:
            canonical_goal_pose = self.goal_poses['right']
        else:
            rospy.logerr("[unicorn_intersection_node] invalid turn type")

        robot_frame_goal_pose = g.SE2.multiply(g.SE2.inverse(g_stop_pose), canonical_goal_pose)

        p, d = g.translation_angle_from_SE2(robot_frame_goal_pose)
        rospy.loginfo(f"goal_pose in robot frame: position {p}, angle {d}")

        vel = g.SE2.algebra_from_group(robot_frame_goal_pose)
        alphas = [x / self.num_waypoints for x in range(1, self.num_waypoints + 1)]
        waypoints = []
        directions = []
        for alpha in alphas:
            rel = g.SE2.group_from_algebra(vel * alpha)
            position, direction = g.translation_angle_from_SE2(rel)
            waypoints.append(position)
            directions.append(direction)

        if self.visualization:
            self.visualize_trajectory(waypoints, directions)
        return waypoints

    def visualize_trajectory(self,waypoints, directions):
        for i in range(len(waypoints)):
            p = Odometry()
            p.header.frame_id = "map"
            p.header.stamp = rospy.Time.now()

            p.pose.pose.position.x = waypoints[i][0]
            p.pose.pose.position.y = waypoints[i][1]
            p.pose.pose.position.z = 0

            p.pose.pose.orientation.x = 0
            p.pose.pose.orientation.y = 0
            p.pose.pose.orientation.z = np.sin(directions[i] / 2)
            p.pose.pose.orientation.w = np.cos(directions[i] / 2)

            self.reference_trajectory_pub.publish(p)

    def reset_odometry(self):
        self.left_encoder_last = None
        self.right_encoder_last = None
        self.encoders_timestamp_last = None
        self.encoders_timestamp_last_local = None
        self.timestamp = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.q = [0.0, 0.0, 0.0, 1.0]
        self.tv = 0.0
        self.rv = 0.0
        self.ticks_per_meter = 656.0
        self.wheelbase = 0.108
        self.iter_ = 0

    # ================================================================
    # HARDCODED EXPERIMENT CALLBACK
    # ================================================================
    def cb_ts_encoders_hardcoded(self, _left_encoder, _right_encoder):
        if self.internal_state != "EXECUTING":
            return

        now = rospy.get_time()
        if self.hc_start_time is None:
            self.hc_start_time = now

        elapsed = now - self.hc_start_time

        if self.turn_type == 0:
            omega    = self.HC_OMEGA_LEFT
            duration = self.HC_DURATION_LEFT
        elif self.turn_type == 1:
            omega    = self.HC_OMEGA_STRAIGHT
            duration = self.HC_DURATION_STRAIGHT
        else:  # turn_type == 2
            omega    = self.HC_OMEGA_RIGHT
            duration = self.HC_DURATION_RIGHT

        if elapsed < duration:
            cmd = Twist2DStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.v     = self.HC_SPEED
            cmd.omega = omega
            self.car_cmd.publish(cmd)
        else:

            self.internal_state      = "READY"
            self.stop_line_pose_received = False
            self.turn_type_received  = False
            self.hc_start_time       = None

            msg_done = BoolStamped()
            msg_done.data = True
            self.pub_int_done.publish(msg_done)
            self.reset_odometry()
            rospy.loginfo("[unicorn_intersection_node] hardcoded intersection complete")
            #self.pub_trans_done.publish(msg_done)
            rospy.loginfo("[unicorn_intersection_node] transition to lane following complete")

    # ================================================================
    # ORIGINAL WAYPOINT / DEAD-RECKONING CALLBACK (kept for reference)
    # ================================================================
    def cb_ts_encoders(self, left_encoder, right_encoder):
        if self.internal_state != "EXECUTING": 
            return

        timestamp_now = rospy.get_time()

        timestamp = (left_encoder.header.stamp.to_sec() + right_encoder.header.stamp.to_sec()) / 2

        if not self.left_encoder_last:
            self.left_encoder_last = left_encoder
            self.right_encoder_last = right_encoder
            self.encoders_timestamp_last = timestamp
            self.encoders_timestamp_last_local = timestamp_now
            return

        dtl = left_encoder.header.stamp - self.left_encoder_last.header.stamp
        dtr = right_encoder.header.stamp - self.right_encoder_last.header.stamp
        if dtl.to_sec() < 0 or dtr.to_sec() < 0:
            self.loginfo("Ignoring stale encoder message")
            return

        left_distance = (left_encoder.data - self.left_encoder_last.data) / self.ticks_per_meter
        right_distance = (right_encoder.data - self.right_encoder_last.data) / self.ticks_per_meter

        distance = (left_distance + right_distance) / 2
        dyaw = (right_distance - left_distance) / self.wheelbase

        dt = max(timestamp - self.encoders_timestamp_last, 1e-6)

        self.tv = distance / dt
        self.rv = dyaw / dt

        dist = self.tv * dt
        dyaw = self.rv * dt

        self.yaw = self.angle_clamp(self.yaw + dyaw)
        self.x = self.x + dist * math.cos(self.yaw)
        self.y = self.y + dist * math.sin(self.yaw)
        self.q = tr.quaternion_from_euler(0, 0, self.yaw)
        self.timestamp = timestamp

        self.left_encoder_last = left_encoder
        self.right_encoder_last = right_encoder
        self.encoders_timestamp_last = timestamp
        self.encoders_timestamp_last_local = timestamp_now

        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()
        car_control_msg.v = self.speed
        car_control_msg.omega = self.compute_omega(self.reference_trajectory[self.iter_], self.x, self.y, self.yaw, dt)
        self.car_cmd.publish(car_control_msg)

        if self.check_point(np.array([self.x, self.y]), self.reference_trajectory[self.iter_]):
            self.iter_ += 1
            if self.iter_ == self.num_waypoints:
                self.internal_state = "READY"
                self.stop_line_pose_received = False
                self.turn_type_received = False
                msg_done = BoolStamped()
                msg_done.data = True
                self.pub_int_done.publish(msg_done)
                self.reset_odometry()
                rospy.loginfo("[unicorn_intersection_node] intersection navigation complete")
                time.sleep(1.5)
                self.pub_trans_done.publish(msg_done)
                rospy.loginfo("[unicorn_intersection_node] transition to lane following complete")





    @staticmethod
    def dictionary_pose_to_geometry(dict_param):
        return g.SE2_from_xytheta([dict_param['x'], dict_param['y'], dict_param['theta']])

    @staticmethod
    def ros_pose_to_geometry(ros_pose):
        return g.SE2_from_xytheta([ros_pose.x, ros_pose.y, ros_pose.theta])

    def cbTurnType(self, msg):
        if self.turn_type_received:
            return

        self.turn_type = msg.data
        self.turn_type_received = True
        rospy.loginfo(f"[unicorn_intersection_node] Received turn type: {self.turn_type} ")
        self.check_if_go()

    def setupParams(self):
        self.use_stop_pose = self.setupParam("~use_stop_pose", False)
        self.num_waypoints = self.setupParam("~num_waypoints", 2)
        self.visualization = self.setupParam("~visualization", True)
        default_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0 }
        self.canonical_goal_pose_right = self.setupParam("~canonical_goal_pose_right", default_pose)
        self.canonical_goal_pose_left = self.setupParam("~canonical_goal_pose_left", default_pose)
        self.canonical_goal_pose_straight = self.setupParam("~canonical_goal_pose_straight", default_pose)
        self.speed = self.setupParam("~speed", 0.30)

        # Hardcoded experiment params (cb_ts_encoders_hardcoded)
        self.HC_SPEED             = self.setupParam("~hc_speed",             0.25)
        self.HC_OMEGA_LEFT        = self.setupParam("~hc_omega_left",        0.75)
        self.HC_OMEGA_STRAIGHT    = self.setupParam("~hc_omega_straight",    0.0)
        self.HC_OMEGA_RIGHT       = self.setupParam("~hc_omega_right",      -1.7)
        self.HC_DURATION_LEFT     = self.setupParam("~hc_duration_left",     4.0)
        self.HC_DURATION_STRAIGHT = self.setupParam("~hc_duration_straight", 4.0)
        self.HC_DURATION_RIGHT    = self.setupParam("~hc_duration_right",    1.6)

    def updateParams(self, event):
        pass

    def setupParam(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)  # Write to parameter server for transparancy
        rospy.loginfo(f"[{self.node_name}] {param_name} = {value} ")
        return value

    def onShutdown(self):
        rospy.loginfo("[UnicornIntersectionNode] Shutdown.")

    @staticmethod
    def angle_clamp(theta):
        if theta > 2 * math.pi:
            return theta - 2 * math.pi
        elif theta < -2 * math.pi:
            return theta + 2 * math.pi
        else:
            return theta      

    def compute_omega(self,targetxy,x,y,current,dt):
        factor = 1 # PARAM 
        target_yaw = np.arctan2( (targetxy[1] - y),(targetxy[0]- x) )
        omega = factor* ((target_yaw - current))

        return omega

    def check_point(self, current_point, target_point):
        dist = np.sqrt((current_point[0] - target_point[0])**2 + (current_point[1] - target_point[1])**2)
        # tighter threshold at the final waypoint to land accurately in the exit lane
        if self.iter_ == (self.num_waypoints - 1):
            return dist < 0.08
        return dist < 0.15
        
    def cbsetFSM(self, fsm_msg):
        self.fsm_state = fsm_msg.state

if __name__ == "__main__":
    unicorn_intersection_node = UnicornIntersectionNode(node_name="unicorn_intersection_node")
    rospy.on_shutdown(unicorn_intersection_node.onShutdown)
    rospy.spin()