#!/usr/bin/env python3
import numpy as np
import time
import rospy
import random
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, FSMState, LanePose, SegmentList, StopLineReading
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String, Float64MultiArray, Int64

class GraphPlannerNode(DTROS):
    """
    Add Planner Description

    Args:
        node_name (:obj:`str'): a unique, descriptive name for the node that ROS will use

    Configuration:

    Subscribers:e lane filter

    Publishers:

    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(GraphPlannerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True)

        # Initialize the parameters
        """self.stop_distance = DTParam("~stop_distance", param_type=ParamType.FLOAT)
        self.min_segs = DTParam("~min_segs", param_type=ParamType.INT)
        self.off_time = DTParam("~off_time", param_type=ParamType.FLOAT)
        self.max_y = DTParam("~max_y", param_type=ParamType.FLOAT)"""

        self.target = None

        self.decision_published = False


        ## publishers and subscribers
        self.sub_target = rospy.Subscriber("~target_location", String, self.cb_init_navigation)

        # TODO this needs to be mapped on the localization node (either EKF or SLAM)
        #self.sub_position = rospy.Subscriber("~lane_pose", Float64MultiArray, self.cb_localize)

        self.sub_stop_line_filter = rospy.Subscriber("~at_stop_line", BoolStamped, self.cb_directional_cmd)
        self.sub_stop_line_filter = rospy.Subscriber("~intersection_go", BoolStamped, self.cb_reset_decided_planning)

        self.pub_arrived_target = rospy.Publisher("~arrived_at_target", BoolStamped, queue_size=1, latch=True)
        self.pub_directional_cmd = rospy.Publisher("~directional_cmd", Int64, queue_size=1)

    def cb_init_navigation(self, target_msg):
        self.target = target_msg.data

    def cb_localize(self, location_msg):
        NotImplemented

    def cb_directional_cmd(self, arrived_at_stop_lane_msg):

        if self.decision_published:
            return

        cmd_msg = Int64()
        # choose random direction
        cmd_msg.data = random.randint(0, 2)

        # wait for april tags turn node to get activated
        time.sleep(0.7)
        rospy.loginfo(f"Planners decision has been made to direction ID {cmd_msg.data}")
        self.pub_directional_cmd.publish(cmd_msg)

        self.decision_published = True

    def cb_reset_decided_planning(self, msg):
        self.decision_published = False



if __name__ == "__main__":
    graph_planner_node = GraphPlannerNode(node_name="graph_planner_node")
    rospy.spin()
