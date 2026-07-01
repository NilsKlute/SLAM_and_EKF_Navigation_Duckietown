#!/usr/bin/env python3
import numpy as np

import rospy
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, FSMState, LanePose, SegmentList, StopLineReading
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String, Float64MultiArray, Int64
from duckietown_msgs.srv import SetFSMState, SetFSMStateResponse, ChangePattern

class TargetSelectionNode(DTROS):
    """
    Add Target Selection Description

    Args:
        node_name (:obj:`str'): a unique, descriptive name for the node that ROS will use

    Configuration:

    Subscribers:

    Publishers:

    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(TargetSelectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True)

        self.srv_target_selection = rospy.Service("~select_target", SetFSMState, self.cb_target_selection)
        
        self.pub_target_specified = rospy.Publisher("~target_specified", BoolStamped, queue_size=1, latch=True)
        self.pub_target_location = rospy.Publisher("~target_location", String, queue_size=1, latch=True)

    def cb_target_selection(self, target_msg):

        specified_msg = BoolStamped()
        specified_msg.header.stamp = rospy.Time.now()
        specified_msg.data = True

        # Report Target Specification to FSM
        self.pub_target_specified.publish(specified_msg)

        send_target_msg = String()
        send_target_msg.data = target_msg.state

        # Transfer Target Label to Planner
        self.pub_target_location.publish(send_target_msg)

        return SetFSMStateResponse()

        

if __name__ == "__main__":
    target_selection_node = TargetSelectionNode(node_name="target_selection_node")
    rospy.spin()
