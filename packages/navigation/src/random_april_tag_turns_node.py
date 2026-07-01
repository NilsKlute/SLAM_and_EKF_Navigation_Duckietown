#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy

import rospy
from duckietown_msgs.msg import AprilTagsWithInfos, FSMState, TurnIDandType, BoolStamped
from std_msgs.msg import Int16, Int64MultiArray, Int64  # Imports msg
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType


class RandomAprilTagTurnsNode(DTROS):
    def __init__(self, node_name):
        super(RandomAprilTagTurnsNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True)

        # Save the name of the node
        self.node_name = node_name
        self.turn_type = -1
        rospy.loginfo(f"[{self.node_name}] Initializing.")

        # Setup publishers
        self.pub_turn_type = rospy.Publisher("~turn_type", Int16, queue_size=1, latch=True)
        #self.pub_id_and_type = rospy.Publisher("~turn_id_and_type", TurnIDandType, queue_size=1, latch=True)
        self.pub_intersection_go = rospy.Publisher("~intersection_go", BoolStamped, queue_size=1)

        # Setup subscribers
        self.sub_topic_tag = rospy.Subscriber("~available_turns", Int64MultiArray, self.decide_cb, queue_size=1)
    
        self.sub_directional_cmd = rospy.Subscriber("~directional_cmd", Int64, self.dir_cmd_cb, queue_size=1)
        

        rospy.loginfo(f"[{self.node_name}] Initialzed.")

    def dir_cmd_cb(self, turn_type_msg):
        self.turn_type == turn_type_msg.data

    def decide_cb(self, avail_turns_msg):
        
        time.sleep(1)

        if self.turn_type == -1:
            rospy.loginfo(f"[{self.node_name}] We havn't received the planners decision yet")
            return
        
        avail_turns = avail_turns_msg.data

        turn_decision_msg = Int16()
        if self.turn_type in avail_turns:
            turn_decision_msg.data = self.turn_type

        else:
            turn_decision_msg.data = 1 if 1 in avail_turns else avail_turns[0]


        self.pub_turn_type.publish(turn_decision_msg)
        rospy.loginfo(f"[{self.node_name}] We decided on turn ID {turn_decision_msg.data}")

        go_msg = BoolStamped()
        go_msg.header.stamp = rospy.Time.now()
        go_msg.data = True
        self.pub_intersection_go.publish(go_msg)


    def setupParameter(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)  # Write to parameter server for transparancy
        # rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    # Initialize the node with rospy
    # rospy.init_node("random_april_tag_turns_node", anonymous=False)

    # Create the NodeName object
    node = RandomAprilTagTurnsNode(node_name="random_april_tag_turns_node")

    # Setup proper shutdown behavior
    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    rospy.spin()
