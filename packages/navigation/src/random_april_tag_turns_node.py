#!/usr/bin/env python3
import time
import rospy
from std_msgs.msg import Int16, Int64MultiArray, Int64
from duckietown.dtros import DTROS, NodeType


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

        # Setup subscribers
        self.sub_topic_tag = rospy.Subscriber("~available_turns", Int64MultiArray, self.decide_cb, queue_size=1)
    
        self.sub_directional_cmd = rospy.Subscriber("~directional_cmd", Int64, self.dir_cmd_cb, queue_size=1)
        

        rospy.loginfo(f"[{self.node_name}] Initialzed.")

    def dir_cmd_cb(self, turn_type_msg):
        self.turn_type = turn_type_msg.data

    def decide_cb(self, avail_turns_msg):


        if self.turn_type == -1:
            rospy.loginfo_throttle(20, f"[{self.node_name}] We havn't received the planners decision yet")
            return
        
        avail_turns = avail_turns_msg.data

        if len(avail_turns) == 0:
            rospy.loginfo("No turns detected by the Intersection Detection Node")
            return

        turn_decision_msg = Int16()
        if self.turn_type in avail_turns:
            turn_decision_msg.data = self.turn_type

        else:
            turn_decision_msg.data = 1 if 1 in avail_turns else avail_turns[0]


        self.pub_turn_type.publish(turn_decision_msg)
        rospy.loginfo(f"[{self.node_name}] We decided on turn ID {turn_decision_msg.data}")
        self.turn_type = -1


    def setupParameter(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)
        return value

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    node = RandomAprilTagTurnsNode(node_name="random_april_tag_turns_node")
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()
