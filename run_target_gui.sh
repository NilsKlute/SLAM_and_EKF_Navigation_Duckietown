#!/usr/bin/env bash
#
# Launch the navigation target-selection GUI on your laptop, joined to the
# robot's ROS master. It runs inside this project's Docker image so the ROS
# message types and the `navigation` package are available without a local ROS
# install.
#
# Adjust the variables below for your setup before running.
set -e

# --- adjust these -----------------------------------------------------------
ROBOT_NAME="myduckiebot"            # "myduckiebot" in simulation, "roboduck" on the real bot
ROBOT_IP="172.17.0.2"               # the robot's CURRENT IP on your network (changes per session!)
LAPTOP_IP="192.168.0.100"           # <-- set this to YOUR laptop's IP on the robot network (ROS_IP)
IMAGE="duckietown/roboduck:ente-amd64"
# ----------------------------------------------------------------------------

xhost +local:root                   # allow the container to use your X display

docker run -it --rm --net=host \
  --add-host ${ROBOT_NAME}.local:${ROBOT_IP} \
  -e DISPLAY=$DISPLAY \
  -e ROS_MASTER_URI=http://${ROBOT_NAME}.local:11311 \
  -e ROS_IP=${LAPTOP_IP} \
  -e VEHICLE_NAME=${ROBOT_NAME} \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ${IMAGE} \
  rosrun navigation target_gui_node.py _veh:=${ROBOT_NAME}
