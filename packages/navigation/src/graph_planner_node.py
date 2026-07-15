#!/usr/bin/env python3
import io
import os
import yaml
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rospy
import math
import heapq
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import BoolStamped, FSMState
from std_msgs.msg import String, Int64, Int32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage


class GraphPlannerNode(DTROS):
    def __init__(self, node_name):
        super(GraphPlannerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True)

        self.target = None
        self.decision_published = False
        self.init_plan = False

        # Intersection-decision debugging
        self.planner_debug = False      # toggled from the GUI
        self._pending_cmd  = None       # computed Int64 decision awaiting manual GO
        self._dbg_current_id = None
        self._dbg_next_id    = None
        self._dbg_diff_deg   = None
        self._dbg_decision   = None

        # --------------------- Params ------------------------------
        self.veh                = rospy.get_param("~veh", rospy.get_namespace().strip("/"))
        self.localization_type  = rospy.get_param("~localization_type", "EKF")

        # Ground-truth localization is only available in simulation (myduckiebot);
        # the real robot (roboduck) has no ground-truth source.
        self.gt_available       = (self.veh == "myduckiebot")
        self.use_ground_truth   = False
        self._last_gt_time      = rospy.Time(0)
        self.use_cached         = rospy.get_param("~use_cached", False)
        self.data_dir           = rospy.get_param("~data_dir", "/data/graph_planner")
        self.dr_overlay         = rospy.get_param("~dr_overlay", False)
        if self.veh == "myduckiebot":
            graph_file_ekf          = rospy.get_param("~graph_file_ekf_myduckiebot")
            graph_file_slam         = rospy.get_param("~graph_file_slam_myduckiebot")
        elif self.veh == "roboduck":
            graph_file_ekf          = rospy.get_param("~graph_file_ekf_roboduck")
            graph_file_slam         = rospy.get_param("~graph_file_ekf_roboduck")

        label_map_file          = rospy.get_param("~label_map_file")

        self.use_clustering     = rospy.get_param("~use_clustering", True)
        spatial_radius          = rospy.get_param("~spatial_radius", 1.8)
        angle_threshold_deg     = rospy.get_param("~angle_threshold_deg", 45.0)
        edge_max_dist              = rospy.get_param("~edge_max_dist", 4.0)
        edge_max_heading_diff      = rospy.get_param("~edge_max_heading_diff_deg", 130.0)
        edge_rescue_dist           = rospy.get_param("~edge_rescue_dist", 4.0)
        edge_cone_threshold_deg    = rospy.get_param("~edge_cone_threshold_deg", 20.0)
        edge_visibility_deg        = rospy.get_param("~edge_visibility_deg", 100.0)
        edge_direction_change_deg  = rospy.get_param("~edge_direction_change_deg", 100.0)

        self.candidate_max_dist         = rospy.get_param("~candidate_max_dist", 1.5)
        self.candidate_k                = rospy.get_param("~candidate_k", 3)
        self.candidate_heading_tol_deg  = rospy.get_param("~candidate_heading_tol_deg", 45.0)
        self.arrival_threshold     = rospy.get_param("~arrival_threshold", 0.5)
        self.arrival_max_angle_diff = rospy.get_param("~arrival_max_angle_diff", 100)
        self.skip_n_nodes          = rospy.get_param("~skip_n_nodes", 2)
        self.turn_angle_thresh_deg = rospy.get_param("~turn_angle_threshold_deg", 35.0)
        self.u_turn_thresh         = np.radians(rospy.get_param("~u_turn_thresh_deg", 120.0))



        # ------------------- Graph loading ------------------------------
        self.label_map = {}
        os.makedirs(self.data_dir, exist_ok=True)
        cached_path = os.path.join(self.data_dir, f"clustered_nodes_{self.localization_type}.txt")

        if self.use_cached and os.path.exists(cached_path):
            self.read_clustered_nodes(cached_path)
            rospy.loginfo(f"[graph_planner] Loaded cached graph from {cached_path}")
        else:
            if self.localization_type == "EKF":
                raw = self.read_keyframes_ekf(graph_file_ekf)
            else:
                raw = self.read_keyframes_slam(graph_file_slam)
            if self.use_clustering:
                self.clustered_nodes = self.cluster_keyframes(raw, spatial_radius, angle_threshold_deg)
                self.write_clustered_nodes(cached_path)
                rospy.loginfo(f"[graph_planner] Built graph with {len(self.clustered_nodes)} clustered nodes"
                              f" (from {len(raw)} keyframes)")
            else:
                self.clustered_nodes = raw
                rospy.loginfo(f"[graph_planner] Clustering DISABLED — using {len(raw)} raw keyframes as nodes")

        self.directed_edges = self.build_directed_edges(
            self.clustered_nodes,
            max_dist=edge_max_dist,
            max_heading_diff_deg=edge_max_heading_diff,
            rescue_dist=edge_rescue_dist,
            cone_threshold_deg=edge_cone_threshold_deg,
            visibility_deg=edge_visibility_deg,
            direction_change_deg=edge_direction_change_deg)
        if rospy.get_param("~edge_prune_shortcuts", False):
            self.directed_edges = self.prune_shortcut_edges(
                self.directed_edges, self.clustered_nodes,
                max_hops=rospy.get_param("~edge_prune_max_hops", 4))
        self.adj_list   = self.build_adjacency_list(self.directed_edges)
        self.nodes_dict = {int(n[0]): n for n in self.clustered_nodes}

        rospy.loginfo(f"[graph_planner] Graph: {len(self.clustered_nodes)} nodes, "
                      f"{len(self.directed_edges)} edges")
        rospy.loginfo(f"[graph_planner] Node IDs: {sorted(self.nodes_dict.keys())}")

        # Label map from YAML (authoritative — overwrites any labels from cache)
        with open(label_map_file) as f:
            self.label_map = yaml.safe_load(f) or {}

        rospy.loginfo(f"[graph_planner] Label map: {self.label_map}")
        for name, nid in self.label_map.items():
            if nid in self.nodes_dict:
                n = self.nodes_dict[nid]
                rospy.loginfo(f"[graph_planner]   '{name}' → node {nid} "
                              f"at ({n[1]:.2f}, {n[2]:.2f}, {np.degrees(n[3]):.0f}°) "
                              f"neighbors: {self.adj_list.get(nid, [])}")
            else:
                rospy.logwarn(f"[graph_planner]   '{name}' → node {nid} NOT IN GRAPH! "
                              f"Valid IDs: {sorted(self.nodes_dict.keys())}")

        self.apriltags = {}
        if self.localization_type == "EKF":
            # AprilTag landmark map (same file the EKF uses) — drawn in the debug
            # image so tag placements can be checked against the graph. Best-effort.
            apriltag_map_file = rospy.get_param("~apriltag_map_file", "")
            self.observed_tags = set()   # tag ids the EKF currently detects
            if apriltag_map_file and os.path.isfile(apriltag_map_file):
                try:
                    with open(apriltag_map_file) as f:
                        tag_map = (yaml.safe_load(f) or {}).get("map", {})
                    self.apriltags = {int(tid): (float(v["position"][0]), float(v["position"][1]))
                                    for tid, v in tag_map.items()}
                    rospy.loginfo(f"[graph_planner] Loaded {len(self.apriltags)} AprilTags for debug overlay")
                except Exception as e:
                    rospy.logwarn(f"[graph_planner] Could not load AprilTag map {apriltag_map_file}: {e}")
            else:
                rospy.logwarn(f"[graph_planner] AprilTag map file not found ('{apriltag_map_file}') — tags not drawn")

        self.plan = None
        self.curr_node = None

        # Dead-reckoning overlay — stores the DR pose transformed into map coords.
        # _dr_origin: DR (x,y,theta) at the moment alignment was first computed.
        # _dr_ekf_ref: EKF map pose at that same moment, used as the reference frame.
        self._latest_dr_pose = None
        self._dr_origin      = None
        self._dr_ekf_ref     = None

        # --------------------- Subscribers ------------------------

        self.sub_target = rospy.Subscriber("~target_location", String, self.cb_init_navigation)

        if self.localization_type == "EKF":
            rospy.Subscriber("ekf_localization_node/pose", Odometry, self.cb_localize_odometry)
        else:
            rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped, self.cb_localize_pose_cov)

        # Ground-truth pose source (simulation only) + GUI toggle to select it.
        if self.gt_available:
            rospy.Subscriber("duckiematrix_interface_node/state", Odometry, self.cb_localize_gt)
        self.sub_use_gt = rospy.Subscriber("~use_ground_truth", BoolStamped, self.cb_use_ground_truth)

        self.sub_stop_line = rospy.Subscriber("~at_stop_line", BoolStamped, self.cb_directional_cmd)
        self.sub_int_done  = rospy.Subscriber("~intersection_done", BoolStamped, self.cb_reset_decided_planning)
        self.sub_fsm_mode  = rospy.Subscriber("fsm_node/mode", FSMState, self.cb_fsm_mode)
        self.sub_detected_tags = rospy.Subscriber("ekf_localization_node/detected_tags", Int32MultiArray, self.cb_detected_tags)
        self.sub_debug_mode = rospy.Subscriber("~debug_mode", BoolStamped, self.cb_debug_mode)
        self.sub_debug_go   = rospy.Subscriber("~debug_go", BoolStamped, self.cb_debug_go)

        # Dead-reckoning pose — comparison overlay in the debug image (dr_overlay: true).
        if self.dr_overlay:
            rospy.Subscriber("deadreckoning_node/odom", Odometry, self.cb_dr_odom)


         # --------------------- Publishers ------------------------

        self.pub_arrived_target  = rospy.Publisher("~arrived_at_target", BoolStamped, queue_size=1, latch=True)
        self.pub_directional_cmd = rospy.Publisher("~directional_cmd", Int64, queue_size=1)
        self.pub_debug_image     = rospy.Publisher("~debug_image/compressed", CompressedImage, queue_size=1)
        self._latest_pose        = (0.0, 0.0, 0.0)
        rospy.Timer(rospy.Duration(0.5), self._debug_image_timer)





    # ---- Localization callbacks ----

    def cb_localize_odometry(self, msg):
        if self.use_ground_truth:      # EKF estimate ignored while GT is selected
            return
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self._localize(x, y, theta)

    def cb_localize_pose_cov(self, msg):
        if self.use_ground_truth:
            return
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self._localize(x, y, theta)

    def cb_localize_gt(self, msg):
        self._last_gt_time = rospy.Time.now()
        if not self.use_ground_truth:  # only drive localization when GT selected
            return
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self._localize(x, y, theta)

    def cb_use_ground_truth(self, msg):
        self.use_ground_truth = bool(msg.data) and self.gt_available
        rospy.loginfo(f"[graph_planner] use_ground_truth = {self.use_ground_truth}")

    def _yaw_from_quaternion(self, q):
        # Z-axis rotation: q = (0, 0, sin(θ/2), cos(θ/2))
        # arctan2(sin(θ/2), cos(θ/2)) = θ/2  →  multiply by 2
        return 2.0 * np.arctan2(q.z, q.w)

    def _localize(self, x, y, theta):

        self._latest_pose = (x, y, theta)

        if self.target is None:
            rospy.loginfo_throttle(30.0, "Localize: no target set yet")
            return

        if self.target not in self.nodes_dict:
            rospy.logerr_throttle(5.0, f"Target node {self.target} not in graph! "
                                       f"Valid IDs: {sorted(self.nodes_dict.keys())}")
            return

        rospy.loginfo_throttle(5.0,
            f"Localize: pos=({x:.2f},{y:.2f},{np.degrees(theta):.0f}°) "
            f"target={self.target} plan={'set' if self.plan else 'None'} "
            f"curr={self.curr_node}")


        if self.curr_node and self.arrived(x, y, self.arrival_threshold, self.arrival_max_angle_diff):
            rospy.loginfo("Target Reached!")
            msg = BoolStamped()
            msg.header.stamp = rospy.Time.now()
            msg.data = True
            self.pub_arrived_target.publish(msg)
            self.plan = None
            self.target = None
            self.init_plan = False
            return


        candidates = self.get_candidate_nodes(
            x, y, theta, max_dist=self.candidate_max_dist, k=self.candidate_k,
            heading_tol_deg=self.candidate_heading_tol_deg)

        # Init Plan after target and first localization are received
        if not self.init_plan and self.plan is None:
            rospy.loginfo("We go into init planning")
            # Right at an intersection the nearby node headings diverge, so the
            # strict heading filter can reject every candidate and we could never
            # start. For initialization only, fall back to the nearest nodes
            # regardless of heading — we just need a valid start node on the graph.
            init_candidates = candidates or self.get_candidate_nodes(
                x, y, theta, max_dist=self.candidate_max_dist,
                k=self.candidate_k, heading_tol_deg=180)
            for candidate in init_candidates:
                self.a_star_planner(candidate, self.target)
                if self.plan is not None and candidate in self.plan:
                    self.curr_node = candidate
                    self.init_plan = True
                    return
            rospy.logwarn_throttle(3.0, "We failed to initialize navigation")
            return

        if not candidates:
            rospy.logwarn_throttle(3.0, "Localization failed: No valid nodes found within search radius.")
            return


        # What if our localization just jumps way ahead on planned path on a node behind intersection?
        # Direction is not yet considered for checking if the candidates are on the plan
        if self.plan is not None:
            for candidate in candidates:
                if candidate in self.plan:
                    self.curr_node = candidate
                    return

        rospy.logwarn_throttle(5.0, f"Off route. Attempting to replan to Target Node {self.target}...")
        path_found = False

        for candidate in candidates:
            self.plan = None
            self.a_star_planner(candidate, self.target)

            if self.plan is not None:
                self.curr_node = candidate
                path_found = True
                rospy.loginfo(f"Replanned from Node {self.curr_node}.")
                break

        if not path_found:
            rospy.logerr_throttle(5.0,
                f"A* Failed: No path from candidates {candidates}. Halting navigation.")
            self.target = None




    # ------------------ Debug visualization -------------------------

    def _debug_image_timer(self, _event=None):
        # Warn if ground truth is selected but no GT pose is arriving — this is
        # why the robot arrow would appear frozen after toggling GT on.
        if self.use_ground_truth:
            age = (rospy.Time.now() - self._last_gt_time).to_sec()
            if age > 1.0:
                rospy.logwarn_throttle(
                    2.0,
                    f"[graph_planner] GROUND TRUTH selected but no pose received on "
                    f"'{self.veh}/duckiematrix_interface_node/state' for {age:.1f}s — "
                    f"is the ground-truth source publishing?")
        rx, ry, rtheta = self._latest_pose
        self._publish_debug_image(rx, ry, rtheta)

        plt.close()

    def _target_label(self):
        if self.target is None:
            return "None"
        reverse = {v: k for k, v in self.label_map.items()}
        return reverse.get(self.target, str(self.target))

    def _publish_debug_image(self, rx, ry, rtheta):
        try:
            fig, ax = plt.subplots(figsize=(14, 14))

            # All graph edges — thin black arrows
            for (id_a, id_b) in self.directed_edges:
                a = self.nodes_dict[id_a]
                b = self.nodes_dict[id_b]
                ax.annotate("", xy=(b[1], b[2]), xytext=(a[1], a[2]),
                            arrowprops=dict(arrowstyle="->", color="black",
                                           lw=5, alpha=0.95, shrinkA=5, shrinkB=5))

            # Planned path — thick gold arrows
            if self.plan is not None and len(self.plan) > 1:
                for i in range(len(self.plan) - 1):
                    a = self.nodes_dict[self.plan[i]]
                    b = self.nodes_dict[self.plan[i + 1]]
                    ax.annotate("", xy=(b[1], b[2]), xytext=(a[1], a[2]),
                                arrowprops=dict(arrowstyle="->", color="gold",
                                               lw=3, alpha=0.9, shrinkA=5, shrinkB=5))

            # All graph nodes — blue dots + heading arrows
            mx = self.clustered_nodes[:, 1]
            my = self.clustered_nodes[:, 2]
            mtheta = self.clustered_nodes[:, 3]
            ax.scatter(mx, my, c='royalblue', s=60, zorder=5,
                       edgecolors='navy', linewidths=1)
            ax.quiver(mx, my, np.cos(mtheta), np.sin(mtheta),
                      color='steelblue', scale=30, width=0.01, alpha=0.5, zorder=6)

            # Node ID labels (skipped for large raw-keyframe graphs — unreadable and slow)
            if len(self.clustered_nodes) <= 300:
                for node in self.clustered_nodes:
                    ax.annotate(str(int(node[0])), (node[1], node[2]),
                                fontsize=12, ha='center', va='bottom', color='navy', zorder=7)

            # AprilTag landmarks (from the EKF map) — verify placement vs graph.
            # Currently-observed tags are highlighted red/larger.
            if self.apriltags:
                for tid, (px, py) in self.apriltags.items():
                    seen = tid in self.observed_tags
                    ax.scatter(px, py, c=('red' if seen else 'purple'),
                               marker='P', s=(250 if seen else 90),
                               zorder=(10 if seen else 8),
                               edgecolors='black', linewidths=3)
                    ax.annotate(str(tid), (px, py), fontsize=14,
                                color=('red' if seen else 'purple'),
                                ha='left', va='bottom', zorder=(10 if seen else 8))

            # Target node — large green circle
            if self.target is not None and self.target in self.nodes_dict:
                tn = self.nodes_dict[self.target]
                ax.scatter(tn[1], tn[2], c='limegreen', s=250, zorder=9,
                           edgecolors='darkgreen', linewidths=2)

            # Current matched node — large red circle
            if self.curr_node is not None and self.curr_node in self.nodes_dict:
                cn = self.nodes_dict[self.curr_node]
                ax.scatter(cn[1], cn[2], c='red', s=250, zorder=9,
                           edgecolors='darkred', linewidths=2)

            # EKF robot pose — red star + heading arrow
            ax.scatter(rx, ry, c='red', s=350, marker='*', zorder=10,
                       edgecolors='darkred', linewidths=1)
            ax.quiver(rx, ry, np.cos(rtheta), np.sin(rtheta),
                      color='red', scale=20, width=0.007, zorder=11)
            ax.annotate("EKF", (rx, ry), fontsize=9, color='red',
                        xytext=(5, 5), textcoords='offset points', zorder=12)

            # Dead-reckoning pose — lime star + heading arrow (for comparison with EKF)
            if self.dr_overlay and self._latest_dr_pose is not None:
                drx, dry, drtheta = self._latest_dr_pose
                ax.scatter(drx, dry, c='lime', s=350, marker='*', zorder=10,
                           edgecolors='darkgreen', linewidths=1)
                ax.quiver(drx, dry, np.cos(drtheta), np.sin(drtheta),
                          color='lime', scale=20, width=0.007, zorder=11)
                ax.annotate("DR", (drx, dry), fontsize=9, color='lime',
                            xytext=(5, -12), textcoords='offset points', zorder=12)
                # Distance between the two estimates
                sep = math.hypot(drx - rx, dry - ry)
                hdiff = abs(np.degrees(self.normalize_angle(drtheta - rtheta)))
                ax.text(0.02, 0.06,
                        f"DR vs EKF: pos_err={sep:.3f}m  hdg_err={hdiff:.1f}°",
                        transform=ax.transAxes, fontsize=9, va='bottom',
                        color='lime',
                        bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.75))

            # --- Intersection-decision debug overlay ---
            if self.planner_debug and self._dbg_next_id is not None \
                    and self._dbg_current_id in self.nodes_dict \
                    and self._dbg_next_id in self.nodes_dict:
                cur = self.nodes_dict[self._dbg_current_id]
                nxt = self.nodes_dict[self._dbg_next_id]
                cx, cy, ctheta = cur[1], cur[2], cur[3]
                thresh = np.radians(self.turn_angle_thresh_deg)
                L = 0.35

                # Lookahead node — orange diamond
                ax.scatter(nxt[1], nxt[2], c='orange', s=320, marker='D', zorder=12,
                           edgecolors='darkorange', linewidths=2)
                # Decision vector current -> lookahead (magenta)
                ax.annotate("", xy=(nxt[1], nxt[2]), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color="magenta",
                                            lw=3, alpha=0.95))
                # Heading ray (solid) + ±threshold decision cone (dashed)
                ax.plot([cx, cx + L*np.cos(ctheta)], [cy, cy + L*np.sin(ctheta)],
                        color='cyan', lw=2, zorder=11)
                for sign in (+1, -1):
                    a = ctheta + sign*thresh
                    ax.plot([cx, cx + L*np.cos(a)], [cy, cy + L*np.sin(a)],
                            color='purple', lw=1.5, ls='--', alpha=0.8, zorder=11)
                ax.annotate(f"{self._dbg_diff_deg:+.0f}°", (cx, cy),
                            fontsize=11, fontweight='bold', color='purple',
                            xytext=(6, 6), textcoords='offset points', zorder=13)

                pending = "YES" if self._pending_cmd is not None else "no"
                ax.text(0.02, 0.90,
                        f"DEBUG ON | thresh ±{self.turn_angle_thresh_deg:.0f}°\n"
                        f"angle {self._dbg_diff_deg:+.0f}° → {self._dbg_decision}\n"
                        f"pending GO: {pending}",
                        transform=ax.transAxes, fontsize=11, va='top', color='purple',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            # Target label in top-left corner
            ax.text(0.02, 0.98, f"Target: {self._target_label()}",
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Graph Planner — Navigation Debug")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='jpeg', dpi=80, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = buf.read()

            # Bypass DTROS FSM switch-gating: this DTPublisher is a no-op
            # whenever the node is OFF for the current state (e.g. joystick
            # control). Debug image should publish regardless of FSM state.
            try:
                self.pub_debug_image.active = True
            except AttributeError:
                pass
            self.pub_debug_image.publish(msg)

        except Exception as e:
            rospy.logwarn_throttle(10.0, f"Debug image render failed: {e}")

    # ---- Graph file readers ----

    def read_keyframes_ekf(self, filename):
        nodes = []
        with open(filename) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                x, y, theta = [float(v) for v in line.split(',')]
                nodes.append([i, x, y, theta])
        return np.array(nodes)

    def read_keyframes_slam(self, filename):
        nodes = []
        with open(filename) as f:
            for line in f:
                if not line.startswith("VERTEX_SE3:QUAT"):
                    continue
                parts = line.split()
                node_id = int(parts[1])
                x, y   = float(parts[2]), float(parts[3])
                # parts[4]=z ignored; parts[5]=qx, parts[6]=qy, parts[7]=qz, parts[8]=qw
                qz, qw = float(parts[7]), float(parts[8])
                theta  = 2.0 * np.arctan2(qz, qw)
                nodes.append([node_id, x, y, theta])
        return np.array(nodes)




    # --------------------- Navigation callbacks -------------------------------

    def cb_init_navigation(self, target_msg):
        label = target_msg.data.strip()
        if label in self.label_map:
            self.target = self.label_map[label]
            rospy.loginfo(f"Target set: '{label}' → node {self.target}")
            return
        # Not a known label — accept a raw node ID typed straight into the GUI.
        try:
            self.target = int(label)
        except ValueError:
            rospy.logerr(f"Target '{label}' is neither a known label nor an integer ID. "
                         f"Known labels: {list(self.label_map.keys())}; "
                         f"valid node IDs: {sorted(self.nodes_dict.keys())}")
            return
        if self.target in self.nodes_dict:
            rospy.loginfo(f"Target set to raw node ID {self.target}")
        else:
            rospy.logwarn(f"Node ID {self.target} not in graph. "
                          f"Valid node IDs: {sorted(self.nodes_dict.keys())}")
        

    def cb_directional_cmd(self, _msg):

        if self.target == None:
            rospy.loginfo_throttle(5, f"[graph_planner_node] cb_directional_cmd: target is not set!")
            return

        if self.decision_published:
            rospy.loginfo_throttle(5, f"We dont need to plan. Decision already published")
            return

        # Debug: a decision is already pending for this intersection; at_stop_line
        # fires repeatedly, so skip to avoid recomputing / re-rendering. Cleared
        # on intersection_done, after the robot has left the stop line.
        if self.planner_debug and self._pending_cmd is not None:
            return

        if self.plan is None:
            rospy.loginfo_throttle(5, f"[graph_planner_node] cb_directional_cmd: We cannot decide on turn! self.plan is: {self.plan}. ")
            return
        
        if self.curr_node is None:
            rospy.loginfo_throttle(5, f"[graph_planner_node] cb_directional_cmd: We cannot decide on turn! We dont have a current node we are localized to.")
            return
        
        if self.curr_node not in self.plan:
            rospy.loginfo_throttle(5, f"cb_directional_cmd: We cannot decide on turn! Our current node is not in our plan.")
            return

        rospy.loginfo("[graph_planner_node] decision making before sleep")
        time.sleep(5)
        rospy.loginfo("[graph_planner_node] decision making after sleep")
        
        cmd_msg = Int64()
        curr_plan_idx = self.plan.index(self.curr_node)
        lookahead_idx = min(curr_plan_idx + self.skip_n_nodes, len(self.plan) - 1)
        next_id    = self.plan[lookahead_idx]
        current_id = self.plan[curr_plan_idx]

        sx, sy, stheta = self.nodes_dict[current_id][1], self.nodes_dict[current_id][2], self.nodes_dict[current_id][3]
        nx, ny = self.nodes_dict[next_id][1], self.nodes_dict[next_id][2]

        vector_angle = np.arctan2(ny - sy, nx - sx)
        diff_deg = np.degrees(self.normalize_angle(vector_angle - stheta))

        if diff_deg > self.turn_angle_thresh_deg:
            cmd_msg.data = 0  # LEFT
            decision = "TURN LEFT"
        elif diff_deg < -self.turn_angle_thresh_deg:
            cmd_msg.data = 2  # RIGHT
            decision = "TURN RIGHT"
        else:
            cmd_msg.data = 1  # STRAIGHT
            decision = "GO STRAIGHT"
        rospy.loginfo(f"[graph_planner_node] Decision: {decision} (angle {diff_deg:+.0f}°, thresh ±{self.turn_angle_thresh_deg:.0f}°)")

        # Store for the debug-image overlay
        self._dbg_current_id = current_id
        self._dbg_next_id    = next_id
        self._dbg_diff_deg   = diff_deg
        self._dbg_decision   = decision
        self._pending_cmd    = cmd_msg

        rospy.loginfo(f"[graph_planner_node] Node ID used for Decision {current_id}")

        if self.planner_debug:
            # Pause: hold the decision, wait for a manual GO from the GUI.
            rospy.loginfo("[graph_planner_node] DEBUG: decision pending — press Intersection GO")
            self._publish_debug_image(*self._latest_pose)
            return

        rospy.Timer(rospy.Duration(0.2), lambda event: self.pub_directional_cmd.publish(cmd_msg), oneshot=True)
        self.decision_published = True

    def cb_reset_decided_planning(self, _msg):
        self.decision_published = False
        self._pending_cmd = None

    def cb_fsm_mode(self, msg):
        # Back to IDLE means the current run is finished/aborted — clear the plan
        # so a stale target/path doesn't carry into the next navigation.
        if msg.state == "IDLE":
            self._reset_planning()

    def cb_detected_tags(self, msg):
        # Currently-observed AprilTag ids from the EKF (empty when none in view).
        self.observed_tags = set(int(i) for i in msg.data)

    def cb_dr_odom(self, msg):
        dr_x     = msg.pose.pose.position.x
        dr_y     = msg.pose.pose.position.y
        q        = msg.pose.pose.orientation
        dr_theta = 2.0 * np.arctan2(q.z, q.w)

        # Align DR to the EKF map frame on first contact (after EKF has settled).
        if self._dr_origin is None:
            ex, ey, etheta = self._latest_pose
            if ex == 0.0 and ey == 0.0:
                return  # EKF hasn't received a real pose yet — wait
            self._dr_origin  = (dr_x, dr_y, dr_theta)
            self._dr_ekf_ref = (ex, ey, etheta)
            rospy.loginfo(
                f"[graph_planner] DR alignment: DR=({dr_x:.3f},{dr_y:.3f},{np.degrees(dr_theta):.1f}°) "
                f"EKF=({ex:.3f},{ey:.3f},{np.degrees(etheta):.1f}°)"
            )

        ox, oy, otheta     = self._dr_origin
        ex, ey, etheta     = self._dr_ekf_ref
        # Displacement in DR's own frame since alignment
        ddx    = dr_x - ox
        ddy    = dr_y - oy
        dtheta = dr_theta - otheta
        # Rotate displacement into EKF map frame (EKF initial heading)
        cos_e, sin_e = np.cos(etheta), np.sin(etheta)
        self._latest_dr_pose = (
            ex + cos_e * ddx - sin_e * ddy,
            ey + sin_e * ddx + cos_e * ddy,
            etheta + dtheta,
        )

    def _reset_planning(self):
        self.plan = None
        self.target = None
        self.curr_node = None
        self.init_plan = False
        self.decision_published = False
        self._pending_cmd = None
        self._dbg_current_id = None
        self._dbg_next_id = None
        self._dbg_diff_deg = None
        self._dbg_decision = None

    def cb_debug_mode(self, msg):
        old_setting = self.planner_debug
        self.planner_debug = msg.data
        if old_setting != self.planner_debug:
            rospy.loginfo(f"[graph_planner_node] planner_debug = {self.planner_debug}")
        if not self.planner_debug:
            self._pending_cmd = None

    def cb_debug_go(self, _msg):
        if not self.planner_debug:
            return
        if self._pending_cmd is None:
            rospy.loginfo("[graph_planner_node] DEBUG GO: no decision pending")
            return
        self.pub_directional_cmd.publish(self._pending_cmd)
        rospy.loginfo(f"[graph_planner_node] DEBUG GO: published decision {self._pending_cmd.data}")
        self.decision_published = True
        self._pending_cmd = None
        self._dbg_current_id = None
        self._dbg_next_id    = None
        self._dbg_diff_deg   = None
        self._dbg_decision   = None






    # -------------------- Helper functions ----------------------------

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def angle_diff(self, a, b):
        return abs(self.normalize_angle(a - b))

    def cluster_keyframes(self, keyframes, spatial_radius=0.2, angle_threshold_deg=45.0):
        visited = np.zeros(len(keyframes), dtype=bool)
        centroids = []
        angle_thresh = np.radians(angle_threshold_deg)
        new_id = 0

        for i in range(len(keyframes)):
            if visited[i]: continue

            x_i, y_i, theta_i = keyframes[i, 1], keyframes[i, 2], keyframes[i, 3]
            cluster_indices = [i]
            visited[i] = True

            for j in range(i + 1, len(keyframes)):
                if visited[j]: continue
                x_j, y_j, theta_j = keyframes[j, 1], keyframes[j, 2], keyframes[j, 3]
                dist = math.hypot(x_j - x_i, y_j - y_i)
                if dist <= spatial_radius and self.angle_diff(theta_i, theta_j) <= angle_thresh:
                    cluster_indices.append(j)
                    visited[j] = True

            cluster_data = keyframes[cluster_indices]
            mean_x = np.mean(cluster_data[:, 1])
            mean_y = np.mean(cluster_data[:, 2])
            sum_sin = np.sum(np.sin(cluster_data[:, 3]))
            sum_cos = np.sum(np.cos(cluster_data[:, 3]))
            mean_theta = np.arctan2(sum_sin, sum_cos)

            centroids.append([new_id, mean_x, mean_y, mean_theta])
            new_id += 1

        return np.array(centroids)

    def build_directed_edges(self, nodes, max_dist=0.5, max_heading_diff_deg=100.0,
                             rescue_dist=4.0, cone_threshold_deg=20.0,
                             visibility_deg=100, direction_change_deg=100):
        edges = []
        max_heading_diff = np.radians(max_heading_diff_deg)
        cone_threshold   = np.radians(cone_threshold_deg)
        visibility       = np.radians(visibility_deg)
        direction_change = np.radians(direction_change_deg)

        for i in range(len(nodes)):
            id_A, x_A, y_A, theta_A = nodes[i]
            strict_candidates = []
            loose_candidates  = []

            for j in range(len(nodes)):
                if i == j: continue
                id_B, x_B, y_B, theta_B = nodes[j]

                if self.angle_diff(theta_A, theta_B) > max_heading_diff: continue
                vector_to_B_angle = np.arctan2(y_B - y_A, x_B - x_A)
                if self.angle_diff(theta_A, vector_to_B_angle) > visibility: continue

                dist = math.hypot(x_B - x_A, y_B - y_A)
                loose_candidates.append((dist, int(id_A), int(id_B), vector_to_B_angle))

                if self.angle_diff(vector_to_B_angle, theta_B) <= direction_change:
                    strict_candidates.append((dist, int(id_A), int(id_B), vector_to_B_angle))

            strict_candidates.sort(key=lambda item: item[0])
            loose_candidates.sort(key=lambda item: item[0])

            added_edges  = 0
            added_angles = []

            for dist, a, b, edge_angle in strict_candidates:
                is_redundant = any(
                    self.angle_diff(edge_angle, ea) < cone_threshold for ea in added_angles)
                if not is_redundant and dist <= max_dist:
                    edges.append((a, b))
                    added_angles.append(edge_angle)
                    added_edges += 1

            if added_edges == 0 and strict_candidates:
                closest_dist, a, b, _ = strict_candidates[0]
                if closest_dist <= rescue_dist:
                    edges.append((a, b))
                    added_edges += 1

            if added_edges == 0 and loose_candidates:
                closest_dist, a, b, _ = loose_candidates[0]
                if closest_dist <= rescue_dist:
                    edges.append((a, b))

        return edges

    def build_adjacency_list(self, edges):
        adj = {}
        for u, v in edges:
            if u not in adj: adj[u] = []
            adj[u].append(v)
        return adj

    def prune_shortcut_edges(self, edges, nodes, max_hops=4):
        """Drop 'shortcut' chords for a cleaner graph.

        A direct edge (A, C) is removed if there is an alternative directed path
        A -> ... -> C of 2..max_hops hops whose longest edge is shorter than the
        direct edge d(A, C). Because the detour must use only edges strictly
        shorter than the chord, the direct edge is never its own detour, genuine
        intersection branches (no short low-cost detour exists) are preserved,
        and reachability is never lost (the detour still connects A to C).
        Longest chords are considered first so the biggest skips go first.
        """
        pos = {int(n[0]): (n[1], n[2]) for n in nodes}

        def elen(u, v):
            (x1, y1), (x2, y2) = pos[u], pos[v]
            return math.hypot(x2 - x1, y2 - y1)

        edge_set = set((int(u), int(v)) for u, v in edges)
        adj = {}
        for u, v in edge_set:
            adj.setdefault(u, []).append(v)

        def has_detour(A, C, chord):
            # DFS up to max_hops; prune any branch whose max edge reaches chord.
            stack = [(A, 0, 0.0)]
            while stack:
                u, hops, maxe = stack.pop()
                if hops >= max_hops:
                    continue
                for v in adj.get(u, []):
                    w = max(maxe, elen(u, v))
                    if w >= chord:            # excludes the direct chord (len == chord)
                        continue
                    if v == C and hops + 1 >= 2:
                        return True
                    stack.append((v, hops + 1, w))
            return False

        removed = 0
        for (A, C) in sorted(edge_set, key=lambda e: elen(*e), reverse=True):
            if (A, C) not in edge_set:
                continue
            chord = elen(A, C)
            # check detour in the CURRENT graph, temporarily hiding the chord
            adj[A].remove(C)
            if has_detour(A, C, chord):
                edge_set.discard((A, C))
                removed += 1
            else:
                adj[A].append(C)          # keep it
        rospy.loginfo(f"[graph_planner] Pruned {removed} shortcut edges "
                      f"({len(edges)} -> {len(edge_set)})")
        return list(edge_set)

    def a_star_planner(self, start_id, target_id):
        if start_id not in self.nodes_dict or target_id not in self.nodes_dict:
            rospy.logerr(f"A*: start={start_id} (in_graph={start_id in self.nodes_dict}), "
                         f"target={target_id} (in_graph={target_id in self.nodes_dict}). "
                         f"Graph node IDs: {sorted(self.nodes_dict.keys())}")
            return

        rospy.loginfo(f"A*: start={start_id} → target={target_id}, "
                      f"start_neighbors={self.adj_list.get(start_id, [])}")

        open_set = []
        heapq.heappush(open_set, (0, start_id))
        came_from = {}
        g_score = {start_id: 0}
        visited = set()

        def heuristic(id_A, id_B):
            A, B = self.nodes_dict[id_A], self.nodes_dict[id_B]
            return math.hypot(B[1] - A[1], B[2] - A[2])

        while open_set:
            _, current_id = heapq.heappop(open_set)

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == target_id:
                path = [current_id]
                while current_id in came_from:
                    current_id = came_from[current_id]
                    path.append(current_id)
                self.plan = path[::-1]
                rospy.loginfo(f"A*: found path of length {len(self.plan)}: {self.plan}")
                return

            if current_id not in self.adj_list:
                continue

            for neighbor_id in self.adj_list[current_id]:
                if current_id in came_from:
                    prev_id    = came_from[current_id]
                    theta_prev = self.nodes_dict[prev_id][3]
                    theta_next = self.nodes_dict[neighbor_id][3]
                    if self.angle_diff(theta_prev, theta_next) > self.u_turn_thresh:
                        continue

                move_cost   = heuristic(current_id, neighbor_id)
                tentative_g = g_score.get(current_id, float('inf')) + move_cost

                if tentative_g < g_score.get(neighbor_id, float('inf')):
                    came_from[neighbor_id]  = current_id
                    g_score[neighbor_id]    = tentative_g
                    f_score = tentative_g + heuristic(neighbor_id, target_id)
                    heapq.heappush(open_set, (f_score, neighbor_id))

        incoming_to_target = [u for u, v in self.directed_edges if v == target_id]
        rospy.logerr(f"A* Failed: no path from {start_id} to {target_id}. "
                     f"Visited {len(visited)}/{len(self.nodes_dict)} nodes. "
                     f"Nodes with edges INTO target: {incoming_to_target}")

    def get_candidate_nodes(self, x, y, theta, max_dist=1.5, heading_tol_deg=45, k=3):
        heading_tol = np.radians(heading_tol_deg)
        ids    = self.clustered_nodes[:, 0]
        coords = self.clustered_nodes[:, 1:3]
        thetas = self.clustered_nodes[:, 3]

        angle_diffs = np.abs(np.arctan2(np.sin(thetas - theta), np.cos(thetas - theta)))
        valid_mask  = angle_diffs <= heading_tol

        if not np.any(valid_mask):
            return []

        dists = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
        dists[~valid_mask] = np.inf
        sorted_indices = np.argsort(dists)

        candidates = []
        for idx in sorted_indices:
            if dists[idx] > max_dist:
                break
            candidates.append(int(ids[idx]))
            if len(candidates) >= k:
                break

        return candidates

    def arrived(self, x, y, max_dist=0.5, max_angle_diff=90):
        tx, ty = self.nodes_dict[self.target][1], self.nodes_dict[self.target][2]
        angle_diff = self.angle_diff(self.nodes_dict[self.target][3], self.nodes_dict[self.curr_node][3])
        return math.hypot(tx - x, ty - y) < max_dist and angle_diff < np.deg2rad(max_angle_diff)

    def write_clustered_nodes(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            reverse_label_map = {v: k for k, v in self.label_map.items()}
            with open(filepath, 'w') as f:
                for node in self.clustered_nodes:
                    node_id = int(node[0])
                    x, y, theta = node[1], node[2], node[3]
                    label = reverse_label_map.get(node_id)
                    if label:
                        f.write(f"{node_id} {x} {y} {theta} {label}\n")
                    else:
                        f.write(f"{node_id} {x} {y} {theta}\n")
            rospy.loginfo(f"Cached {len(self.clustered_nodes)} clustered nodes to {filepath}")
        except Exception as e:
            rospy.logerr(f"Failed to write clustered nodes: {e}")

    def read_clustered_nodes(self, filename):
        self.label_map = {}
        nodes_data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4 or len(parts) > 5: continue
                node_id = int(parts[0])
                x, y    = float(parts[1]), float(parts[2])
                theta   = float(parts[3])
                nodes_data.append([node_id, x, y, theta])
                if len(parts) == 5:
                    self.label_map[parts[4]] = node_id
        self.clustered_nodes = np.array(nodes_data)


if __name__ == "__main__":
    graph_planner_node = GraphPlannerNode(node_name="graph_planner_node")
    rospy.spin()
