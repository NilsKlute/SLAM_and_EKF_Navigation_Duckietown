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
from duckietown_msgs.msg import BoolStamped
from std_msgs.msg import String, Int64
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

        # --- Params ---
        self.localization_type  = rospy.get_param("~localization_type", "EKF")
        self.use_cached         = rospy.get_param("~use_cached", False)
        self.data_dir           = rospy.get_param("~data_dir", "/data/graph_planner")
        graph_file_ekf          = rospy.get_param("~graph_file_ekf")
        graph_file_slam         = rospy.get_param("~graph_file_slam")
        label_map_file          = rospy.get_param("~label_map_file")

        self.use_clustering     = rospy.get_param("~use_clustering", True)
        spatial_radius          = rospy.get_param("~spatial_radius", 1.8)
        angle_threshold_deg     = rospy.get_param("~angle_threshold_deg", 45.0)
        edge_max_dist           = rospy.get_param("~edge_max_dist", 4.0)
        edge_max_heading_diff   = rospy.get_param("~edge_max_heading_diff_deg", 130.0)

        self.candidate_max_dist    = rospy.get_param("~candidate_max_dist", 1.5)
        self.candidate_k           = rospy.get_param("~candidate_k", 3)
        self.arrival_threshold     = rospy.get_param("~arrival_threshold", 0.5)
        self.skip_n_nodes          = rospy.get_param("~skip_n_nodes", 2)
        self.turn_angle_thresh_deg = rospy.get_param("~turn_angle_threshold_deg", 35.0)
        self.u_turn_thresh         = np.radians(rospy.get_param("~u_turn_thresh_deg", 120.0))

        # --- Graph loading ---
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
            self.clustered_nodes, max_dist=edge_max_dist, max_heading_diff_deg=edge_max_heading_diff)
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

        self.plan = None
        self.curr_node = None

        # --- Subscribers ---
        self.sub_target = rospy.Subscriber("~target_location", String, self.cb_init_navigation)

        if self.localization_type == "EKF":
            rospy.Subscriber("ekf_localization_node/pose", Odometry, self.cb_localize_odometry)
        else:
            rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped, self.cb_localize_pose_cov)

        self.sub_stop_line = rospy.Subscriber("~at_stop_line", BoolStamped, self.cb_directional_cmd)
        self.sub_int_go    = rospy.Subscriber("~intersection_go", BoolStamped, self.cb_reset_decided_planning)

        # --- Publishers ---
        self.pub_arrived_target  = rospy.Publisher("~arrived_at_target", BoolStamped, queue_size=1, latch=True)
        self.pub_directional_cmd = rospy.Publisher("~directional_cmd", Int64, queue_size=1)
        self.pub_debug_image     = rospy.Publisher("~debug_image/compressed", CompressedImage, queue_size=1)
        self._latest_pose        = (0.0, 0.0, 0.0)
        rospy.Timer(rospy.Duration(3), self._debug_image_timer)

    # ---- Localization callbacks ----

    def cb_localize_odometry(self, msg):
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self._localize(x, y, theta)

    def cb_localize_pose_cov(self, msg):
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self._localize(x, y, theta)

    def _yaw_from_quaternion(self, q):
        # Z-axis rotation: q = (0, 0, sin(θ/2), cos(θ/2))
        # arctan2(sin(θ/2), cos(θ/2)) = θ/2  →  multiply by 2
        return 2.0 * np.arctan2(q.z, q.w)

    def _localize(self, x, y, theta):
        if self.target is None:
            rospy.loginfo_throttle(5.0, "Localize: no target set yet")
            return

        if self.target not in self.nodes_dict:
            rospy.logerr_throttle(5.0, f"Target node {self.target} not in graph! "
                                       f"Valid IDs: {sorted(self.nodes_dict.keys())}")
            return

        rospy.loginfo_throttle(5.0,
            f"Localize: pos=({x:.2f},{y:.2f},{np.degrees(theta):.0f}°) "
            f"target={self.target} plan={'set' if self.plan else 'None'} "
            f"curr={self.curr_node}")

        if self.arrived(x, y, self.arrival_threshold):
            rospy.loginfo("Target Reached!")
            msg = BoolStamped()
            msg.header.stamp = rospy.Time.now()
            msg.data = True
            self.pub_arrived_target.publish(msg)
            self.plan = None
            self.target = None
            self.init_plan = False
            return

        self._latest_pose = (x, y, theta)

        candidates = self.get_candidate_nodes(
            x, y, theta, max_dist=self.candidate_max_dist, k=self.candidate_k)

        if not candidates:
            rospy.logwarn_throttle(3.0, "Localization failed: No valid nodes found within search radius.")
            return

        # Init Plan after target and first localization are received
        if not self.init_plan and self.plan is None:
            rospy.loginfo("We go into init planning")
            for candidate in candidates:
                self.a_star_planner(candidate, self.target)
                if self.plan is not None and candidate in self.plan:
                    self.curr_node = candidate
                    self.init_plan = True
                    return
            rospy.logwarn_throttle(3.0, "We failed to initialize navigation")
        

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
                rospy.loginfo(f"Replanned from Node {self.curr_node}. Path: {self.plan}")
                break

        if not path_found:
            rospy.logerr_throttle(5.0,
                f"A* Failed: No path from candidates {candidates}. Halting navigation.")
            self.target = None

    # ---- Debug visualization ----

    def _debug_image_timer(self, _event=None):
        rx, ry, rtheta = self._latest_pose
        self._publish_debug_image(rx, ry, rtheta)

    def _target_label(self):
        if self.target is None:
            return "None"
        reverse = {v: k for k, v in self.label_map.items()}
        return reverse.get(self.target, str(self.target))

    def _publish_debug_image(self, rx, ry, rtheta):
        rospy.loginfo("Debug Image cb called")
        try:
            fig, ax = plt.subplots(figsize=(10, 10))

            # All graph edges — thin black arrows
            for (id_a, id_b) in self.directed_edges:
                a = self.nodes_dict[id_a]
                b = self.nodes_dict[id_b]
                ax.annotate("", xy=(b[1], b[2]), xytext=(a[1], a[2]),
                            arrowprops=dict(arrowstyle="->", color="black",
                                           lw=3, alpha=0.95, shrinkA=5, shrinkB=5))

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
                       edgecolors='navy', linewidths=0.5)
            ax.quiver(mx, my, np.cos(mtheta), np.sin(mtheta),
                      color='steelblue', scale=30, width=0.004, alpha=0.5, zorder=6)

            # Node ID labels (skipped for large raw-keyframe graphs — unreadable and slow)
            if len(self.clustered_nodes) <= 300:
                for node in self.clustered_nodes:
                    ax.annotate(str(int(node[0])), (node[1], node[2]),
                                fontsize=7, ha='center', va='bottom', color='navy', zorder=7)

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

            # Robot pose — red star + heading arrow
            ax.scatter(rx, ry, c='red', s=350, marker='*', zorder=10,
                       edgecolors='darkred', linewidths=1)
            ax.quiver(rx, ry, np.cos(rtheta), np.sin(rtheta),
                      color='red', scale=20, width=0.007, zorder=11)

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

    # ---- Navigation callbacks ----

    def cb_init_navigation(self, target_msg):
        if target_msg.data in self.label_map:
            self.target = self.label_map[target_msg.data]
            rospy.loginfo(f"Target set: '{target_msg.data}' → node {self.target}")
        else:
            rospy.logwarn(f"Label '{target_msg.data}' not in label map {self.label_map}. "
                          f"Trying as raw integer ID.")
            try:
                self.target = int(target_msg.data)
                rospy.loginfo(f"Target set to raw node ID {self.target}")
            except ValueError:
                rospy.logerr(f"Target '{target_msg.data}' is neither a known label nor a valid integer ID.")
                return
        

    def cb_directional_cmd(self, _msg):
        if self.decision_published or self.plan is None or self.curr_node is None or self.curr_node not in self.plan:
            rospy.loginfo(f"We cannot plan! Decision published: {self.decision_published}.\n self.plan is: {self.plan}. ")
            return
        
        time.sleep(5)

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
            rospy.loginfo("TURN LEFT")
        elif diff_deg < -self.turn_angle_thresh_deg:
            cmd_msg.data = 2  # RIGHT
            rospy.loginfo("TURN RIGHT")
        else:
            cmd_msg.data = 1  # STRAIGHT
            rospy.loginfo("GO STRAIGHT")

        rospy.Timer(rospy.Duration(0.2), lambda event: self.pub_directional_cmd.publish(cmd_msg), oneshot=True)
        self.decision_published = True
        rospy.loginfo("Planner decided and published directional cmd")

    def cb_reset_decided_planning(self, _msg):
        self.decision_published = False

    # ---- Helper functions ----

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

    def arrived(self, x, y, max_dist=0.5):
        tx, ty = self.nodes_dict[self.target][1], self.nodes_dict[self.target][2]
        return math.hypot(tx - x, ty - y) < max_dist

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
