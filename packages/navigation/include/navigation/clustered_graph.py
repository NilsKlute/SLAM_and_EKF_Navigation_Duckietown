#!/usr/bin/env python3
import cv2
import rospy
import numpy as np
import tf
from multiprocessing import Lock
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle, Arc
import math

def normalize_angle(angle):
    """Normalizes an angle to be within [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def angle_diff(a, b):
    """Returns the absolute difference between two angles in radians"""
    return abs(normalize_angle(a - b))

def cluster_nodes(keyframes, spatial_radius=1.0, angle_threshold_deg=45.0):
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

            if dist <= spatial_radius and angle_diff(theta_i, theta_j) <= angle_thresh:
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

def build_directed_edges(nodes, max_dist=2.5, max_heading_diff_deg=100.0, rescue_dist=10.0, cone_threshold_deg=20.0, visibility_deg=100, direction_change_deg=100):

    edges = []
    max_heading_diff = np.radians(max_heading_diff_deg)
    cone_threshold = np.radians(cone_threshold_deg)
    visibility = np.radians(visibility_deg)
    direction_change= np.radians(direction_change_deg)

    for i in range(len(nodes)):
        id_A, x_A, y_A, theta_A = nodes[i]
        strict_candidates = []
        loose_candidates = []

        for j in range(len(nodes)):
            if i == j: continue

            id_B, x_B, y_B, theta_B = nodes[j]

            if angle_diff(theta_A, theta_B) > max_heading_diff: continue

            vector_to_B_angle = np.arctan2(y_B - y_A, x_B - x_A)
            if angle_diff(theta_A, vector_to_B_angle) > visibility: continue

            dist = math.hypot(x_B - x_A, y_B - y_A)
            loose_candidates.append((dist, int(id_A), int(id_B), vector_to_B_angle))

            if angle_diff(vector_to_B_angle, theta_B) <= direction_change:
                strict_candidates.append((dist, int(id_A), int(id_B), vector_to_B_angle))

        strict_candidates.sort(key=lambda item: item[0])
        loose_candidates.sort(key=lambda item: item[0])

        added_edges = 0
        added_angles = []

        for dist, a, b, edge_angle in strict_candidates:
            is_redundant = False
            for existing_angle in added_angles:
                if angle_diff(edge_angle, existing_angle) < cone_threshold:
                    is_redundant = True
                    break

            if not is_redundant and dist <= max_dist:
                edges.append((a, b))
                added_angles.append(edge_angle)
                added_edges += 1

        if added_edges == 0 and len(strict_candidates) > 0:
            closest_dist, a, b, _ = strict_candidates[0]
            if closest_dist <= rescue_dist:
                edges.append((a, b))
                added_edges += 1

        if added_edges == 0 and len(loose_candidates) > 0:
            closest_dist, a, b, _ = loose_candidates[0]
            if closest_dist <= rescue_dist:
                edges.append((a, b))

    return edges

def visualize_clustered_graph(mean_nodes, edges, annotations=None, show=True,
                            title="Clustered Graph"):
    fig, ax = plt.subplots(figsize=(8, 8))

    for (id_A, id_B) in edges:
        A = mean_nodes[mean_nodes[:, 0] == id_A][0]
        B = mean_nodes[mean_nodes[:, 0] == id_B][0]
        ax.annotate("", xy=(B[1], B[2]), xycoords='data',
                    xytext=(A[1], A[2]), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", lw=2,
                                    alpha=0.7, shrinkA=8, shrinkB=8))

    mx, my, mtheta = mean_nodes[:, 1], mean_nodes[:, 2], mean_nodes[:, 3]
    ax.scatter(mx, my, c='blue', s=80, zorder=5, edgecolors='black',
            label="Standard Lane Node")

    if annotations:
        ix, iy = [], []
        for node in mean_nodes:
            if annotations[int(node[0])].get("is_intersection"):
                ix.append(node[1]); iy.append(node[2])
        if ix:
            ax.scatter(ix, iy, c='gold', s=100, zorder=6, marker='o',
                    label="Intersection Approach Node")

    ax.quiver(mx, my, np.cos(mtheta), np.sin(mtheta), color='red',
            scale=30, width=0.005, zorder=7, label='Heading')

    ax.set_title(title)
    ax.set_xlabel("X Position (m)"); ax.set_ylabel("Y Position (m)")
    ax.xaxis.set_major_locator(MultipleLocator(0.6))
    ax.yaxis.set_major_locator(MultipleLocator(0.6))
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.axis('equal')

    if show:
        plt.show()
    return fig, ax