#!/usr/bin/env python3

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle
from itertools import product
from scipy.spatial.distance import cdist

# ============================================================
# 0. Tile geometry constants
# ============================================================
TILE_SIZE = 0.6
LANE_OFFSET = TILE_SIZE * 0.15
EDGE_MARGIN = TILE_SIZE * 0.15
ROAD_WIDTH = TILE_SIZE - 2 * EDGE_MARGIN

RIGHT_LANE = EDGE_MARGIN + ROAD_WIDTH * 0.25
LEFT_LANE = EDGE_MARGIN + ROAD_WIDTH * 0.75

N_SAMPLES = 10  # for making the tiles that are later fitted

INNER_R = ROAD_WIDTH * 0.25 + EDGE_MARGIN
OUTER_R = ROAD_WIDTH * 0.75 + EDGE_MARGIN

# ============================================================
# 1. Canonical tile-type / direction system
# ============================================================
TILE_TYPES = [
    "empty", "N-S", "E-W", "NE", "ES", "SW", "WN",
    "NES", "ESW", "SWN", "WNE", "4-way"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TILE_TYPES)}
NUM_TYPES = len(TILE_TYPES)

DIRECTIONS = ["N", "E", "S", "W"]
OPPOSITE_DIR = {"N": "S", "S": "N", "E": "W", "W": "E"}

DIR_SETS = {
    "empty": set(),
    "N-S": {"N", "S"},
    "E-W": {"E", "W"},
    "NE": {"N", "E"},
    "ES": {"E", "S"},
    "SW": {"S", "W"},
    "WN": {"W", "N"},
    "NES": {"N", "E", "S"},
    "ESW": {"E", "S", "W"},
    "SWN": {"S", "W", "N"},
    "WNE": {"W", "N", "E"},
    "4-way": {"N", "E", "S", "W"},
}

# Color map for tile types
_TYPE_COLOR_MAP = {
    "N-S": "#4477AA", "E-W": "#66AA55",
    "NE": "#FF8844", "ES": "#CC6677", "SW": "#AA4499", "WN": "#88CCEE",
    "NES": "#FFD700", "ESW": "#DAA520", "SWN": "#B8860B", "WNE": "#CD853F",
    "4-way": "#FF4500", "empty": "#F5F5F5",
}

# All 12 ordered (entry_edge -> exit_edge) directional lane tokens
TILE_DIRECTIONS = {
    # Straights
    "S→N": ("S", "N"),
    "N→S": ("N", "S"),
    "W→E": ("W", "E"),
    "E→W": ("E", "W"),

    # Inner lane curves (quarter circles)
    "S→E": ("S", "E"),
    "E→N": ("E", "N"),
    "N→W": ("N", "W"),
    "W→S": ("W", "S"),

    # Outer lane curves (quarter circles)
    "E→S": ("E", "S"),
    "N→E": ("N", "E"),
    "W→N": ("W", "N"),
    "S→W": ("S", "W"),

    # U-turns (half circles)
    "N→N": ("N", "N"),
    "S→S": ("S", "S"),
    "E→E": ("E", "E"),
    "W→W": ("W", "W"),
}

LANE_TO_STRUCTURE = {
    "S\u2192N": "N-S", "N\u2192S": "N-S",
    "W\u2192E": "E-W", "E\u2192W": "E-W",
    "S\u2192E": "ES",  "E\u2192S": "ES",
    "E\u2192N": "NE",  "N\u2192E": "NE",
    "N\u2192W": "WN",  "W\u2192N": "WN",
    "W\u2192S": "SW",  "S\u2192W": "SW",
}

BP_TO_CLASSIFICATION = {
    "empty": "empty",
    "N-S": "straight", "E-W": "straight",
    "NE": "curve", "ES": "curve", "SW": "curve", "WN": "curve",
    "NES": "intersection_3way", "ESW": "intersection_3way",
    "SWN": "intersection_3way", "WNE": "intersection_3way",
    "4-way": "intersection_4way",
}

# ============================================================
# Fitting Pipeline
# ============================================================

def sample_straight(direction):
    if direction == "S->N":
        y = np.linspace(0, TILE_SIZE, N_SAMPLES)
        x = np.full_like(y, LEFT_LANE)

    elif direction == "N->S":
        y = np.linspace(TILE_SIZE, 0, N_SAMPLES)
        x = np.full_like(y, RIGHT_LANE)

    elif direction == "W->E":
        x = np.linspace(0, TILE_SIZE, N_SAMPLES)
        y = np.full_like(x, RIGHT_LANE)

    elif direction == "E->W":
        x = np.linspace(TILE_SIZE, 0, N_SAMPLES)
        y = np.full_like(x, LEFT_LANE)

    else:
        raise ValueError(direction)

    theta = np.arctan2(np.gradient(y), np.gradient(x))
    return np.column_stack((x, y, theta))


def sample_curve(entry, exit):
    """
    Quarter-circle lane centerline.
    """

    inner_turns = {
        ('S','E'),
        ('E','N'),
        ('N','W'),
        ('W','S'),
    }

    outer_turns = {
        ('E','S'),
        ('N','E'),
        ('W','N'),
        ('S','W'),
    }

    if (entry, exit) in inner_turns:
        r = INNER_R
    elif (entry, exit) in outer_turns:
        r = OUTER_R
    else:
        raise ValueError((entry, exit))

    table = {
        # inner lane
        ('S','E'): ((TILE_SIZE, 0),          np.pi,       np.pi/2),
        ('E','N'): ((TILE_SIZE, TILE_SIZE), -np.pi/2,   -np.pi),
        ('N','W'): ((0, TILE_SIZE),           0,         -np.pi/2),
        ('W','S'): ((0, 0),                   np.pi/2,    0),

        # outer lane
        ('E','S'): ((TILE_SIZE, 0),           np.pi/2,    np.pi),
        ('N','E'): ((TILE_SIZE, TILE_SIZE),   -np.pi,      -np.pi/2),
        ('W','N'): ((0, TILE_SIZE),           -np.pi/2,   0),
        ('S','W'): ((0, 0),                   0,          np.pi/2),
    }

    center, phi0, phi1 = table[(entry, exit)]

    phi = np.linspace(phi0, phi1, N_SAMPLES)

    cx, cy = center

    x = cx + r * np.cos(phi)
    y = cy + r * np.sin(phi)

    theta = np.arctan2(
        np.gradient(y),
        np.gradient(x)
    )

    return np.column_stack((x, y, theta))


def sample_halfcircle_uturn(direction):
    """
    U-turn with a half circle in the middle of the tile.
    Direction indicates entry side and exit side (same side).
    """
    HALF_CIRCLE_R = ROAD_WIDTH * 0.25
    
    if direction == "N->N":
        y1 = np.linspace(TILE_SIZE, TILE_SIZE/2 + HALF_CIRCLE_R, N_SAMPLES//2)
        x1 = np.full_like(y1, RIGHT_LANE)
        
        phi = np.linspace(np.pi, 2*np.pi, N_SAMPLES)
        x_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.cos(phi)
        y_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.sin(phi)
        
        y2 = np.linspace(TILE_SIZE/2 + HALF_CIRCLE_R, TILE_SIZE, N_SAMPLES//2)
        x2 = np.full_like(y2, LEFT_LANE)
        
        x = np.concatenate([x1, x_circle, x2])
        y = np.concatenate([y1, y_circle, y2])
        
    elif direction == "S->S":
        y1 = np.linspace(0, TILE_SIZE/2 - HALF_CIRCLE_R, N_SAMPLES//2)
        x1 = np.full_like(y1, LEFT_LANE)
        
        phi = np.linspace(0, np.pi, N_SAMPLES)
        x_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.cos(phi)
        y_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.sin(phi)
        
        y2 = np.linspace(TILE_SIZE/2 - HALF_CIRCLE_R, 0, N_SAMPLES//2)
        x2 = np.full_like(y2, RIGHT_LANE)
        
        x = np.concatenate([x1, x_circle, x2])
        y = np.concatenate([y1, y_circle, y2])
        
    elif direction == "E->E":
        x1 = np.linspace(TILE_SIZE, TILE_SIZE/2 + HALF_CIRCLE_R, N_SAMPLES//2)
        y1 = np.full_like(x1, LEFT_LANE)
        
        phi = np.linspace(np.pi/2, np.pi/2*3, N_SAMPLES)
        x_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.cos(phi)
        y_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.sin(phi)
        
        x2 = np.linspace(TILE_SIZE/2 + HALF_CIRCLE_R, TILE_SIZE, N_SAMPLES//2)
        y2 = np.full_like(x2, RIGHT_LANE)
        
        x = np.concatenate([x1, x_circle, x2])
        y = np.concatenate([y1, y_circle, y2])
        
    elif direction == "W->W":
        x1 = np.linspace(0, TILE_SIZE/2 - HALF_CIRCLE_R, N_SAMPLES//2)
        y1 = np.full_like(x1, RIGHT_LANE)
        
        phi = np.linspace(-np.pi/2, np.pi/2, N_SAMPLES)
        x_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.cos(phi)
        y_circle = TILE_SIZE/2 + HALF_CIRCLE_R * np.sin(phi)
        
        x2 = np.linspace(TILE_SIZE/2 - HALF_CIRCLE_R, 0, N_SAMPLES//2)
        y2 = np.full_like(x2, LEFT_LANE)
        
        x = np.concatenate([x1, x_circle, x2])
        y = np.concatenate([y1, y_circle, y2])
    
    else:
        raise ValueError(direction)
    
    theta = np.arctan2(np.gradient(y), np.gradient(x))
    return np.column_stack((x, y, theta))


def reverse_template(pts):
    pts = pts[::-1].copy()
    pts[:,2] += np.pi
    pts[:,2] = np.arctan2(np.sin(pts[:,2]), np.cos(pts[:,2]))
    return pts


templates = {
    # straights
    "S→N": sample_straight("S->N"),
    "N→S": sample_straight("N->S"),
    "W→E": sample_straight("W->E"),
    "E→W": sample_straight("E->W"),

    # curves (inner lane)
    "S→E": sample_curve("S", "E"),
    "E→N": sample_curve("E", "N"),
    "N→W": sample_curve("N", "W"),
    "W→S": sample_curve("W", "S"),

    # curves (outer lane)
    "E→S": sample_curve("E", "S"),
    "N→E": sample_curve("N", "E"),
    "W→N": sample_curve("W", "N"),
    "S→W": sample_curve("S", "W"),
    
    # U-turns (half circles)
    "N→N": sample_halfcircle_uturn("N->N"),
    "S→S": sample_halfcircle_uturn("S->S"),
    "E→E": sample_halfcircle_uturn("E->E"),
    "W→W": sample_halfcircle_uturn("W->W"),
}


def entry_exit(tile_type):
    return TILE_DIRECTIONS[tile_type]


# ------------------------------------------------------------
# 1. Generate connected 3 tile hypotheses
# ------------------------------------------------------------
OPPOSITE = {
    "N":"S",
    "S":"N",
    "E":"W",
    "W":"E"
}


def generate_tile_sequences(tile_types, n_tiles=3):
    all_sequences = []

    # Define U-turn types
    UTURNS = {"N→N", "S→S", "E→E", "W→W"}
    CURVES = {"S→E", "E→N", "N→W", "W→S", "E→S", "N→E", "W→N", "S→W"}
    STRAIGHTS = {"S→N", "N→S", "W→E", "E→W"}

    for k in range(2, n_tiles + 1):
        for seq in product(tile_types, repeat=k):
            valid = True

            # Check if connections are valid
            for i in range(k - 1):
                _, exit_i = entry_exit(seq[i])
                entry_next, _ = entry_exit(seq[i+1])

                if exit_i != OPPOSITE[entry_next]:
                    valid = False
                    break

            # Check for consecutive U-turns
            if valid:
                for i in range(k - 1):
                    if seq[i] in UTURNS and seq[i+1] in UTURNS:
                        valid = False
                        break

            # Check for 'curve has to be followed by straight' validation (original rule)
            if valid:
                for i in range(k - 1):
                    if seq[i] in CURVES and seq[i+1] not in STRAIGHTS:
                        valid = False
                        break

            if valid:
                all_sequences.append(list(seq))

    return all_sequences


# ------------------------------------------------------------
# 2. Transform local templates into world coordinates
# ------------------------------------------------------------
MOVE = {
    "N": np.array([0, TILE_SIZE]),
    "S": np.array([0, -TILE_SIZE]),
    "E": np.array([TILE_SIZE, 0]),
    "W": np.array([-TILE_SIZE, 0]),
}


def rotate_heading(theta, offset):
    return np.arctan2(
        np.sin(theta + offset),
        np.cos(theta + offset)
    )


def build_3tile_template(tile_sequence, start_tile, templates):
    tile_origin = np.array([
        start_tile[0] * TILE_SIZE,
        start_tile[1] * TILE_SIZE
    ])

    result = []
    current_origin = tile_origin.copy()

    for i, tile_type in enumerate(tile_sequence):
        pts = templates[tile_type].copy()
        pts[:,0:2] += current_origin
        result.append(pts)

        _, exit_dir = entry_exit(tile_type)
        current_origin += MOVE[exit_dir]

    return np.concatenate(result, axis=0)


# ------------------------------------------------------------
# 3. Trajectory fitting loss
# ------------------------------------------------------------
def angle_difference(a, b):
    return np.arctan2(
        np.sin(a-b),
        np.cos(a-b)
    )


def trajectory_loss(trajectory_segment, template, heading_weight=0.2):
    template_xy = template[:,:2]
    template_theta = template[:,2]
    
    dist_sq = cdist(trajectory_segment[:, :2], template_xy, 'sqeuclidean')
    idx = np.argmin(dist_sq, axis=1)
    d_sq = dist_sq[np.arange(len(trajectory_segment)), idx]
    
    theta_traj = trajectory_segment[:, 2]
    theta_template = template_theta[idx]
    dtheta = angle_difference(theta_traj, theta_template)
    
    loss = np.sum(d_sq + heading_weight * dtheta * dtheta)
    return loss / len(trajectory_segment)


# ------------------------------------------------------------
# 4. Convert losses to probabilities
# ------------------------------------------------------------
def loss_to_probability(losses, sigma=1.0):
    keys = list(losses.keys())
    L = np.array([losses[k] for k in keys])
    weights = np.exp(-L/sigma)
    probs = weights / np.sum(weights)
    return {k: p for k, p in zip(keys, probs)}


# ------------------------------------------------------------
# Helper: trajectory length along path
# ------------------------------------------------------------
def trajectory_distance(points):
    d = np.diff(points[:,:2], axis=0)
    return np.sum(np.linalg.norm(d, axis=1))


# ------------------------------------------------------------
# Find segment covering 3 tiles
# ------------------------------------------------------------
def take_three_tile_segment(queue, max_tiles=3):
    travelled = 0.0
    tile_start = 0
    tile_counts = []

    for i in range(1, len(queue)):
        travelled += np.linalg.norm(queue[i, :2] - queue[i - 1, :2])

        if travelled >= TILE_SIZE:
            tile_counts.append(i - tile_start + 1)
            tile_start = i + 1
            travelled = 0.0

            if len(tile_counts) >= max_tiles:
                end_idx = i
                segment = queue[:end_idx + 1]
                return segment, end_idx + 1, tile_counts

    return None, None, None


# ------------------------------------------------------------
# Main inference function with debug flag
# ------------------------------------------------------------
def infer_map(trajectory, templates, debug=False):
    """
    Winner-takes-all approach: count how often each tile type wins as the first tile.
    
    Parameters:
    -----------
    trajectory : np.ndarray
        Trajectory points with (x, y, theta)
    templates : dict
        Dictionary of tile templates
    debug : bool
        If True, print debug information
    
    Returns:
    --------
    probabilities, uncertainty, fitted_sections, source_sections, vote_counts, total_visits
    """
    if trajectory is None or len(trajectory) < 20:
        if debug:
            print("Warning: Trajectory too short for inference")
        return {}, {}, [], [], {}, {}
    
    first_point = trajectory[0, :2]
    start_tile = np.floor(first_point / TILE_SIZE).astype(int)
    tile_types = list(templates.keys())
    sequences = generate_tile_sequences(tile_types)

    queue = trajectory.copy()
    queue_ids = np.arange(len(trajectory))

    vote_counts = {}
    total_visits = {}
    fitted_sections = []
    source_sections = []
    fitted_trajectory = []

    current_tile = np.array(start_tile, dtype=int)
    tile_nr = 1
    last_picked_exit_direction = None # None for the very first tile, no prior constraint


    OPPOSITE_MAP = {
        "S→N": "N-S", "N→S": "N-S",
        "W→E": "E-W", "E→W": "E-W",
        "N→W": "WN", "W→N": "WN",
        "S→E": "ES", "E→S": "ES",
        "E→N": "NE", "N→E": "NE",
        "W→S": "SW", "S→W": "SW",
        "N→N": "N→N", "S→S": "S→S",
        "E→E": "E→E", "W→W": "W→W",
    }

    iteration = 0
    while len(queue) > 20 and iteration < 100:  # Add max iterations to prevent infinite loop
        iteration += 1
        segment, remove_count, tile_counts = take_three_tile_segment(queue)
        if segment is None:
            break

        segment_ids = queue_ids[:remove_count]
        losses = {}
        fitted_candidates = {}

        #  Filter sequences based on the last_picked_exit_direction
        current_sequences_to_test = sequences
        if last_picked_exit_direction is not None:
            required_entry_direction = OPPOSITE[last_picked_exit_direction]
            filtered_sequences = []
            for seq in sequences:
                first_tile_entry, _ = entry_exit(seq[0])
                if first_tile_entry == required_entry_direction:
                    filtered_sequences.append(seq)
            current_sequences_to_test = filtered_sequences
            if not current_sequences_to_test: # If no sequences match, cannot proceed
                if debug:
                    print(f"[{tuple(current_tile)}] No sequences found matching required entry direction: {required_entry_direction}. Stopping.")
                break

        # Test all 3 tile hypotheses using the filtered sequences
        for seq in current_sequences_to_test: # USE THE FILTERED LIST

            template = build_3tile_template(seq, current_tile, templates)
            loss = trajectory_loss(segment, template)
            loss = loss + trajectory_loss(template, segment)
            name = "|".join(seq)
            losses[name] = loss
            fitted_candidates[name] = template

        probs = loss_to_probability(losses)

        tile_probs = {}
        for name, p in probs.items():
            first_tile = name.split("|")[0]
            tile_probs[first_tile] = tile_probs.get(first_tile, 0) + p

        best_sequence = max(probs, key=probs.get)
        best_first_tile = best_sequence.split("|")[0]
        
        if best_first_tile == "empty":
            winner = "empty"
        else:
            winner = OPPOSITE_MAP.get(best_first_tile, best_first_tile)

        tile_probs_summed = {}
        for tile_type, prob in tile_probs.items():
            if tile_type == "empty":
                tile_probs_summed["empty"] = tile_probs_summed.get("empty", 0) + prob
            else:
                canonical = OPPOSITE_MAP.get(tile_type, tile_type)
                tile_probs_summed[canonical] = tile_probs_summed.get(canonical, 0) + prob

        tile_pos = tuple(current_tile)
        
        if tile_pos not in vote_counts:
            vote_counts[tile_pos] = {}
            total_visits[tile_pos] = 0
        
        vote_counts[tile_pos][winner] = vote_counts[tile_pos].get(winner, 0) + 1
        total_visits[tile_pos] += 1

        best_template = fitted_candidates[best_sequence]
        best_tile_names = best_sequence.split("|")

        tile_lengths = [len(templates[t]) for t in best_tile_names]
        first_tile_template = best_template[:tile_lengths[0]].copy()
        
        fitted_sections.append(first_tile_template)
        source_sections.append(segment_ids)

        if debug:
            print("\n-----------------------------")
            print("tile_nr", tile_nr)
            print("Tile position:", tuple(current_tile))
            print("First tile probabilities (summed):")
            total = sum(tile_probs_summed.values())
            for k, v in sorted(tile_probs_summed.items(), key=lambda x: x[1], reverse=True)[:8]:
                pct = (v / total * 100) if total > 0 else 0
                print(f"  {k}: {v:.3f} ({pct:.1f}%)")
            print(f"\nBest 3 tile fit: {best_sequence}")
            print(f"WINNER (first tile): {best_first_tile} -> {winner}")
            print(f"Total visits to this tile: {total_visits[tile_pos]}")
            print("\nVote counts so far:")
            for k, v in sorted(vote_counts[tile_pos].items(), key=lambda x: x[1], reverse=True):
                pct = (v / total_visits[tile_pos] * 100)
                print(f"  {k}: {v} votes ({pct:.1f}%)")
            print(f"Probability of best sequence: {probs[best_sequence]:.6f}")

        tile_nr += 1
        fitted_trajectory.append(best_template)

        template_xy = best_template[:, :2]
        assigned_tile_idx = np.empty(len(segment), dtype=int)
        
        for j in range(len(segment)):
            dist = np.linalg.norm(template_xy - segment[j, :2], axis=1)
            closest_idx = np.argmin(dist)
            assigned_tile_idx[j] = np.searchsorted(np.cumsum(tile_lengths), closest_idx, side="right")

        remove_mask = assigned_tile_idx == 0
        keep_mask = ~remove_mask

        queue = np.concatenate([segment[keep_mask], queue[remove_count:]], axis=0)
        queue_ids = np.concatenate([segment_ids[keep_mask], queue_ids[remove_count:]], axis=0)

        first_tile_of_best_sequence = best_tile_names[0]
        _, exit_dir_of_best_sequence = entry_exit(first_tile_of_best_sequence)
        current_tile = (current_tile + MOVE[exit_dir_of_best_sequence] / TILE_SIZE)
        current_tile = np.round(current_tile).astype(int)

        # NEW: Update the last picked exit direction for the next iteration
        last_picked_exit_direction = exit_dir_of_best_sequence

    probabilities = {}
    uncertainty = {}
    
    for tile_pos, votes in vote_counts.items():
        total = total_visits[tile_pos]
        probs = {}
        for tile_type, count in votes.items():
            probs[tile_type] = count / total
        probabilities[tile_pos] = probs
        
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * np.log(p)
        uncertainty[tile_pos] = entropy

    if len(fitted_trajectory):
        fitted_trajectory = np.concatenate(fitted_trajectory, axis=0)
    else:
        fitted_trajectory = np.empty((0, 3))

    if debug:
        print("\n" + "="*50)
        print("SUMMARY:")
        print(f"Total unique tiles: {len(probabilities)}")
        print(f"Total observations: {sum(total_visits.values())}")
        print("="*50)
    
    return probabilities, uncertainty, fitted_sections, source_sections, vote_counts, total_visits

