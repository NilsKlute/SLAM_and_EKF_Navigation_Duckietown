#!/usr/bin/env python3
"""
navigation.tile_graph
======================
Duckietown tile-map reconstruction: from a raw (smoothed) trajectory to a
classified tile map, a belief-propagation-refined tile map, and a lane-level
street graph -- plus plotting utilities for all three stages.

Usage:
    from navigation.tile_graph import *
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle

warnings.filterwarnings('ignore')

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
#    (defined BEFORE anything that references it as a default arg
#    or at import time -- this was the root cause of the NameErrors)
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

# All 12 ordered (entry_edge -> exit_edge) directional lane tokens, e.g.
# "S\u2192E" means "entered the tile through its South edge, exited through
# its East edge". This was referenced throughout the pipeline but never
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


# Maps a directional lane token to the undirected structural tile type it
# implies. Defined once at module scope so analyze_tile_traversals_tile_types
# and map_traversals_to_tile_types can't drift out of sync with each other.
LANE_TO_STRUCTURE = {
    "S\u2192N": "N-S", "N\u2192S": "N-S",
    "W\u2192E": "E-W", "E\u2192W": "E-W",
    "S\u2192E": "ES",  "E\u2192S": "ES",
    "E\u2192N": "NE",  "N\u2192E": "NE",
    "N\u2192W": "WN",  "W\u2192N": "WN",
    "W\u2192S": "SW",  "S\u2192W": "SW",
}


def connects(tile_type, direction):
    return direction in DIR_SETS[tile_type]


# ============================================================
# 2. Compatibility matrix for belief propagation
# ============================================================
COMPATIBILITY_MATRIX = {d: np.zeros((NUM_TYPES, NUM_TYPES)) for d in DIRECTIONS}

for d in DIRECTIONS:
    opp_d = OPPOSITE_DIR[d]
    for i, t1 in enumerate(TILE_TYPES):
        for j, t2 in enumerate(TILE_TYPES):
            t1_conn = connects(t1, d)
            t2_conn = connects(t2, opp_d)

            if t1 != "empty" and t2 != "empty":
                if t1_conn != t2_conn:
                    continue  # a connection must be mutual

                if t1_conn:
                    is_t1_inter = t1 in ["NES", "ESW", "SWN", "WNE", "4-way"]
                    is_t2_inter = t2 in ["NES", "ESW", "SWN", "WNE", "4-way"]
                    is_t1_curve = t1 in ["NE", "ES", "SW", "WN"]
                    is_t2_curve = t2 in ["NE", "ES", "SW", "WN"]
                    is_t1_straight = t1 in ["N-S", "E-W"]
                    is_t2_straight = t2 in ["N-S", "E-W"]

                    if is_t1_inter and not is_t2_straight:
                        continue
                    if is_t2_inter and not is_t1_straight:
                        continue
                    if is_t1_curve and not is_t2_straight:
                        continue
                    if is_t2_curve and not is_t1_straight:
                        continue

            elif t1 != "empty" and t2 == "empty":
                if t1_conn:
                    continue
            elif t1 == "empty" and t2 != "empty":
                if t2_conn:
                    continue

            COMPATIBILITY_MATRIX[d][i, j] = 1.0


# ============================================================
# 3. Geometry helpers
# ============================================================
def tile_index(x, y, tile_size=TILE_SIZE):
    return int(np.floor(x / tile_size)), int(np.floor(y / tile_size))


def get_neighbor_pos(pos, direction):
    x, y = pos
    if direction == "N":
        return (x, y + 1)
    if direction == "S":
        return (x, y - 1)
    if direction == "E":
        return (x + 1, y)
    if direction == "W":
        return (x - 1, y)
    return None


def direction_to_delta(direction):
    return {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[direction]


def map_to_canonical(tile_type):
    """Maps directed trajectory transitions to their canonical undirected road legs."""
    if tile_type in ["S\u2192N", "N\u2192S", "N-S"]:
        return "N-S"
    if tile_type in ["W\u2192E", "E\u2192W", "E-W"]:
        return "E-W"
    if tile_type in ["N\u2192E", "E\u2192N", "NE"]:
        return "NE"
    if tile_type in ["S\u2192E", "E\u2192S", "ES"]:
        return "ES"
    if tile_type in ["S\u2192W", "W\u2192S", "SW"]:
        return "SW"
    if tile_type in ["N\u2192W", "W\u2192N", "WN"]:
        return "WN"
    if tile_type in ["NES", "ESW", "SWN", "WNE", "4-way"]:
        return tile_type
    return "empty"


def safe_traj(traj):
    if traj is None:
        return False
    return isinstance(traj, np.ndarray) and traj.shape[0] > 1


# ============================================================
# 4. Simple (legacy) tile traversal pipeline
# ============================================================
def analyze_tile_traversals(traj, tile_size=TILE_SIZE):
    if traj is None or len(traj) < 2:
        return {}
    tile_counts = {}
    opposite = OPPOSITE_DIR

    def ensure_tile(t):
        if t not in tile_counts:
            tile_counts[t] = {d: [0, 0] for d in "NESW"}

    prev_tile = tile_index(traj[0, 0], traj[0, 1], tile_size)
    ensure_tile(prev_tile)

    for k in range(1, len(traj)):
        cur_tile = tile_index(traj[k, 0], traj[k, 1], tile_size)
        ensure_tile(cur_tile)
        if cur_tile == prev_tile:
            continue

        di = cur_tile[0] - prev_tile[0]
        dj = cur_tile[1] - prev_tile[1]
        moves = []
        if di != 0:
            moves.append(('E' if di > 0 else 'W', int(np.sign(di))))
        if dj != 0:
            moves.append(('N' if dj > 0 else 'S', int(np.sign(dj))))

        step_tile = prev_tile
        for direction, sign in moves:
            tile_counts[step_tile][direction][1] += 1  # exit
            step_tile = (step_tile[0] + sign, step_tile[1]) if direction in 'EW' \
                        else (step_tile[0], step_tile[1] + sign)
            ensure_tile(step_tile)
            tile_counts[step_tile][opposite[direction]][0] += 1  # enter

        prev_tile = cur_tile

    return tile_counts


def classify_tiles(tile_counts, min_events=1):
    classification = {}
    for tile, dirs in tile_counts.items():
        active = [d for d, (e, x) in dirs.items() if (e + x) >= min_events]
        n = len(active)
        if n == 0:
            classification[tile] = "empty"
        elif n == 2:
            classification[tile] = "straight" if frozenset(active) in \
                ({frozenset(['N', 'S']), frozenset(['E', 'W'])}) else "curve"
        elif n == 3:
            classification[tile] = "intersection_3way"
        elif n == 4:
            classification[tile] = "intersection_4way"
        else:
            classification[tile] = "unknown"
    return classification


def get_active_sides(tile_counts, min_events=1):
    active = {}
    for tile, dirs in tile_counts.items():
        active[tile] = [
            d for d, (e, x) in dirs.items()
            if (e + x) >= min_events
        ]
    return active


# ============================================================
# 5. Road rendering primitives
# ============================================================
def draw_tile_road(ax, i, j, ttype, active_sides,
                    tile_size=TILE_SIZE,
                    lane_offset=LANE_OFFSET):

    x0, y0 = i * tile_size, j * tile_size
    cx, cy = x0 + tile_size / 2, y0 + tile_size / 2

    bg = '#2e7d32' if ttype == 'empty' else 'black'
    ax.add_patch(Rectangle((x0, y0), tile_size, tile_size,
                            facecolor=bg, edgecolor='none', zorder=1))

    if ttype == 'empty':
        return

    def white(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='white', linewidth=2, zorder=2)

    def yellow(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#f4d500', linewidth=1.5,
                linestyle=(0, (3, 2)), zorder=2)

    if ttype == "straight":
        if "N" in active_sides and "S" in active_sides:
            yellow((cx, y0), (cx, y0 + tile_size))
            white((x0 + lane_offset, y0), (x0 + lane_offset, y0 + tile_size))
            white((x0 + tile_size - lane_offset, y0),
                  (x0 + tile_size - lane_offset, y0 + tile_size))
        else:
            yellow((x0, cy), (x0 + tile_size, cy))
            white((x0, y0 + lane_offset), (x0 + tile_size, y0 + lane_offset))
            white((x0, y0 + tile_size - lane_offset),
                  (x0 + tile_size, y0 + tile_size - lane_offset))

    elif ttype == "curve":
        sides = active_sides[:2]
        key = frozenset(sides)

        corners = {
            frozenset(["N", "E"]): (x0 + tile_size, y0 + tile_size, 180, 270),
            frozenset(["N", "W"]): (x0, y0 + tile_size, 270, 360),
            frozenset(["S", "E"]): (x0 + tile_size, y0, 90, 180),
            frozenset(["S", "W"]): (x0, y0, 0, 90),
        }

        if key not in corners:
            return

        cx0, cy0, t1, t2 = corners[key]

        ax.add_patch(Arc(
            (cx0, cy0), tile_size, tile_size,
            theta1=t1, theta2=t2,
            color='#f4d500', linewidth=1.5,
            linestyle=(0, (3, 2)), zorder=2
        ))

    elif ttype in ("intersection_3way", "intersection_4way"):
        mid = {
            "N": (cx, y0 + tile_size),
            "S": (cx, y0),
            "E": (x0 + tile_size, cy),
            "W": (x0, cy),
        }

        for s in active_sides:
            yellow((cx, cy), mid[s])

        if "N" not in active_sides:
            white((x0, y0 + tile_size - lane_offset),
                  (x0 + tile_size, y0 + tile_size - lane_offset))
        if "S" not in active_sides:
            white((x0, y0 + lane_offset), (x0 + tile_size, y0 + lane_offset))
        if "E" not in active_sides:
            white((x0 + tile_size - lane_offset, y0),
                  (x0 + tile_size - lane_offset, y0 + tile_size))
        if "W" not in active_sides:
            white((x0 + lane_offset, y0), (x0 + lane_offset, y0 + tile_size))


# ============================================================
# 6. Tile-road plot (fixed: ax reuse + show + always returns fig, ax)
# ============================================================
def plot_tile_roads(classification,
                     tile_counts,
                     tile_size=TILE_SIZE,
                     min_events=1,
                     title="Duckietown Map",
                     ax=None,
                     show=False):

    active = get_active_sides(tile_counts, min_events)

    tiles = list(classification.keys())
    if not tiles:
        raise ValueError("classification is empty -- nothing to plot")

    i_min, i_max = min(t[0] for t in tiles), max(t[0] for t in tiles)
    j_min, j_max = min(t[1] for t in tiles), max(t[1] for t in tiles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            ttype = classification.get((i, j), "empty")
            sides = active.get((i, j), [])
            draw_tile_road(ax, i, j, ttype, sides, tile_size)

    ax.set_xlim(i_min * tile_size, (i_max + 1) * tile_size)
    ax.set_ylim(j_min * tile_size, (j_max + 1) * tile_size)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, linewidth=0.5, alpha=0.4)

    if show:
        plt.show()

    return fig, ax


# ============================================================
# 7. Street graph construction + plot
# ============================================================
def build_street_graph(classification,
                        tile_counts,
                        tile_size=TILE_SIZE,
                        lane_offset=LANE_OFFSET,
                        min_events=1):

    active = get_active_sides(tile_counts, min_events)

    node_map = {}
    node_pose = {}
    edges = []
    next_id = 0

    def get_node(pos):
        nonlocal next_id
        key = (round(pos[0], 4), round(pos[1], 4))
        if key not in node_map:
            node_map[key] = next_id
            node_pose[next_id] = pos
            next_id += 1
        return node_map[key]

    def lane_nodes(i, j):
        x0, y0 = i * tile_size, j * tile_size
        cx, cy = x0 + tile_size / 2, y0 + tile_size / 2

        return {
            "N": ((cx, y0 + tile_size), (cx, y0 + tile_size)),
            "S": ((cx, y0), (cx, y0)),
            "E": ((x0 + tile_size, cy), (x0 + tile_size, cy)),
            "W": ((x0, cy), (x0, cy)),
        }

    for (i, j), ttype in classification.items():
        sides = active.get((i, j), [])
        if not sides:
            continue

        lanes = lane_nodes(i, j)

        for a in sides:
            for b in sides:
                if a == b:
                    continue
                n1 = get_node(lanes[a][0])
                n2 = get_node(lanes[b][1])
                edges.append((n1, n2))

    return node_pose, edges


def plot_street_graph(classification,
                       tile_counts,
                       tile_size=TILE_SIZE,
                       min_events=1,
                       ax=None,
                       show=False):

    node_pose, edges = build_street_graph(
        classification, tile_counts, tile_size, LANE_OFFSET, min_events
    )

    fig, ax = plot_tile_roads(
        classification, tile_counts, tile_size, min_events,
        title="Street Graph", ax=ax
    )

    for a, b in edges:
        x1, y1 = node_pose[a]
        x2, y2 = node_pose[b]

        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2, alpha=0.8),
            zorder=5
        )

    xs = [p[0] for p in node_pose.values()]
    ys = [p[1] for p in node_pose.values()]
    ax.scatter(xs, ys, s=20, c="deepskyblue", edgecolors="black", zorder=6)
    
    # Use classification (not tile_probabilities) to get tile indices
    tile_indices_x = sorted({pos[0] for pos in classification.keys()})
    tile_indices_y = sorted({pos[1] for pos in classification.keys()})
    x_ticks = np.arange(min(tile_indices_x), max(tile_indices_x) + 2) * tile_size
    y_ticks = np.arange(min(tile_indices_y), max(tile_indices_y) + 2) * tile_size
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)

    if show:
        plt.show()

    return fig, ax, node_pose, edges


# ============================================================
# 8. Probabilistic tile pipeline (trajectory -> per-tile belief)
# ============================================================
def analyze_tile_traversals_tile_types(traj, tile_size=TILE_SIZE, tile_types=TILE_TYPES):
    if traj is None or len(traj) < 2:
        return {}, {}, {}, {}

    tile_counts = {}
    opposite = OPPOSITE_DIR

    def ensure_tile(t):
        if t not in tile_counts:
            tile_counts[t] = {td: 0 for td in TILE_DIRECTIONS.keys()}

    def get_move_dir(from_t, to_t):
        di = to_t[0] - from_t[0]
        dj = to_t[1] - from_t[1]
        if di > 0:
            return 'E', 1
        if di < 0:
            return 'W', -1
        if dj > 0:
            return 'N', 1
        if dj < 0:
            return 'S', -1
        return None, 0

    # 1. First pass: reconstruct orthogonal steps from trajectory transitions
    tile_history = []
    prev_tile = tile_index(traj[0, 0], traj[0, 1], tile_size)
    tile_history.append(prev_tile)

    for k in range(1, len(traj)):
        cur_tile = tile_index(traj[k, 0], traj[k, 1], tile_size)
        if cur_tile == prev_tile:
            continue

        di = cur_tile[0] - prev_tile[0]
        dj = cur_tile[1] - prev_tile[1]

        step_tile = prev_tile
        if di != 0:
            sign = int(np.sign(di))
            for _ in range(abs(di)):
                step_tile = (step_tile[0] + sign, step_tile[1])
                tile_history.append(step_tile)
        if dj != 0:
            sign = int(np.sign(dj))
            for _ in range(abs(dj)):
                step_tile = (step_tile[0], step_tile[1] + sign)
                tile_history.append(step_tile)

        prev_tile = cur_tile

    # 2. Second pass: calculate internal lane traversals
    for idx in range(1, len(tile_history) - 1):
        prev_t = tile_history[idx - 1]
        curr_t = tile_history[idx]
        next_t = tile_history[idx + 1]

        ensure_tile(curr_t)

        in_dir_raw, _ = get_move_dir(prev_t, curr_t)
        if in_dir_raw is None:
            continue
        entry_edge = opposite[in_dir_raw]

        exit_edge, _ = get_move_dir(curr_t, next_t)
        if exit_edge is None:
            continue

        matched_token = None
        for token, (src, dest) in TILE_DIRECTIONS.items():
            if src == entry_edge and dest == exit_edge:
                matched_token = token
                break

        if matched_token:
            tile_counts[curr_t][matched_token] += 1

    # 3. Distinct entry-exit traversal visit counts
    total_visits = {}
    if tile_history:
        total_visits[tile_history[0]] = 1
        for idx in range(1, len(tile_history)):
            if tile_history[idx] != tile_history[idx - 1]:
                t_pos = tile_history[idx]
                total_visits[t_pos] = total_visits.get(t_pos, 0) + 1

    # 4. Translate lane data into structure profiles, probabilities, and entropy
    vote_counts = {}
    tile_probabilities = {}
    uncertainty = {}

    for pos in total_visits.keys():
        vote_counts[pos] = {t: 0 for t in tile_types}
        tile_probabilities[pos] = {t: 1.0 / len(tile_types) for t in tile_types}

        has_votes = False
        if pos in tile_counts:
            for lane_token, count in tile_counts[pos].items():
                if count > 0 and lane_token in LANE_TO_STRUCTURE:
                    struct_type = LANE_TO_STRUCTURE[lane_token]
                    if struct_type in vote_counts[pos]:
                        vote_counts[pos][struct_type] += count
                        has_votes = True

        if not has_votes:
            vote_counts[pos]["empty"] = 1

        raw_votes = np.array([vote_counts[pos][t] for t in tile_types], dtype=float)
        sum_votes = np.sum(raw_votes)

        if sum_votes > 0:
            probs = raw_votes / sum_votes
            for i, t in enumerate(tile_types):
                tile_probabilities[pos][t] = probs[i]
        else:
            probs = np.array([tile_probabilities[pos][t] for t in tile_types])

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        uncertainty[pos] = entropy

    return tile_probabilities, uncertainty, vote_counts, total_visits


def map_traversals_to_tile_types(traversal_counts):
    """Translates directional lane transition counts into structural tile-type votes."""
    structural_vote_counts = {}

    for pos, counts in traversal_counts.items():
        structural_vote_counts[pos] = {t: 0 for t in TILE_TYPES}
        has_votes = False

        for lane_token, vote_value in counts.items():
            if vote_value > 0 and lane_token in LANE_TO_STRUCTURE:
                target_tile_type = LANE_TO_STRUCTURE[lane_token]
                if target_tile_type in structural_vote_counts[pos]:
                    structural_vote_counts[pos][target_tile_type] += vote_value
                    has_votes = True

        if not has_votes:
            structural_vote_counts[pos]["empty"] = 1

    return structural_vote_counts


# ============================================================
# 9. Global belief propagation engine
# ============================================================
def propagate_constraints(vote_counts, total_visits, observed_tiles,
                           max_iterations=25, damping=0.4,
                           debug_pos=None, verbose=False):
    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING SUM-PRODUCT BELIEF PROPAGATION ENGINE")
        print("=" * 70)

    all_positions = set(observed_tiles.keys()) | set(vote_counts.keys())
    for pos in list(all_positions):
        for d in DIRECTIONS:
            all_positions.add(get_neighbor_pos(pos, d))

    unaries = {}
    for pos in all_positions:
        u = np.zeros(NUM_TYPES)

        if pos in observed_tiles:
            canon = map_to_canonical(observed_tiles[pos])
            u[TYPE_TO_IDX[canon]] = 15.0
        elif pos in vote_counts and total_visits.get(pos, 0) > 0:
            tot = total_visits[pos]
            has_real_votes = any(k != "empty" and v > 0 for k, v in vote_counts[pos].items())

            if has_real_votes:
                u[TYPE_TO_IDX["empty"]] = 0.1
                for v_type, count in vote_counts[pos].items():
                    canon = map_to_canonical(v_type)
                    u[TYPE_TO_IDX[canon]] += (count / tot) * 4.0
            else:
                u[TYPE_TO_IDX["empty"]] = 15.0
        else:
            u[TYPE_TO_IDX["empty"]] = 12.0

        unaries[pos] = np.exp(u - np.max(u))
        unaries[pos] /= np.sum(unaries[pos])

    messages = {pos: {d: np.ones(NUM_TYPES) / NUM_TYPES for d in DIRECTIONS} for pos in all_positions}

    for iteration in range(max_iterations):
        new_messages = {pos: {} for pos in all_positions}
        max_msg_diff = 0.0

        if verbose and debug_pos in all_positions:
            print(f"\n--- [Iter {iteration}] Diagnostics for tile {debug_pos} ---")

        for pos in all_positions:
            for d in DIRECTIONS:
                neighbor = get_neighbor_pos(pos, d)
                if neighbor not in all_positions:
                    new_messages[pos][d] = messages[pos][d]
                    continue

                incoming = unaries[pos].copy()

                for other_d in DIRECTIONS:
                    if other_d != d:
                        other_neighbor = get_neighbor_pos(pos, other_d)
                        if other_neighbor in all_positions:
                            incoming *= messages[other_neighbor][OPPOSITE_DIR[other_d]]

                msg_out = COMPATIBILITY_MATRIX[d].T @ incoming

                if np.sum(msg_out) > 0:
                    msg_out /= np.sum(msg_out)
                else:
                    msg_out = np.ones(NUM_TYPES) / NUM_TYPES

                damped_msg = (1.0 - damping) * msg_out + damping * messages[pos][d]
                new_messages[pos][d] = damped_msg

                max_msg_diff = max(max_msg_diff, np.max(np.abs(damped_msg - messages[pos][d])))

        messages = new_messages
        if max_msg_diff < 1e-4:
            if verbose:
                print(f"Messages converged at iteration {iteration}.")
            break

    final_types = {}
    intersection_directions = {}

    for pos in all_positions:
        beliefs = unaries[pos].copy()
        for d in DIRECTIONS:
            neighbor = get_neighbor_pos(pos, d)
            if neighbor in all_positions:
                beliefs *= messages[neighbor][OPPOSITE_DIR[d]]

        if np.sum(beliefs) > 0:
            beliefs /= np.sum(beliefs)

        max_idx = np.argmax(beliefs)
        chosen_type = TILE_TYPES[max_idx]

        if beliefs[max_idx] < 0.30:
            chosen_type = "empty"

        final_types[pos] = chosen_type

        if chosen_type in ["NES", "ESW", "SWN", "WNE", "4-way"]:
            dir_probs = {}
            for d in DIRECTIONS:
                neighbor = get_neighbor_pos(pos, d)
                if neighbor in all_positions and final_types.get(neighbor, "empty") != "empty":
                    dir_probs[d] = 1.0 if connects(final_types[neighbor], OPPOSITE_DIR[d]) else 0.0
                else:
                    dir_probs[d] = 1.0 if connects(chosen_type, d) else 0.0
            intersection_directions[pos] = dir_probs

    return final_types, intersection_directions, observed_tiles


# ============================================================
# 10. Plotting: probabilistic belief, tile types, intersections
# ============================================================
_TYPE_COLOR_MAP = {
    "N-S": "#4477AA", "E-W": "#66AA55",
    "NE": "#FF8844", "ES": "#CC6677", "SW": "#AA4499", "WN": "#88CCEE",
    "NES": "#FFD700", "ESW": "#DAA520", "SWN": "#B8860B", "WNE": "#CD853F",
    "4-way": "#FF4500", "empty": "#F5F5F5",
}


def plot_tile_probabilities(tile_probabilities, uncertainty,
                             tile_size=TILE_SIZE, ax=None, show=False):
    """
    Visualizes the raw per-tile belief BEFORE global constraint propagation:
    the most likely structural type per tile, shaded by confidence
    (opaque = low entropy / high confidence, faint = high entropy).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if not tile_probabilities:
        ax.set_title("Tile Probabilities (no data)")
        if show:
            plt.show()
        return fig, ax

    max_entropy = max(uncertainty.values()) if uncertainty else 1.0
    max_entropy = max_entropy if max_entropy > 0 else 1.0

    xs, ys = [], []
    for pos, probs in tile_probabilities.items():
        best_type = max(probs, key=probs.get)
        best_prob = probs[best_type]

        x, y = pos
        xs.append(x)
        ys.append(y)

        entropy = uncertainty.get(pos, 0.0)
        confidence = 1.0 - min(entropy / max_entropy, 1.0)
        alpha = 0.25 + 0.65 * confidence

        rect = Rectangle((x * tile_size, y * tile_size), tile_size, tile_size,
                          facecolor=_TYPE_COLOR_MAP.get(best_type, "#AAAAAA"),
                          alpha=alpha, edgecolor='black', linewidth=0.6)
        ax.add_patch(rect)

        if best_type != "empty":
            cx = x * tile_size + tile_size / 2
            cy = y * tile_size + tile_size / 2
            ax.text(cx, cy, f"{best_type}\n{best_prob:.2f}",
                    ha="center", va="center", fontsize=6)

    ax.set_xlim(min(xs) * tile_size - 1.0, max(xs) * tile_size + 1.0)
    ax.set_ylim(min(ys) * tile_size - 1.0, max(ys) * tile_size + 1.0)
    ax.set_aspect("equal")
    # After setting xlim/ylim
    tile_indices_x = sorted({pos[0] for pos in tile_probabilities.keys()})  # or use i_min, i_max from tiles
    tile_indices_y = sorted({pos[1] for pos in tile_probabilities.keys()})
    x_ticks = np.arange(min(tile_indices_x), max(tile_indices_x) + 2) * tile_size
    y_ticks = np.arange(min(tile_indices_y), max(tile_indices_y) + 2) * tile_size
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_title("Per-Tile Belief (Pre-Propagation)", fontsize=12, pad=10)

    if own_fig:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_tile_types(final_types, observed_tiles, tile_size=TILE_SIZE,
                     ax=None, show=False):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure

    xs, ys = [], []
    for pos, tile_type in final_types.items():
        if tile_type == "empty":
            continue
        x, y = pos
        xs.append(x)
        ys.append(y)
        cx = x * tile_size + tile_size / 2
        cy = y * tile_size + tile_size / 2

        rect = Rectangle((x * tile_size, y * tile_size), tile_size, tile_size,
                          facecolor=_TYPE_COLOR_MAP.get(tile_type, "#AAAAAA"),
                          alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)

        if pos in observed_tiles:
            ax.text(cx - tile_size / 4, cy + tile_size / 4, "\u2605", fontsize=10, color='black')

        ax.text(cx, cy, tile_type, ha="center", va="center", fontsize=7, fontweight='bold')

    if xs:
        ax.set_xlim(min(xs) * tile_size - 1.0, max(xs) * tile_size + 1.0)
        ax.set_ylim(min(ys) * tile_size - 1.0, max(ys) * tile_size + 1.0)

    ax.set_aspect("equal")
    
    # Use final_types (not tile_probabilities) to get tile indices
    tile_indices_x = sorted({pos[0] for pos in final_types.keys()})
    tile_indices_y = sorted({pos[1] for pos in final_types.keys()})
    x_ticks = np.arange(min(tile_indices_x), max(tile_indices_x) + 2) * tile_size
    y_ticks = np.arange(min(tile_indices_y), max(tile_indices_y) + 2) * tile_size
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_title("Globally Consistent Inferred Road Layout", fontsize=12, pad=10)

    if own_fig:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_intersections(intersection_directions, tile_size=TILE_SIZE,
                        ax=None, show=False):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    if not intersection_directions:
        ax.set_title("No structural intersections detected")
        if own_fig:
            fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    xs, ys = [], []
    for pos, dir_probs in intersection_directions.items():
        x, y = pos
        xs.append(x)
        ys.append(y)
        cx = x * tile_size + tile_size / 2
        cy = y * tile_size + tile_size / 2

        ax.add_patch(Circle((cx, cy), tile_size * 0.35, facecolor='gold',
                             edgecolor='black', linewidth=1.5))

        for d, prob in dir_probs.items():
            if prob > 0.5:
                dx, dy = direction_to_delta(d)
                ax.arrow(cx, cy, dx * tile_size * 0.35, dy * tile_size * 0.35,
                          head_width=0.06, head_length=0.08, fc='black', ec='black')
        ax.text(cx, cy, f"({x},{y})", ha="center", va="center", fontsize=8,
                color='black', fontweight='bold')

    ax.set_xlim(min(xs) * tile_size - 1.0, max(xs) * tile_size + 1.0)
    ax.set_ylim(min(ys) * tile_size - 1.0, max(ys) * tile_size + 1.0)
    ax.set_aspect("equal")
    ax.set_title("Intersection Topology and Inferred Active Directions", fontsize=12)

    if own_fig:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


# ============================================================
# 11. Bridge: belief-propagation output -> street-graph input
# ============================================================
BP_TO_CLASSIFICATION = {
    "empty": "empty",
    "N-S": "straight", "E-W": "straight",
    "NE": "curve", "ES": "curve", "SW": "curve", "WN": "curve",
    "NES": "intersection_3way", "ESW": "intersection_3way",
    "SWN": "intersection_3way", "WNE": "intersection_3way",
    "4-way": "intersection_4way",
}


def bp_to_street_graph_inputs(final_types):
    """Builds a classification dict + a synthetic tile_counts dict
    (1 in / 1 out per active side) from the BP engine's final_types."""
    classification = {}
    tile_counts = {}
    for pos, t in final_types.items():
        classification[pos] = BP_TO_CLASSIFICATION.get(t, "empty")
        active = DIR_SETS.get(t, set())
        tile_counts[pos] = {d: ([1, 1] if d in active else [0, 0]) for d in "NESW"}
    return classification, tile_counts


# ============================================================
# 12. Combined side-by-side overview (for publishing as one image)
# ============================================================
def plot_pipeline_overview(tile_probabilities, uncertainty,
                            final_types, observed_tiles,
                            classification, tile_counts,
                            tile_size=TILE_SIZE, min_events=1,
                            figsize=(24, 8), show=False):
    """
    Renders plot_tile_probabilities, plot_tile_types, and plot_street_graph
    side by side in a single figure -- convenient for publishing one combined
    image from a ROS node instead of three separate ones.

    Returns
    -------
    fig, axes : the combined Matplotlib figure and its 3 Axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_tile_probabilities(tile_probabilities, uncertainty,
                             tile_size=tile_size, ax=axes[0])

    plot_tile_types(final_types, observed_tiles,
                     tile_size=tile_size, ax=axes[1])

    plot_street_graph(classification, tile_counts,
                       tile_size=tile_size, min_events=min_events, ax=axes[2])

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
