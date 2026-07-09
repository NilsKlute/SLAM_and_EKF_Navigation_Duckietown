#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle, Arc

# -----------------------------
# Constants
# -----------------------------
TILE_SIZE = 0.61
LANE_OFFSET = TILE_SIZE * 0.15

def tile_index(x, y, tile_size=TILE_SIZE):
    return int(np.floor(x / tile_size)), int(np.floor(y / tile_size))


def safe_traj(traj):
    if traj is None:
        return False
    return isinstance(traj, np.ndarray) and traj.shape[0] > 1

def analyze_tile_traversals(traj, tile_size=TILE_SIZE):
    if traj is None or len(traj) < 2:
        return {}
    tile_counts = {}
    opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

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


# -----------------------------
# Helpers
# -----------------------------
def get_active_sides(tile_counts, min_events=1):
    active = {}
    for tile, dirs in tile_counts.items():
        active[tile] = [
            d for d, (e, x) in dirs.items()
            if (e + x) >= min_events
        ]
    return active


# -----------------------------
# Road rendering
# -----------------------------
def draw_tile_road(ax, i, j, ttype, active_sides,
                   tile_size=TILE_SIZE,
                   lane_offset=LANE_OFFSET):

    x0, y0 = i * tile_size, j * tile_size
    cx, cy = x0 + tile_size / 2, y0 + tile_size / 2

    # background
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

    def red(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='red', linewidth=3, zorder=3)

    # -----------------------------
    # Straight
    # -----------------------------
    if ttype == "straight":
        if "N" in active_sides and "S" in active_sides:
            yellow((cx, y0), (cx, y0 + tile_size))
            white((x0 + lane_offset, y0),
                  (x0 + lane_offset, y0 + tile_size))
            white((x0 + tile_size - lane_offset, y0),
                  (x0 + tile_size - lane_offset, y0 + tile_size))
        else:
            yellow((x0, cy), (x0 + tile_size, cy))
            white((x0, y0 + lane_offset),
                  (x0 + tile_size, y0 + lane_offset))
            white((x0, y0 + tile_size - lane_offset),
                  (x0 + tile_size, y0 + tile_size - lane_offset))

    # -----------------------------
    # Curve
    # -----------------------------
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
            (cx0, cy0),
            2 * tile_size / 2,
            2 * tile_size / 2,
            theta1=t1,
            theta2=t2,
            color='#f4d500',
            linewidth=1.5,
            linestyle=(0, (3, 2)),
            zorder=2
        ))

    # -----------------------------
    # Intersections
    # -----------------------------
    elif ttype in ("intersection_3way", "intersection_4way"):

        mid = {
            "N": (cx, y0 + tile_size),
            "S": (cx, y0),
            "E": (x0 + tile_size, cy),
            "W": (x0, cy),
        }

        for s in active_sides:
            yellow((cx, cy), mid[s])

        # lane lines
        if "N" not in active_sides:
            white((x0, y0 + tile_size - lane_offset),
                  (x0 + tile_size, y0 + tile_size - lane_offset))
        if "S" not in active_sides:
            white((x0, y0 + lane_offset),
                  (x0 + tile_size, y0 + lane_offset))
        if "E" not in active_sides:
            white((x0 + tile_size - lane_offset, y0),
                  (x0 + tile_size - lane_offset, y0 + tile_size))
        if "W" not in active_sides:
            white((x0 + lane_offset, y0),
                  (x0 + lane_offset, y0 + tile_size))


# -----------------------------
# Main plotting function
# -----------------------------
def plot_tile_roads(classification,
                    tile_counts,
                    tile_size=TILE_SIZE,
                    min_events=1,
                    title="Duckietown Map"):

    active = get_active_sides(tile_counts, min_events)

    tiles = list(classification.keys())
    i_min, i_max = min(t[0] for t in tiles), max(t[0] for t in tiles)
    j_min, j_max = min(t[1] for t in tiles), max(t[1] for t in tiles)

    fig, ax = plt.subplots(figsize=(10, 10))

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

    return fig, ax


# -----------------------------
# Graph construction
# -----------------------------
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


# -----------------------------
# Full street graph plot
# -----------------------------
def plot_street_graph(classification,
                      tile_counts,
                      tile_size=TILE_SIZE,
                      min_events=1,
                      show=False):

    node_pose, edges = build_street_graph(
        classification,
        tile_counts,
        tile_size,
        LANE_OFFSET,
        min_events
    )

    fig, ax = plot_tile_roads(
        classification,
        tile_counts,
        tile_size,
        min_events,
        title="Street Graph"
    )

    for a, b in edges:
        x1, y1 = node_pose[a]
        x2, y2 = node_pose[b]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->",
                            color="cyan",
                            lw=1.2,
                            alpha=0.8),
            zorder=5
        )

    xs = [p[0] for p in node_pose.values()]
    ys = [p[1] for p in node_pose.values()]

    ax.scatter(xs, ys, s=20, c="deepskyblue", edgecolors="black", zorder=6)

    if show:
        plt.show()

    return fig, ax, node_pose, edges

