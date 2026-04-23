"""
=============================================================================
AI Lab: Search Algorithms for Shortest Path Finding
With Obstacles and Risk Factors on a 2D Grid Map
=============================================================================
Author  : [Your Name]
Course  : Artificial Intelligence Lab
Topic   : Search Algorithms — BFS, DFS, Dijkstra, A*, Greedy BFS, UCS
=============================================================================
"""

import heapq
import time
import tracemalloc
from collections import deque
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────────────
# GRID / ENVIRONMENT DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

# Cell type constants
NORMAL   = 0   # cost = 1
BLOCKED  = -1  # impassable
RISKY    = 5   # cost = 5  (road with hazard / high traffic)
START    = 2   # marked for display
GOAL     = 3   # marked for display

def build_grid(layout):
    """
    Convert a list-of-lists layout into a NumPy grid.
    Values: 0=normal, -1=blocked, positive-int=risk-cost
    """
    return np.array(layout, dtype=int)


def cell_cost(grid, row, col):
    """Return traversal cost for a cell (ignore blocked / start / goal markers)."""
    v = grid[row, col]
    if v == BLOCKED:
        return None          # cannot enter
    if v == START or v == GOAL:
        return 1             # treat S/G as normal cost
    return max(1, v)         # 0 → 1, RISKY → 5


def get_neighbors(grid, row, col):
    """4-directional neighbors (up, down, left, right)."""
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < cols:
            cost = cell_cost(grid, r, c)
            if cost is not None:
                neighbors.append(((r, c), cost))
    return neighbors


def reconstruct_path(came_from, start, goal):
    """Walk back through came_from dict to build the path."""
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []           # no path
    path.append(start)
    path.reverse()
    return path


# ──────────────────────────────────────────────────────────────────────────────
# HEURISTIC
# ──────────────────────────────────────────────────────────────────────────────

def manhattan(a, b):
    """Manhattan distance heuristic for a 4-directional grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM 1 — BFS (Breadth-First Search)
# ──────────────────────────────────────────────────────────────────────────────

def bfs(grid, start, goal):
    """
    BFS explores all nodes at depth d before depth d+1.
    Finds the path with the fewest *edges* (ignores edge weights).
    Time  : O(V + E)
    Space : O(V)
    Optimal for unit-cost graphs; NOT optimal for weighted graphs.
    """
    frontier = deque([start])
    came_from = {start: None}
    visited   = set([start])
    nodes_explored = 0

    while frontier:
        current = frontier.popleft()
        nodes_explored += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = sum(cell_cost(grid, r, c) for r, c in path[1:])
            return path, cost, nodes_explored

        for (neighbor, _) in get_neighbors(grid, *current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                frontier.append(neighbor)

    return [], float('inf'), nodes_explored   # no path


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM 2 — DFS (Depth-First Search)
# ──────────────────────────────────────────────────────────────────────────────

def dfs(grid, start, goal):
    """
    DFS dives as deep as possible before backtracking.
    Uses a LIFO stack.
    Time  : O(V + E)
    Space : O(V)  — but recursion depth can be large
    NOT optimal (may return very long paths).
    Can get trapped in deep branches; poor for large graphs.
    """
    frontier = [(start, [start])]
    visited  = set()
    nodes_explored = 0

    while frontier:
        current, path = frontier.pop()

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            cost = sum(cell_cost(grid, r, c) for r, c in path[1:])
            return path, cost, nodes_explored

        for (neighbor, _) in get_neighbors(grid, *current):
            if neighbor not in visited:
                frontier.append((neighbor, path + [neighbor]))

    return [], float('inf'), nodes_explored


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM 3 — Dijkstra's Algorithm (Uniform Cost Search variant)
# ──────────────────────────────────────────────────────────────────────────────

def dijkstra(grid, start, goal):
    """
    Dijkstra expands the node with the lowest cumulative cost first.
    Uses a min-heap priority queue.
    Time  : O((V + E) log V)
    Space : O(V)
    OPTIMAL for non-negative edge weights.
    """
    dist      = {start: 0}
    came_from = {start: None}
    heap      = [(0, start)]
    visited   = set()
    nodes_explored = 0

    while heap:
        cost, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, cost, nodes_explored

        for (neighbor, edge_cost) in get_neighbors(grid, *current):
            new_cost = cost + edge_cost
            if neighbor not in dist or new_cost < dist[neighbor]:
                dist[neighbor]      = new_cost
                came_from[neighbor] = current
                heapq.heappush(heap, (new_cost, neighbor))

    return [], float('inf'), nodes_explored


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM 4 — A* Search
# ──────────────────────────────────────────────────────────────────────────────

def astar(grid, start, goal):
    """
    A* combines Dijkstra's cost-so-far (g) with a heuristic estimate (h).
    f(n) = g(n) + h(n)
    Uses Manhattan distance as admissible heuristic (never over-estimates
    because each step costs at least 1).
    Time  : O((V + E) log V)  in practice much faster than Dijkstra
    Space : O(V)
    OPTIMAL and more EFFICIENT than Dijkstra when heuristic is admissible.
    """
    g         = {start: 0}
    came_from = {start: None}
    h         = manhattan(start, goal)
    heap      = [(h, 0, start)]     # (f, g, node)
    visited   = set()
    nodes_explored = 0

    while heap:
        f, g_cost, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, g_cost, nodes_explored

        for (neighbor, edge_cost) in get_neighbors(grid, *current):
            new_g = g_cost + edge_cost
            if neighbor not in g or new_g < g[neighbor]:
                g[neighbor]         = new_g
                came_from[neighbor] = current
                f_val = new_g + manhattan(neighbor, goal)
                heapq.heappush(heap, (f_val, new_g, neighbor))

    return [], float('inf'), nodes_explored


# ──────────────────────────────────────────────────────────────────────────────
# BONUS ALGORITHM 5 — Greedy Best-First Search
# ──────────────────────────────────────────────────────────────────────────────

def greedy_bfs(grid, start, goal):
    """
    Greedy Best-First uses only the heuristic h(n) — ignores cost so far.
    Very fast in practice but NOT optimal (ignores actual path cost).
    """
    heap      = [(manhattan(start, goal), start)]
    came_from = {start: None}
    visited   = set()
    nodes_explored = 0

    while heap:
        _, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = sum(cell_cost(grid, r, c) for r, c in path[1:])
            return path, cost, nodes_explored

        for (neighbor, _) in get_neighbors(grid, *current):
            if neighbor not in visited:
                came_from[neighbor] = current
                heapq.heappush(heap, (manhattan(neighbor, goal), neighbor))

    return [], float('inf'), nodes_explored


# ──────────────────────────────────────────────────────────────────────────────
# BONUS ALGORITHM 6 — Uniform Cost Search (explicit UCS)
# ──────────────────────────────────────────────────────────────────────────────

def ucs(grid, start, goal):
    """
    UCS is Dijkstra re-stated as a search algorithm.
    Expands nodes in non-decreasing order of path cost.
    Equivalent to Dijkstra; included for completeness.
    OPTIMAL for non-negative costs.
    """
    return dijkstra(grid, start, goal)   # functionally identical


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARKING WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

def run_algorithm(name, func, grid, start, goal):
    """Run a search algorithm and capture timing + memory."""
    tracemalloc.start()
    t0   = time.perf_counter()
    path, cost, nodes = func(grid, start, goal)
    t1   = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "Algorithm"     : name,
        "Path Length"   : len(path),
        "Path Cost"     : cost,
        "Nodes Explored": nodes,
        "Time (ms)"     : round((t1 - t0) * 1000, 4),
        "Peak Mem (KB)" : round(peak_mem / 1024, 2),
        "Path"          : path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

CELL_COLORS = {
    NORMAL  : "#FFFFFF",    # white — normal road
    BLOCKED : "#2C2C2C",    # dark  — wall/obstacle
    RISKY   : "#FF8C00",    # orange — risky road
    START   : "#00B050",    # green
    GOAL    : "#FF0000",    # red
}

def visualise_grid(grid, start, goal, path, title, filename):
    """
    Draw the 2-D grid with colour-coded cells.
    Overlays the solution path as blue squares.
    """
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(max(6, cols * 0.7), max(5, rows * 0.7)))

    for r in range(rows):
        for c in range(cols):
            v = grid[r, c]
            if (r, c) == start:
                color = CELL_COLORS[START]
                label = "S"
            elif (r, c) == goal:
                color = CELL_COLORS[GOAL]
                label = "G"
            elif v == BLOCKED:
                color = CELL_COLORS[BLOCKED]
                label = "#"
            elif v == RISKY:
                color = CELL_COLORS[RISKY]
                label = "R"
            else:
                color = CELL_COLORS[NORMAL]
                label = ""

            rect = plt.Rectangle([c, rows - r - 1], 1, 1,
                                  facecolor=color, edgecolor="#AAAAAA", linewidth=0.8)
            ax.add_patch(rect)
            if label:
                ax.text(c + 0.5, rows - r - 0.5, label,
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        color='white' if color == CELL_COLORS[BLOCKED] else 'black')

    # Draw path
    for (r, c) in path:
        if (r, c) not in (start, goal):
            rect = plt.Rectangle([c + 0.15, rows - r - 0.85], 0.7, 0.7,
                                  facecolor="#1F77B4", edgecolor="none", zorder=3)
            ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=CELL_COLORS[NORMAL],  edgecolor='grey', label='Normal (cost=1)'),
        mpatches.Patch(facecolor=CELL_COLORS[RISKY],   edgecolor='grey', label='Risky   (cost=5)'),
        mpatches.Patch(facecolor=CELL_COLORS[BLOCKED], edgecolor='grey', label='Blocked'),
        mpatches.Patch(facecolor=CELL_COLORS[START],   edgecolor='grey', label='Start (S)'),
        mpatches.Patch(facecolor=CELL_COLORS[GOAL],    edgecolor='grey', label='Goal  (G)'),
        mpatches.Patch(facecolor="#1F77B4",             edgecolor='grey', label='Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {filename}")


def visualise_comparison(results, title, filename):
    """Bar-chart comparison of all algorithms on 3 metrics."""
    algos   = [r["Algorithm"]      for r in results]
    costs   = [r["Path Cost"]      for r in results if r["Path Cost"] < 1e9]
    nodes   = [r["Nodes Explored"] for r in results]
    times   = [r["Time (ms)"]      for r in results]

    x = np.arange(len(algos))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    bars_data = [
        (axes[0], [r["Path Cost"]       for r in results], "Path Cost",        "#4472C4"),
        (axes[1], [r["Nodes Explored"]  for r in results], "Nodes Explored",   "#ED7D31"),
        (axes[2], [r["Time (ms)"]       for r in results], "Execution Time (ms)", "#70AD47"),
    ]
    for ax, vals, ylabel, color in bars_data:
        bars = ax.bar(x, vals, color=color, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT GRIDS
# ──────────────────────────────────────────────────────────────────────────────

# Legend inside layouts:
#  0  = normal cell   (cost 1)
# -1  = blocked cell  (wall)
#  5  = risky cell    (cost 5)
#  2  = start marker
#  3  = goal  marker

SMALL_GRID_LAYOUT = [
    [ 2,  0,  0, -1,  0,  0,  0],
    [ 0, -1,  0, -1,  0, -1,  0],
    [ 0, -1,  0,  0,  0, -1,  0],
    [ 0,  0,  5,  5,  5, -1,  0],
    [-1, -1, -1,  0, -1, -1,  0],
    [ 0,  0,  0,  0,  0,  0,  3],
]

MEDIUM_GRID_LAYOUT = [
    [ 2,  0,  0,  0, -1,  0,  0,  0,  0,  0],
    [ 0, -1, -1,  0, -1,  0, -1, -1,  0,  0],
    [ 0,  0, -1,  0,  0,  0, -1,  0,  0,  0],
    [ 0,  0, -1,  5,  5,  5, -1,  0, -1,  0],
    [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0],
    [ 0,  0, -1,  0, -1,  0, -1,  5,  5,  0],
    [ 0, -1, -1,  0,  0,  0, -1,  0, -1,  0],
    [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  3],
    [ 0, -1,  0,  0, -1,  0, -1,  0, -1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
]

LARGE_GRID_LAYOUT = []
rng = np.random.default_rng(42)
for row_i in range(15):
    row = []
    for col_i in range(15):
        if row_i == 0 and col_i == 0:
            row.append(2)   # start
        elif row_i == 14 and col_i == 14:
            row.append(3)   # goal
        else:
            r = rng.random()
            if r < 0.18:
                row.append(-1)   # blocked
            elif r < 0.32:
                row.append(5)    # risky
            else:
                row.append(0)
    LARGE_GRID_LAYOUT.append(row)

# Simple grid with no obstacles — pure risk path comparison
OPEN_GRID_LAYOUT = [
    [ 2,  0,  0,  0,  0,  0],
    [ 0,  0,  5,  5,  5,  0],
    [ 0,  0,  5,  0,  5,  0],
    [ 0,  0,  5,  0,  5,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  3],
]


# ──────────────────────────────────────────────────────────────────────────────
# PRINT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(results):
    header = f"{'Algorithm':<22} {'Path Len':>9} {'Path Cost':>10} {'Nodes':>8} {'Time(ms)':>10} {'Mem(KB)':>9}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['Algorithm']:<22} {r['Path Length']:>9} {r['Path Cost']:>10} "
              f"{r['Nodes Explored']:>8} {r['Time (ms)']:>10.4f} {r['Peak Mem (KB)']:>9.2f}")
    print(sep)


def print_grid(grid, start, goal, path):
    """ASCII art grid to console."""
    rows, cols = grid.shape
    path_set = set(path)
    symbols = {-1: '▓▓', 0: '  ', 5: 'RR'}
    print()
    for r in range(rows):
        row_str = ''
        for c in range(cols):
            if (r, c) == start:
                row_str += ' S'
            elif (r, c) == goal:
                row_str += ' G'
            elif (r, c) in path_set:
                row_str += ' ·'
            elif grid[r, c] == -1:
                row_str += ' #'
            elif grid[r, c] == 5:
                row_str += ' R'
            else:
                row_str += ' .'
        print(row_str)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL EXPERIMENTS
# ──────────────────────────────────────────────────────────────────────────────

ALGORITHMS = [
    ("BFS",          bfs),
    ("DFS",          dfs),
    ("Dijkstra",     dijkstra),
    ("A*",           astar),
    ("Greedy BFS",   greedy_bfs),
    ("UCS",          ucs),
]

def run_experiment(name, layout, output_dir):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    grid  = build_grid(layout)
    rows, cols = grid.shape

    # Locate S and G
    start = tuple(map(int, np.argwhere(grid == START )[0]))
    goal  = tuple(map(int, np.argwhere(grid == GOAL  )[0]))
    grid[start] = 0   # normalise for cost calculations
    grid[goal]  = 0

    print(f"  Grid size  : {rows}×{cols}")
    print(f"  Start      : {start}   Goal: {goal}")

    results = []
    for alg_name, func in ALGORITHMS:
        res = run_algorithm(alg_name, func, grid, start, goal)
        results.append(res)
        print(f"  {alg_name:<14} path_cost={res['Path Cost']:>6}  "
              f"nodes={res['Nodes Explored']:>5}  time={res['Time (ms)']:.4f}ms")

    print_results_table(results)

    # Visualise best-path (A*)
    astar_res = next(r for r in results if r["Algorithm"] == "A*")
    grid_vis = build_grid(layout)    # re-use original for colours
    vis_path = astar_res["Path"]

    grid_vis[start] = START
    grid_vis[goal]  = GOAL

    img_path   = os.path.join(output_dir, f"{name.replace(' ', '_')}_grid.png")
    chart_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_chart.png")

    visualise_grid(grid_vis, start, goal, vis_path,
                   title=f"{name} — A* Path (cost={astar_res['Path Cost']})",
                   filename=img_path)
    visualise_comparison(results,
                         title=f"Algorithm Comparison — {name}",
                         filename=chart_path)

    # ASCII print
    print("  A* path on grid:")
    print_grid(grid, start, goal, vis_path)

    return results, img_path, chart_path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT = "/mnt/user-data/outputs/ai_search_outputs"
    os.makedirs(OUT, exist_ok=True)

    experiments = [
        ("Exp1_Small_Grid",   SMALL_GRID_LAYOUT),
        ("Exp2_Medium_Grid",  MEDIUM_GRID_LAYOUT),
        ("Exp3_Large_Grid",   LARGE_GRID_LAYOUT),
        ("Exp4_Open_Grid",    OPEN_GRID_LAYOUT),
    ]

    all_results = {}
    all_images  = {}

    for exp_name, layout in experiments:
        res, img, chart = run_experiment(exp_name, layout, OUT)
        all_results[exp_name] = res
        all_images[exp_name]  = (img, chart)

    # ── Aggregate summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY — A* vs Dijkstra vs BFS (all experiments)")
    print(f"{'='*60}")
    for exp_name, results in all_results.items():
        print(f"\n  {exp_name}:")
        for r in results:
            if r["Algorithm"] in ("BFS", "Dijkstra", "A*", "Greedy BFS"):
                print(f"    {r['Algorithm']:<14} cost={r['Path Cost']:>6}  "
                      f"nodes={r['Nodes Explored']:>5}")

    print(f"\n[DONE] All outputs saved to: {OUT}/\n")
