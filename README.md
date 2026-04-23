# AI Search with Heuristics

## Overview

This project implements and compares classic AI search algorithms for shortest path finding in two scenarios:
- **2D Grid Maps** with obstacles and risk factors
- **Real-World City Map** (Dhaka) with multi-objective costs (distance, time, risk, blocked roads)

Algorithms implemented:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Dijkstra’s Algorithm (Uniform Cost Search)
- A* Search
- Greedy Best-First Search

## 1. Grid-Based Search

### Features
- **Grid Representation:** 2D numpy arrays with normal, blocked, risky, start, and goal cells.
- **Cell Costs:** Normal (1), Risky (5), Blocked (impassable).
- **Visualization:** Matplotlib plots for grids, paths, and algorithm comparisons.
- **Experiments:** Small, medium, large, and open grids.

### Algorithms
- **BFS:** Finds shortest path in steps (ignores weights).
- **DFS:** Explores deep paths, not optimal for cost.
- **Dijkstra/UCS:** Finds lowest-cost path, optimal for weighted graphs.
- **A\* Search:** Combines cost-so-far and heuristic (Manhattan distance).
- **Greedy BFS:** Uses only heuristic, not optimal.

### Results
- Each algorithm is benchmarked for path cost, nodes explored, time, and memory.
- Visual outputs: grid with path, bar charts comparing algorithms.

## 2. Map-Based Search (Dhaka City)

### Features
- **Graph Representation:** Nodes are locations, edges have length, time, incidents, blocked status, and type (primary/alley).
- **Multi-Objective Cost:** Customizable by user profile (gender, group size, time of day, preference for fastest/safest).
- **Blocked Roads:** Some roads are impassable.
- **Visualization:** NetworkX graph with color-coded edges and blocked roads.

### Algorithms
- **BFS/DFS:** Pathfinding ignoring edge weights.
- **UCS/Dijkstra, Greedy, A\*:** Use custom cost and heuristic functions.

### Evaluation
- **Profiles:** Multiple user profiles (gender, group size, day/night, preference).
- **Metrics:** Path cost, time, nodes checked, path taken.
- **Comprehensive Analysis:** All algorithms run for all profiles, results tabulated.

## How to Run

1. **Grid Search:**  
	 Run `ai__Grid__search.py` to execute all grid experiments and generate visualizations.
2. **Map Search:**  
	 Open and run `ai__map_search.ipynb` for city map experiments and visualizations.

## Example Outputs

- **Grid Path Visualization:**  
	![Grid Example](outputs/Exp1_Small_Grid_grid.png)
- **Algorithm Comparison Chart:**  
	![Comparison Chart](outputs/Exp1_Small_Grid_chart.png)
- **Dhaka Map Visualization:**  
	![Dhaka Map](outputs/dhaka_map.png)

## Report Summary

- **BFS** is optimal for unweighted grids but inefficient for weighted/risky paths.
- **DFS** is not optimal and can get stuck in deep branches.
- **Dijkstra/UCS** always finds the lowest-cost path but can be slower than A*.
- **A\*** is fastest and optimal when heuristic is admissible.
- **Greedy** is fastest but not always optimal.
- **Map Search** shows how user profile and risk factors affect path selection, especially for vulnerable groups at night.

## Author

- Mehedi Hasan Sakib
- CSEDU
