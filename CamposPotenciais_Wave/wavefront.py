# wavefront.py
import numpy as np
from collections import deque

def world_to_grid(x, y, origem_x, origem_y, resolucao):
    col = int((x - origem_x) / resolucao)
    row = int((y - origem_y) / resolucao)
    return row, col

def grid_to_world(row, col, origem_x, origem_y, resolucao):
    x = origem_x + (col * resolucao) + (resolucao / 2.0)
    y = origem_y + (row * resolucao) + (resolucao / 2.0)
    return x, y

def compute_wavefront(grid, goal_row, goal_col):
    wavefront = np.array(grid, dtype=float)
    rows, cols = wavefront.shape
    
    for r in range(rows):
        for c in range(cols):
            if wavefront[r, c] != 999:
                wavefront[r, c] = float('inf')
                
    wavefront[goal_row, goal_col] = 0
    queue = deque([(goal_row, goal_col)])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while queue:
        curr_r, curr_c = queue.popleft()
        current_val = wavefront[curr_r, curr_c]
        
        for dr, dc in directions:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cost = 1.414 if dr != 0 and dc != 0 else 1.0
                if wavefront[nr, nc] > current_val + cost and wavefront[nr, nc] != 999:
                    wavefront[nr, nc] = current_val + cost
                    queue.append((nr, nc))
                    
    return wavefront

def get_next_waypoint(r_row, r_col, wavefront, origem_x, origem_y, resolucao):
    rows, cols = wavefront.shape
    min_val = float('inf')
    best_r, best_c = r_row, r_col
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for dr, dc in directions:
        nr, nc = r_row + dr, r_col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if wavefront[nr, nc] < min_val:
                min_val = wavefront[nr, nc]
                best_r, best_c = nr, nc
                
    return grid_to_world(best_r, best_c, origem_x, origem_y, resolucao)