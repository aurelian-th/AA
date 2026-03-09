import pygame
import random
import math
import numpy as np
import time
import argparse
import sys
import heapq
from collections import deque

# --- CONFIGURATION ---
GRID_AREA_SIZE = 600 
UI_WIDTH = 340
WIDTH = GRID_AREA_SIZE + UI_WIDTH
HEIGHT = GRID_AREA_SIZE
DEFAULT_GRID_SIZE = 50 

# Colors
BLACK = (15, 15, 20)
WHITE = (240, 240, 240)
RED = (220, 50, 50)       
GREEN = (50, 220, 50)     
BLUE = (50, 150, 255)     
YELLOW = (255, 200, 50)   
PURPLE = (198, 120, 221)  
CYAN = (86, 182, 194)     
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)
WALL_COLOR = (40, 44, 52)
START_COLOR = (152, 195, 121)
BUTTON_COLOR = (70, 70, 80)
BUTTON_HOVER = (100, 100, 110)
ACTIVE_TOOL_COLOR = (229, 192, 123)

EMPTY = 0
WALL = 1
START = 2
TARGET = 3

class AudioEngine:
    def __init__(self):
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        self.sample_rate, self.format, self.channels = pygame.mixer.get_init()
        self.enabled = True
    
    def play_tone(self, distance, max_distance):
        if not self.enabled: 
            return
            
        normalized = max(0, min(1, 1 - (distance / max_distance))) if max_distance > 0 else 0
        freq = 150 + (normalized * 800)
        duration = 0.03
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        wave = 0.1 * np.sign(np.sin(2 * np.pi * freq * t)) 
        audio_data = (wave * 32767).astype(np.int16)
        
        if self.channels > 1:
            audio_data = np.column_stack((audio_data, audio_data))
            
        sound = pygame.sndarray.make_sound(audio_data)
        sound.set_volume(0.05)
        sound.play(fade_ms=5)

class Button:
    def __init__(self, x, y, width, height, text, mode_key):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.mode_key = mode_key
        self.hovered = False

    def draw(self, surface, font):
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=5)
        
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

class MazeVisualizer:
    def __init__(self, headless=False):
        self.headless = headless
        self.grid = []
        self.grid_size = DEFAULT_GRID_SIZE
        self.cell_size = GRID_AREA_SIZE / self.grid_size
        self.current_mode = "Sparse"
        self.draw_mode = 'P' 
        
        self.start_node = (0, 0)
        self.target_node = (self.grid_size - 1, self.grid_size - 1)
        self.max_dist = 0
        
        self.searching = False
        self.found = False
        self.algo_generator = None
        
        self.nodes_visited = 0
        self.max_memory = 0
        self.start_time = 0
        self.time_elapsed = 0.0
        self.visited_set = set()
        self.frontier_set = set()
        self.current_head = None
        self.recursive_found_flag = False

        self.algorithms = {
            "1. BFS (Basic - List)": self.bfs_basic,
            "2. BFS (Optimized - Deque)": self.bfs_optimized,
            "3. DFS (Basic - Recursive)": self.dfs_basic,
            "4. DFS (Optimized - Iterative)": self.dfs_optimized,
            "5. A* Search (Optimal Heuristic)": self.astar_search,
            "6. Greedy BFS (Speed Heuristic)": self.greedy_search
        }
        self.current_algo_idx = 0
        self.algo_keys = list(self.algorithms.keys())

        if not self.headless:
            pygame.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Graph Traversal Laboratory")
            self.clock = pygame.time.Clock()
            self.audio = AudioEngine()
            
            self.font = pygame.font.SysFont('Consolas', 14)
            self.title_font = pygame.font.SysFont('Consolas', 15, bold=True)
            self.btn_font = pygame.font.SysFont('Consolas', 14, bold=True)

            btn_width = 140
            btn_height = 35
            start_x = GRID_AREA_SIZE + 20
            
            self.buttons = [
                Button(start_x, 340, btn_width, btn_height, "Empty", "Empty"),
                Button(start_x + btn_width + 10, 340, btn_width, btn_height, "Sparse", "Sparse"),
                Button(start_x, 385, btn_width, btn_height, "Dense", "Dense"),
                Button(start_x + btn_width + 10, 385, btn_width, btn_height, "Maze Walls", "Walls"),
            ]
            
            self.btn_minus = Button(start_x + 130, 290, 30, 30, "-", "Minus")
            self.btn_plus = Button(start_x + 170, 290, 30, 30, "+", "Plus")
            self.reset_btn = Button(start_x, 440, btn_width * 2 + 10, btn_height, "Clear Graph Stats", "Clear")

        self.reset_grid("Sparse")

    def reset_grid(self, mode=None, custom_size=None, worst_case_target=False):
        self.clear_stats()
        if mode is not None: 
            self.current_mode = mode
            
        self.grid_size = custom_size if custom_size else self.grid_size
        
        if not self.headless: 
            self.cell_size = GRID_AREA_SIZE / self.grid_size
            
        self.start_node = (0, 0)
        
        if worst_case_target: 
            self.target_node = (0, 1) 
        else:
            tr = min(self.grid_size - 1, self.grid_size - 1)
            tc = min(self.grid_size - 1, self.grid_size - 1)
            self.target_node = (tr, tc)
        
        self.max_dist = abs(self.start_node[0] - self.target_node[0]) + abs(self.start_node[1] - self.target_node[1])
        self.grid = [[EMPTY for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        mode_to_use = self.current_mode
        if mode_to_use == "Sparse":
            self._generate_random_walls(0.25)
            self._force_organic_path()
        elif mode_to_use == "Dense":
            self._generate_random_walls(0.45)
            self._force_organic_path()
        elif mode_to_use == "Walls":
            self._generate_true_maze()

        self.grid[self.start_node[0]][self.start_node[1]] = START
        self.grid[self.target_node[0]][self.target_node[1]] = TARGET

    def clear_stats(self):
        self.searching = False
        self.found = False
        self.nodes_visited = 0
        self.max_memory = 0
        self.time_elapsed = 0.0
        self.visited_set.clear()
        self.frontier_set.clear()
        self.current_head = None
        self.recursive_found_flag = False

    def _generate_random_walls(self, density):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if random.random() < density: 
                    self.grid[r][c] = WALL

    def _generate_true_maze(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size): 
                self.grid[r][c] = WALL
                
        stack = [(0, 0)]
        self.grid[0][0] = EMPTY
        visited = {(0, 0)}
        
        while stack:
            r, c = stack[-1]
            neighbors = []
            
            for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and (nr, nc) not in visited:
                    neighbors.append((nr, nc, r + dr//2, c + dc//2))
                    
            if neighbors:
                nr, nc, wall_r, wall_c = random.choice(neighbors)
                self.grid[wall_r][wall_c] = EMPTY
                self.grid[nr][nc] = EMPTY
                visited.add((nr, nc))
                stack.append((nr, nc))
            else: 
                stack.pop()
                
        self.grid[self.target_node[0]][self.target_node[1]] = EMPTY
        tr, tc = self.target_node
        
        while True:
            if self._is_connected_to_empty(tr, tc): 
                break
                
            if tr > 0 and tc > 0:
                if random.random() < 0.5: 
                    tr -= 1
                else: 
                    tc -= 1
            elif tr > 0: 
                tr -= 1
            elif tc > 0: 
                tc -= 1
                
            self.grid[tr][tc] = EMPTY

    def _is_connected_to_empty(self, r, c):
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.grid[nr][nc] == EMPTY and (nr, nc) != self.target_node:
                return True
        return False

    def _force_organic_path(self):
        r, c = self.start_node
        tr, tc = self.target_node
        
        while r != tr or c != tc:
            self.grid[r][c] = EMPTY
            if r < tr and c < tc:
                if random.random() < 0.5: 
                    r += 1
                else: 
                    c += 1
            elif r < tr: 
                r += 1
            elif c < tc: 
                c += 1
                
        self.grid[tr][tc] = EMPTY

    def handle_draw(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_x < GRID_AREA_SIZE:
            c = int(mouse_x // self.cell_size)
            r = int(mouse_y // self.cell_size)
            
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if pygame.mouse.get_pressed()[0]:
                    if self.draw_mode == 'P' and (r, c) != self.start_node and (r, c) != self.target_node: 
                        self.grid[r][c] = WALL
                    elif self.draw_mode == 'R' and (r, c) != self.start_node and (r, c) != self.target_node: 
                        self.grid[r][c] = EMPTY
                    elif self.draw_mode == 'E' and (r, c) != self.start_node:
                        self.grid[self.target_node[0]][self.target_node[1]] = EMPTY
                        self.target_node = (r, c)
                        self.grid[r][c] = TARGET
                        self.max_dist = abs(self.start_node[0] - self.target_node[0]) + abs(self.start_node[1] - self.target_node[1])
                    elif self.draw_mode == 'S' and (r, c) != self.target_node:
                        self.grid[self.start_node[0]][self.start_node[1]] = EMPTY
                        self.start_node = (r, c)
                        self.grid[r][c] = START
                        self.max_dist = abs(self.start_node[0] - self.target_node[0]) + abs(self.start_node[1] - self.target_node[1])

    def draw(self):
        if self.headless: 
            return
            
        self.window.fill(BLACK)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = c * self.cell_size
                y = r * self.cell_size
                color = BLACK
                
                if self.grid[r][c] == WALL: 
                    color = WALL_COLOR
                elif self.grid[r][c] == START: 
                    color = START_COLOR
                elif self.grid[r][c] == TARGET: 
                    color = GREEN
                elif (r, c) == self.current_head: 
                    color = YELLOW
                elif (r, c) in self.frontier_set: 
                    if "A*" in self.algo_keys[self.current_algo_idx]: 
                        color = PURPLE
                    elif "Greedy" in self.algo_keys[self.current_algo_idx]: 
                        color = CYAN
                    else: 
                        color = BLUE
                elif (r, c) in self.visited_set: 
                    color = RED
                
                pygame.draw.rect(self.window, color, (x, y, math.ceil(self.cell_size), math.ceil(self.cell_size)))
                if self.grid_size <= 60: 
                    pygame.draw.rect(self.window, DARK_GRAY, (x, y, math.ceil(self.cell_size), math.ceil(self.cell_size)), 1)
        
        panel_x = GRID_AREA_SIZE
        pygame.draw.rect(self.window, GRAY, (panel_x, 0, UI_WIDTH, HEIGHT))
        pygame.draw.rect(self.window, DARK_GRAY, (panel_x, 0, UI_WIDTH, HEIGHT), 3)
        
        tx = panel_x + 15
        algo_text = self.title_font.render("Selected Algorithm:", True, WHITE)
        algo_name = self.title_font.render(self.algo_keys[self.current_algo_idx], True, YELLOW)
        self.window.blit(algo_text, (tx, 20))
        self.window.blit(algo_name, (tx, 45))
        
        pygame.draw.line(self.window, WHITE, (tx, 80), (tx + 290, 80))
        self.window.blit(self.font.render(f"Time:       {self.time_elapsed:.3f}s", True, WHITE), (tx, 95))
        self.window.blit(self.font.render(f"Visited:    {self.nodes_visited}", True, WHITE), (tx, 120))
        self.window.blit(self.font.render(f"Max Memory: {self.max_memory}", True, WHITE), (tx, 145))
        pygame.draw.line(self.window, WHITE, (tx, 175), (tx + 290, 175))
        
        self.window.blit(self.font.render("Current Tool:", True, WHITE), (tx, 190))
        
        if self.draw_mode == 'P': 
            tool_str = "PLACE WALLS"
        elif self.draw_mode == 'R': 
            tool_str = "REMOVE WALLS"
        elif self.draw_mode == 'E': 
            tool_str = "MOVE TARGET"
        else: 
            tool_str = "MOVE START"
            
        self.window.blit(self.title_font.render(tool_str, True, ACTIVE_TOOL_COLOR), (tx + 120, 189))
        
        self.window.blit(self.font.render("Hotkeys:", True, WHITE), (tx, 220))
        self.window.blit(self.font.render("[SPACE] Start/Stop | [M] Mute", True, YELLOW), (tx, 240))
        self.window.blit(self.font.render("[UP/DN] Algo | [P,R,E,S] Tools", True, YELLOW), (tx, 260))
        self.window.blit(self.font.render(f"Size: {self.grid_size}x{self.grid_size}", True, WHITE), (tx, 297))
        
        for btn in self.buttons: 
            btn.draw(self.window, self.btn_font)
            
        self.btn_minus.draw(self.window, self.btn_font)
        self.btn_plus.draw(self.window, self.btn_font)
        self.reset_btn.draw(self.window, self.btn_font)
        pygame.display.update()

    def update_display(self, current_node):
        self.nodes_visited += 1
        if self.headless: 
            yield
            return
            
        self.current_head = current_node
        if self.searching: 
            self.time_elapsed = time.time() - self.start_time
            
        draw_freq = 1 if self.grid_size <= 40 else (5 if self.grid_size <= 80 else 20)
        
        if self.nodes_visited % draw_freq == 0:
            dist = abs(current_node[0] - self.target_node[0]) + abs(current_node[1] - self.target_node[1])
            self.audio.play_tone(dist, self.max_dist)
            self.draw()
            yield

    def get_neighbors(self, r, c):
        neighbors = []
        for dr, dc in [(1, 0), (0, -1), (-1, 0), (0, 1)]: 
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if self.grid[nr][nc] != WALL: 
                    neighbors.append((nr, nc))
        return neighbors

    def bfs_basic(self):
        queue = [self.start_node]
        self.visited_set.add(self.start_node)
        
        while queue:
            self.max_memory = max(self.max_memory, len(queue))
            current = queue.pop(0)
            self.frontier_set.discard(current)
            
            yield from self.update_display(current)
            
            if current == self.target_node: 
                self.found = True
                return
                
            for neighbor in self.get_neighbors(current[0], current[1]):
                if neighbor not in self.visited_set:
                    self.visited_set.add(neighbor)
                    self.frontier_set.add(neighbor)
                    queue.append(neighbor)

    def bfs_optimized(self):
        queue = deque([self.start_node])
        self.visited_set.add(self.start_node)
        
        while queue:
            self.max_memory = max(self.max_memory, len(queue))
            current = queue.popleft()
            self.frontier_set.discard(current)
            
            yield from self.update_display(current)
            
            if current == self.target_node: 
                self.found = True
                return
                
            for neighbor in self.get_neighbors(current[0], current[1]):
                if neighbor not in self.visited_set:
                    self.visited_set.add(neighbor)
                    self.frontier_set.add(neighbor)
                    queue.append(neighbor)

    def dfs_basic(self):
        self.recursive_found_flag = False
        self.visited_set.add(self.start_node)
        yield from self._dfs_recursive_helper(self.start_node, 1)

    def _dfs_recursive_helper(self, current, depth):
        if self.recursive_found_flag: 
            return
            
        self.max_memory = max(self.max_memory, depth)
        yield from self.update_display(current)
        
        if current == self.target_node: 
            self.recursive_found_flag = True
            self.found = True
            return
            
        for neighbor in self.get_neighbors(current[0], current[1]):
            if neighbor not in self.visited_set and not self.recursive_found_flag:
                self.visited_set.add(neighbor)
                yield from self._dfs_recursive_helper(neighbor, depth + 1)

    def dfs_optimized(self):
        stack = [self.start_node]
        
        while stack:
            self.max_memory = max(self.max_memory, len(stack))
            current = stack.pop()
            
            if current not in self.visited_set:
                self.visited_set.add(current)
                yield from self.update_display(current)
                
                if current == self.target_node: 
                    self.found = True
                    return
                    
                for neighbor in reversed(self.get_neighbors(current[0], current[1])):
                    if neighbor not in self.visited_set: 
                        stack.append(neighbor)

    def astar_search(self):
        count = 0
        pq = [(0, count, self.start_node)]
        self.visited_set.add(self.start_node)
        g_score = {self.start_node: 0}
        
        while pq:
            self.max_memory = max(self.max_memory, len(pq))
            current_f, _, current = heapq.heappop(pq)
            self.frontier_set.discard(current)
            
            yield from self.update_display(current)
            
            if current == self.target_node: 
                self.found = True
                return
                
            for neighbor in self.get_neighbors(current[0], current[1]):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    h_score = abs(neighbor[0] - self.target_node[0]) + abs(neighbor[1] - self.target_node[1])
                    f_score = tentative_g_score + h_score
                    
                    if neighbor not in self.visited_set:
                        self.visited_set.add(neighbor)
                        self.frontier_set.add(neighbor)
                        count += 1
                        heapq.heappush(pq, (f_score, count, neighbor))

    def greedy_search(self):
        count = 0
        pq = [(0, count, self.start_node)]
        self.visited_set.add(self.start_node)
        
        while pq:
            self.max_memory = max(self.max_memory, len(pq))
            _, _, current = heapq.heappop(pq)
            self.frontier_set.discard(current)
            
            yield from self.update_display(current)
            
            if current == self.target_node: 
                self.found = True
                return
                
            for neighbor in self.get_neighbors(current[0], current[1]):
                if neighbor not in self.visited_set:
                    self.visited_set.add(neighbor)
                    self.frontier_set.add(neighbor)
                    h_score = abs(neighbor[0] - self.target_node[0]) + abs(neighbor[1] - self.target_node[1])
                    count += 1
                    heapq.heappush(pq, (h_score, count, neighbor))

    def run(self):
        running = True
        while running:
            self.clock.tick(240)
            mouse_pos = pygame.mouse.get_pos()
            
            for btn in self.buttons: 
                btn.check_hover(mouse_pos)
            self.btn_minus.check_hover(mouse_pos)
            self.btn_plus.check_hover(mouse_pos)
            self.reset_btn.check_hover(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: 
                    for btn in self.buttons:
                        if btn.is_clicked(mouse_pos) and not self.searching: 
                            self.reset_grid(btn.mode_key)
                    if self.btn_minus.is_clicked(mouse_pos) and not self.searching: 
                        self.grid_size = max(10, self.grid_size - 10)
                        self.reset_grid()
                    if self.btn_plus.is_clicked(mouse_pos) and not self.searching: 
                        self.grid_size = min(150, self.grid_size + 10)
                        self.reset_grid()
                    if self.reset_btn.is_clicked(mouse_pos) and not self.searching: 
                        self.clear_stats()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p: self.draw_mode = 'P'
                    if event.key == pygame.K_r: self.draw_mode = 'R'
                    if event.key == pygame.K_e: self.draw_mode = 'E'
                    if event.key == pygame.K_s: self.draw_mode = 'S'
                    if event.key == pygame.K_m: self.audio.enabled = not self.audio.enabled
                    if event.key == pygame.K_SPACE and not self.searching:
                        self.clear_stats()
                        self.searching = True
                        self.start_time = time.time()
                        algo_func = self.algorithms[self.algo_keys[self.current_algo_idx]]
                        sys.setrecursionlimit(500000)
                        self.algo_generator = algo_func()
                    if event.key == pygame.K_DOWN and not self.searching:
                        self.current_algo_idx = (self.current_algo_idx + 1) % len(self.algorithms)
                        self.clear_stats() 
                    if event.key == pygame.K_UP and not self.searching:
                        self.current_algo_idx = (self.current_algo_idx - 1) % len(self.algorithms)
                        self.clear_stats()
                    
            if pygame.mouse.get_pressed()[0] and not self.searching:
                is_hovering_btn = any(btn.hovered for btn in self.buttons) or self.btn_minus.hovered or self.btn_plus.hovered or self.reset_btn.hovered
                if not is_hovering_btn: 
                    self.handle_draw()

            if self.searching:
                try: 
                    next(self.algo_generator)
                    self.time_elapsed = time.time() - self.start_time
                except StopIteration: 
                    self.searching = False
                    self.time_elapsed = time.time() - self.start_time
                    self.draw()

            if not self.searching: 
                self.draw()
                
        pygame.quit()

    def generate_graphs(self):
        import matplotlib.pyplot as plt
        sys.setrecursionlimit(500000) 
        test_sizes = [50, 100, 200, 300] 
        
        short_names = {
            "1. BFS (Basic - List)": "BFS",
            "2. BFS (Optimized - Deque)": "BFS(O)",
            "3. DFS (Basic - Recursive)": "DFS",
            "4. DFS (Optimized - Iterative)": "DFS(O)",
            "5. A* Search (Optimal Heuristic)": "A*",
            "6. Greedy BFS (Speed Heuristic)": "Greedy"
        }

        all_results = {mode: {size: {} for size in test_sizes} for mode in ["Empty", "Sparse", "Dense", "Walls"]}
        
        print("\n" + "="*90)
        print(" GRAPH TRAVERSAL BENCHMARK ")
        print("="*90)

        modes = ["Empty", "Sparse", "Dense", "Walls"]
        for mode in modes:
            print(f"\nProcessing Data Distribution: {mode}")
            plt.figure(figsize=(10, 6))
            for algo_name in self.algo_keys:
                algo_times = []
                for size in test_sizes:
                    self.reset_grid(mode=mode, custom_size=size, worst_case_target=False)
                    gen = self.algorithms[algo_name]()
                    start_time = time.perf_counter()
                    exec_time = None
                    try:
                        for _ in gen: pass
                        exec_time = time.perf_counter() - start_time
                    except RecursionError: 
                        pass
                    except Exception: 
                        pass
                        
                    algo_times.append(exec_time)
                    all_results[mode][size][algo_name] = exec_time
                
                valid_sizes = [s for s, t in zip(test_sizes, algo_times) if t is not None]
                valid_times = [t for t in algo_times if t is not None]
                plt.plot(valid_sizes, valid_times, label=short_names[algo_name], marker='o', linewidth=2)
            
            plt.title(f"Performance Scaling: {mode} Environment", fontsize=14, fontweight='bold')
            plt.xlabel("Grid Dimensions (N x N)", fontsize=12)
            plt.ylabel("Execution Time (seconds)", fontsize=12)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            filename = f"graph_benchmark_{mode.lower()}.png"
            plt.savefig(filename, dpi=300)
            plt.close() 
            print(f"  Saved {filename}")

        print("\n" + "="*90)
        print(" BENCHMARK RESULTS (seconds)")
        print("="*90)
        
        header = f"{'Size':<8} | " + " | ".join([f"{short_names[a]:<8}" for a in self.algo_keys])
        separator = "-" * len(header)
        
        for mode in modes:
            print(f"\n[ Distribution: {mode.upper()} ]")
            print(separator)
            print(header)
            print(separator)
            
            for n in test_sizes:
                row_str = f"{str(n)+'x'+str(n):<8} | "
                for algo in self.algo_keys:
                    val = all_results[mode][n][algo]
                    if val is None: 
                        row_str += f"{'ERROR':<8} | "
                    else: 
                        row_str += f"{val:<8.4f} | "
                print(row_str[:-3]) 
            print(separator)

        print("\nAll tasks complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    
    if args.graph:
        app = MazeVisualizer(headless=True)
        app.generate_graphs()
    else:
        app = MazeVisualizer(headless=False)
        app.run()