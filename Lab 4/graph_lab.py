import pygame
import random
import math
import numpy as np
import time
import argparse
import sys
import heapq
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 700
UI_HEIGHT = 150
CANVAS_HEIGHT = HEIGHT - UI_HEIGHT

# Colors
BLACK = (15, 15, 20)
WHITE = (240, 240, 240)
RED = (220, 50, 50)
GREEN = (50, 220, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 200, 50)
PURPLE = (198, 120, 221)
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)

class AudioEngine:
    def __init__(self):
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sample_rate, self.format, self.channels = pygame.mixer.get_init()
        self.enabled = True
    
    def play_tone(self, weight, max_weight):
        if not self.enabled: return
        normalized = max(0, min(1, 1 - (weight / max_weight))) if max_weight > 0 else 0
        freq = 200 + (normalized * 600)
        duration = 0.05
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        wave = 0.1 * np.sign(np.sin(2 * np.pi * freq * t))
        audio_data = (wave * 32767).astype(np.int16)
        if self.channels > 1:
            audio_data = np.column_stack((audio_data, audio_data))
        sound = pygame.sndarray.make_sound(audio_data)
        sound.set_volume(0.05)
        sound.play(fade_ms=5)

class DPGraphVisualizer:
    def __init__(self, headless=False):
        self.headless = headless
        self.V = 15 
        self.density = 0.25
        
        self.adj_matrix = []
        self.adj_list = {}
        self.node_positions = []
        self.max_weight = 20
        
        self.animating = False
        self.algo_generator = None
        self.start_node = 0
        
        self.algorithms = {
            "1. Dijkstra (Single Source DP Relaxation)": self.dijkstra_generator,
            "2. Floyd-Warshall (All-Pairs DP Matrix)": self.fw_generator
        }
        self.algo_keys = list(self.algorithms.keys())
        self.current_algo_idx = 0
        
        self.distances = {}
        self.visited = set()
        self.current_edge = None
        
        self.fw_i = None
        self.fw_j = None
        self.fw_k = None
        self.fw_dist = []
        self.fw_completed = False # BUG FIX: Added specific flag for completion
        
        if not self.headless:
            pygame.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Dynamic Programming: Graph Algorithms")
            self.clock = pygame.time.Clock()
            self.audio = AudioEngine()
            self.font = pygame.font.SysFont('Consolas', 14)
            self.title_font = pygame.font.SysFont('Consolas', 18, bold=True)
            self.generate_graph()

    def generate_graph(self):
        self.adj_matrix = [[float('inf')] * self.V for _ in range(self.V)]
        self.adj_list = {i: [] for i in range(self.V)}
        self.node_positions = []
        
        for _ in range(self.V):
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            self.node_positions.append((x, y))
            
        for i in range(1, self.V):
            target = random.randint(0, i - 1)
            w = random.randint(1, self.max_weight)
            self.adj_matrix[i][target] = w
            self.adj_matrix[target][i] = w
            self.adj_list[i].append((target, w))
            self.adj_list[target].append((i, w))

        for i in range(self.V):
            self.adj_matrix[i][i] = 0
            for j in range(i + 1, self.V):
                if random.random() < self.density and self.adj_matrix[i][j] == float('inf'):
                    w = random.randint(1, self.max_weight)
                    self.adj_matrix[i][j] = w
                    self.adj_matrix[j][i] = w
                    self.adj_list[i].append((j, w))
                    self.adj_list[j].append((i, w))
                    
        self.reset_traversal()

    def reset_traversal(self):
        self.distances = {node: float('inf') for node in range(self.V)}
        self.distances[self.start_node] = 0
        self.visited = set()
        self.current_edge = None
        
        self.fw_i = self.fw_j = self.fw_k = None
        self.fw_dist = [[self.adj_matrix[i][j] for j in range(self.V)] for i in range(self.V)]
        self.fw_completed = False # BUG FIX: Reset the completion flag
        
        self.animating = False

    def dijkstra_generator(self):
        pq = [(0, self.start_node)]
        while pq:
            current_dist, u = heapq.heappop(pq)
            if u in self.visited: continue
            self.visited.add(u)
            
            for v, weight in self.adj_list[u]:
                self.current_edge = (u, v)
                self.audio.play_tone(weight, self.max_weight)
                
                if self.distances[u] + weight < self.distances[v]:
                    self.distances[v] = self.distances[u] + weight
                    heapq.heappush(pq, (self.distances[v], v))
                yield

    def fw_generator(self):
        for k in range(self.V):
            self.fw_k = k
            for i in range(self.V):
                self.fw_i = i
                for j in range(self.V):
                    self.fw_j = j
                    
                    if self.fw_dist[i][k] + self.fw_dist[k][j] < self.fw_dist[i][j]:
                        self.fw_dist[i][j] = self.fw_dist[i][k] + self.fw_dist[k][j]
                        self.audio.play_tone(self.fw_dist[i][j], self.max_weight * 3)
                    
                    yield
                    
        self.fw_i = self.fw_j = self.fw_k = None
        self.fw_completed = True # BUG FIX: Mark matrix as fully computed

    def draw(self):
        self.window.fill(BLACK)
        
        current_algo = self.algo_keys[self.current_algo_idx]
        is_fw = "Floyd" in current_algo
        
        hovered_node = None
        # BUG FIX: Only check for hover if FW is 100% completed
        if is_fw and self.fw_completed:
            mx, my = pygame.mouse.get_pos()
            for n, pos in enumerate(self.node_positions):
                if math.hypot(mx - pos[0], my - pos[1]) < 12:
                    hovered_node = n
                    break

        for u in range(self.V):
            for v, w in self.adj_list[u]:
                if u < v: 
                    color = DARK_GRAY
                    width = 1
                    
                    if is_fw and self.fw_completed and hovered_node is not None:
                        if u == hovered_node or v == hovered_node:
                            color = GRAY
                            
                    if not is_fw and self.current_edge and (self.current_edge == (u, v) or self.current_edge == (v, u)):
                        color = YELLOW
                        width = 3
                    
                    pygame.draw.line(self.window, color, self.node_positions[u], self.node_positions[v], width)

        if is_fw and self.animating and self.fw_i is not None and self.fw_j is not None and self.fw_k is not None:
            pygame.draw.line(self.window, PURPLE, self.node_positions[self.fw_i], self.node_positions[self.fw_k], 3)
            pygame.draw.line(self.window, YELLOW, self.node_positions[self.fw_k], self.node_positions[self.fw_j], 3)
            pygame.draw.line(self.window, RED, self.node_positions[self.fw_i], self.node_positions[self.fw_j], 1)

        for n, pos in enumerate(self.node_positions):
            color = BLUE
            
            if not is_fw:
                if n == self.start_node: color = GREEN
                elif n in self.visited: color = PURPLE
            else:
                if self.animating:
                    if n == self.fw_i: color = GREEN
                    elif n == self.fw_j: color = RED
                    elif n == self.fw_k: color = PURPLE
                elif hovered_node is not None and n == hovered_node:
                    color = GREEN 

            pygame.draw.circle(self.window, color, pos, 8)
            
            label_val = float('inf')
            if not is_fw:
                label_val = self.distances.get(n, float('inf'))
            elif is_fw:
                if self.animating and self.fw_i is not None:
                    label_val = self.fw_dist[self.fw_i][n]
                # BUG FIX: Only show matrix row values if fully completed
                elif self.fw_completed and hovered_node is not None:
                    label_val = self.fw_dist[hovered_node][n]

            if label_val != float('inf') and label_val != 0:
                text = self.font.render(str(label_val), True, WHITE)
                self.window.blit(text, (pos[0] + 10, pos[1] - 10))

        pygame.draw.rect(self.window, GRAY, (0, CANVAS_HEIGHT, WIDTH, UI_HEIGHT))
        pygame.draw.rect(self.window, DARK_GRAY, (0, CANVAS_HEIGHT, WIDTH, UI_HEIGHT), 3)
        
        self.window.blit(self.title_font.render(f"Algorithm: {current_algo}", True, YELLOW), (20, CANVAS_HEIGHT + 20))
        
        if is_fw:
            if self.animating:
                self.window.blit(self.font.render("Green = Start (i)  |  Red = Dest (j)  |  Purple = Intermediate Bridge (k)", True, WHITE), (20, CANVAS_HEIGHT + 50))
                self.window.blit(self.font.render("Visualizing O(V^3) Triangulation matrix scans.", True, PURPLE), (20, CANVAS_HEIGHT + 75))
            elif self.fw_completed:
                self.window.blit(self.font.render("Matrix Computed. HOVER MOUSE over any node to view distances from that source.", True, GREEN), (20, CANVAS_HEIGHT + 50))
            else:
                # BUG FIX: Specific UI state for before the algorithm runs
                self.window.blit(self.font.render("Press SPACE to compute the All-Pairs DP Matrix.", True, WHITE), (20, CANVAS_HEIGHT + 50))
        else:
            self.window.blit(self.font.render("Green = Source Node  |  Purple = Visited/Memoized  |  Yellow = Active Relaxation", True, WHITE), (20, CANVAS_HEIGHT + 50))
            
        self.window.blit(self.font.render("Controls:", True, YELLOW), (20, CANVAS_HEIGHT + 105))
        self.window.blit(self.font.render("[SPACE] Run Traversal | [UP/DOWN] Change Algo | [R] New Graph | [S] Mute", True, WHITE), (20, CANVAS_HEIGHT + 125))
        
        pygame.display.update()

    def run(self):
        running = True
        while running:
            self.clock.tick(240 if "Floyd" in self.algo_keys[self.current_algo_idx] else 30)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.animating:
                        self.reset_traversal()
                        self.animating = True
                        algo_func = self.algorithms[self.algo_keys[self.current_algo_idx]]
                        self.algo_generator = algo_func()
                    
                    if event.key == pygame.K_DOWN and not self.animating:
                        self.current_algo_idx = (self.current_algo_idx + 1) % len(self.algorithms)
                        self.reset_traversal()
                        
                    if event.key == pygame.K_UP and not self.animating:
                        self.current_algo_idx = (self.current_algo_idx - 1) % len(self.algorithms)
                        self.reset_traversal()

                    if event.key == pygame.K_r and not self.animating:
                        self.generate_graph()
                    if event.key == pygame.K_s:
                        self.audio.enabled = not self.audio.enabled

            if self.animating:
                try:
                    next(self.algo_generator)
                except StopIteration:
                    self.animating = False
                    self.current_edge = None

            self.draw()
        pygame.quit()

    def fw_benchmark(self, adj_matrix):
        V = len(adj_matrix)
        dist = [[adj_matrix[i][j] for j in range(V)] for i in range(V)]
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def all_pairs_dijkstra_benchmark(self, adj_list):
        V = len(adj_list)
        all_distances = []
        for start_node in range(V):
            distances = {node: float('inf') for node in range(V)}
            distances[start_node] = 0
            pq = [(0, start_node)]
            while pq:
                current_dist, u = heapq.heappop(pq)
                if current_dist > distances[u]: continue
                for v, weight in adj_list[u]:
                    distance = current_dist + weight
                    if distance < distances[v]:
                        distances[v] = distance
                        heapq.heappush(pq, (distance, v))
            all_distances.append(distances)
        return all_distances

    def run_headless_benchmarks(self):
        print("\n" + "="*80)
        print(" GRAPH ALGORITHM BENCHMARK INITIATED: DIJKSTRA VS FLOYD-WARSHALL")
        print("="*80)
        
        test_sizes = [25, 50, 100, 150, 250]
        densities = {"Sparse (15% Edges)": 0.15, "Dense (85% Edges)": 0.85}
        all_results = {mode: {size: {} for size in test_sizes} for mode in densities.keys()}

        for mode_name, density in densities.items():
            print(f"\n--- Testing Environment: {mode_name} ---")
            fw_times, dijk_times = [], []
            plt.figure(figsize=(10, 6))
            
            for V in test_sizes:
                self.V = V; self.density = density; self.generate_graph() 
                
                t0 = time.perf_counter()
                self.fw_benchmark(self.adj_matrix)
                t_fw = time.perf_counter() - t0
                fw_times.append(t_fw)
                
                t0 = time.perf_counter()
                self.all_pairs_dijkstra_benchmark(self.adj_list)
                t_dijk = time.perf_counter() - t0
                dijk_times.append(t_dijk)
                
                all_results[mode_name][V]["Floyd-Warshall"] = t_fw
                all_results[mode_name][V]["All-Pairs Dijkstra"] = t_dijk

            plt.plot(test_sizes, fw_times, label="Floyd-Warshall $O(V^3)$", marker='o', color='red', linewidth=2)
            plt.plot(test_sizes, dijk_times, label="All-Pairs Dijkstra $O(V E \log V)$", marker='s', color='blue', linewidth=2)
            plt.title(f"Algorithm Scaling in {mode_name} Graphs", fontsize=14, fontweight='bold')
            plt.xlabel("Number of Vertices (V)", fontsize=12)
            plt.ylabel("Execution Time (seconds)", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            filename = f"benchmark_{mode_name.split()[0].lower()}.png"
            plt.savefig(filename, dpi=300)
            plt.close()

        print("\n\n" + "="*60)
        print(" BENCHMARK RESULTS (Time in Seconds)")
        print("="*60)
        for mode in densities.keys():
            print(f"\n[ Distribution: {mode} ]")
            print("-" * 55)
            print(f"{'Nodes (V)':<12} | {'Floyd-Warshall':<18} | {'All-Pairs Dijkstra':<18}")
            print("-" * 55)
            for n in test_sizes:
                fw_val = all_results[mode][n]["Floyd-Warshall"]
                dijk_val = all_results[mode][n]["All-Pairs Dijkstra"]
                print(f"{n:<12} | {fw_val:<18.4f} | {dijk_val:<18.4f}")
            print("-" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="Run headless empirical benchmarks")
    args = parser.parse_args()
    if args.graph:
        app = DPGraphVisualizer(headless=True)
        app.run_headless_benchmarks()
    else:
        app = DPGraphVisualizer(headless=False)
        app.run()