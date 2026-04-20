import pygame
import random
import time
import argparse
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
WIDTH, HEIGHT = 950, 650
UI_HEIGHT = 150
CANVAS_HEIGHT = HEIGHT - UI_HEIGHT

# Colors
BLACK = (15, 15, 20)
WHITE = (240, 240, 240)
GREEN = (50, 220, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 200, 50)
PURPLE = (198, 120, 221)
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]: self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]: self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False

    def draw(self, surface, font):
        color = GRAY if self.hovered else DARK_GRAY
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, WHITE, self.rect, 1, border_radius=4)
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

class GreedyMSTVisualizer:
    def __init__(self, headless=False):
        self.headless = headless
        self.V = 20 
        self.density = 0.20 
        self.uniform_mode = False 
        
        self.adj_matrix = []
        self.edge_list = []
        self.node_positions = []
        
        self.animating = False
        self.algo_generator = None
        
        # Kept only the absolute best implementations
        self.algorithms = {
            "1. Kruskal O(E log E) (Union-Find)": self.kruskal_generator,
            "2. Prim O(V^2) (Adjacency Matrix)": self.prim_matrix_generator
        }
        self.algo_keys = list(self.algorithms.keys())
        self.current_algo_idx = 0
        
        self.mst_edges = []
        self.current_edge = None
        self.visited_nodes = set()
        
        if not self.headless:
            pygame.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Lab 5: Kruskal vs Prim")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Consolas', 13)
            self.small_font = pygame.font.SysFont('Consolas', 11)
            self.title_font = pygame.font.SysFont('Consolas', 16, bold=True)
            
            self.btn_v_sub = Button(530, CANVAS_HEIGHT + 15, 30, 25, "-")
            self.btn_v_add = Button(570, CANVAS_HEIGHT + 15, 30, 25, "+")
            self.btn_d_sub = Button(530, CANVAS_HEIGHT + 50, 30, 25, "-")
            self.btn_d_add = Button(570, CANVAS_HEIGHT + 50, 30, 25, "+")
            self.btn_toggle = Button(450, CANVAS_HEIGHT + 85, 150, 25, "Uniform: OFF")
            
            self.generate_graph()

    def generate_graph(self):
        self.adj_matrix = [[float('inf')] * self.V for _ in range(self.V)]
        self.edge_list = []
        self.node_positions = []
        
        edge_set = set()
        
        num_dense = int(self.V * self.density) if not self.uniform_mode else self.V
        
        for i in range(self.V):
            if not self.uniform_mode and i < num_dense:
                x = random.randint(WIDTH//4, 3*WIDTH//4)
                y = random.randint(CANVAS_HEIGHT//4, 3*CANVAS_HEIGHT//4)
            else:
                x = random.randint(30, WIDTH - 30)
                y = random.randint(30, CANVAS_HEIGHT - 30)
            self.node_positions.append((x, y))
            
        def add_edge(u, v, w):
            if u > v: u, v = v, u
            if (u, v) not in edge_set:
                edge_set.add((u, v))
                
                # Global Edge List (For Kruskal)
                self.edge_list.append((w, u, v))
                
                # Adjacency Matrix (For Classical Array Prim)
                self.adj_matrix[u][v] = w
                self.adj_matrix[v][u] = w

        # 1. Base Spanning Tree (Guarantees Connectivity)
        for i in range(1, self.V):
            target = random.randint(0, i - 1)
            w = random.randint(10, 99)
            add_edge(i, target, w)

        # 2. Add extra edges based on mode
        if self.uniform_mode:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if random.random() < self.density:
                        w = random.randint(10, 99)
                        add_edge(i, j, w)
        else:
            for i in range(num_dense):
                for j in range(i + 1, num_dense):
                    w = random.randint(10, 99)
                    add_edge(i, j, w)
                        
        self.reset_traversal()

    def reset_traversal(self):
        self.mst_edges = []
        self.current_edge = None
        self.visited_nodes = set()
        self.animating = False

    # --- UI GENERATORS ---
    def kruskal_generator(self):
        sorted_edges = sorted(self.edge_list, key=lambda item: item[0])
        uf = UnionFind(self.V)
        for weight, u, v in sorted_edges:
            self.current_edge = (u, v)
            yield 
            if uf.union(u, v):
                self.mst_edges.append((u, v))
                self.visited_nodes.add(u)
                self.visited_nodes.add(v)
            yield

    def prim_matrix_generator(self):
        visited = [False] * self.V
        min_weight = [float('inf')] * self.V
        parent = [-1] * self.V
        
        min_weight[0] = 0
        edges_found = 0
        self.visited_nodes.add(0)

        while edges_found < self.V - 1:
            u = -1
            for i in range(self.V):
                if not visited[i] and (u == -1 or min_weight[i] < min_weight[u]):
                    u = i

            if min_weight[u] == float('inf'):
                break

            visited[u] = True
            self.visited_nodes.add(u)

            if parent[u] != -1:
                self.mst_edges.append((parent[u], u))
                self.current_edge = (parent[u], u)
                edges_found += 1
                yield

            for v in range(self.V):
                weight = self.adj_matrix[u][v]
                if not visited[v] and weight < min_weight[v]:
                    min_weight[v] = weight
                    parent[v] = u
            yield

    # --- UI DRAW & RUN ---
    def draw(self):
        self.window.fill(BLACK)
        
        for weight, u, v in self.edge_list:
            pos_u, pos_v = self.node_positions[u], self.node_positions[v]
            pygame.draw.line(self.window, DARK_GRAY, pos_u, pos_v, 1)
            if len(self.edge_list) <= 150:
                mx, my = (pos_u[0] + pos_v[0]) / 2, (pos_u[1] + pos_v[1]) / 2
                self.window.blit(self.small_font.render(str(weight), True, GRAY), (mx, my))

        if self.animating and self.current_edge:
            pygame.draw.line(self.window, YELLOW, self.node_positions[self.current_edge[0]], self.node_positions[self.current_edge[1]], 2)

        for u, v in self.mst_edges:
            pygame.draw.line(self.window, GREEN, self.node_positions[u], self.node_positions[v], 3)

        for n, pos in enumerate(self.node_positions):
            num_dense = int(self.V * self.density)
            is_dense_core = not self.uniform_mode and n < num_dense
            
            if n in self.visited_nodes: color = PURPLE
            elif is_dense_core: color = (100, 200, 255)
            else: color = BLUE
            pygame.draw.circle(self.window, color, pos, 6)

        pygame.draw.rect(self.window, GRAY, (0, CANVAS_HEIGHT, WIDTH, UI_HEIGHT))
        pygame.draw.rect(self.window, DARK_GRAY, (0, CANVAS_HEIGHT, WIDTH, UI_HEIGHT), 2)
        
        current_algo = self.algo_keys[self.current_algo_idx]
        self.window.blit(self.title_font.render(f"Algorithm: {current_algo}", True, YELLOW), (20, CANVAS_HEIGHT + 15))
        self.window.blit(self.font.render(f"Nodes (V): {self.V} | Edges (E): {len(self.edge_list)}", True, WHITE), (20, CANVAS_HEIGHT + 45))
        
        mode_text = f"Uniform Density ({int(self.density*100)}%)" if self.uniform_mode else f"Split: {int(self.density*100)}% Dense Core / {100-int(self.density*100)}% Sparse Periphery"
        self.window.blit(self.font.render(f"Topology: {mode_text}", True, WHITE), (20, CANVAS_HEIGHT + 70))
        self.window.blit(self.font.render("[SPACE] Run | [UP/DN] Change Algo | [R] New Graph", True, GREEN), (20, CANVAS_HEIGHT + 105))
        
        self.window.blit(self.font.render(f"Set V: {self.V}", True, WHITE), (450, CANVAS_HEIGHT + 20))
        self.btn_v_sub.draw(self.window, self.font)
        self.btn_v_add.draw(self.window, self.font)
        
        lbl_dens = "Set Dens:" if self.uniform_mode else "Core %:"
        self.window.blit(self.font.render(f"{lbl_dens} {int(self.density*100)}%", True, WHITE), (430, CANVAS_HEIGHT + 55))
        self.btn_d_sub.draw(self.window, self.font)
        self.btn_d_add.draw(self.window, self.font)
        
        self.btn_toggle.text = "Uniform: ON" if self.uniform_mode else "Uniform: OFF"
        self.btn_toggle.draw(self.window, self.font)

        pygame.display.update()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            mouse_pos = pygame.mouse.get_pos()
            
            for btn in [self.btn_v_sub, self.btn_v_add, self.btn_d_sub, self.btn_d_add, self.btn_toggle]:
                btn.hovered = btn.rect.collidepoint(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.animating:
                    if self.btn_v_sub.is_clicked(mouse_pos) and self.V > 5:
                        self.V -= 5; self.generate_graph()
                    if self.btn_v_add.is_clicked(mouse_pos) and self.V < 100:
                        self.V += 5; self.generate_graph()
                    if self.btn_d_sub.is_clicked(mouse_pos) and self.density > 0.0:
                        self.density = round(self.density - 0.1, 1); self.generate_graph()
                    if self.btn_d_add.is_clicked(mouse_pos) and self.density < 1.0:
                        self.density = round(self.density + 0.1, 1); self.generate_graph()
                    if self.btn_toggle.is_clicked(mouse_pos):
                        self.uniform_mode = not self.uniform_mode; self.generate_graph()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.animating:
                        self.reset_traversal()
                        self.animating = True
                        self.algo_generator = self.algorithms[self.algo_keys[self.current_algo_idx]]()
                    if event.key == pygame.K_DOWN and not self.animating:
                        self.current_algo_idx = (self.current_algo_idx + 1) % len(self.algorithms)
                        self.reset_traversal()
                    if event.key == pygame.K_UP and not self.animating:
                        self.current_algo_idx = (self.current_algo_idx - 1) % len(self.algorithms)
                        self.reset_traversal()
                    if event.key == pygame.K_r and not self.animating:
                        self.generate_graph()

            if self.animating:
                try: next(self.algo_generator)
                except StopIteration:
                    self.animating = False
                    self.current_edge = None

            self.draw()
        pygame.quit()

    # --- HEADLESS BENCHMARKS ---
    def kruskal_bench(self, V, edges):
        """ Kruskal O(E log E) using Sorting + Union-Find """
        sorted_edges = sorted(edges, key=lambda item: item[0])
        uf = UnionFind(V)
        mst = []
        for weight, u, v in sorted_edges:
            if uf.union(u, v):
                mst.append((u, v))
                if len(mst) == V - 1: break
        return mst

    def prim_dense_bench(self, V, adj_matrix):
        """ Classical O(V^2) Prim's Algorithm using an Adjacency Matrix """
        visited = [False] * V
        min_weight = [float('inf')] * V
        parent = [-1] * V
        
        min_weight[0] = 0
        mst = []

        for _ in range(V):
            u = -1
            for i in range(V):
                if not visited[i] and (u == -1 or min_weight[i] < min_weight[u]):
                    u = i

            if min_weight[u] == float('inf'):
                break

            visited[u] = True
            if parent[u] != -1:
                mst.append((parent[u], u))

            for v in range(V):
                weight = adj_matrix[u][v]
                if not visited[v] and weight < min_weight[v]:
                    min_weight[v] = weight
                    parent[v] = u
                    
        return mst

    def run_headless_benchmarks(self):
        test_sizes = [50, 100, 250, 500, 1000, 1500] 
        self.uniform_mode = False 
        
        densities = {
            "Sparse (0% - No Cycles)": 0.00,
            "Medium (20% Dense Core)": 0.20, 
            "Medium (50% Dense Core)": 0.50, 
            "Medium (80% Dense Core)": 0.80,
            "Dense (100% - 1:1)": 1.00
        }
        
        for mode_name, density in densities.items():
            print(f"\nRunning Benchmark: {mode_name}...")
            k_times, p_dense_times = [], []
            plt.figure(figsize=(10, 6))
            
            for V in test_sizes:
                self.V = V; self.density = density; self.generate_graph() 
                
                # Kruskal O(E log E)
                t0 = time.perf_counter()
                self.kruskal_bench(self.V, self.edge_list)
                t_k = time.perf_counter() - t0
                k_times.append(t_k)

                # Prim Matrix O(V^2)
                t0 = time.perf_counter()
                self.prim_dense_bench(self.V, self.adj_matrix)
                t_p_dense = time.perf_counter() - t0
                p_dense_times.append(t_p_dense)
                
                print(f" V={V:<4} | E={len(self.edge_list):<7} | Kruskal: {t_k:.4f}s | Prim(Matrix): {t_p_dense:.4f}s")

            plt.plot(test_sizes, k_times, label="Kruskal O(E log E)", marker='o', color='blue', linewidth=2)
            plt.plot(test_sizes, p_dense_times, label="Prim (Matrix) O(V^2)", marker='s', color='green', linewidth=2)
            
            plt.title(f"Algorithm Scaling: {mode_name}", fontsize=14, fontweight='bold')
            plt.xlabel("Number of Vertices (V)", fontsize=12)
            plt.ylabel("Execution Time (seconds)", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            filename = f"lab5_benchmark_{int(density*100)}.png"
            plt.savefig(filename, dpi=300)
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if args.graph:
        app = GreedyMSTVisualizer(headless=True)
        app.run_headless_benchmarks()
    else:
        app = GreedyMSTVisualizer(headless=False)
        app.run()