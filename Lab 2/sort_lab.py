import pygame
import random
import math
import numpy as np
import time
import argparse

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 750
BAR_WIDTH = 4
FPS = 120
CONTROLS_HEIGHT = 200
ARRAY_SIZE = WIDTH // BAR_WIDTH
MAX_VAL = HEIGHT - CONTROLS_HEIGHT - 20

# Colors
BLACK = (15, 15, 20)
WHITE = (240, 240, 240)
RED = (220, 50, 50)
GREEN = (50, 220, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 200, 50)
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)
BUTTON_COLOR = (70, 70, 80)
BUTTON_HOVER = (100, 100, 110)
RESET_BTN_COLOR = (200, 50, 50)

class AudioEngine:
    def __init__(self):
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        self.sample_rate, self.format, self.channels = pygame.mixer.get_init()
        self.enabled = True
    
    def play_tone(self, value, max_val):
        if not self.enabled: return
        freq = 100 + (value / max_val) * 900
        duration = 0.05
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        wave = 0.1 * np.sign(np.sin(2 * np.pi * freq * t)) 
        audio_data = (wave * 32767).astype(np.int16)
        if self.channels > 1:
            audio_data = np.column_stack((audio_data, audio_data))
        sound = pygame.sndarray.make_sound(audio_data)
        sound.set_volume(0.1)
        sound.play(fade_ms=10)

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

class IconButton:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.hovered = False
        self.rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)

    def check_hover(self, mouse_pos):
        dist = math.hypot(mouse_pos[0] - self.x, mouse_pos[1] - self.y)
        self.hovered = dist <= self.radius

    def is_clicked(self, mouse_pos):
        dist = math.hypot(mouse_pos[0] - self.x, mouse_pos[1] - self.y)
        return dist <= self.radius

    def draw(self, surface):
        color = BUTTON_HOVER if self.hovered else RESET_BTN_COLOR
        pygame.draw.circle(surface, color, (self.x, self.y), self.radius)
        pygame.draw.circle(surface, WHITE, (self.x, self.y), self.radius, 2)

        center_x, center_y = self.x, self.y
        r = self.radius * 0.6
        rect = pygame.Rect(center_x - r, center_y - r, r*2, r*2)
        pygame.draw.arc(surface, WHITE, rect, 0, 1.5 * math.pi, 3)
        
        tip_x = center_x + r
        tip_y = center_y
        pygame.draw.polygon(surface, WHITE, [(tip_x + 5, tip_y), (tip_x - 5, tip_y - 5), (tip_x - 5, tip_y + 5)])

class SortingVisualizer:
    def __init__(self, headless=False):
        self.headless = headless
        self.array = []
        self.current_mode = "Random" 
        
        self.sorting = False
        self.algo_generator = None
        self.comparisons = 0
        self.accesses = 0
        self.start_time = 0
        self.time_elapsed = 0.0

        self.algorithms = {
            "1. QuickSort (Basic)": self.quicksort_basic,
            "2. QuickSort (Optimized)": self.quicksort_optimized,
            "3. MergeSort (Basic)": self.mergesort_basic,
            "4. MergeSort (Optimized)": self.mergesort_optimized,
            "5. HeapSort (Basic)": self.heapsort_basic,
            "6. HeapSort (Optimized)": self.heapsort_optimized,
            "7. Flash Sort (Basic)": self.flashsort_basic,
            "8. Flash Sort (Optimized)": self.flashsort_optimized
        }
        self.current_algo_idx = 0
        self.algo_keys = list(self.algorithms.keys())

        if not self.headless:
            pygame.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Sorting Algorithm Laboratory")
            self.clock = pygame.time.Clock()
            self.audio = AudioEngine()
            
            self.font = pygame.font.SysFont('Consolas', 18)
            self.title_font = pygame.font.SysFont('Consolas', 24, bold=True)
            self.btn_font = pygame.font.SysFont('Consolas', 16, bold=True)

            btn_width = 160
            btn_height = 40
            btn_y = HEIGHT - 60 
            gap = 20
            total_width = (btn_width * 5) + (gap * 4)
            start_x = (WIDTH - total_width) // 2

            self.buttons = [
                Button(start_x, btn_y, btn_width, btn_height, "Random", "Random"),
                Button(start_x + (btn_width + gap), btn_y, btn_width, btn_height, "Sorted", "Sorted"),
                Button(start_x + (btn_width + gap)*2, btn_y, btn_width, btn_height, "Reverse", "Reverse"),
                Button(start_x + (btn_width + gap)*3, btn_y, btn_width, btn_height, "V-Curve", "V-Curve"),
                Button(start_x + (btn_width + gap)*4, btn_y, btn_width, btn_height, "Bell Curve", "Bell Curve"),
            ]
            
            self.reset_btn = IconButton(WIDTH - 50, HEIGHT - CONTROLS_HEIGHT + 50, 25)

        self.reset_array("Random")

    def reset_array(self, mode=None, custom_size=None):
        self.sorting = False
        self.comparisons = 0
        self.accesses = 0
        self.time_elapsed = 0.0
        
        if mode is not None:
            self.current_mode = mode
            
        # Determine size: Use custom size for benchmarks, or default ARRAY_SIZE for visualizer
        n = custom_size if custom_size else ARRAY_SIZE
        mode_to_use = self.current_mode
        self.array = [0] * n

        if mode_to_use == "Random":
            self.array = [random.randint(10, MAX_VAL) for _ in range(n)]
        elif mode_to_use == "Sorted":
            for i in range(n):
                self.array[i] = int(10 + (i / n) * (MAX_VAL - 10))
        elif mode_to_use == "Reverse":
            for i in range(n):
                self.array[i] = int(MAX_VAL - (i / n) * (MAX_VAL - 10))
        elif mode_to_use == "V-Curve":
            mid = n // 2
            for i in range(n):
                dist = abs(i - mid) / mid if mid != 0 else 0
                self.array[i] = int(10 + dist * (MAX_VAL - 10))
        elif mode_to_use == "Bell Curve":
            mid = n // 2
            for i in range(n):
                dist = 1.0 - (abs(i - mid) / mid) if mid != 0 else 0
                self.array[i] = int(10 + dist * (MAX_VAL - 10))
                
    def handle_draw(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_y < HEIGHT - CONTROLS_HEIGHT:
            idx = int(mouse_x / BAR_WIDTH)
            val = MAX_VAL - (mouse_y - 20) 
            val = max(10, min(MAX_VAL, val))
            brush_size = 5
            for i in range(max(0, idx - brush_size), min(ARRAY_SIZE, idx + brush_size)):
                self.array[i] = val

    def draw(self, color_overrides={}):
        if self.headless: return
        self.window.fill(BLACK)
        for i, val in enumerate(self.array):
            color = color_overrides.get(i, BLUE)
            x = i * BAR_WIDTH
            y = (HEIGHT - CONTROLS_HEIGHT) - val
            pygame.draw.rect(self.window, color, (x, y, BAR_WIDTH, val))
        
        ui_y = HEIGHT - CONTROLS_HEIGHT
        pygame.draw.rect(self.window, GRAY, (0, ui_y, WIDTH, CONTROLS_HEIGHT))
        pygame.draw.rect(self.window, DARK_GRAY, (0, ui_y, WIDTH, CONTROLS_HEIGHT), 3)
        
        algo_text = self.title_font.render(f"Algorithm: {self.algo_keys[self.current_algo_idx]}", True, WHITE)
        stats_str = f"Time: {self.time_elapsed:.2f}s | Comparisons: {self.comparisons} | Access: {self.accesses}"
        stats_text = self.font.render(stats_str, True, YELLOW)
        controls_text = self.font.render(
            "Controls: [SPACE] Start | [D] Draw Mode | [UP/DOWN] Change Algo | [S] Sound", 
            True, WHITE
        )
        
        self.window.blit(algo_text, (20, ui_y + 20))
        self.window.blit(stats_text, (20, ui_y + 60))
        self.window.blit(controls_text, (20, ui_y + 100))

        for btn in self.buttons:
            btn.draw(self.window, self.btn_font)
            
        self.reset_btn.draw(self.window)
        pygame.display.update()

    def update_display(self, bar1=None, bar2=None, pivot=None):
        self.accesses += 1
        if self.headless:
            yield # Just yield instantly for headless benchmark mode
            return
            
        colors = {}
        if bar1 is not None: colors[bar1] = RED
        if bar2 is not None: colors[bar2] = RED
        if pivot is not None: colors[pivot] = GREEN
        
        if self.accesses % 2 == 0: 
            if self.sorting:
                self.time_elapsed = time.time() - self.start_time
            self.draw(colors)
            if bar1 is not None: self.audio.play_tone(self.array[bar1], MAX_VAL)
            yield

    # --- ALGORITHMS ---
    def quicksort_basic(self, start, end):
        if start >= end: return
        pivot_val = self.array[end]
        self.comparisons += 1
        left = start
        for i in range(start, end):
            self.comparisons += 1
            yield from self.update_display(i, left, end)
            if self.array[i] < pivot_val:
                self.array[i], self.array[left] = self.array[left], self.array[i]
                left += 1
        self.array[left], self.array[end] = self.array[end], self.array[left]
        yield from self.update_display(left, end)
        yield from self.quicksort_basic(start, left - 1)
        yield from self.quicksort_basic(left + 1, end)

    def quicksort_optimized(self, start, end):
        count = end - start + 1
        if count < 10:
            yield from self.insertion_sort(start, end)
            return
        mid = (start + end) // 2
        if self.array[start] > self.array[mid]: self.array[start], self.array[mid] = self.array[mid], self.array[start]
        if self.array[start] > self.array[end]: self.array[start], self.array[end] = self.array[end], self.array[start]
        if self.array[mid] > self.array[end]: self.array[mid], self.array[end] = self.array[end], self.array[mid]
        self.array[mid], self.array[end-1] = self.array[end-1], self.array[mid]
        pivot = self.array[end-1]
        left = start + 1
        right = end - 2
        while True:
            while self.array[left] < pivot:
                left += 1
                self.comparisons += 1
                yield from self.update_display(left, right, end-1)
            while self.array[right] > pivot:
                right -= 1
                self.comparisons += 1
                yield from self.update_display(left, right, end-1)
            if left >= right: break
            self.array[left], self.array[right] = self.array[right], self.array[left]
            left += 1
            right -= 1
            yield from self.update_display(left, right, end-1)
        self.array[left], self.array[end-1] = self.array[end-1], self.array[left]
        yield from self.quicksort_optimized(start, left - 1)
        yield from self.quicksort_optimized(left + 1, end)

    def insertion_sort(self, start, end):
        for i in range(start + 1, end + 1):
            key = self.array[i]
            j = i - 1
            while j >= start and self.array[j] > key:
                self.comparisons += 1
                self.array[j + 1] = self.array[j]
                j -= 1
                yield from self.update_display(j, i)
            self.array[j + 1] = key

    def mergesort_basic(self, start, end):
        if start >= end: return
        mid = (start + end) // 2
        yield from self.mergesort_basic(start, mid)
        yield from self.mergesort_basic(mid + 1, end)
        yield from self.merge(start, mid, end)

    def merge(self, start, mid, end):
        temp = []
        i, j = start, mid + 1
        while i <= mid and j <= end:
            self.comparisons += 1
            yield from self.update_display(i, j)
            if self.array[i] < self.array[j]:
                temp.append(self.array[i])
                i += 1
            else:
                temp.append(self.array[j])
                j += 1
        while i <= mid: temp.append(self.array[i]); i += 1
        while j <= end: temp.append(self.array[j]); j += 1
        for idx, val in enumerate(temp):
            self.array[start + idx] = val
            yield from self.update_display(start + idx)

    def mergesort_optimized(self, start, end):
        width = 1
        n = len(self.array)
        while width < n:
            l = 0
            while l < n:
                mid = l + width - 1
                r = min(l + 2 * width - 1, n - 1)
                if mid < r:
                    yield from self.merge(l, mid, r)
                l += 2 * width
            width *= 2

    def heapsort_basic(self, start, end):
        n = len(self.array)
        for i in range(n // 2 - 1, -1, -1):
            yield from self.heapify_recursive(n, i)
        for i in range(n - 1, 0, -1):
            self.array[i], self.array[0] = self.array[0], self.array[i]
            yield from self.update_display(i, 0)
            yield from self.heapify_recursive(i, 0)

    def heapify_recursive(self, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        self.comparisons += 1
        if l < n and self.array[l] > self.array[largest]: largest = l
        self.comparisons += 1
        if r < n and self.array[r] > self.array[largest]: largest = r
        if largest != i:
            self.array[i], self.array[largest] = self.array[largest], self.array[i]
            yield from self.update_display(i, largest)
            yield from self.heapify_recursive(n, largest)

    def heapsort_optimized(self, start, end):
        n = len(self.array)
        for i in range(n // 2 - 1, -1, -1):
            yield from self.heapify_floyd(n, i)
        for i in range(n - 1, 0, -1):
            self.array[i], self.array[0] = self.array[0], self.array[i]
            yield from self.update_display(i, 0)
            yield from self.heapify_floyd(i, 0)

    def heapify_floyd(self, n, i):
        root = i
        child = 2 * root + 1
        temp = self.array[root]
        while child < n:
            self.comparisons += 1
            if child + 1 < n and self.array[child + 1] > self.array[child]:
                child += 1
            self.array[root] = self.array[child]
            yield from self.update_display(root, child)
            root = child
            child = 2 * root + 1
        self.array[root] = temp
        yield from self.update_display(root)
        while root > i:
            parent = (root - 1) // 2
            self.comparisons += 1
            if self.array[parent] < self.array[root]:
                self.array[parent], self.array[root] = self.array[root], self.array[parent]
                yield from self.update_display(parent, root)
                root = parent
            else: break

    def flashsort_basic(self, start, end):
        if end - start <= 1: return
        min_val = self.array[start]
        max_val = 0
        max_idx = -1
        for i in range(start, end + 1):
            self.accesses += 1
            val = self.array[i]
            if val < min_val: min_val = val
            if val > max_val: max_val = val; max_idx = i
            yield from self.update_display(i)
        if min_val == max_val: return
        n = end - start + 1
        m = int(0.43 * n)
        if m < 2: m = 2
        l = [0] * m
        c = (m - 1) / (max_val - min_val) if max_val != min_val else 0
        for i in range(start, end + 1):
            self.accesses += 1
            k = int(c * (self.array[i] - min_val))
            l[k] += 1
            yield from self.update_display(i)
        for k in range(1, m): l[k] += l[k-1]
        self.array[max_idx], self.array[start] = self.array[start], self.array[max_idx]
        yield from self.update_display(start, max_idx)
        move_count = 0
        k = m - 1
        num_moves = end - start + 1 
        flash = self.array[start]
        idx = start
        while move_count < num_moves - 1:
            while idx > (start + l[k] - 1):
                idx += 1
                k = int(c * (self.array[idx] - min_val))
                flash = self.array[idx]
            k = int(c * (flash - min_val))
            target_idx = start + l[k] - 1
            l[k] -= 1
            temp = self.array[target_idx]
            self.array[target_idx] = flash
            flash = temp
            self.accesses += 2
            self.comparisons += 1
            yield from self.update_display(idx, target_idx, pivot=target_idx)
            move_count += 1
        yield from self.insertion_sort(start, end)

    def flashsort_optimized(self, start, end):
        if end - start <= 20:
            yield from self.insertion_sort(start, end)
            return
        min_val = self.array[start]
        max_val = self.array[start]
        for i in range(start, end + 1):
            val = self.array[i]
            if val < min_val: min_val = val
            if val > max_val: max_val = val
        if min_val == max_val: return 
        n = end - start + 1
        m = int(0.43 * n)
        c = (m - 1) / (max_val - min_val) if max_val != min_val else 0
        l = [0] * m
        for i in range(start, end + 1):
            k = int(c * (self.array[i] - min_val))
            l[k] += 1
        for k in range(1, m): l[k] += l[k-1]
        self.array[start], self.array[start] = self.array[start], self.array[start] 
        move_count = 0
        idx = start
        k = m - 1
        flash = self.array[start]
        while move_count < n - 1:
            while idx > (start + l[k] - 1):
                idx += 1
                k = int(c * (self.array[idx] - min_val))
                flash = self.array[idx]
            k = int(c * (flash - min_val))
            target_idx = start + l[k] - 1
            l[k] -= 1
            temp = self.array[target_idx]
            self.array[target_idx] = flash
            flash = temp
            if move_count % 2 == 0: yield from self.update_display(idx, target_idx, pivot=target_idx)
            move_count += 1
        yield from self.insertion_sort(start, end)

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            mouse_pos = pygame.mouse.get_pos()
            
            for btn in self.buttons:
                btn.check_hover(mouse_pos)
            self.reset_btn.check_hover(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: 
                        for btn in self.buttons:
                            if btn.is_clicked(mouse_pos) and not self.sorting:
                                self.reset_array(btn.mode_key)
                        
                        if self.reset_btn.is_clicked(mouse_pos):
                            self.reset_array() 

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.sorting:
                        self.sorting = True
                        self.start_time = time.time()
                        algo_func = self.algorithms[self.algo_keys[self.current_algo_idx]]
                        self.algo_generator = algo_func(0, len(self.array) - 1)
                    
                    if event.key == pygame.K_DOWN:
                        self.current_algo_idx = (self.current_algo_idx + 1) % len(self.algorithms)
                        self.reset_array() 
                        
                    if event.key == pygame.K_UP:
                        self.current_algo_idx = (self.current_algo_idx - 1) % len(self.algorithms)
                        self.reset_array()
                    
                    if event.key == pygame.K_s:
                        self.audio.enabled = not self.audio.enabled

            if pygame.mouse.get_pressed()[0] and not self.sorting:
                is_hovering_btn = any(btn.hovered for btn in self.buttons) or self.reset_btn.hovered
                if not is_hovering_btn:
                    self.handle_draw()

            if self.sorting:
                try:
                    next(self.algo_generator)
                    self.time_elapsed = time.time() - self.start_time
                except StopIteration:
                    self.sorting = False
                    self.time_elapsed = time.time() - self.start_time
                    self.draw({i: GREEN for i in range(len(self.array))})

            if not self.sorting:
                self.draw()

        pygame.quit()

    def generate_graphs(self):
        import matplotlib.pyplot as plt
        import sys
        
        # Boost recursion limit so Basic QuickSort doesn't instantly crash on sorted data
        sys.setrecursionlimit(25000) 

        modes = ["Random", "Sorted", "Reverse", "V-Curve", "Bell Curve"]
        test_sizes = [1000, 2500, 5000, 7500, 10000] 
        
        # Short names for the table and graph legends
        short_names = {
            "1. QuickSort (Basic)": "Quick",
            "2. QuickSort (Optimized)": "Quick(O)",
            "3. MergeSort (Basic)": "Merge",
            "4. MergeSort (Optimized)": "Merge(O)",
            "5. HeapSort (Basic)": "Heap",
            "6. HeapSort (Optimized)": "Heap(O)",
            "7. Flash Sort (Basic)": "Flash",
            "8. Flash Sort (Optimized)": "Flash(O)"
        }

        # Dictionary to store all our timings for the text tables
        all_results = {mode: {size: {} for size in test_sizes} for mode in modes}
        
        print("Running benchmarks (1000 to 10000 elements).")
        print("This may take some time. Fell free to grab a coffe...............\n")
        
        for mode in modes:
            print(f"--- Processing Data Distribution: {mode} ---")
            
            # Setup a new figure for the line graphs
            plt.figure(figsize=(10, 6))
            
            for algo_name in self.algo_keys:
                algo_times = []
                
                for n in test_sizes:
                    self.reset_array(mode=mode, custom_size=n)
                    algo_func = self.algorithms[algo_name]
                    gen = algo_func(0, len(self.array) - 1)
                    
                    start_time = time.perf_counter()
                    
                    try:
                        # Consume the generator fully
                        for _ in gen: pass
                        exec_time = time.perf_counter() - start_time
                    except RecursionError:
                        # If Basic QuickSort blows up the stack, catch it cleanly
                        exec_time = None 
                    
                    algo_times.append(exec_time)
                    all_results[mode][n][algo_name] = exec_time
                
                # Plot the line for this algorithm (Matplotlib handles 'None' by skipping that point)
                plt.plot(test_sizes, algo_times, label=short_names[algo_name], marker='o', linewidth=2, markersize=5)
            
            # Format the Graph
            plt.title(f"Time Complexity Analysis: {mode} Distribution", fontsize=14, fontweight='bold')
            plt.xlabel("Number of Elements (N)", fontsize=12)
            plt.ylabel("Execution Time (seconds)", fontsize=12)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save as separate file
            filename = f"benchmark_{mode.replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300)
            plt.close() # Close figure to free memory
            print(f"  Saved {filename}")

        # ====================
        # PRINT CONSOLE TABLES
        # ====================
        print("\n\n" + "="*90)
        print(" BENCHMARK RESULTS (Execution Time in Seconds)")
        print("="*90)
        
        # Build the table header dynamically
        header = f"{'Size':<6} | " + " | ".join([f"{short_names[a]:<8}" for a in self.algo_keys])
        separator = "-" * len(header)
        
        for mode in modes:
            print(f"\n[ Distribution: {mode.upper()} ]")
            print(separator)
            print(header)
            print(separator)
            
            for n in test_sizes:
                row_str = f"{n:<6} | "
                for algo in self.algo_keys:
                    val = all_results[mode][n][algo]
                    if val is None:
                        row_str += f"{'ERROR':<8} | " # Show ERROR if RecursionError hit
                    else:
                        row_str += f"{val:<8.4f} | "
                print(row_str[:-3]) # Strip the last trailing pipe
            print(separator)
            
        print("\nAll tasks complete!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sorting Algorithm Laboratory")
    parser.add_argument("--graph", action="store_true", help="Generate benchmark graphs instead of running the GUI")
    args = parser.parse_args()
    
    if args.graph:
        app = SortingVisualizer(headless=True)
        app.generate_graphs()
    else:
        app = SortingVisualizer(headless=False)
        app.run()