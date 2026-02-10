import time
import sys
import random
import tracemalloc
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

# Configuration
sys.setrecursionlimit(5000)
getcontext().prec = 10000  # High precision for Binet

# ALGORITHMS

def fib_recursive(n):
    if n <= 1: return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_dp_array(n):
    if n <= 1: return n
    fib_list = [0, 1]
    for i in range(2, n + 1):
        fib_list.append(fib_list[i-1] + fib_list[i-2])
    return fib_list[n]

def fib_space_optimized(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fib_matrix(n):
    def multiply(F, M):
        x = F[0][0] * M[0][0] + F[0][1] * M[1][0]
        y = F[0][0] * M[0][1] + F[0][1] * M[1][1]
        z = F[1][0] * M[0][0] + F[1][1] * M[1][0]
        w = F[1][0] * M[0][1] + F[1][1] * M[1][1]
        F[0][0], F[0][1], F[1][0], F[1][1] = x, y, z, w

    def power(F, n):
        if n <= 1: return
        M = [[1, 1], [1, 0]]
        power(F, n // 2)
        multiply(F, F)
        if n % 2 != 0: multiply(F, M)

    if n == 0: return 0
    F = [[1, 1], [1, 0]]
    power(F, n - 1)
    return F[0][0]

def fib_fast_doubling(n):
    def _fib(n):
        if n == 0: return (0, 1)
        a, b = _fib(n >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if n & 1: return (d, c + d)
        else: return (c, d)
    return _fib(n)[0]

def fib_binet(n):
    phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
    return int(round((phi**n - (1 - phi)**n) / Decimal(5).sqrt()))

# REPORT GENERATOR (GRAPHS & TABLES)

def generate_report_data():
    print("\n" + "="*60)
    print(" GENERATING LABORATORY REPORT DATA (GRAPHS + TABLES)")
    print("="*60)

    # 1. Recursive data (small N only)
    print("\n--- Recursive method ---")
    rec_inputs = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40]
    rec_times = []
    
    # Calculate all times first
    for n in rec_inputs:
        start = time.perf_counter()
        fib_recursive(n)
        end = time.perf_counter()
        rec_times.append(end - start)

    # Row 1: N values
    print(f"{'N-Term:':<10}", end=" | ")
    for n in rec_inputs:
        print(f"{n:<8}", end="")
    print("\n" + "-" * 140)
    
    # Row 2: Times
    print(f"{'Time(s):':<10}", end=" | ")
    for t in rec_times:
        print(f"{t:<8.5f}", end="")
    print("\n")

    # Plot Recursive Graph
    plt.figure(figsize=(10, 6))
    plt.plot(rec_inputs, rec_times, marker='o', color='red')
    plt.title('Recursive Method: Exponential Time Complexity O(2^n)')
    plt.xlabel('N-th Term')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.savefig('graph_1_recursive.png')
    print(">> Saved graph_1_recursive.png")

    # 2. Fast algorithms data (large N)
    # Generate 15 sorted random numbers up to 25,000
    fast_inputs = sorted(random.sample(range(100, 25001), 15))
    
    # Define algorithms configuration
    algos_config = [
        {"name": "DP Array",      "func": fib_dp_array,        "color": "blue",   "file": "graph_2_dp.png"},
        {"name": "Space Opt",     "func": fib_space_optimized, "color": "green",  "file": "graph_3_space.png"},
        {"name": "Matrix Pow",    "func": fib_matrix,          "color": "orange", "file": "graph_4_matrix.png"},
        {"name": "Fast Dbl",      "func": fib_fast_doubling,   "color": "purple", "file": "graph_5_doubling.png"},
        {"name": "Binet Form",    "func": fib_binet,           "color": "brown",  "file": "graph_6_binet.png"}
    ]
    
    # Initialize storage for graphing later
    results_storage = {algo["name"]: [] for algo in algos_config}

    print("\n--- Optimized algorithms comparison--")
    
    # Print header Row
    header = f"{'N-th Term':<10} |"
    for algo in algos_config:
        header += f" {algo['name']:<12} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Loop through N inputs (Rows)
    for n in fast_inputs:
        row_str = f"{n:<10} |"
        
        # Loop through Algorithms (Columns)
        for algo in algos_config:
            start = time.perf_counter()
            algo["func"](n)
            end = time.perf_counter()
            elapsed = end - start
            
            # Store for graph
            results_storage[algo["name"]].append(elapsed)
            
            # Add to table row
            row_str += f" {elapsed:.8f}   |"
        
        print(row_str)

    # Generate graphs (using stored data)
    print("\n>> Generating Graphs...")

    # Individual Graphs
    for algo in algos_config:
        times = results_storage[algo["name"]]
        plt.figure(figsize=(10, 6))
        plt.plot(fast_inputs, times, marker='o', color=algo["color"])
        plt.title(f'{algo["name"]} Time Complexity')
        plt.xlabel('N-th Term')
        plt.ylabel('Time (s)')
        plt.grid(True)
        plt.savefig(algo["file"])
        print(f">> Saved {algo['file']}")

    # 3. Combined graph (the 7th graph)
    plt.figure(figsize=(12, 8))
    for algo in algos_config:
        times = results_storage[algo["name"]]
        plt.plot(fast_inputs, times, marker='o', label=algo["name"])
    
    plt.title('Combined Comparison of All Optimized Algorithms')
    plt.xlabel('N-th Term')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph_7_combined.png')
    print(">> Saved graph_7_combined.png")
    print("\n[DONE] All graphs and tables generated successfully.")

# User input mode (with memory check)
def user_input_mode():
    try:
        n = int(input("\nEnter N to calculate (e.g., 20000): "))
    except ValueError:
        print("Invalid input.")
        return

    print(f"\nCalculating F({n})... This may take a moment.")
    print("-" * 105)
    # Header showing BOTH measurement types
    print(f"{'ALGORITHM':<20} | {'TIME (s)':<12} | {'RESULT SIZE':<20} | {'PEAK RAM':<20} |")
    print("-" * 105)

    # 1. Recursive (Skip if too big)
    if n <= 35:
        start = time.perf_counter()
        res = fib_recursive(n)
        t = time.perf_counter() - start
        size = sys.getsizeof(res)
        print(f"{'Recursive':<20} | {t:.6f}       | {size:<20} | {'(High Stack)':<20} |")
    else:
        print(f"{'Recursive':<20} | {'SKIPPED':<12} | {'(Too Slow)':<20} | {'-':<20} |")

    # 2. Optimized Algorithms
    algorithms = [
        ("DP (Array)", fib_dp_array),
        ("Space Opt", fib_space_optimized),
        ("Matrix Power", fib_matrix),
        ("Fast Doubling", fib_fast_doubling),
        ("Binet Formula", fib_binet)
    ]

    for name, func in algorithms:
        # Start Memory Tracking
        tracemalloc.start()
        
        start_time = time.perf_counter()
        result = func(n)
        end_time = time.perf_counter()
        
        # Get Memory Stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed = end_time - start_time
        old_size_bytes = sys.getsizeof(result)
        
        if peak > 1024 * 1024:
            new_peak_str = f"{peak / (1024*1024):.2f} MB"
        elif peak > 1024:
            new_peak_str = f"{peak / 1024:.2f} KB"
        else:
            new_peak_str = f"{peak} bytes"

        print(f"{name:<20} | {elapsed:.8f}   | {old_size_bytes:<20} | {new_peak_str:<20} |")
    
    print("-" * 105)

# Main menu
while True:
        print("\n--- FIBONACCI LABORATORY ---")
        print("1. Run User Input Test (Compare Memory)")
        print("2. Generate Report (7 Graphs + Tables)")
        print("3. Exit")
        
        choice = input("Select option: ")
        
        if choice == '1':
            user_input_mode()
        elif choice == '2':
            generate_report_data()
        elif choice == '3':
            break
