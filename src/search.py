import pandas as pd
import heapq
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.csp import WarehouseCSP

class LayoutOptimizer:
    def __init__(self, csp_instance, demand_dict):
        self.csp = csp_instance
        self.demand = demand_dict
        
        # Identify top 30 products by demand[cite: 16]
        sorted_products = sorted(self.demand.keys(), key=lambda p: self.demand[p], reverse=True)
        self.top_30 = sorted_products[:30]
        
        # FIX: Calculate the 30 slots closest to the depot (slot 0) directly here[cite: 16]
        slots_by_depot = sorted(self.csp.slots, key=lambda s: self.csp.manhattan_distance(s, 0))
        self.top_30_slots = set(slots_by_depot[:30])

    def calc_weighted_distance(self, layout):
        """Calculates demand-weighted walking distance from the depot[cite: 16]."""
        return sum(self.demand.get(p, 0) * self.csp.manhattan_distance(s, 0) for p, s in layout.items())

    def is_swap_valid(self, p1, s2, p2, s1, layout):
        """Optimized rule checker: only checks constraints for the two moved items."""
        layout[p1], layout[p2] = s2, s1  # Temporarily apply
        valid = True
        
        for p, s in [(p1, s2), (p2, s1)]:
            aisle, _ = self.csp.get_aisle_and_pos(s)
            cat = self.csp.product_cats[p]
            
            for other_p, other_s in layout.items():
                if p == other_p: continue
                
                # Check Incompatible Categories
                if (cat, self.csp.product_cats[other_p]) in self.csp.incompatible_cats:
                    other_aisle, _ = self.csp.get_aisle_and_pos(other_s)
                    if aisle == other_aisle:
                        valid = False; break
                        
                # Check Co-bought pairs distance
                if (p, other_p) in self.csp.top_pairs or (other_p, p) in self.csp.top_pairs:
                    if self.csp.manhattan_distance(s, other_s) > self.csp.max_distance:
                        valid = False; break
            if not valid: break
            
        layout[p1], layout[p2] = s1, s2  # Revert
        return valid

    def get_valid_swaps(self, layout):
        """Generates valid layouts by swapping a Top 30 item with an item in a Top 30 slot[cite: 16]."""
        products_near_depot = [p for p, s in layout.items() if s in self.top_30_slots]
        
        swap_pairs = set()
        for p1 in self.top_30:
            for p2 in products_near_depot:
                if p1 != p2:
                    swap_pairs.add(tuple(sorted([p1, p2])))
                    
        neighbors = []
        for p1, p2 in swap_pairs:
            s1, s2 = layout[p1], layout[p2]
            
            # Heavy items rule
            if p1 in self.csp.heavy_items and s2 not in self.csp.low_shelves: continue
            if p2 in self.csp.heavy_items and s1 not in self.csp.low_shelves: continue
                
            if self.is_swap_valid(p1, s2, p2, s1, layout):
                new_layout = layout.copy()
                new_layout[p1], new_layout[p2] = s2, s1
                neighbors.append(new_layout)
                
        return neighbors

    def dict_to_tuple(self, layout):
        return tuple(sorted(layout.items()))

    def optimize(self, initial_layout, algorithm='Greedy', max_swaps=6, max_nodes=1500):
        initial_cost = self.calc_weighted_distance(initial_layout)
        pq = [(initial_cost if algorithm != 'UCS' else 0, 0, 0, initial_layout)]
        visited = {self.dict_to_tuple(initial_layout)}
        
        best_layout = initial_layout
        best_cost = initial_cost
        nodes_expanded = 0
        tie_breaker = 0
        target_distance = initial_cost * 0.90 
        
        while pq:
            if nodes_expanded >= max_nodes:
                break
                
            score, swaps_made, _, current_layout = heapq.heappop(pq)
            current_cost = self.calc_weighted_distance(current_layout)
            nodes_expanded += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_layout = current_layout
                
            if current_cost <= target_distance or swaps_made >= max_swaps:
                continue
                
            for neighbor_layout in self.get_valid_swaps(current_layout):
                n_tuple = self.dict_to_tuple(neighbor_layout)
                if n_tuple not in visited:
                    visited.add(n_tuple)
                    tie_breaker += 1
                    neighbor_cost = self.calc_weighted_distance(neighbor_layout)
                    
                    if algorithm == 'Greedy':
                        priority = neighbor_cost 
                    elif algorithm == 'UCS':
                        priority = swaps_made + 1 
                    elif algorithm == 'A*':
                        priority = ((swaps_made + 1) * 2000) + neighbor_cost 
                        
                    heapq.heappush(pq, (priority, swaps_made + 1, tie_breaker, neighbor_layout))
                    
        return best_layout, best_cost, nodes_expanded

if __name__ == "__main__":
    print("Preparing Data and Generating Initial CSP Layout...")
    df_train = pd.read_csv('data/train.csv')
    
    product_demand = df_train.groupby('StockCode')['Total_Units'].sum().to_dict()
    product_demand = {str(k): v for k, v in product_demand.items()}
    top_120 = sorted(product_demand.keys(), key=lambda k: product_demand[k], reverse=True)[:120]
    
    pairs_df = pd.read_csv('data/top_30_pairs.csv')
    top_pairs = list(zip(pairs_df['StockCode_x'].astype(str), pairs_df['StockCode_y'].astype(str)))
    
    csp = WarehouseCSP(top_120, top_pairs)
    csp.max_distance = 12
    initial_layout = csp.solve(top_120, algorithm='FC')
    
    optimizer = LayoutOptimizer(csp, product_demand)
    initial_cost = optimizer.calc_weighted_distance(initial_layout)
    print(f"\nInitial Layout Demand-Weighted Distance: {initial_cost:,.0f}")
    
    print("\n--- EXPERIMENT 1: Search Algorithm Comparison ---")
    for algo in ['UCS', 'Greedy', 'A*']:
        print(f"\nRunning {algo} Search (Max 6 Swaps, Bounded to 1500 Nodes)...")
        start_t = time.time()
        final_layout, final_cost, nodes = optimizer.optimize(initial_layout, algorithm=algo, max_swaps=6)
        
        print(f"Algorithm:      {algo}")
        print(f"Final Distance: {final_cost:,.0f} (Reduced by {initial_cost - final_cost:,.0f})")
        print(f"Nodes Expanded: {nodes}")
        print(f"Time Taken:     {time.time() - start_t:.2f} seconds")