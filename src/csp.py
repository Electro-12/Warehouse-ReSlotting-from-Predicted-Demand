import pandas as pd
import numpy as np
import time

class WarehouseCSP:
    def __init__(self, products, top_pairs, num_aisles=10, slots_per_aisle=12):
        self.products = products
        self.num_slots = num_aisles * slots_per_aisle
        self.slots = list(range(self.num_slots))
        self.num_aisles = num_aisles
        self.slots_per_aisle = slots_per_aisle
        
        np.random.seed(42)
        
        # 10% Heavy items[cite: 16]
        num_heavy = int(len(self.products) * 0.1)
        self.heavy_items = set(np.random.choice(self.products, size=num_heavy, replace=False)) if num_heavy > 0 else set()
        self.low_shelves = set([i for i in self.slots if i % self.slots_per_aisle < 2])
        
        # Sparse Incompatibilities dynamically scaled to fit small test batches[cite: 16]
        self.product_cats = {p: 'Standard' for p in self.products}
        num_special = min(8, len(self.products) // 4) 
        
        if num_special > 0:
            hazardous = np.random.choice(self.products, size=num_special, replace=False)
            food = np.random.choice([p for p in self.products if p not in hazardous], size=num_special, replace=False)
            for p in hazardous: self.product_cats[p] = 'Hazardous'
            for p in food: self.product_cats[p] = 'Food'
            
        self.incompatible_cats = {('Hazardous', 'Food'), ('Food', 'Hazardous')}
        
        # Co-bought pairs[cite: 16]
        self.top_pairs = set(top_pairs)
        self.max_distance = 6  
        
        self.assignments = 0
        self.backtracks = 0
        self.backtrack_limit = 2000  

    def get_aisle_and_pos(self, slot):
        return divmod(slot, self.slots_per_aisle)

    def manhattan_distance(self, slot1, slot2):
        a1, p1 = self.get_aisle_and_pos(slot1)
        a2, p2 = self.get_aisle_and_pos(slot2)
        return abs(a1 - a2) + abs(p1 - p2)

    def is_valid(self, product, slot, assignment):
        if slot in assignment.values(): return False
        if product in self.heavy_items and slot not in self.low_shelves: return False
            
        aisle, _ = self.get_aisle_and_pos(slot)
        cat = self.product_cats[product]
        
        for assigned_prod, assigned_slot in assignment.items():
            assigned_aisle, _ = self.get_aisle_and_pos(assigned_slot)
            assigned_cat = self.product_cats[assigned_prod]
            
            if aisle == assigned_aisle and (cat, assigned_cat) in self.incompatible_cats:
                return False
                
            if (product, assigned_prod) in self.top_pairs or (assigned_prod, product) in self.top_pairs:
                if self.manhattan_distance(slot, assigned_slot) > self.max_distance:
                    return False
                    
        return True

    def solve(self, unassigned_products, algorithm='FC'):
        self.assignments = 0
        self.backtracks = 0
        domains = {p: set(self.slots) for p in self.products}
        return self._backtrack({}, domains, algorithm)

    def _backtrack(self, assignment, domains, algorithm):
        if len(assignment) == len(self.products):
            return assignment
            
        if self.backtracks >= self.backtrack_limit:
            return None 
            
        unassigned = [p for p in self.products if p not in assignment]
        var = min(unassigned, key=lambda p: (p not in self.heavy_items, len(domains[p])))
        
        valid_values = sorted(list(domains[var]), key=lambda s: self.manhattan_distance(s, 0))
        
        for val in valid_values: 
            if self.is_valid(var, val, assignment):
                assignment[var] = val
                self.assignments += 1
                
                pruned = []
                valid_fc = True
                if algorithm in ['FC', 'AC-3']:
                    for future_var in unassigned:
                        if future_var != var and val in domains[future_var]:
                            domains[future_var].remove(val)
                            pruned.append(future_var)
                            if not domains[future_var]:
                                valid_fc = False
                                break
                                
                if valid_fc:
                    result = self._backtrack(assignment, domains, algorithm)
                    if result is not None:
                        return result
                        
                for future_var in pruned:
                    domains[future_var].add(val)
                    
                del assignment[var]
                self.backtracks += 1
                
        return None