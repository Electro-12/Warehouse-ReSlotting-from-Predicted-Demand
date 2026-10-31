import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.csp import WarehouseCSP

def run_test_suite():
    print("=== WAREHOUSE CSP TEST SUITE ===\n")
    
    def execute_test(name, csp, products, expect_fail=False):
        print(f"{name:<55}", end="")
        csp.backtrack_limit = 50  
        layout = csp.solve(products, algorithm='FC')
        
        if layout:
            if expect_fail:
                print(" [ FAILED - Unexpectedly found a solution ]")
            else:
                print(" [ PASSED - Solution Found ]")
        else:
            if expect_fail:
                print(" [ PASSED - NO SOLUTION (Clear message triggered as expected) ]")
            else:
                print(" [ FAILED - Unexpectedly found no solution ]")

    print("--- 5 NORMAL CASES ---")
    base_products = [str(i) for i in range(15)]
    execute_test("Normal 1: Small batch basic layout", WarehouseCSP(base_products, []), base_products)
    execute_test("Normal 2: Batch with 1 co-bought pair", WarehouseCSP(base_products, [('0', '1')]), base_products)
    execute_test("Normal 3: Standard 2-aisle fit", WarehouseCSP(base_products, [], num_aisles=2, slots_per_aisle=10), base_products)
    execute_test("Normal 4: Oversized warehouse capacity", WarehouseCSP(base_products, [], num_aisles=20, slots_per_aisle=5), base_products)
    execute_test("Normal 5: Dense packing", WarehouseCSP(base_products, [('1', '2')], num_aisles=2, slots_per_aisle=8), base_products)

    print("\n--- 3 EDGE CASES ---")
    execute_test("Edge 1: Single item warehouse", WarehouseCSP(['0'], []), ['0'])
    execute_test("Edge 2: Empty co-bought pairs constraints list", WarehouseCSP(base_products, []), base_products)
    execute_test("Edge 3: Exact max grid capacity (20 items, 20 slots)", WarehouseCSP([str(i) for i in range(20)], [], num_aisles=2, slots_per_aisle=10), [str(i) for i in range(20)])

    print("\n--- 2 IMPOSSIBLE CASES ---")
    imp1 = WarehouseCSP([str(i) for i in range(5)], [], num_aisles=1, slots_per_aisle=5)
    imp1.heavy_items = {str(i) for i in range(5)}
    execute_test("Impossible 1: Too many heavy items for low shelves", imp1, [str(i) for i in range(5)], expect_fail=True)

    imp2 = WarehouseCSP(['0', '1'], [], num_aisles=1, slots_per_aisle=5)
    imp2.product_cats = {'0': 'Hazardous', '1': 'Food'}
    # FIX: Bidirectional incompatibility
    imp2.incompatible_cats = {('Hazardous', 'Food'), ('Food', 'Hazardous')} 
    execute_test("Impossible 2: Incompatible categories in single aisle", imp2, ['0', '1'], expect_fail=True)

if __name__ == "__main__":
    run_test_suite()