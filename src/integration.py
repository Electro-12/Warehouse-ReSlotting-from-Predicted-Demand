import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.csp import WarehouseCSP
from src.search import LayoutOptimizer
from src.nn import DemandPredictor
from src.data import WarehouseDataset

class IntegrationSimulator:
    def __init__(self):
        print("Loading Data and NN Model...")
        self.df_test = pd.read_csv('data/test.csv')
        self.df_val = pd.read_csv('data/val.csv')
        
        # Isolate exactly 10 test weeks for Experiment 3[cite: 17]
        test_weeks = sorted(self.df_test['Week_Index'].unique())[:10]
        self.df_test = self.df_test[self.df_test['Week_Index'].isin(test_weeks)]
        
        # Load Top 30 Pairs[cite: 17]
        pairs_df = pd.read_csv('data/top_30_pairs.csv')
        self.top_pairs = list(zip(pairs_df['StockCode_x'].astype(str), pairs_df['StockCode_y'].astype(str)))
        
        self.top_120 = sorted(self.df_test['StockCode'].unique().astype(str).tolist())
        
        # Load trained Neural Network[cite: 1, 16]
        self.model = DemandPredictor()
        self.model.load_state_dict(torch.load('models/best_model.pt', weights_only=True))
        self.model.eval()

    def get_demand_profiles(self):
        """Generates the demand profiles used to build the different layouts."""
        print("Generating Demand Profiles...")
        
        # 1. Last-Week Demand (using the final validation week: week 84)[cite: 17]
        df_last_week = self.df_val[self.df_val['Week_Index'] == 84]
        last_week_demand = df_last_week.set_index('StockCode')['Total_Units'].to_dict()
        last_week_demand = {str(k): v for k, v in last_week_demand.items()}
        self.last_week_demand = {p: last_week_demand.get(p, 0) for p in self.top_120}
        
        # 2. NN Predicted Demand (average forecast over the test weeks)[cite: 16, 17]
        test_dataset = WarehouseDataset('data/test.csv')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        features, _ = next(iter(test_loader))
        with torch.no_grad():
            log_preds = self.model(features).numpy()
            
        # Convert log units back to real units: exp(log_units) - 1
        real_preds = np.expm1(log_preds)
        
        self.df_test['NN_Pred_Units'] = real_preds[:len(self.df_test)]
        nn_demand = self.df_test.groupby('StockCode')['NN_Pred_Units'].mean().to_dict()
        self.nn_demand = {str(k): v for k, v in nn_demand.items()}

    def generate_layouts(self):
        """Generates all 4 required layouts for comparison[cite: 17]."""
        print("\nGenerating 4 Warehouse Layouts...")
        self.layouts = {}
        
        csp = WarehouseCSP(self.top_120, self.top_pairs)
        
        # Helper function to auto-relax distance if the CSP gets stuck[cite: 16]
        def safe_solve(product_order):
            csp.products = product_order
            for dist in range(12, 24, 2):
                csp.max_distance = dist
                layout = csp.solve(self.top_120, algorithm='FC')
                if layout is not None:
                    return layout
            return None
        
        # 1. Random Layout
        print("- Building Random Layout...")
        shuffled_products = self.top_120.copy()
        np.random.shuffle(shuffled_products)
        self.layouts['Random'] = safe_solve(shuffled_products)
        
        # 2. Last-Week Demand Layout[cite: 17]
        print("- Building Last-Week Demand Layout...")
        last_week_order = sorted(self.top_120, key=lambda p: self.last_week_demand[p], reverse=True)
        self.layouts['Last-Week'] = safe_solve(last_week_order)
        
        # 3. NN Demand Layout[cite: 17]
        print("- Building NN Demand Layout...")
        nn_order = sorted(self.top_120, key=lambda p: self.nn_demand[p], reverse=True)
        self.layouts['NN Demand'] = safe_solve(nn_order)
        
        # 4. NN + Swaps Layout (Search Phase)[cite: 16, 17]
        print("- Optimizing NN Layout with Greedy Search (NN + Swaps)...")
        optimizer = LayoutOptimizer(csp, self.nn_demand)
        best_layout, _, _ = optimizer.optimize(self.layouts['NN Demand'].copy(), algorithm='Greedy', max_swaps=6)
        self.layouts['NN + Swaps'] = best_layout

    def simulate_picker_distance(self):
        """Calculates true walking distance over the 10 test weeks using actual sales[cite: 17]."""
        print("\nSimulating Picker Walking Distance over 10 Test Weeks...")
        csp = WarehouseCSP(self.top_120, self.top_pairs)
        
        weeks = sorted(self.df_test['Week_Index'].unique())
        results = {name: [] for name in self.layouts.keys()}
        
        for week in weeks:
            week_data = self.df_test[self.df_test['Week_Index'] == week]
            actual_demand = {str(row['StockCode']): row['Total_Units'] for _, row in week_data.iterrows()}
            
            for name, layout in self.layouts.items():
                if layout is None:
                    distance = float('nan') 
                else:
                    distance = sum(actual_demand.get(p, 0) * csp.manhattan_distance(s, 0) for p, s in layout.items())
                results[name].append(distance)
                
        self.plot_experiment_3(weeks, results)

    def plot_experiment_3(self, weeks, results):
        """Generates the final plot for Experiment 3[cite: 1, 17]."""
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        colors = {'Random': 'gray', 'Last-Week': 'orange', 'NN Demand': 'blue', 'NN + Swaps': 'green'}
        markers = {'Random': 'x', 'Last-Week': 's', 'NN Demand': 'o', 'NN + Swaps': '^'}
        
        for name, distances in results.items():
            plt.plot(weeks, distances, label=name, color=colors[name], marker=markers[name], linewidth=2)
            print(f"Average Weekly Distance ({name}): {np.mean(distances):,.0f}")
            
        plt.title('Experiment 3: Picker Distance Over 10 Held-Out Weeks')
        plt.xlabel('Test Week Index')
        plt.ylabel('True Demand-Weighted Walking Distance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig('plots/experiment3_distances.png')
        print("\nSaved Experiment 3 plot to plots/experiment3_distances.png")

if __name__ == "__main__":
    print("=== P8: WAREHOUSE RE-SLOTTING INTEGRATED DEMO ===")
    sim = IntegrationSimulator()
    sim.get_demand_profiles()
    sim.generate_layouts()
    sim.simulate_picker_distance()
    print("=== PIPELINE COMPLETE ===")