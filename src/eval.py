import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure Python can find the src module when running from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_dataloaders
from src.nn import DemandPredictor

def evaluate_model():
    print("Loading test data and best model...")
    _, _, test_loader = get_dataloaders(batch_size=32)
    
    model = DemandPredictor()
    # Load the best model weights saved from nn.py[cite: 1]
    model.load_state_dict(torch.load('models/best_model.pt', weights_only=True))
    model.eval()
    
    nn_losses = []
    last_week_losses = []
    four_week_losses = []
    
    criterion = nn.MSELoss()
    
    print("Evaluating against baselines...")
    with torch.no_grad():
        for features, targets in test_loader:
            # 1. Neural Network Prediction
            nn_preds = model(features)
            
            # 2. Baselines extracted from feature vector (indices 0-7 contain the 8-week history)
            # Index 7 is the most recent past week[cite: 17]
            last_week_preds = features[:, 7] 
            
            # Indices 4,5,6,7 are the last 4 weeks[cite: 17]
            four_week_preds = features[:, 4:8].mean(dim=1) 
            
            # Calculate batch Mean Squared Errors
            nn_losses.append(criterion(nn_preds, targets).item())
            last_week_losses.append(criterion(last_week_preds, targets).item())
            four_week_losses.append(criterion(four_week_preds, targets).item())
            
    avg_nn_mse = np.mean(nn_losses)
    avg_last_week_mse = np.mean(last_week_losses)
    avg_four_week_mse = np.mean(four_week_losses)
    
    print("\n--- Demand Forecast Evaluation (Test Set MSE) ---")
    print(f"Neural Network MSE:     {avg_nn_mse:.4f}")
    print(f"Last-Week Baseline MSE: {avg_last_week_mse:.4f}")
    print(f"4-Week-Avg Baseline MSE:{avg_four_week_mse:.4f}")
    
    # Generate the required plot for the report[cite: 1]
    os.makedirs('plots', exist_ok=True)
    labels = ['Neural Network', 'Last-Week', '4-Week-Avg']
    mses = [avg_nn_mse, avg_last_week_mse, avg_four_week_mse]
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, mses, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Mean Squared Error (Log Units)')
    plt.title('Demand Forecast Performance vs Baselines')
    for i, v in enumerate(mses):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    plt.savefig('plots/forecast_evaluation.png')
    print("\nSaved evaluation plot to plots/forecast_evaluation.png")

if __name__ == "__main__":
    evaluate_model()