import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

# Ensure Python can find the src module when running from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_dataloaders

class DemandPredictor(nn.Module):
    def __init__(self):
        super(DemandPredictor, self).__init__()
        # Inputs: 8 weeks log units + 1 week index + 1 price + 1 invoice count = 11 features[cite: 16]
        self.net = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

def train_model():
    print("Initializing data loaders and PyTorch neural network...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    
    model = DemandPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    best_val_loss = float('inf')
    
    os.makedirs('models', exist_ok=True)
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_model.pt')
            
    print("Training complete. Best model saved to models/best_model.pt[cite: 1]")

if __name__ == "__main__":
    train_model()