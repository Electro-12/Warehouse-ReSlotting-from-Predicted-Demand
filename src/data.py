import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# ==========================================
# DELIVERABLE #3: DATA DICTIONARY DEFINITION
# ==========================================
DATA_DICTIONARY = {
    "StockCode": {"Type": "Real (Categorical/String)", "Description": "Unique product identifier from the Online Retail II dataset."},
    "Absolute_Week": {"Type": "Real (Integer)", "Description": "Continuous chronological week index spanning across multiple years."},
    "Week": {"Type": "Real (Integer)", "Description": "Calendar week of the year (1-52)."},
    "Total_Units": {"Type": "Real (Integer)", "Description": "Actual units sold for the product in that specific week (derived from real transactions)."},
    "Mean_Price": {"Type": "Real (Float)", "Description": "Average unit price of the product during that week."},
    "Invoice_Count": {"Type": "Real (Integer)", "Description": "Number of unique transactions/invoices containing this product in that week."},
    "Log_Units": {"Type": "Real/Generated (Float)", "Description": "Log-transformed target variable calculated as ln(1 + Total_Units) for neural network stability."},
    "Week_Index": {"Type": "Generated (Integer)", "Description": "Normalized relative week index starting from 0 for chronological splitting."}
}

class WarehouseDataset(Dataset):
    """
    PyTorch Dataset wrapper for product-week demand forecasting.
    Uses past 8 weeks of log units, week-of-year, mean price, and invoice count 
    to predict the next week's log units.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data = df.sort_values(['StockCode', 'Absolute_Week']).reset_index(drop=True)
        
        # Group by product to easily extract historical windows
        self.products = list(self.data['StockCode'].unique())
        self.samples = []
        
        for stock_code, group in self.data.groupby('StockCode'):
            group = group.sort_values('Absolute_Week').reset_index(drop=True)
            log_units = group['Log_Units'].values
            weeks = group['Week'].values
            prices = group['Mean_Price'].values
            invoices = group['Invoice_Count'].values
            
            # We need at least 8 historical weeks to predict the 9th
            if len(group) >= 9:
                for i in range(8, len(group)):
                    history_units = log_units[i-8:i] # Last 8 weeks of log(1+units)[cite: 16]
                    current_week = weeks[i]
                    current_price = prices[i]
                    current_invoice = invoices[i]
                    target = log_units[i]           # Target next-week units[cite: 16]
                    
                    # Feature vector: 8 historical units + week-of-year + price + invoice count
                    features = np.concatenate([
                        history_units, 
                        np.array([current_week, current_price, current_invoice], dtype=np.float32)
                    ])
                    
                    self.samples.append((features, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, target = self.samples[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def get_dataloaders(batch_size=32):
    train_dataset = WarehouseDataset('data/train.csv')
    val_dataset = WarehouseDataset('data/val.csv')
    test_dataset = WarehouseDataset('data/test.csv')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader