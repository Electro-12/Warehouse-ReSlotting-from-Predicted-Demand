import pandas as pd
import numpy as np
import os
import urllib.request

# 1. Reproducibility: Fixed seed[cite: 1]
np.random.seed(42)

def prepare_warehouse_data():
    file_path = "online_retail_II.xlsx"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    
    if not os.path.exists(file_path):
        print("Downloading Online Retail II Excel file from UCI...")
        urllib.request.urlretrieve(url, file_path)
        
    print("Loading Excel file into Pandas (reading both sheets)...")
    df = pd.concat(pd.read_excel(file_path, sheet_name=None), ignore_index=True)
    
    # 2. Clean Data: Remove cancellations and enforce string types
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df.dropna(subset=['StockCode', 'InvoiceDate', 'Price'])
    df['StockCode'] = df['StockCode'].astype(str) # <-- FIX: Enforce string type
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Week'] = df['InvoiceDate'].dt.isocalendar().week
    df['Year'] = df['InvoiceDate'].dt.isocalendar().year
    df['Absolute_Week'] = (df['Year'] - df['Year'].min()) * 52 + df['Week']
    
    # 3. Filter for Top 120 products by volume[cite: 16]
    volume_per_product = df.groupby('StockCode')['Quantity'].sum()
    top_120_codes = volume_per_product.nlargest(120).index
    df_top = df[df['StockCode'].isin(top_120_codes)].copy()
    
    # 4. Aggregate to product-week level
    weekly_data = df_top.groupby(['StockCode', 'Absolute_Week', 'Week']).agg(
        Total_Units=('Quantity', 'sum'),
        Mean_Price=('Price', 'mean'),
        Invoice_Count=('Invoice', 'nunique')
    ).reset_index()
    
    weekly_data['Log_Units'] = np.log1p(weekly_data['Total_Units'])
    
    # 5. Chronological Split (Train/Val/Test)[cite: 16]
    min_week = weekly_data['Absolute_Week'].min()
    weekly_data['Week_Index'] = weekly_data['Absolute_Week'] - min_week
    
    train_mask = weekly_data['Week_Index'] < 70
    val_mask = (weekly_data['Week_Index'] >= 70) & (weekly_data['Week_Index'] < 85)
    test_mask = weekly_data['Week_Index'] >= 85
    
    df_train = weekly_data[train_mask]
    df_val = weekly_data[val_mask]
    df_test = weekly_data[test_mask]
    
    os.makedirs('data', exist_ok=True)
    df_train.to_csv('data/train.csv', index=False)
    df_val.to_csv('data/val.csv', index=False)
    df_test.to_csv('data/test.csv', index=False)
    
    # 6. Extract Co-purchase pairs strictly from training weeks[cite: 17]
    train_invoices = df_top[df_top['Absolute_Week'] - min_week < 70]
    pairs = pd.merge(train_invoices[['Invoice', 'StockCode']], 
                     train_invoices[['Invoice', 'StockCode']], 
                     on='Invoice')
    pairs = pairs[pairs['StockCode_x'] < pairs['StockCode_y']]
    pair_counts = pairs.groupby(['StockCode_x', 'StockCode_y']).size().reset_index(name='Count')
    top_30_pairs = pair_counts.nlargest(30, 'Count')
    
    top_30_pairs.to_csv('data/top_30_pairs.csv', index=False)
    print("Saved data/top_30_pairs.csv successfully.")

if __name__ == "__main__":
    prepare_warehouse_data()