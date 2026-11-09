# Warehouse Re-Slotting from Predicted Demand (P8)

An end-to-end AI logistics pipeline that optimizes warehouse product placement by combining deep learning demand forecasting with Constraint Satisfaction Problem (CSP) algorithms and heuristic search.

## Architecture & Modules
1. **Data Pipeline (`src/make_data.py`, `src/data.py`)**: Processes the UCI Online Retail II dataset, engineering an 8-week sliding window feature set and extracting the top 30 co-bought product pairs.
2. **Demand Forecasting (`src/nn.py`, `src/eval.py`)**: A PyTorch Neural Network trained to predict weekly product demand, outperforming 4-week-average baselines in Test MSE.
3. **Constraint Satisfaction (`src/csp.py`)**: A highly optimized Backtracking solver with Forward Checking and Minimum Remaining Values (MRV). It enforces physical warehouse constraints (heavy items, incompatible categories, co-bought distances) and features an auto-relaxation loop to prevent search traps.
4. **Layout Optimization (`src/search.py`)**: Explores the swap-space of the CSP-generated layout. Implements bounded Uniform Cost Search (UCS), Greedy Best-First, and A* algorithms to minimize the demand-weighted walking distance for order pickers.
5. **Integrated Simulation (`src/integration.py`)**: Simulates 10 weeks of held-out real order data to evaluate the true walking distance across Random, Last-Week Demand, and NN-Optimized layouts.

## Individual Contribution Statement
**Author:** Yousef Maged Samy Aziz  
**Contribution:** 100% (Solo Project)

As the sole author of this project, I independently designed, implemented, and evaluated all modules:
- **Data & Neural Network:** Preprocessed the UCI Online Retail II dataset, engineered sliding-window features, and trained the PyTorch demand forecasting model against baselines.
- **Constraint Satisfaction:** Modeled the 120-item warehouse grid and built the CSP backtracking solver with Forward Checking, MRV, and auto-relaxation loops.
- **Heuristic Search & Integration:** Implemented A*, UCS, and Greedy Best-First algorithms with node counters, built the 10-week order simulator, and ran multi-seed experiments.
- **Deliverables:** Authored the technical report, data dictionary, unit test suite, and presentation materials.


## Environment & Setup
Install the pinned dependency requirements:
```bash
pip install -r requirements.txt
```

## Usage
Generate/preprocess data reproducibly:
```bash
python src/make_data.py
```

Execute the full integrated pipeline demonstration:
```bash
python run_demo.py
```

Run unit tests across normal, edge, and impossible scenarios:
```bash
python -m unittest discover tests
```
