# Assignment 9: RNN Model for Stock Price Prediction

## Objective
Implement a Recurrent Neural Network (RNN) to predict stock prices using historical time-series data.

## Dataset
**Apple Inc. (AAPL) Stock Prices** — Real historical closing prices embedded directly in the code.

> **Why embedded?** Stock price CSVs cause file path errors across different lab systems.
> Embedding real data directly in the Python file ensures it runs on any college PC
> without any file dependencies, internet access, or setup required.

- 200 real AAPL daily closing prices
- Price range: ~$131 – $203
- No download or file required

## What This Assignment Covers
| Step | Task |
|------|------|
| 1 | Load and scale data using MinMaxScaler |
| 2 | Create sliding window sequences (look-back = 10 steps) |
| 3 | Split into 80% train / 20% test |
| 4 | Build a Stacked SimpleRNN model |
| 5 | Train with EarlyStopping |
| 6 | Evaluate using RMSE and MAE |
| 7 | Visualize actual vs predicted prices |

## RNN Architecture
```
Input (10 time steps, 1 feature)
  → SimpleRNN(64, return_sequences=True) + Dropout(0.2)
  → SimpleRNN(32) + Dropout(0.2)
  → Dense(16, relu)
  → Dense(1)  ← predicted next price
```

## How to Run
```bash
python assignment9_rnn_stock.py
```

### Requirements
```
tensorflow
numpy
matplotlib
scikit-learn
```

## Output
- Console: model summary, training logs, RMSE, MAE
- Plot: `assignment9_output.png` with:
  1. Full AAPL price history with train/test split line
  2. Training loss curve (MSE)
  3. Training set: Actual vs Predicted
  4. Test set: Actual vs Predicted

## Key Concepts
- **Look-back window**: Use last 10 prices to predict the next one
- **MinMaxScaler**: Scales prices to [0,1] for stable training
- **SimpleRNN**: Processes sequences with memory of past steps
- **RMSE/MAE**: Measures prediction error in dollar terms

## GitHub Folder Structure
```
Assignment_09_RNN_StockPrice/
├── assignment9_rnn_stock.py
└── README.md
```
