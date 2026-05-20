# Assignment 10: LSTM Weather Prediction

## Objective
Use a Long Short-Term Memory (LSTM) network to predict future weather temperatures from historical data.

## Dataset
**Mumbai Daily Temperature Data** — Real historical temperatures (°C) embedded directly in the code.

> **Why embedded?** Weather CSV files cause file path errors across different lab systems.
> Embedding real data directly in the Python file ensures it runs on any college PC
> without any file dependencies, internet access, or setup required.

- 270 days of real Mumbai average daily temperatures
- Temperature range: ~23.6°C – 31.5°C (realistic seasonal variation)
- No download or file required

## What This Assignment Covers
| Step | Task |
|------|------|
| 1 | Load and scale temperature data with MinMaxScaler |
| 2 | Create sliding window sequences (look-back = 15 days) |
| 3 | Split into 80% train / 20% test |
| 4 | Build a Stacked LSTM model (3 layers) |
| 5 | Train with EarlyStopping + ReduceLROnPlateau |
| 6 | Evaluate with RMSE and MAE |
| 7 | Forecast next 10 days of temperature |
| 8 | Visualize results with 6 plots |

## LSTM Architecture
```
Input (15 time steps, 1 feature)
  → LSTM(64, return_sequences=True) + Dropout(0.2)
  → LSTM(32, return_sequences=True) + Dropout(0.2)
  → LSTM(16, return_sequences=False) + Dropout(0.2)
  → Dense(16, relu)
  → Dense(1)  ← predicted next day's temperature
```

## LSTM vs RNN (Key Difference)
| Feature | SimpleRNN | LSTM |
|---------|-----------|------|
| Memory | Short-term only | Long + short term |
| Vanishing gradient | Prone to it | Handles it well |
| Best for | Short sequences | Long sequences |
| Gates | None | Input, Forget, Output |

## How to Run
```bash
python assignment10_lstm_weather.py
```

### Requirements
```
tensorflow
numpy
matplotlib
scikit-learn
```

## Output
- Console: model summary, training logs, RMSE, MAE, 10-day forecast
- Plot: `assignment10_output.png` with:
  1. Full temperature history with train/test split
  2. Training loss (MSE) curve
  3. Training MAE curve
  4. Training set: Actual vs Predicted
  5. Test set: Actual vs Predicted
  6. Future 10-day forecast with annotated temperatures

## Expected Results
- Test RMSE: ~0.3–0.8°C
- Training Time: ~3–6 minutes on CPU

## GitHub Folder Structure
```
Assignment_10_LSTM_Weather/
├── assignment10_lstm_weather.py
└── README.md
```
