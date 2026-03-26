# Stock AI - Stock Market Direction Predictor

An advanced machine learning system for predicting stock market direction (UP or DOWN) using historical price data, technical indicators, and sentiment analysis.

## 📊 Project Overview

This project builds a binary classification model to predict whether a stock's price will go UP or DOWN for the next day. It combines multiple data sources and techniques:

- **Historical Data**: OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Machine Learning**: Random Forest classifier with evaluation metrics
- **Interactive UI**: Streamlit web application for easy model training and predictions

## ✅ Completed Components

### 1. **Data Loading Module** (`src/data_loader.py`)
- Fetches historical stock data using yfinance
- Validates data integrity and handles missing values
- Supports custom date ranges
- Returns clean OHLCV DataFrame with Date as a column
- Includes error handling and logging

### 2. **Feature Engineering Module** (`src/features.py`)
- Calculates technical indicators using the `ta` library
- **Implemented Features:**
  - Simple Moving Average (SMA) - 10 and 50 day periods
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands (upper and lower bands)
- Standard scaling/normalization of features
- Handles NaN values effectively

### 3. **Sentiment Analysis Module** (`src/sentiment.py`)
- Uses Hugging Face transformers library
- Pretrained sentiment model (DistilBERT-based)
- Functions for:
  - Single text analysis
  - Batch processing of headlines
  - Daily sentiment aggregation
- Output: -1 (negative), 0 (neutral), +1 (positive)

### 4. **Model Training Module** (`src/model.py`)
- **Classifier**: Random Forest (with XGBoost support)
- Train/test split with time series consideration
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
- Feature importance analysis
- Clean, modular training functions

### 5. **Training Pipeline** (`src/train.py`)
- Orchestrates the full ML workflow:
  1. Data fetching and loading
  2. Feature engineering
  3. Target variable creation (next-day direction)
  4. Model training
  5. Latest prediction generation
- Returns comprehensive results: model, metrics, dataset, feature columns
- Proper error handling and TypedDict definitions

### 6. **Streamlit Web Application** (`app/streamlit_app.py`)
- Interactive user interface for model training and predictions
- **Configuration Options**:
  - Ticker symbol input (default: RELIANCE.NS)
  - Custom date range selection
  - Train and predict button
- **Display Features**:
  - Real-time accuracy, precision, recall metrics
  - Latest predicted direction (UP/DOWN)
  - Enriched dataset preview (last 20 rows)
  - Feature columns list

## 🚀 How to Use

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies (add these to requirements.txt)
pip install pandas numpy scikit-learn yfinance ta transformers torch streamlit fastapi uvicorn
```

### Running the Application
```bash
# From the project root directory
streamlit run app/streamlit_app.py
```

Then open your browser and navigate to `http://localhost:8501`

### Running the REST API (FastAPI)
```bash
# From the project root directory
uvicorn app.api:app --reload
```

Open API docs:
- `http://127.0.0.1:8000/docs`

Available endpoints:
- `POST /train`
- `POST /predict`
- `GET /metrics`

Persistence behavior:
- `/train` stores model artifacts in `models/model.pkl`, `models/return_model.pkl`, `models/multiclass_model.pkl`
- `/train` also stores API metadata in `models/api_state.json`
- After server restart, `POST /predict` can run with `{"retrain_if_missing": false}` by loading persisted models and rebuilding the latest features from market data (no retraining required)

Example `POST /train` body:
```json
{
  "symbol": "RELIANCE.NS",
  "start_date": "2020-01-01",
  "end_date": "2026-03-26",
  "model_family": "ensemble",
  "buy_threshold": 0.7,
  "sell_threshold": 0.3,
  "sl_stop": 0.02,
  "tp_stop": 0.04,
  "save_models": true
}
```

Example `POST /predict` body:
```json
{
  "retrain_if_missing": false
}
```

### Automated Daily Predictions

The project now includes scheduler scripts for unattended daily prediction runs.

#### 1) Run once (manual trigger)
```bash
python src/run_daily_once.py
```

Artifacts are written to `data/realtime/`:
- `latest_<SYMBOL>.json`
- `prediction_<SYMBOL>_<TIMESTAMP>.json`
- `predictions_<SYMBOL>.csv`

#### 2) Run continuously with built-in scheduler (APScheduler)
```bash
python src/realtime_scheduler.py
```

Optional environment variables:
- `SYMBOL` (default: `RELIANCE.NS`)
- `MODEL_FAMILY` (default: `ensemble`)
- `BUY_THRESHOLD` (default: `0.7`)
- `SELL_THRESHOLD` (default: `0.3`)
- `SL_STOP` (default: `0.02`)
- `TP_STOP` (default: `0.04`)
- `TIMEZONE` (default: `Asia/Kolkata`)
- `RUN_HOUR` (default: `18`)
- `RUN_MINUTE` (default: `0`)

#### 3) Windows Task Scheduler (daily run)
Create a basic task that runs daily and set Program/script to your virtual environment Python executable, with arguments:
```bash
src/run_daily_once.py
```
Set Start in to the project root folder.

#### 4) Cron (Linux/macOS)
Run daily at 18:00 Asia/Kolkata equivalent server time:
```bash
0 18 * * * cd /path/to/stock-ai && /path/to/python src/run_daily_once.py >> logs/realtime.log 2>&1
```

### Using Programmatically
```python
from src.train import train_stock_model, predict_latest_direction

# Train model
result = train_stock_model(
    symbol="RELIANCE.NS",
    start_date="2020-01-01",
    end_date="2024-01-01"
)

# Get prediction
prediction = predict_latest_direction(
    result["model"],
    result["dataset"],
    result["feature_columns"]
)
```

## 📁 Project Structure

```
stock-ai/
├── app/
│   └── streamlit_app.py          # Interactive web interface
├── src/
│   ├── data_loader.py            # Yahoo Finance data fetching
│   ├── features.py               # Technical indicator engineering
│   ├── sentiment.py              # News sentiment analysis
│   ├── model.py                  # ML model training and prediction
│   └── train.py                  # Full pipeline orchestration
├── data/                         # Data storage (future)
├── models/                       # Saved model storage (future)
├── notebooks/                    # Jupyter notebooks (future)
└── README.md                     # This file
```

## 🔧 Technical Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost (optional)
- **Technical Indicators**: ta (Technical Analysis library)
- **Data Source**: yfinance
- **NLP/Sentiment**: Hugging Face transformers, PyTorch
- **Web Framework**: Streamlit
- **Visualization**: matplotlib (integrated in model.py)

## 📊 Data Flow

```
Raw Stock Data (yfinance)
         ↓
Feature Engineering (Technical Indicators)
         ↓
Target Creation (Next-day Direction)
         ↓
Train/Test Split
         ↓
Model Training (Random Forest)
         ↓
Evaluation Metrics
         ↓
Latest Direction Prediction
         ↓
Streamlit UI Display
```

## ⚙️ Configuration

You can customize the following:
- **Ticker symbols**: Any valid Yahoo Finance ticker
- **Date ranges**: Custom start and end dates
- **Technical indicators**: Modify periods in `features.py`
- **Model parameters**: Adjust Random Forest hyperparameters in `model.py`

## 📈 Output Metrics

The system provides:
- **Accuracy**: Overall prediction correctness
- **Precision**: Correct UP predictions / Total UP predictions
- **Recall**: Correct UP predictions / Actual UP movements
- **Feature Columns**: List of all features used in training

## 🚧 Known Limitations & Future Work

- Current implementation uses only technical indicators (sentiment analysis module present but not fully integrated)
- Single stock prediction (extensible to multiple stocks)
- Basic Random Forest model (can be enhanced with deep learning)
- No real-time data updates (future: live prediction service)
- Production deployment considerations needed (database, scheduling, etc.)

## 📝 Development Notes

- All modules include comprehensive docstrings and type hints
- Error handling for invalid inputs and missing data
- Modular design allows easy feature additions
- Clean separation of concerns (data, features, model, UI)

## 🎯 Next Steps

1. Integrate sentiment analysis into the training pipeline
2. Add multiple stock comparison
3. Implement model persistence (save/load trained models)
4. Add backtesting functionality
5. Deploy as production service
6. Create REST API for programmatic access

---

**Last Updated**: March 2026  
**Status**: Core functionality complete, ready for enhancement
