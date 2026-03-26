# Stock AI

Stock AI is a machine-learning powered stock direction and trade-setup assistant for Indian market tickers (for example RELIANCE.NS).

It combines price-based features, technical indicators, optional sentiment, and backtesting to produce:

- BUY, SELL, HOLD signal
- Confidence score
- Suggested entry, target, and stop-loss levels
- Multi-timeframe view
- Risk and performance summary

Important: This project is for research and educational use, not financial advice.

## Highlights

- Interactive Streamlit app for analysis and visualization
- Multi-model workflow (Random Forest, XGBoost, LightGBM, ensemble)
- Technical indicators and market context features
- Backtesting with trade logs and metrics
- FastAPI endpoints for train and predict workflows
- Daily scheduler scripts for automated runs

## Project Structure

```text
stock-ai/
  app/
    Streamlit_main_app.py
    streamlit_app.py
    api.py
  src/
    train.py
    model.py
    features.py
    data_loader.py
    backtesting.py
    sentiment.py
    sentiment_data.py
    realtime_scheduler.py
    run_daily_once.py
  data/
  models/
  notebooks/
  README.md
  DEVELOPER_GUIDE.md
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If requirements.txt is not present, install common dependencies manually:

```bash
pip install pandas numpy scikit-learn yfinance ta transformers torch streamlit fastapi uvicorn xgboost lightgbm plotly
```

### 2. Run Streamlit app

```bash
streamlit run app/Streamlit_main_app.py
```

Open http://localhost:8501

### 3. Run API

```bash
uvicorn app.api:app --reload
```

Open docs at http://127.0.0.1:8000/docs

## Supported Workflows

- Manual analysis from Streamlit UI
- API-based training and prediction
- Scheduled daily prediction via src/run_daily_once.py or src/realtime_scheduler.py

## Notes

- Use Yahoo Finance ticker format correctly (example: INFY.NS, TCS.NS).
- First sentiment run may be slower because the transformer model is downloaded and cached.
- Results can vary by timeframe and market regime.

## Developer Documentation

For implementation details, architecture, module-level behavior, and maintenance guidance, see DEVELOPER_GUIDE.md.
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
