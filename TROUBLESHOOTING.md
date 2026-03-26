# Stock-AI Troubleshooting Guide

## Quick Fixes for Common Issues

### 1. ❌ Streamlit Not Starting

**Symptom:** `streamlit: command not found` or `ModuleNotFoundError`

**Solution:**
```powershell
# Use Python module directly
cd 'd:\Projects\New folder\stock-ai'
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -m streamlit run app/Streamlit_main_app.py
```

---

### 2. ❌ Port 8501 Already in Use

**Symptom:** `Address already in use` when starting Streamlit

**Solution (PowerShell):**
```powershell
# Kill processes on ports 8501-8505
$procIds = Get-NetTCPConnection -State Listen | Where-Object { 
    $_.LocalPort -in 8501,8502,8503,8504,8505 
} | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($id in $procIds) { 
    try { Stop-Process -Id $id -Force } catch {} 
}

# Start on different port
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -m streamlit run app/Streamlit_main_app.py --server.port 8502
```

---

### 3. ❌ Stock Data Not Fetching

**Symptom:** `ValueError: No data returned for {symbol}`

**Causes & Solutions:**
```python
# ✓ Use correct ticker format
"RELIANCE.NS"      # India: .NS suffix required
"INFY.NS"          # Correct
"INFY"             # ✗ Wrong - won't work for India stocks

# ✓ Check date range
start_date = "2020-01-01"  # Older data is more stable
end_date = "2024-03-26"    # Don't use future dates

# ✓ Verify yfinance works
import yfinance as yf
df = yf.download("RELIANCE.NS", start="2024-01-01", end="2024-03-26")
print(df.head())  # Should show OHLCV data
```

---

### 4. ❌ Sentiment Model Taking Too Long

**Symptom:** First run takes 1+ minute, lots of downloading messages

**Reason:** HuggingFace model downloads on first use (268MB)

**Solution:**
```powershell
# Pre-download the model (one-time)
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -c "
from transformers import pipeline
pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
print('✓ Model cached')
"

# Now subsequent runs will be fast
```

**Alternative:** Skip sentiment in initial tests
```python
# In Streamlit app, sentiment is optional - if it fails, app continues
```

---

### 5. ❌ NewsAPI Returns No Headlines

**Symptom:** `No recent headlines available` message

**Reasons:**
- Free tier API key required (free tier has limits)
- Date range too old/recent
- Ticker not recognized

**Solution:**
```python
# Get free API key from: https://newsapi.org/pricing
# Set it in your code:
from src.sentiment_data import fetch_news_headlines

df = fetch_news_headlines(
    symbol="RELIANCE.NS",
    start_date="2024-01-01",
    end_date="2024-03-26",
    api_key="YOUR_FREE_API_KEY_HERE"  # Add free key
)

print(df.head())  # Should show headlines
```

---

### 6. ❌ Features Not Computing

**Symptom:** `ValueError: Features or indices are not properly aligned`

**Solution:**
```python
# Check that data has at least 50+ rows (indicators need history)
from src.data_loader import fetch_stock_data

df = fetch_stock_data("RELIANCE.NS", "2024-01-01", "2024-03-26")
print(f"Rows: {len(df)}")  # Should be > 50

# If too few rows, extend date range:
df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2024-03-26")  # Larger window
```

---

### 7. ❌ Model Not Training / Training Too Slow

**Symptom:** `XGBoost` or training hangs

**Check:**
```python
# Verify sufficient data
from src.data_loader import fetch_stock_data

df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2024-03-26")
print(f"Data shape: {df.shape}")  # Should be (500+, 6)

# Check for NaN values
print(df.isnull().sum())  # Should be minimal

# For development/testing, reduce data
df_sample = df.tail(200)  # Use last 200 rows only
# Then add features and train
```

---

### 8. ❌ Import Errors (`ModuleNotFoundError`)

**Symptom:** `No module named 'ta'` or similar

**Solution:**
```powershell
# Verify packages installed
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -m pip list | grep -E "ta|yfinance|streamlit"

# Install missing packages
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -m pip install ta yfinance streamlit

# Verify specific imports
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -c "import ta; import yfinance; print('✓ OK')"
```

---

### 9. ❌ Backtesting Shows Wrong Numbers

**Symptom:** Win rate = 0% or negative profit with positive predictions

**This is normal - reasons:**
- Market moved against prediction even with correct direction
- Slippage + brokerage costs reduced profit
- Stop loss triggered before target

**Verify:**
```python
from src.backtesting import Backtester

# Check individual trade details
bt_results = backtest_obj.calculate_metrics()
print(bt_results['trade_history'])  # Each trade should show cost breakdown
```

---

### 10. ❌ Streamlit Showing Blank/Error Page

**Symptom:** White page or red error box after clicking "Analyze"

**Check:**
```
1. Check terminal for Python errors
2. Verify all imports work: 
   python -c "from src.train import train_stock_model; print('OK')"
3. Clear Streamlit cache:
   Delete ~/.streamlit/cache
4. Restart app with:
   streamlit run app/Streamlit_main_app.py --logger.level=debug
```

---

## Verification Commands

### Test All Modules
```powershell
$py = 'd:/Projects/New folder/.venv/Scripts/python.exe'

Write-Host "Testing imports..." -ForegroundColor Green
& $py -c "
from src.train import train_stock_model
from src.data_loader import fetch_stock_data  
from src.features import add_features
from src.sentiment import analyze_sentiment
from src.backtesting import Backtester
print('✓ All imports successful')
"
```

### Test Data Pipeline
```powershell
$py = 'd:/Projects/New folder/.venv/Scripts/python.exe'

Write-Host "Testing data fetch..." -ForegroundColor Green
& $py -c "
from src.data_loader import fetch_stock_data
df = fetch_stock_data('RELIANCE.NS', '2024-01-01', '2024-03-26')
print(f'✓ Fetched {len(df)} rows')
print(f'✓ Columns: {list(df.columns)}')
"
```

### Test Models
```powershell
$py = 'd:/Projects/New folder/.venv/Scripts/python.exe'

Write-Host "Testing model training..." -ForegroundColor Green
& $py -c "
from src.train import train_stock_model
result = train_stock_model('RELIANCE.NS', '2024-01-01', '2024-03-26')
print(f'✓ Model trained: {result[\"accuracy_train\"]:.2%} accuracy')
"
```

---

## Performance Optimization

### Make It Faster

**1. Use Cached Data:**
```python
# Don't re-fetch if already have data
import pickle

# Save data
with open('reliance_cache.pkl', 'wb') as f:
    pickle.dump(df, f)

# Load data
with open('reliance_cache.pkl', 'rb') as f:
    df = pickle.load(f)
```

**2. Reduce Training Size:**
```python
# Use only last N trading days instead of 5 years
df_recent = df.tail(252)  # Last 1 year of trading days
```

**3. Simplify Ensemble:**
```python
# Use only XGBoost instead of Random Forest + XGBoost + LightGBM
model_family = "xgboost"  # instead of "ensemble"
```

**4. Cache Models:**
```python
# Don't retrain - load saved model
import joblib
model = joblib.load('models/stock_model.pkl')
predictions = model.predict(X_test)
```

---

## Logging & Debugging

### Enable Debug Mode
```powershell
# Run with verbose logging
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -m streamlit run app/Streamlit_main_app.py --logger.level=debug 2>&1 | Tee-Object streamlit.log
```

### Check Errors
```powershell
# View saved logs
Get-Content streamlit.log | Select-String "Error|error|Exception" -Context 3
```

### Manual Testing
```python
# Test step by step
import sys
sys.path.insert(0, 'd:\\Projects\\New folder\\stock-ai')

from src.data_loader import fetch_stock_data
print("Step 1: Fetch data")
df = fetch_stock_data("RELIANCE.NS", "2024-01-01", "2024-03-26")

from src.features import add_features
print("Step 2: Add features")
df = add_features(df)

from src.model import train_model
print("Step 3: Train model")
model = train_model(df)

print("✓ All steps successful!")
```

---

## Still Having Issues?

**Check the Diagnostic Report:**
```
cat d:\Projects\New folder\stock-ai\DIAGNOSTIC_REPORT.md
```

**Common Paths:**
- Python: `d:/Projects/New folder/.venv/Scripts/python.exe`
- Project: `d:\Projects\New folder\stock-ai`
- Models: `d:\Projects\New folder\stock-ai\models`
- Data: `d:\Projects\New folder\stock-ai\data`
- Cache: `C:\Users\{username}\.cache\huggingface`

---

💡 **Pro Tips:**
1. Always use the venv Python, not system Python
2. Start Streamlit from inside the project directory
3. Keep ticker symbols in correct format (e.g., "RELIANCE.NS" not "RELIANCE")
4. Give yfinance 2-3 seconds to download data
5. First sentiment analysis always takes 30-60 seconds (expected)
