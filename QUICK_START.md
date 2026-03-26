# Stock-AI Quick Start Guide

## 🚀 Launch the Application (30 seconds)

### Step 1: Open PowerShell
```
Press: Windows Key + X → Select "Windows PowerShell (Admin)"
```

### Step 2: Navigate to Project
```powershell
cd 'd:\Projects\New folder\stock-ai'
```

### Step 3: Start Streamlit
```powershell
& 'd:/Projects/New folder/.venv/Scripts/python.exe' -m streamlit run app/Streamlit_main_app.py
```

### Step 4: Open Browser
```
Automatic: Click link in terminal, OR
Manual:   http://localhost:8501
```

**✅ App is now running!**

---

## 🎯 Using the Application

### Input Section (Top Bar)
```
📍 Ticker     → Enter: RELIANCE.NS (or INFY.NS, TCS.NS, etc.)
📊 Timeframe  → Select: Intraday, 1 Day, 1 Week, 1 Month, 6 Months
💵 Investment → Enter: Amount to invest (e.g., 10000 for ₹10,000)
🎲 Strategy   → Select: Conservative, Moderate, Aggressive
🔍 [Analyze]  → Click to run AI analysis
```

### Output Sections

#### 1. AI Signal Card
```
Shows: BUY / SELL / HOLD (in large text)
Confidence: Progress bar (0-100%)
Risk Level: High / Medium / Low
⏱️  Analysis Time: How long it took
```

#### 2. Multi-Timeframe Analysis
```
Shows signal for each timeframe:
[Intraday: BUY] [1D: SELL] [1W: BUY] [1M: SELL] [6M: BUY]
Compare across different time horizons
```

#### 3. Trade Setup
```
Entry Price   : ₹123.45 (Blue card)
Target Price  : ₹135.67 (Green card)
Stop Loss     : ₹115.23 (Red card)
Calculate position size based on investment
```

#### 4. Profit/Loss Calculator
```
Max Profit    : ₹1,234 (68%)
Max Loss      : ₹567 (32%)
R:R Ratio     : 2.17 (Risk-Reward)
Visual bar showing profit vs loss potential
```

#### 5. Price Chart
```
Candlestick chart with:
• Entry level (blue line)
• Target level (green line)
• Stop loss level (red line)
Interactive: Hover for values, zoom, pan
```

#### 6. Sentiment & News
```
Sentiment Score: -0.45 to +1.00
Sentiment Label: Negative / Neutral / Positive
News Impact: High / Medium / Low
Top Headlines: Recent news affecting stock
```

#### 7. Backtest Performance
```
Win Rate      : 62.5% (of recent backtests)
Total Profit  : ₹5,432 (cumulative from backtest period)
Max Drawdown  : 8.3% (largest peak-to-trough decline)
Equity Curve  : Line chart showing profit over time
Trade History: Detailed table with each trade
```

#### 8. Risk Disclaimer
```
⚠️  Reminder: This is AI prediction, not financial advice
    Always use stop-loss and manage risk responsibly
```

---

## 📊 Understanding the Signals

### What Does BUY Mean?
```
✅ AI model indicates HIGH probability (>75%) of price increase
✅ Confidence level shows how certain the model is
✓ Good for: Entering long positions
⚠️ Not a guarantee - always use stop loss!
```

### What Does SELL Mean?
```
⚠️ AI model indicates HIGH probability (>75%) of price decrease
⚠️ Confidence level shows how certain the model is
✓ Good for: Closing positions or shorting (if trading permits)
⚠️ Only for experienced traders in India (check broker rules)
```

### What Does HOLD Mean?
```
⏸️ AI model indicates UNCERTAINTY (25-75% probability range)
⏸️ Confidence is medium - mixed signals
✓ Good for: Staying in existing positions / Wait for clarity
⚠️ Don't open new positions during HOLD
```

---

## 💡 Example Workflow

### Scenario: Want to trade RELIANCE.NS

1. **Input your parameters:**
   - Ticker: `RELIANCE.NS`
   - Timeframe: `1 Day` (for swing trading)
   - Investment: `10000` (₹10,000)
   - Strategy: `Moderate`
   - Click: `Analyze`

2. **Wait 12-20 seconds for analysis**
   - Real-time timer shows progress
   - App fetches stock data
   - Trains ML model
   - Backtests strategy
   - Fetches news/sentiment

3. **Review the AI Signal**
   ```
   If signal = BUY with 82% confidence:
   ✅ Consider entering position
   Entry Price: ₹2,456 (current market)
   Stop Loss: ₹2,401 (set at 2.2% below)
   Target: ₹2,567 (set at 4.5% above)
   Risk-Reward: 2.05:1 (Good ratio)
   
   Decision: BUY 4 shares (₹10,000 / ₹2,456 = 4.07 shares)
   If target hits: Profit = (₹2,567 - ₹2,456) × 4 = ₹444
   If stop hits: Loss = (₹2,401 - ₹2,456) × 4 = -₹220
   ```

4. **Check Backtest Performance**
   ```
   Last 252 days trading:
   Win Rate: 58%
   Total Profit: ₹8,532
   Max Drawdown: 6.2%
   
   Interpretation: Model won ~6/10 trades historically
   Average winner > average loser (good R:R)
   ```

5. **Check Multi-Timeframe**
   ```
   Intraday: BUY (short-term up)
   1 Day: SELL (daily is bearish)
   1 Week: BUY (weekly is bullish)
   
   Interpretation: Conflicting signals!
   ⚠️ Maybe wait for alignment before trading
   ```

6. **Consider News/Sentiment**
   ```
   Sentiment: Negative (-0.34)
   Impact: Medium
   Recent headlines: Mixed (some positive, some negative)
   
   Decision: Counter-signal to BUY
   May wait for sentiment to improve
   OR proceed with smaller position size
   ```

---

## ⚙️ Settings Guide

### Timeframes
| Timeframe | Use For | Data Points |
|-----------|---------|------------|
| Intraday | Day trading | 60 days |
| 1 Day | Swing trading | ~180 days |
| 1 Week | Mid-term | ~365 days |
| 1 Month | Position trading | ~730 days |
| 6 Months | Long-term | ~1,825 days |

### Strategies
| Strategy | Risk Level | Entry Threshold | Exit Threshold |
|----------|-----------|-----------------|----------------|
| Conservative | Low | 75% confidence | 25% confidence |
| Moderate | Medium | 70% confidence | 30% confidence |
| Aggressive | High | 60% confidence | 40% confidence |

**Note:** Higher threshold = fewer trades but higher win rate

### Stock Symbols (NSE Format)
```
Reliance:         RELIANCE.NS
Infosys:          INFY.NS
TCS:              TCS.NS
HDFC Bank:        HDFCBANK.NS
Bajaj Auto:       BAJAJ-AUTO.NS
ITC:              ITC.NS
Hindustan Unilever: HINDUNILVR.NS
State Bank:       SBIN.NS

Check NSE website for complete list
```

---

## 🛟 Quick Troubleshooting

### App shows blank page
**Solution:** Click "Analyze" button again

### "No data returned" error
**Solution:** Check ticker format (must end with .NS for India)

### Analysis takes too long (1+ min)
**Solution:** Normal on first run - sentiment model downloads
Subsequent runs will be 15-20 seconds

### Sentiment always shows 0/neutral
**Solution:** Set up free NewsAPI key from https://newsapi.org

### Profit/Loss shows negative on positive prediction
**Solution:** Normal! Market slippage + costs reduce profits

---

## 📈 Advanced Usage

### Running Backtests on Historical Data
```python
# From terminal/notebook:
python -c "
from src.train import train_stock_model
result = train_stock_model(
    'RELIANCE.NS',
    start_date='2023-01-01',  # 3 years back
    end_date='2024-03-26'
)
print(f\"Accuracy: {result['accuracy_train']:.2%}\")
print(f\"Backtest Win Rate: {result['backtest_custom']['metrics']['win_rate']:.2%}\")
"
```

### Batch Analysis (Multiple Stocks)
```python
# Analyze multiple stocks programmatically:
stocks = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS']
for symbol in stocks:
    # In Streamlit: UI doesn't support this natively
    # Use train_stock_model() directly in Python script
    pass
```

### Export Results
```python
# Results are available in Python:
result['dataset']              # Full OHLCV data + features
result['accuracy_train']       # Model accuracy
result['backtest_custom']      # Trade-by-trade history
result['feature_importance']   # Top important features
```

---

## 📞 Need Help?

**Check these files in the project:**
1. `DIAGNOSTIC_REPORT.md` - Full system status
2. `TROUBLESHOOTING.md` - Common issues & solutions
3. `README.md` - Project overview

**Quick Test Command:**
```powershell
$py = 'd:/Projects/New folder/.venv/Scripts/python.exe'
& $py -c "from src.train import train_stock_model; print('✓ System OK')"
```

---

## 🎓 Learning Resources

- **Streamlit:** https://docs.streamlit.io/library
- **yfinance:** https://github.com/ranaroussi/yfinance
- **Technical Indicators:** https://github.com/bukosabino/ta
- **Machine Learning:** scikit-learn.org

---

**Happy Trading! 📊**

⚠️ Remember: This tool is for educational/research purposes. Always do your own due diligence and consult financial advisors before trading.
