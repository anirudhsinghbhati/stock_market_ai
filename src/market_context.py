# Fetch index data (NIFTY, SENSEX) using yfinance
# Calculate:
# - daily returns
# - 5-day trend
# - volatility (rolling std)
#
# Merge with stock dataset on date
# Prefix columns with index name (e.g., nifty_return)