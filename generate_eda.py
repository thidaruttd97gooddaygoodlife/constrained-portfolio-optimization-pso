import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "CSCO",
    "JPM", "V", "MA", "BAC", "WFC",
    "LLY", "UNH", "JNJ", "ABBV", "MRK",
    "XOM", "CVX",
    "PG", "HD", "COST", "KO", "PEP",
    "GE", "CAT",
    "NFLX", "DIS",
    "LIN", "NEE"
]

os.makedirs("outputs", exist_ok=True)

# 1. Data Downloading (Preprocessing Step 1)
print("Downloading Data for EDA...")
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=10*365)

prices = yf.download(TICKERS, start=start_date, end=end_date, auto_adjust=True)
if isinstance(prices.columns, pd.MultiIndex):
    data = prices['Close'].copy()
else:
    data = prices.copy()
    
data.dropna(axis=1, thresh=int(len(data)*0.8), inplace=True)
data.ffill(inplace=True)
data.bfill(inplace=True)

# 2. Historical Prices Plot (EDA)
plt.figure(figsize=(12, 6))
for ticker in TICKERS[:10]: # plot top 10 for visibility
    plt.plot(data.index, data[ticker] / data[ticker].iloc[0] * 100, label=ticker)
plt.title("Normalized Historical Prices (Top 10 Stocks)")
plt.xlabel("Date")
plt.ylabel("Normalized Price (Base 100)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/eda_historical_prices.png", dpi=300)
plt.close()

# 3. Calculate Daily Returns (Preprocessing Step 2)
returns = data.pct_change().dropna()

# 4. Correlation Heatmap (EDA Feature Relationship)
plt.figure(figsize=(14, 12))
corr_matrix = returns.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, vmin=-1, vmax=1)
plt.title("Stock Returns Correlation Heatmap (Feature Relationship)")
plt.tight_layout()
plt.savefig("outputs/eda_correlation_heatmap.png", dpi=300)
plt.close()

# 5. Risk vs Return Scatter for individual stocks (EDA)
annual_return = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)

plt.figure(figsize=(10, 8))
plt.scatter(annual_volatility, annual_return, alpha=0.7, color='blue')
for i, ticker in enumerate(TICKERS):
    plt.annotate(ticker, (annual_volatility.iloc[i], annual_return.iloc[i]), fontsize=8, alpha=0.7)
plt.title("Individual Stocks: Risk vs Expected Return")
plt.xlabel("Annual Volatility (Risk)")
plt.ylabel("Annual Expected Return")
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig("outputs/eda_risk_return_scatter.png", dpi=300)
plt.close()

print("EDA Images generated successfully in outputs/ folder.")
