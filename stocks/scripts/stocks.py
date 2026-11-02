import json
import numpy as np
import yfinance as yf

symbol = "AAPL"
df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
close = df["Close"].squeeze().dropna()

r = np.log(close / close.shift(1)).dropna()
sigma = r.std(ddof=1) * np.sqrt(252)
mu_log = r.mean() * 252
mu = mu_log + 0.5 * sigma**2

params = {
    "symbol": symbol,
    "S0": float(close.iloc[-1]),
    "mu": float(mu),
    "mu_log": float(mu_log),
    "sigma": float(sigma),
}

with open("params.json", "w") as f:
    json.dump(params, f, indent=2)

print(params)
