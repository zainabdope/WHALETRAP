# app.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
import time

st.set_page_config(page_title="Whale Trap Scanner", layout="wide")
st.title("🐳 Whale Trap Dashboard")

# ---------------- PARAMETERS ----------------
TOP_N = 50
CANDLE_INTERVAL = "15m"
CANDLE_LIMIT = 50
REFRESH_SECONDS = 60

# ---------------- HELPERS ----------------
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_top_gainers():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url, timeout=10).json()
    df = pd.DataFrame(data)
    df["priceChangePercent"] = df["priceChangePercent"].astype(float)
    return (
        df.sort_values("priceChangePercent", ascending=False)
        .head(TOP_N)[["symbol", "lastPrice", "priceChangePercent", "volume"]]
    )

def get_candles(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={CANDLE_INTERVAL}&limit={CANDLE_LIMIT}"
    data = requests.get(url, timeout=10).json()
    df = pd.DataFrame(
        data,
        columns=[
            "OpenTime","Open","High","Low","Close","Volume",
            "CloseTime","QuoteVol","Trades","TBBase","TBQuote","Ignore"
        ],
    )
    df[["Open","High","Low","Close","Volume"]] = df[
        ["Open","High","Low","Close","Volume"]
    ].astype(float)
    return df

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def wick_ratio(df):
    upper = df["High"] - df[["Open","Close"]].max(axis=1)
    lower = df[["Open","Close"]].min(axis=1) - df["Low"]
    lw = lower.mean()
    return round(upper.mean() / lw, 2) if lw != 0 else 0

def futures_metrics(symbol):
    try:
        fr = requests.get(
            f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1",
            timeout=10
        ).json()[0]["fundingRate"]
        fr = float(fr) * 100
    except:
        fr = np.nan

    try:
        oi = requests.get(
            f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}",
            timeout=10
        ).json()["openInterest"]
        oi = float(oi)
    except:
        oi = np.nan

    return fr, oi

def trap_score(rsi, wick, funding):
    score = 0
    if rsi > 70: score += 1
    if wick > 1.2: score += 1
    if funding < 0: score += 1
    return score

# ---------------- MAIN ----------------
def main():
    gainers = fetch_top_gainers()
    rows = []

    for sym in gainers["symbol"]:
        try:
            candles = get_candles(sym)
            rsi = compute_rsi(candles["Close"])
            wick = wick_ratio(candles)
            fr, oi = futures_metrics(sym)
            score = trap_score(rsi, wick, fr)

            flag = (
                "🔴 Short Trap"
                if score == 3 else
                "🟠 Trap Forming"
                if score == 2 else
                "🟢 Pumping"
            )

            base = gainers[gainers["symbol"] == sym].iloc[0]
            rows.append({
                "Symbol": sym,
                "Price": base["lastPrice"],
                "%Change": base["priceChangePercent"],
                "Volume": base["volume"],
                "RSI": rsi,
                "WickRatio": wick,
                "Funding%": fr,
                "OpenInterest": oi,
                "Signal": flag,
            })
        except:
            continue

    df = pd.DataFrame(rows)
    st.dataframe(df.sort_values("%Change", ascending=False), use_container_width=True)
    st.caption("Auto-refresh every 60s • Binance public data • No API keys")

main()
