import streamlit as st
import pandas as pd
import requests
import numpy as np
import time

# ================= CONFIG =================
st.set_page_config(page_title="Whale Trap Scanner", layout="wide")
st.title("🐳 Whale Trap Dashboard")

TOP_N = 20                 # Scan only top 20 coins
CANDLE_INTERVAL = "15m"
CANDLE_LIMIT = 50
REFRESH_SECONDS = 60

# ================= RSI (NO ta LIB) =================
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

# ================= DATA FETCH =================
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_top_gainers():
    url = "https://api.binance.com/api/v3/ticker/24hr"

    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except:
        return pd.DataFrame(columns=["symbol","lastPrice","priceChangePercent","volume"])

    if not isinstance(data, list):
        return pd.DataFrame(columns=["symbol","lastPrice","priceChangePercent","volume"])

    df = pd.DataFrame(data)

    cols = ["symbol","lastPrice","priceChangePercent","volume"]
    if not all(c in df.columns for c in cols):
        return pd.DataFrame(columns=cols)

    df = df[cols]
    df["priceChangePercent"] = pd.to_numeric(df["priceChangePercent"], errors="coerce")

    return df.sort_values("priceChangePercent", ascending=False).head(TOP_N)

def get_candles(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={CANDLE_INTERVAL}&limit={CANDLE_LIMIT}"

    try:
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data, columns=[
            "OpenTime","Open","High","Low","Close","Volume",
            "CloseTime","QAV","Trades","TB","TQ","Ignore"
        ])
        df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
        return df
    except:
        return None

def wick_ratio(df):
    upper = df["High"] - df[["Open","Close"]].max(axis=1)
    lower = df[["Open","Close"]].min(axis=1) - df["Low"]
    if lower.mean() == 0:
        return 0
    return round(upper.mean() / lower.mean(), 2)

def fetch_futures(symbol):
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

# ================= SCORING =================
def trap_score(rsi, wick, funding):
    score = 0
    if rsi > 70: score += 1
    if wick > 1: score += 1
    if funding < 0: score += 1
    return score

# ================= MAIN =================
def main():
    top = fetch_top_gainers()

    if top.empty or "symbol" not in top.columns:
        st.warning("⚠️ Binance API unavailable. Retrying...")
        st.stop()

    results = []

    for sym in top["symbol"]:
        df = get_candles(sym)
        if df is None or len(df) < 20:
            continue

        rsi = compute_rsi(df["Close"])
        wick = wick_ratio(df)
        fr, oi = fetch_futures(sym)
        score = trap_score(rsi, wick, fr)

        flag = (
            "🔴 Short Trap" if score == 3 else
            "🟠 Trap Forming" if score == 2 else
            "🟢 Pumping"
        )

        row = top[top["symbol"] == sym].iloc[0]

        results.append({
            "Symbol": sym,
            "Price": row["lastPrice"],
            "%Change": row["priceChangePercent"],
            "Volume": row["volume"],
            "RSI": rsi,
            "WickRatio": wick,
            "Funding%": fr,
            "OpenInterest": oi,
            "Flag": flag
        })

    if not results:
        st.warning("No valid data yet.")
        return

    df = pd.DataFrame(results)
    st.dataframe(df.sort_values("%Change", ascending=False), use_container_width=True)

    st.caption(
        f"Auto-refresh: {REFRESH_SECONDS}s | Top {TOP_N} Binance coins | 15m candles"
    )

# ================= RUN =================
main()
