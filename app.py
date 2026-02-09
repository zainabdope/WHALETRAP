# whale_trap_dashboard.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
from ta.momentum import RSIIndicator
import time

st.set_page_config(page_title="Whale Trap Scanner", layout="wide")
st.title("🐳 Whale Trap Dashboard")

# --- PARAMETERS ---
TOP_N = 50          # Number of coins to scan
CANDLE_INTERVAL = "15m"
CANDLE_LIMIT = 50   # Candles for RSI & wick
REFRESH_SECONDS = 60  # Auto-refresh

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=REFRESH_SECONDS)
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_top_gainers():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    resp = requests.get(url, timeout=10)

    try:
        data = resp.json()
    except:
        return pd.DataFrame()

    # 🚨 CRITICAL FIX
    if not isinstance(data, list):
        st.warning("Binance API temporarily unavailable. Retrying...")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    required = {"symbol", "lastPrice", "priceChangePercent", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["priceChangePercent"] = pd.to_numeric(df["priceChangePercent"], errors="coerce")

    return (
        df.sort_values("priceChangePercent", ascending=False)
        .head(TOP_N)[["symbol", "lastPrice", "priceChangePercent", "volume"]]
    )


def get_candles(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={CANDLE_INTERVAL}&limit={CANDLE_LIMIT}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['OpenTime','Open','High','Low','Close','Volume','CloseTime','QuoteAssetVol','Trades','TBBaseVol','TBQuoteVol','Ignore'])
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
    return df

def compute_rsi(symbol):
    df = get_candles(symbol)
    close_prices = df['Close']
    rsi = RSIIndicator(close_prices, window=14).rsi().iloc[-1]
    return round(rsi, 2)

def wick_ratio(df):
    df['upper_wick'] = df['High'] - df[['Close','Open']].max(axis=1)
    df['lower_wick'] = df[['Close','Open']].min(axis=1) - df['Low']
    ratio = (df['upper_wick'].mean() / df['lower_wick'].mean()) if df['lower_wick'].mean() != 0 else 0
    return round(ratio, 2)

def fetch_futures_data(symbol):
    # Funding rate
    try:
        fr = requests.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1").json()[0]['fundingRate']
        fr = float(fr) * 100  # %
    except:
        fr = np.nan
    # Open interest
    try:
        oi = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}").json()['openInterest']
        oi = float(oi)
    except:
        oi = np.nan
    return fr, oi

def short_trap_score(rsi, wick, funding):
    score = 0
    if rsi > 70: score += 1
    if wick > 1: score += 1
    if funding < 0: score +=1
    return score

# --- MAIN DASHBOARD ---
def main():
    top_gainers = fetch_top_gainers()
    symbols = top_gainers['symbol'].tolist()
    results = []

    for sym in symbols:
        try:
            rsi = compute_rsi(sym)
            candles = get_candles(sym)
            wick = wick_ratio(candles)
            fr, oi = fetch_futures_data(sym)
            score = short_trap_score(rsi, wick, fr)
            if score >= 3:
                flag = "🔴 Short Trap High"
            elif score == 2:
                flag = "🟠 Trap Forming"
            else:
                flag = "🟢 Pumping"
            results.append({
                "Symbol": sym,
                "Price": top_gainers[top_gainers['symbol']==sym]['lastPrice'].values[0],
                "%Change": top_gainers[top_gainers['symbol']==sym]['priceChangePercent'].values[0],
                "Volume": top_gainers[top_gainers['symbol']==sym]['volume'].values[0],
                "RSI": rsi,
                "WickRatio": wick,
                "Funding%": fr,
                "OpenInterest": oi,
                "Flag": flag
            })
        except Exception as e:
            print(f"Error {sym}: {e}")
            continue

    df_results = pd.DataFrame(results)
    st.dataframe(df_results.sort_values("%Change", ascending=False), use_container_width=True)
    st.caption(f"Auto-refresh every {REFRESH_SECONDS}s | Top {TOP_N} Binance coins | 15m candles")

# Auto-refresh loop
if st.button("Refresh Now") or True:
    main()
