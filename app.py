# whale_trap_dashboard.py
import streamlit as st 
import pandas as pd
import requests
import numpy as np
from ta.momentum import RSIIndicator
from datetime import datetime
import time

st.set_page_config(page_title="Whale Trap Scanner", layout="wide")
st.title("🐳 Whale Trap Dashboard")

# --- PARAMETERS ---
TOP_N = st.sidebar.slider("Number of coins to scan", 10, 100, 50)
CANDLE_INTERVAL = st.sidebar.selectbox("Candle Interval", ["5m", "15m", "30m", "1h", "4h"], index=1)
REFRESH_SECONDS = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)

# Initialize session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_top_gainers(top_n):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        resp = requests.get(url, timeout=10).json()
        df = pd.DataFrame(resp)
        df['priceChangePercent'] = df['priceChangePercent'].astype(float)
        top_gainers = df.sort_values('priceChangePercent', ascending=False).head(top_n)
        top_gainers = top_gainers[['symbol','lastPrice','priceChangePercent','volume']]
        return top_gainers
    except Exception as e:
        st.error(f"Error fetching top gainers: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_candles(symbol, interval):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data, columns=['OpenTime','Open','High','Low','Close','Volume','CloseTime',
                                          'QuoteAssetVol','Trades','TBBaseVol','TBQuoteVol','Ignore'])
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching candles for {symbol}: {e}")
        return pd.DataFrame()

def compute_rsi(symbol, interval):
    df = get_candles(symbol, interval)
    if df.empty:
        return 0
    close_prices = df['Close']
    rsi = RSIIndicator(close_prices, window=14).rsi().iloc[-1]
    return round(rsi, 2)

def wick_ratio(df):
    if df.empty:
        return 0
    df['upper_wick'] = df['High'] - df[['Close','Open']].max(axis=1)
    df['lower_wick'] = df[['Close','Open']].min(axis=1) - df['Low']
    df = df[df['lower_wick'] > 0]  # Filter out candles with no lower wick
    if len(df) == 0:
        return 0
    ratio = (df['upper_wick'].mean() / df['lower_wick'].mean())
    return round(ratio, 2)

def fetch_futures_data(symbol):
    # Funding rate
    try:
        fr_data = requests.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1", timeout=10).json()
        fr = float(fr_data[0]['fundingRate']) * 100  # %
    except:
        fr = np.nan
    
    # Open interest
    try:
        oi_data = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}", timeout=10).json()
        oi = float(oi_data['openInterest'])
    except:
        oi = np.nan
    return fr, oi

def short_trap_score(rsi, wick, funding):
    score = 0
    if rsi > 70: score += 1
    if wick > 1.5: score += 1  # Increased threshold for better accuracy
    if funding < -0.01: score += 1  # Only count significant negative funding
    return score

# --- MAIN DASHBOARD ---
def main():
    st.sidebar.header("Settings")
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Manual refresh button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Whale Trap Signals")
    with col2:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner("Fetching data from Binance..."):
        top_gainers = fetch_top_gainers(TOP_N)
        
        if top_gainers.empty:
            st.error("Failed to fetch data from Binance. Please try again.")
            return
            
        symbols = top_gainers['symbol'].tolist()
        results = []
        
        progress_bar = st.progress(0)
        total_symbols = len(symbols)
        
        for i, sym in enumerate(symbols):
            try:
                # Update progress
                progress_bar.progress((i + 1) / total_symbols)
                
                # Fetch all data
                rsi = compute_rsi(sym, CANDLE_INTERVAL)
                candles = get_candles(sym, CANDLE_INTERVAL)
                wick = wick_ratio(candles)
                fr, oi = fetch_futures_data(sym)
                score = short_trap_score(rsi, wick, fr)
                
                # Determine flag
                if score >= 3:
                    flag = "🔴 Short Trap High"
                elif score == 2:
                    flag = "🟠 Trap Forming"
                else:
                    flag = "🟢 Pumping"
                
                results.append({
                    "Symbol": sym,
                    "Price": float(top_gainers[top_gainers['symbol']==sym]['lastPrice'].values[0]),
                    "%Change": round(float(top_gainers[top_gainers['symbol']==sym]['priceChangePercent'].values[0]), 2),
                    "Volume": float(top_gainers[top_gainers['symbol']==sym]['volume'].values[0]),
                    "RSI": rsi,
                    "WickRatio": wick,
                    "Funding%": round(fr, 4) if not np.isnan(fr) else "N/A",
                    "OpenInterest": f"{oi:,.0f}" if not np.isnan(oi) else "N/A",
                    "TrapScore": score,
                    "Flag": flag
                })
            except Exception as e:
                continue
        
        progress_bar.empty()
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            # Sort by TrapScore descending
            df_results = df_results.sort_values("TrapScore", ascending=False)
            
            # Display results
            st.dataframe(
                df_results,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.4f"),
                    "%Change": st.column_config.NumberColumn("24h %", format="%.2f%%"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "WickRatio": st.column_config.NumberColumn("Wick Ratio", format="%.2f"),
                    "Funding%": st.column_config.NumberColumn("Funding", format="%.4f%%"),
                    "TrapScore": st.column_config.NumberColumn("Score", width="small"),
                    "Flag": st.column_config.TextColumn("Signal", width="medium")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            st.subheader("📊 Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Coins Scanned", len(df_results))
            with col2:
                high_traps = len(df_results[df_results['TrapScore'] >= 3])
                st.metric("High Trap Signals", high_traps)
            with col3:
                st.metric("Avg. RSI", f"{df_results['RSI'].mean():.1f}")
            
            # Last updated timestamp
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"Top {TOP_N} Binance coins | {CANDLE_INTERVAL} candles | "
                      f"Auto-refresh: {'On' if st.session_state.auto_refresh else 'Off'}")
        else:
            st.warning("No data to display. Please check your internet connection.")

# Run the main function
main()

# Auto-refresh logic
if st.session_state.auto_refresh:
    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_SECONDS:
        st.session_state.last_refresh = current_time
        st.rerun()
