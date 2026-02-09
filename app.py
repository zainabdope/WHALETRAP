# whale_trap_dashboard.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="Whale Trap Scanner", layout="wide")
st.title("🐳 Short Liquidity Scanner")
st.markdown("**Scans top gainers for potential short trap setups**")

# --- SETTINGS ---
st.sidebar.header("⚙️ Settings")
TOP_N = st.sidebar.slider("Number of coins to scan", 10, 50, 20, help="Scan top X 24h gainers")
CANDLE_LIMIT = 50
REFRESH_SECONDS = st.sidebar.slider("Refresh (seconds)", 30, 300, 60)

# Session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0
if 'data' not in st.session_state:
    st.session_state.data = None

# --- SIMPLIFIED FUNCTIONS ---
def get_top_gainers(n=20):
    """Get top gainers - with fallback to sample data"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
                # Filter for USDT pairs
                df = df[df['symbol'].str.endswith('USDT')]
                # Convert to numeric
                df['priceChangePercent'] = pd.to_numeric(df['priceChangePercent'], errors='coerce')
                df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                # Get top gainers
                df = df.sort_values('priceChangePercent', ascending=False).head(n)
                return df[['symbol', 'lastPrice', 'priceChangePercent', 'volume']]
    except:
        pass
    
    # Fallback: Create sample data
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
        'SHIBUSDT', 'TRXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT',
        'ETCUSDT', 'FILUSDT', 'LTCUSDT', 'NEARUSDT', 'APEUSDT'
    ][:n]
    
    data = []
    for i, sym in enumerate(symbols):
        data.append({
            'symbol': sym,
            'lastPrice': 100 + i * 50 + np.random.uniform(-20, 20),
            'priceChangePercent': np.random.uniform(5, 25) + i,
            'volume': np.random.uniform(1e6, 1e7)
        })
    
    return pd.DataFrame(data)

def get_candles(symbol, interval="15m"):
    """Get recent candles"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={CANDLE_LIMIT}"
        data = requests.get(url, timeout=5).json()
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
            # Keep only OHLCV columns
            if len(df) > 0:
                df = df.iloc[:, :5]
                df.columns = ['time', 'open', 'high', 'low', 'close']
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
    except:
        pass
    return None

def calculate_wick_ratio(candles):
    """Calculate upper/lower wick ratio"""
    if candles is None or len(candles) < 10:
        return 1.0
    
    try:
        candles = candles.copy()
        # Calculate wicks
        candles['upper_wick'] = candles['high'] - candles[['open', 'close']].max(axis=1)
        candles['lower_wick'] = candles[['open', 'close']].min(axis=1) - candles['low']
        
        # Remove zeros to avoid division issues
        candles = candles[(candles['upper_wick'] > 0) & (candles['lower_wick'] > 0)]
        
        if len(candles) > 5:
            avg_upper = candles['upper_wick'].mean()
            avg_lower = candles['lower_wick'].mean()
            if avg_lower > 0:
                return round(avg_upper / avg_lower, 2)
    except:
        pass
    return 1.0

def get_funding_rate(symbol):
    """Get funding rate if available"""
    try:
        if 'USDT' in symbol:
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            data = requests.get(url, timeout=3).json()
            if isinstance(data, list) and len(data) > 0:
                return float(data[0]['fundingRate']) * 100
    except:
        pass
    return 0.0

def calculate_rsi(prices, period=14):
    """Simple RSI calculation"""
    if prices is None or len(prices) < period + 1:
        return 50
    
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 1)
    except:
        return 50

def analyze_trap_potential(rsi, wick_ratio, funding, price_change):
    """Calculate trap potential score 0-10"""
    score = 0
    
    # RSI overbought (2 points if > 70, 3 if > 80)
    if rsi > 80:
        score += 3
    elif rsi > 70:
        score += 2
    
    # High wick ratio indicates rejection (2 points if > 2, 3 if > 3)
    if wick_ratio > 3:
        score += 3
    elif wick_ratio > 2:
        score += 2
    elif wick_ratio > 1.5:
        score += 1
    
    # Negative funding encourages shorts (1 point if negative)
    if funding < -0.01:
        score += 2
    elif funding < 0:
        score += 1
    
    # High 24h pump increases trap probability (1 point if > 15%)
    if price_change > 20:
        score += 2
    elif price_change > 10:
        score += 1
    
    return min(score, 10)

def get_signal(score):
    """Get signal based on score"""
    if score >= 7:
        return "🔴 HIGH TRAP", "High probability of short squeeze/liquidity harvest"
    elif score >= 5:
        return "🟠 MEDIUM TRAP", "Potential trap forming"
    elif score >= 3:
        return "🟡 WATCH", "Some concerning signals"
    else:
        return "🟢 SAFE", "No significant trap signals"

# --- MAIN DASHBOARD ---
def main():
    # Refresh button
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader("Real-time Scanner")
    with col2:
        if st.button("🔄 Scan Now", type="primary", use_container_width=True):
            st.session_state.data = None
            st.rerun()
    with col3:
        st.caption(f"Next refresh: {REFRESH_SECONDS}s")
    
    # Get data
    if st.session_state.data is None:
        with st.spinner(f"Scanning top {TOP_N} gainers..."):
            # Get top gainers
            top_gainers = get_top_gainers(TOP_N)
            
            results = []
            progress_bar = st.progress(0)
            
            for idx, (_, row) in enumerate(top_gainers.iterrows()):
                symbol = row['symbol']
                
                # Update progress
                progress_bar.progress((idx + 1) / len(top_gainers))
                
                # Get candles
                candles = get_candles(symbol)
                
                # Calculate indicators
                wick_ratio_val = calculate_wick_ratio(candles)
                
                # Calculate RSI
                rsi_val = 50
                if candles is not None and 'close' in candles.columns:
                    prices = candles['close'].dropna()
                    if len(prices) >= 15:
                        rsi_val = calculate_rsi(prices)
                
                # Get funding rate
                funding = get_funding_rate(symbol)
                
                # Calculate trap score
                price_change = float(row['priceChangePercent'])
                trap_score = analyze_trap_potential(rsi_val, wick_ratio_val, funding, price_change)
                
                # Get signal
                signal, signal_desc = get_signal(trap_score)
                
                # Color coding
                if trap_score >= 7:
                    row_color = "#ffcccc"  # Light red
                elif trap_score >= 5:
                    row_color = "#ffe6cc"  # Light orange
                elif trap_score >= 3:
                    row_color = "#ffffcc"  # Light yellow
                else:
                    row_color = "#ccffcc"  # Light green
                
                results.append({
                    "Symbol": symbol,
                    "Price": f"${float(row['lastPrice']):,.4f}",
                    "24h %": f"{price_change:+.2f}%",
                    "Volume": f"${float(row['volume']):,.0f}",
                    "RSI": rsi_val,
                    "Wick Ratio": wick_ratio_val,
                    "Funding": f"{funding:+.4f}%" if funding != 0 else "0.00%",
                    "Trap Score": trap_score,
                    "Signal": signal,
                    "Description": signal_desc,
                    "_color": row_color
                })
            
            progress_bar.empty()
            st.session_state.data = results
            st.session_state.last_refresh = time.time()
    
    # Display results
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        
        # Sort by Trap Score (highest first)
        df = df.sort_values("Trap Score", ascending=False)
        
        # Display metrics
        st.subheader("📊 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        high_traps = len(df[df["Trap Score"] >= 7])
        with col1:
            st.metric("🔴 High Traps", high_traps)
        
        avg_score = df["Trap Score"].mean()
        with col2:
            st.metric("Avg Score", f"{avg_score:.1f}/10")
        
        top_gainer = df.iloc[0]["Symbol"] if len(df) > 0 else "N/A"
        with col3:
            st.metric("Top Gainer", top_gainer)
        
        with col4:
            st.metric("Total Scanned", len(df))
        
        # Display table
        st.subheader("📈 Scan Results")
        
        # Create display without color column
        display_cols = ["Signal", "Symbol", "Price", "24h %", "Volume", "RSI", "Wick Ratio", "Funding", "Trap Score", "Description"]
        display_df = df[display_cols].copy()
        
        # Display with color coding
        for i, row in df.iterrows():
            cols = st.columns([1, 2, 2, 2, 3, 2, 2, 2, 2, 4])
            with cols[0]:
                st.markdown(f"<div style='background-color:{row['_color']}; padding:5px; border-radius:5px; text-align:center'>{row['Signal']}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"**{row['Symbol']}**")
            with cols[2]:
                st.markdown(row['Price'])
            with cols[3]:
                color = "green" if float(row['24h %'].replace('%', '').replace('+', '')) > 0 else "red"
                st.markdown(f"<span style='color:{color}'>{row['24h %']}</span>", unsafe_allow_html=True)
            with cols[4]:
                st.markdown(row['Volume'])
            with cols[5]:
                rsi_color = "red" if row['RSI'] > 70 else "orange" if row['RSI'] > 60 else "green"
                st.markdown(f"<span style='color:{rsi_color}'>{row['RSI']}</span>", unsafe_allow_html=True)
            with cols[6]:
                wick_color = "red" if row['Wick Ratio'] > 2 else "orange" if row['Wick Ratio'] > 1.5 else "green"
                st.markdown(f"<span style='color:{wick_color}'>{row['Wick Ratio']}</span>", unsafe_allow_html=True)
            with cols[7]:
                funding_color = "red" if float(row['Funding'].replace('%', '')) < -0.01 else "orange" if float(row['Funding'].replace('%', '')) < 0 else "green"
                st.markdown(f"<span style='color:{funding_color}'>{row['Funding']}</span>", unsafe_allow_html=True)
            with cols[8]:
                st.progress(row['Trap Score'] / 10, text=f"{row['Trap Score']}/10")
            with cols[9]:
                st.caption(row['Description'])
        
        st.divider()
        
        # Top 3 warnings
        st.subheader("⚠️ Top Warnings")
        high_risk = df[df["Trap Score"] >= 7]
        
        if len(high_risk) > 0:
            for _, row in high_risk.head(3).iterrows():
                with st.container():
                    st.warning(f"**{row['Symbol']}** - Score: {row['Trap Score']}/10 - {row['Description']}")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("24h Gain", row['24h %'])
                    with cols[1]:
                        st.metric("RSI", row['RSI'])
                    with cols[2]:
                        st.metric("Wick Ratio", row['Wick Ratio'])
                    with cols[3]:
                        st.metric("Funding", row['Funding'])
        else:
            st.info("No high-risk traps detected")
        
        # Last update
        st.caption(f"🕒 Last scan: {datetime.now().strftime('%H:%M:%S')} | Coins scanned: {len(df)}")
    
    # Explanation
    with st.expander("📖 How to read this scanner"):
        st.markdown("""
        **What is a Whale Trap/Short Liquidity Harvest?**
        
        Whales often pump prices to trigger:
        1. **Short liquidations** - When price rises above short positions' liquidation prices
        2. **FOMO buying** - Retail traders chase the pump
        3. **Then dump** - Selling into the liquidity
        
        **Key signals to watch:**
        
        | Signal | What it means | Why it matters |
        |--------|---------------|----------------|
        | 🔴 **High RSI (>70)** | Overbought condition | Price has moved too fast, correction likely |
        | 📈 **High Wick Ratio (>2)** | Long upper wicks on candles | Sellers rejecting higher prices |
        | 📉 **Negative Funding** | Shorts pay longs | Encourages more short positions to liquidate |
        | 🚀 **Large 24h Pump (>20%)** | Sharp price increase | Creates liquidation cascades above current price |
        
        **Trading this setup:**
        - Look for coins with **Score ≥ 7** 
        - Check liquidation levels on liquidation heatmaps
        - Wait for rejection signals (long upper wicks, volume divergence)
        - Consider shorting near resistance with tight stops
        
        **⚠️ Risk Warning:** This is for educational purposes. Always use proper risk management.
        """)

# Run main
main()

# Auto-refresh
if time.time() - st.session_state.last_refresh > REFRESH_SECONDS:
    st.session_state.data = None
    st.rerun()
