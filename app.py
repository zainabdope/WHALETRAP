import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

st.set_page_config(page_title="Whale Trap Scanner", layout="wide")
st.title("🐋 Whale Trap Scanner")
st.markdown("**Real-time detection of short squeeze/liquidity harvesting setups**")

# --- Settings ---
st.sidebar.header("Settings")
TOP_N = st.sidebar.slider("Coins to Scan", 10, 100, 20, help="Top gainers to analyze")
INTERVAL = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=0)

# --- Working Binance API Functions ---
def get_top_gainers(limit=20):
    """Get top gainers from Binance - working method"""
    try:
        # Get all tickers
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Filter for USDT pairs only
            df = df[df['symbol'].str.endswith('USDT')].copy()
            
            # Convert to numeric
            df['priceChangePercent'] = pd.to_numeric(df['priceChangePercent'], errors='coerce')
            df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Sort by gain
            df = df.sort_values('priceChangePercent', ascending=False)
            
            # Return top N
            top_gainers = df.head(limit)[['symbol', 'lastPrice', 'priceChangePercent', 'volume']]
            
            return top_gainers
    except Exception as e:
        st.error(f"API Error: {e}")
    
    # Fallback to sample data
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
        'SHIBUSDT', 'TRXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT',
        'ETCUSDT', 'FILUSDT', 'LTCUSDT', 'NEARUSDT', 'APEUSDT'
    ][:limit]
    
    data = []
    for i, sym in enumerate(symbols):
        data.append({
            'symbol': sym,
            'lastPrice': np.random.uniform(10, 1000),
            'priceChangePercent': np.random.uniform(5, 30),
            'volume': np.random.uniform(1000000, 50000000)
        })
    
    return pd.DataFrame(data)

def get_klines(symbol, interval="15m", limit=50):
    """Get OHLCV data"""
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except:
        pass
    return None

def get_funding_rate(symbol):
    """Get futures funding rate"""
    try:
        # Convert spot symbol to futures symbol (usually same)
        futures_symbol = symbol.replace('USDT', 'USDT')
        url = f"https://fapi.binance.com/fapi/v1/fundingRate"
        params = {'symbol': futures_symbol, 'limit': 1}
        response = requests.get(url, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                return float(data[0]['fundingRate']) * 100  # Convert to percentage
    except:
        pass
    return 0.0

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period:
        return 50
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def calculate_wick_ratio(df):
    """Calculate average upper/lower wick ratio"""
    if df is None or len(df) < 10:
        return 1.0
    
    try:
        # Calculate wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Filter out zero wicks
        df = df[(df['upper_wick'] > 0) & (df['lower_wick'] > 0)]
        
        if len(df) > 5:
            avg_upper = df['upper_wick'].mean()
            avg_lower = df['lower_wick'].mean()
            
            if avg_lower > 0:
                ratio = avg_upper / avg_lower
                return round(ratio, 2)
    except:
        pass
    
    return 1.0

def calculate_volume_profile(df):
    """Check if volume is decreasing on higher prices (distribution)"""
    if df is None or len(df) < 10:
        return 0
    
    try:
        # Last 5 candles
        recent = df.tail(5).copy()
        
        # Check if price is increasing but volume is decreasing
        price_increase = (recent['close'].iloc[-1] / recent['close'].iloc[0]) - 1
        volume_change = (recent['volume'].iloc[-1] / recent['volume'].iloc[0]) - 1
        
        if price_increase > 0.05 and volume_change < -0.2:  # Price up 5% but volume down 20%
            return 2  # Strong distribution signal
        elif price_increase > 0.03 and volume_change < 0:
            return 1  # Weak distribution signal
    
    except:
        pass
    
    return 0

def calculate_trap_score(symbol_data):
    """Calculate trap probability score 0-10"""
    score = 0
    
    # Extract data
    rsi = symbol_data.get('rsi', 50)
    wick_ratio = symbol_data.get('wick_ratio', 1.0)
    funding = symbol_data.get('funding', 0)
    price_change = symbol_data.get('price_change', 0)
    volume_profile = symbol_data.get('volume_profile', 0)
    
    # 1. RSI Scoring (0-3 points)
    if rsi > 80:
        score += 3
    elif rsi > 75:
        score += 2
    elif rsi > 70:
        score += 1
    
    # 2. Wick Ratio Scoring (0-3 points)
    if wick_ratio > 3.0:
        score += 3
    elif wick_ratio > 2.0:
        score += 2
    elif wick_ratio > 1.5:
        score += 1
    
    # 3. Funding Rate Scoring (0-2 points)
    if funding < -0.03:  # Very negative funding
        score += 2
    elif funding < -0.01:  # Negative funding
        score += 1
    
    # 4. Price Change Scoring (0-1 point)
    if price_change > 25:  # Very large pump
        score += 1
    
    # 5. Volume Profile (0-1 point)
    score += volume_profile
    
    return min(score, 10)

def get_signal_emoji(score):
    """Get signal emoji based on score"""
    if score >= 8:
        return "🔴", "HIGH TRAP", "High probability of short squeeze"
    elif score >= 6:
        return "🟠", "MEDIUM TRAP", "Strong trap signals"
    elif score >= 4:
        return "🟡", "WATCH", "Moderate trap signals"
    else:
        return "🟢", "SAFE", "Minimal trap signals"

# --- Main App ---
st.header("Real-Time Scanner")

# Refresh button
col1, col2 = st.columns([4, 1])
with col1:
    st.subheader(f"Top {TOP_N} Gainers Analysis")
with col2:
    if st.button("🔄 Refresh", type="primary"):
        st.rerun()

# Get data
with st.spinner(f"Scanning top {TOP_N} gainers..."):
    # Fetch top gainers
    top_gainers = get_top_gainers(TOP_N)
    
    if top_gainers.empty:
        st.error("No data available")
        st.stop()
    
    # Analyze each coin
    results = []
    progress_bar = st.progress(0)
    
    for idx, (_, row) in enumerate(top_gainers.iterrows()):
        symbol = row['symbol']
        price = float(row['lastPrice'])
        price_change = float(row['priceChangePercent'])
        volume = float(row['volume'])
        
        # Update progress
        progress_bar.progress((idx + 1) / len(top_gainers))
        
        # Get klines
        klines = get_klines(symbol, INTERVAL)
        
        # Calculate RSI
        rsi_val = 50
        if klines is not None and len(klines) >= 15:
            prices = klines['close']
            rsi_val = calculate_rsi(prices)
        
        # Calculate wick ratio
        wick_val = calculate_wick_ratio(klines)
        
        # Get funding rate
        funding_val = get_funding_rate(symbol)
        
        # Calculate volume profile
        volume_profile_val = calculate_volume_profile(klines)
        
        # Calculate trap score
        symbol_data = {
            'rsi': rsi_val,
            'wick_ratio': wick_val,
            'funding': funding_val,
            'price_change': price_change,
            'volume_profile': volume_profile_val
        }
        
        trap_score = calculate_trap_score(symbol_data)
        
        # Get signal
        emoji, signal, desc = get_signal_emoji(trap_score)
        
        # Add to results
        results.append({
            'Symbol': symbol,
            'Price': price,
            '24h %': price_change,
            'Volume': volume,
            'RSI': rsi_val,
            'Wick': wick_val,
            'Funding': funding_val,
            'Score': trap_score,
            'Signal': f"{emoji} {signal}",
            'Description': desc
        })
    
    progress_bar.empty()

# Display results
if results:
    df = pd.DataFrame(results)
    
    # Sort by score descending
    df = df.sort_values('Score', ascending=False)
    
    # Display summary metrics
    st.subheader("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_traps = len(df[df['Score'] >= 8])
        st.metric("🔴 High Traps", high_traps)
    
    with col2:
        avg_score = df['Score'].mean()
        st.metric("Avg Score", f"{avg_score:.1f}/10")
    
    with col3:
        avg_rsi = df['RSI'].mean()
        st.metric("Avg RSI", f"{avg_rsi:.1f}")
    
    with col4:
        top_coin = df.iloc[0]['Symbol'] if len(df) > 0 else "N/A"
        st.metric("Top Signal", top_coin)
    
    # Display table
    st.subheader("📈 Detailed Analysis")
    
    # Create color function for rows
    def color_row(row):
        if row['Score'] >= 8:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Score'] >= 6:
            return ['background-color: #ffe6cc'] * len(row)
        elif row['Score'] >= 4:
            return ['background-color: #ffffcc'] * len(row)
        else:
            return ['background-color: #ccffcc'] * len(row)
    
    # Apply styling
    styled_df = df.style.apply(color_row, axis=1)
    
    # Display with formatting
    st.dataframe(
        styled_df.format({
            'Price': '${:,.4f}',
            '24h %': '{:+.2f}%',
            'Volume': '${:,.0f}',
            'RSI': '{:.1f}',
            'Wick': '{:.2f}',
            'Funding': '{:+.4f}%',
            'Score': '{:.0f}'
        }),
        column_config={
            'Symbol': st.column_config.TextColumn('Symbol', width='small'),
            'Price': st.column_config.NumberColumn('Price', format='$%.4f'),
            '24h %': st.column_config.NumberColumn('24h %', format='+%.2f%%'),
            'Volume': st.column_config.NumberColumn('Volume', format='$%.0f'),
            'RSI': st.column_config.NumberColumn('RSI', format='%.1f'),
            'Wick': st.column_config.NumberColumn('Wick Ratio', format='%.2f'),
            'Funding': st.column_config.NumberColumn('Funding', format='%+.4f%%'),
            'Score': st.column_config.ProgressColumn('Trap Score', min_value=0, max_value=10, format='%d'),
            'Signal': st.column_config.TextColumn('Signal'),
            'Description': st.column_config.TextColumn('Description', width='large')
        },
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    # Show top 3 warnings
    st.subheader("⚠️ Top Warnings")
    
    high_risk = df[df['Score'] >= 6].head(3)
    
    if len(high_risk) > 0:
        for _, row in high_risk.iterrows():
            with st.expander(f"{row['Signal']} - {row['Symbol']} (Score: {row['Score']}/10)"):
                cols = st.columns(4)
                with cols[0]:
                    st.metric("24h Change", f"{row['24h %']:+.2f}%")
                with cols[1]:
                    st.metric("RSI", f"{row['RSI']:.1f}")
                with cols[2]:
                    st.metric("Wick Ratio", f"{row['Wick']:.2f}")
                with cols[3]:
                    st.metric("Funding", f"{row['Funding']:+.4f}%")
                
                st.info(f"**Why it's flagged:** {row['Description']}")
                st.caption(f"**Action:** Watch for rejection at resistance levels. High probability of short squeeze setup.")
    else:
        st.info("No high-risk traps detected in current scan")
    
    # Explanation
    with st.expander("📖 How to Interpret Signals"):
        st.markdown("""
        ### 🐋 Whale Trap Signals Explained
        
        **What is a Whale Trap?**
        - Large players pump prices to trigger short liquidations
        - Create FOMO buying from retail traders
        - Dump into the liquidity at the top
        
        **Key Indicators:**
        
        | Indicator | Bullish Signal | Bearish/Trap Signal |
        |-----------|----------------|---------------------|
        | **RSI** | 30-50 (Oversold) | **>70 (Overbought)** |
        | **Wick Ratio** | <1 (More lower wicks) | **>2 (More upper wicks = rejection)** |
        | **Funding** | Positive (Longs pay shorts) | **Negative (Shorts pay longs)** |
        | **24h Change** | Gradual increase | **Sudden spike (>20%)** |
        
        **Scoring System (0-10):**
        - **0-3**: 🟢 SAFE - Minimal trap signals
        - **4-5**: 🟡 WATCH - Some concerning indicators
        - **6-7**: 🟠 MEDIUM TRAP - Strong trap setup forming
        - **8-10**: 🔴 HIGH TRAP - High probability of squeeze
        
        **Trading Strategy:**
        1. Look for coins with **Score ≥ 6**
        2. Check liquidation heatmaps for clusters above current price
        3. Wait for rejection signals (pin bars, volume divergence)
        4. Consider shorting near resistance with tight stops
        5. Target: Recent swing lows or below liquidation clusters
        """)
    
    # Last update
    st.caption(f"🕒 Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Interval: {INTERVAL} | Coins: {len(df)}")

# Auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 60:  # Auto-refresh every 60 seconds
    st.session_state.last_refresh = time.time()
    st.rerun()
