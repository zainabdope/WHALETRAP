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

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# --- HELPER FUNCTIONS ---
def test_binance_connection():
    """Test if Binance API is accessible"""
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_top_gainers_alternative():
    """Alternative method to fetch top gainers using a different endpoint"""
    try:
        # Try using the 24hr ticker endpoint
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        
        # Debug: Check response
        st.write(f"Response status: {response.status_code}")
        st.write(f"Response content type: {response.headers.get('content-type')}")
        
        if response.status_code != 200:
            st.error(f"API Error: Status code {response.status_code}")
            return None
            
        data = response.json()
        
        # Check if data is a list
        if isinstance(data, list):
            # Success - we got the expected data
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            df['priceChangePercent'] = pd.to_numeric(df['priceChangePercent'], errors='coerce')
            df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Filter for USDT pairs and sort by gain
            df = df[df['symbol'].str.endswith('USDT')]
            df = df.sort_values('priceChangePercent', ascending=False)
            
            return df.head(TOP_N)
        else:
            # We got an error object instead of a list
            st.error(f"Unexpected response format: {data}")
            return None
            
    except Exception as e:
        st.error(f"Error in fetch_top_gainers: {str(e)}")
        return None

def get_top_gainers_simple():
    """Simpler method to get top gainers"""
    try:
        # Use a more reliable endpoint
        url = "https://api.binance.com/api/v3/ticker/price"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None
            
        all_prices = response.json()
        
        # Get 24hr ticker for each symbol (limit to 100 to avoid rate limiting)
        top_symbols = [item['symbol'] for item in all_prices if 'USDT' in item['symbol']][:100]
        
        results = []
        for symbol in top_symbols[:50]:  # Limit to 50 for speed
            try:
                ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                ticker_response = requests.get(ticker_url, timeout=5)
                
                if ticker_response.status_code == 200:
                    ticker_data = ticker_response.json()
                    results.append({
                        'symbol': ticker_data.get('symbol'),
                        'lastPrice': float(ticker_data.get('lastPrice', 0)),
                        'priceChangePercent': float(ticker_data.get('priceChangePercent', 0)),
                        'volume': float(ticker_data.get('volume', 0))
                    })
            except:
                continue
        
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('priceChangePercent', ascending=False)
            return df.head(TOP_N)
        else:
            return None
            
    except Exception as e:
        st.error(f"Error in get_top_gainers_simple: {e}")
        return None

def get_top_gainers_fallback():
    """Fallback method using static data or cached data"""
    try:
        # Try to load from a backup file or use sample data
        st.info("Using fallback data - Binance API might be temporarily unavailable")
        
        # Create sample data for testing
        sample_data = []
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 
                  'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LINKUSDT', 'MATICUSDT']
        
        for i, symbol in enumerate(symbols[:TOP_N]):
            sample_data.append({
                'symbol': symbol,
                'lastPrice': 1000 + i * 100,
                'priceChangePercent': 5 + i * 2,
                'volume': 1000000 + i * 500000
            })
        
        df = pd.DataFrame(sample_data)
        return df
        
    except Exception as e:
        st.error(f"Fallback also failed: {e}")
        return None

@st.cache_data(ttl=30)
def get_candles(symbol, interval):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return pd.DataFrame()
            
        data = response.json()
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
            
        columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
                  'CloseTime', 'QuoteAssetVol', 'Trades', 'TBBaseVol', 
                  'TBQuoteVol', 'Ignore']
        
        df = pd.DataFrame(data, columns=columns)
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except Exception as e:
        return pd.DataFrame()

def compute_rsi(symbol, interval):
    df = get_candles(symbol, interval)
    if df.empty or len(df) < 14:
        return 50
    
    try:
        close_prices = df['Close'].dropna()
        if len(close_prices) < 14:
            return 50
            
        rsi_indicator = RSIIndicator(close=close_prices, window=14)
        rsi_values = rsi_indicator.rsi()
        
        if rsi_values.empty:
            return 50
            
        return round(rsi_values.iloc[-1], 2)
    except:
        return 50

def wick_ratio(df):
    if df.empty or len(df) < 5:
        return 1.0
    
    try:
        df = df.copy()
        df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        
        df = df[(df['upper_wick'] > 0) & (df['lower_wick'] > 0)]
        
        if len(df) == 0:
            return 1.0
            
        avg_upper_wick = df['upper_wick'].mean()
        avg_lower_wick = df['lower_wick'].mean()
        
        if avg_lower_wick == 0:
            return 2.0
            
        ratio = avg_upper_wick / avg_lower_wick
        return round(ratio, 2)
    except:
        return 1.0

def fetch_futures_data(symbol):
    fr, oi = np.nan, np.nan
    
    # Only try futures for USDT pairs
    if 'USDT' not in symbol:
        return fr, oi
    
    try:
        fr_url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        fr_response = requests.get(fr_url, timeout=5)
        
        if fr_response.status_code == 200:
            fr_data = fr_response.json()
            if fr_data and isinstance(fr_data, list) and len(fr_data) > 0:
                fr = float(fr_data[0]['fundingRate']) * 100
    except:
        pass
    
    try:
        oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        oi_response = requests.get(oi_url, timeout=5)
        
        if oi_response.status_code == 200:
            oi_data = oi_response.json()
            if 'openInterest' in oi_data:
                oi = float(oi_data['openInterest'])
    except:
        pass
    
    return fr, oi

def short_trap_score(rsi, wick, funding):
    score = 0
    
    if rsi > 75:
        score += 2
    elif rsi > 70:
        score += 1
    
    if wick > 2.0:
        score += 2
    elif wick > 1.5:
        score += 1
    
    if not np.isnan(funding) and funding < -0.02:
        score += 2
    elif not np.isnan(funding) and funding < 0:
        score += 1
    
    return min(score, 5)

# --- MAIN DASHBOARD ---
def main():
    st.sidebar.header("Settings")
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Connection test
    if st.sidebar.button("Test Connection"):
        if test_binance_connection():
            st.sidebar.success("✓ Connected to Binance API")
        else:
            st.sidebar.error("✗ Cannot connect to Binance API")
    
    # Manual refresh button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Whale Trap Signals")
    with col2:
        if st.button("🔄 Refresh Now", type="primary"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    # Check connection first
    if not test_binance_connection():
        st.error("⚠️ Cannot connect to Binance API. Please check:")
        st.write("1. Your internet connection")
        st.write("2. If Binance is accessible from your location")
        st.write("3. Try using a VPN if needed")
        
        # Offer to use demo mode
        use_demo = st.checkbox("Use demo mode with sample data", value=True)
        if use_demo:
            top_gainers = get_top_gainers_fallback()
        else:
            return
    else:
        # Try multiple methods to get data
        st.info("Fetching data from Binance...")
        
        # Try main method first
        top_gainers = fetch_top_gainers_alternative()
        
        # If main method fails, try simple method
        if top_gainers is None or top_gainers.empty:
            st.warning("Trying alternative method...")
            top_gainers = get_top_gainers_simple()
        
        # If still no data, use fallback
        if top_gainers is None or top_gainers.empty:
            st.warning("Using fallback data...")
            top_gainers = get_top_gainers_fallback()
    
    if top_gainers is None or top_gainers.empty:
        st.error("Could not fetch any data. Please try again later.")
        return
    
    st.success(f"Loaded {len(top_gainers)} coins")
    
    # Display some debug info in expander
    with st.expander("Data Preview"):
        st.dataframe(top_gainers.head(10))
    
    symbols = top_gainers['symbol'].tolist()
    results = []
    
    if len(symbols) > 0:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_symbols = len(symbols)
        
        for i, sym in enumerate(symbols):
            try:
                # Update progress
                progress_percentage = (i + 1) / total_symbols
                progress_bar.progress(progress_percentage)
                status_text.text(f"Scanning: {sym} ({i+1}/{total_symbols})")
                
                # Get coin data
                coin_data = top_gainers[top_gainers['symbol'] == sym]
                if coin_data.empty:
                    continue
                    
                coin_data = coin_data.iloc[0]
                
                # Calculate indicators
                rsi = compute_rsi(sym, CANDLE_INTERVAL)
                candles = get_candles(sym, CANDLE_INTERVAL)
                wick = wick_ratio(candles)
                fr, oi = fetch_futures_data(sym)
                score = short_trap_score(rsi, wick, fr)
                
                # Determine signal
                if score >= 4:
                    flag = "🔴 HIGH TRAP"
                    color = "#ffcccc"
                elif score == 3:
                    flag = "🟠 MEDIUM TRAP"
                    color = "#ffe6cc"
                elif score == 2:
                    flag = "🟡 WATCH"
                    color = "#ffffcc"
                else:
                    flag = "🟢 SAFE"
                    color = "#ccffcc"
                
                results.append({
                    "Symbol": sym,
                    "Price": float(coin_data['lastPrice']) if pd.notna(coin_data['lastPrice']) else 0,
                    "24h %": round(float(coin_data['priceChangePercent']), 2) if pd.notna(coin_data['priceChangePercent']) else 0,
                    "Volume": f"${float(coin_data['volume']):,.0f}" if pd.notna(coin_data['volume']) else "N/A",
                    "RSI": rsi,
                    "Wick": wick,
                    "Funding": f"{fr:.4f}%" if not np.isnan(fr) else "N/A",
                    "OI": f"{oi:,.0f}" if not np.isnan(oi) else "N/A",
                    "Score": score,
                    "Signal": flag,
                    "Color": color
                })
                
            except Exception as e:
                continue
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values("Score", ascending=False)
            
            # Display results
            st.subheader("📈 Scan Results")
            
            # Create a styled DataFrame
            def apply_color(row):
                return [f'background-color: {row["Color"]}'] * len(row)
            
            styled_df = df_results.drop('Color', axis=1).style.apply(apply_color, axis=1)
            
            # Display with container width
            st.dataframe(
                styled_df,
                column_config={
                    "Symbol": "Symbol",
                    "Price": st.column_config.NumberColumn("Price", format="$%.4f"),
                    "24h %": st.column_config.NumberColumn("24h %", format="+%.2f%%"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "Wick": st.column_config.NumberColumn("Wick Ratio", format="%.2f"),
                    "Score": st.column_config.ProgressColumn("Trap Score", min_value=0, max_value=5),
                    "Signal": "Signal"
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Summary
            st.subheader("📊 Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_signals = len(df_results[df_results['Score'] >= 4])
                st.metric("High Risk Signals", high_signals)
            
            with col2:
                avg_rsi = df_results['RSI'].mean()
                st.metric("Average RSI", f"{avg_rsi:.1f}")
            
            with col3:
                st.metric("Total Coins", len(df_results))
            
            # Last update
            st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"Auto-refresh: {'On' if st.session_state.auto_refresh else 'Off'}")
        else:
            st.warning("No data to display after scanning.")
    else:
        st.warning("No symbols found to scan.")

# Run the main function
main()

# Auto-refresh logic
if st.session_state.auto_refresh:
    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_SECONDS:
        st.session_state.last_refresh = current_time
        st.rerun()
