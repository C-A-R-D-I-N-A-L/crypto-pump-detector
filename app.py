import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# -App Configuration & Styling
st.set_page_config(page_title="Crypto Pump Detector", layout="wide", page_icon="🕵️‍♂️")

# Override Streamlit default theme to ensure consistent Dark Mode
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;}
    [data-testid="stAppViewContainer"] { background-color: #0E1117; }
    [data-testid="stHeader"] { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #262730; }
    h1, h2, h3, p, span, div { color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ AI Crypto Pump Detector")
st.markdown("Автоматизована система виявлення ринкових аномалій на базі **Isolation Forest**")

# -Sidebar Controls
with st.sidebar:
    st.header("⚙️ Параметри аналізу")

    # Defaulting to PEPE/USDT as a high-volatility example suitable for demo
    symbol = st.text_input("Торгова пара (Ticker)", value="PEPE/USDT")

    timeframe = st.selectbox("Таймфрейм", ["5m", "15m", "1h", "4h"], index=1)

    # Contamination parameter: expected proportion of outliers in the dataset
    contamination = st.slider("Чутливість (Contamination)", 1, 10, 1) / 100
    st.caption("ℹ️ Визначає очікувану частку аномалій у вибірці (поріг чутливості).")


# -Core Logic

def fetch_data(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetches historical OHLCV data from Binance API via CCXT.
    Returns a DataFrame with standardized timestamp.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
            },
        'userAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Note: No proxy needed if deployed in EU/non-restricted region
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates technical indicators required for the ML model.
    Key features: Returns, Volatility (20p), Volume Ratio (vs SMA50).
    """
    data = df.copy()

    data['returns'] = data['close'].pct_change()

    # Rolling standard deviation as a proxy for volatility
    data['volatility'] = data['returns'].rolling(window=20).std()

    # Relative volume to detect sudden liquidity injections
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=50).mean()

    # Drop NaNs created by rolling windows to prevent model errors
    data.dropna(inplace=True)
    return data


def detect_anomalies(df: pd.DataFrame, contamination_rate: float) -> pd.DataFrame:
    """
    Trains Isolation Forest on the fly and flags anomalies.
    Returns the dataframe with 'anomaly_score' and 'is_pump' boolean.
    """
    # random_state fixed for reproducibility
    model = IsolationForest(contamination=contamination_rate, random_state=42)

    features = ['returns', 'volume_ratio', 'volatility']

    # fit_predict returns -1 for outliers and 1 for inliers
    df['anomaly_score'] = model.fit_predict(df[features])
    df['is_pump'] = df['anomaly_score'] == -1

    return df


# -Main Execution Flow

if st.sidebar.button("🚀 Запустити сканування", type="primary"):
    with st.spinner(f'Отримання даних для {symbol}...'):
        try:
            # 1. Data Acquisition & Processing
            raw_data = fetch_data(symbol, timeframe)
            processed_data = feature_engineering(raw_data)

            # 2. Model Inference
            result_df = detect_anomalies(processed_data, contamination)
            anomalies = result_df[result_df['is_pump']]

            # 3. KPI Display
            last_close = result_df['close'].iloc[-1]
            prev_close = result_df['close'].iloc[-2]
            pct_change = (last_close - prev_close) / prev_close * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Поточна ціна", f"{last_close:.8f}", f"{pct_change:.2f}%")
            col2.metric("Виявлено аномалій", len(anomalies))
            col3.metric("Макс. сплеск об'єму", f"{result_df['volume_ratio'].max():.1f}x")

            # 4. Visualization
            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=result_df['timestamp'],
                open=result_df['open'], high=result_df['high'],
                low=result_df['low'], close=result_df['close'],
                name='Price Action'
            ))

            # Anomaly markers
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['high'],
                mode='markers',
                name='Detected Anomaly',
                marker=dict(color='#FF4B4B', size=10, symbol='triangle-down')
            ))

            fig.update_layout(
                title=f'Результати аналізу: {symbol} ({timeframe})',
                xaxis_title='Час',
                yaxis_title='Ціна (USDT)',
                height=600,
                template="plotly_dark",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # 5. Detailed Logs
            st.subheader("📋 Журнал аномальної активності")
            st.dataframe(
                anomalies[['timestamp', 'close', 'volume_ratio', 'volatility']]
                .sort_values(by='timestamp', ascending=False)
                .style.background_gradient(subset=['volume_ratio'], cmap='Reds'),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Помилка виконання: {str(e)}. Перевірте правильність тікера.")
else:

    st.info("👈 Для початку роботи ініціюйте аналіз через панель управління.")
