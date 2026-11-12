
import streamlit as st
import pandas as pd
import joblib

import live_data
import tv_chart
import paper_trading
import economic_calender as economic_calendar
import report
import multi_horizon


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="GoldSilver AI | Intelligence Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# PROFESSIONAL UI DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #0B1220;
    color: #E5EAF2;
}

[data-testid="stHeader"] {
    background: rgba(11,18,32,0.95);
}

[data-testid="stSidebar"] {
    background: #101B2D;
    border-right: 1px solid #26364C;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1 {
    color: #E8C76A !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #E5EAF2 !important;
    font-weight: 700 !important;
}

p, label {
    color: #AAB7CA;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg,#17243A,#111C2E);
    border: 1px solid #34445D;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.20);
}

[data-testid="stMetricLabel"] {
    color: #AAB7CA !important;
}

[data-testid="stMetricValue"] {
    color: #E8C76A !important;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: #111C2E;
    padding: 10px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #B8C4D5;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #E8C76A !important;
}

.stButton button,
.stDownloadButton button {
    background: #D6B45C;
    color: #101827;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #E8C76A;
    color: #101827;
    border: none;
}

[data-testid="stDataFrame"] {
    border: 1px solid #34445D;
    border-radius: 12px;
}

hr {
    border-color: #26364C;
}

[data-testid="stExpander"] {
    background: #111C2E;
    border: 1px solid #34445D;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA AND MODEL LOADING
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("gold_silver_dashboard_data.csv")


@st.cache_resource
def load_models():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="surajkumar2618/gold-silver-model",
        filename="gold_silver_models.pkl"
    )

    return joblib.load(path)


df = load_data()
models = load_models()


# ==========================================
# MACHINE LEARNING PREDICTIONS
# ==========================================

features = models["features"]

latest = df.iloc[-1]

X_latest = df[features].iloc[[-1]].astype(float)

gold_price = float(
    models["gold_price_model"].predict(X_latest)[0]
)

silver_price = float(
    models["silver_price_model"].predict(X_latest)[0]
)

gold_direction = str(
    models["gold_direction_model"].predict(X_latest)[0]
)

silver_direction = str(
    models["silver_direction_model"].predict(X_latest)[0]
)


# ==========================================
# LIVE MARKET DATA
# ==========================================

live = live_data.get_live_prices()

gold_live = (
    live["Gold"]["price"]
    if live and live.get("Gold")
    else float(latest["Gold_Close"])
)

silver_live = (
    live["Silver"]["price"]
    if live and live.get("Silver")
    else float(latest["Silver_Close"])
)

prices = {
    "Gold": gold_live,
    "Silver": silver_live
}

preds = {
    "Gold": {
        "next": gold_price,
        "direction": gold_direction
    },
    "Silver": {
        "next": silver_price,
        "direction": silver_direction
    }
}

signals = {
    "Gold": gold_direction,
    "Silver": silver_direction
}


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div style="
    background:linear-gradient(110deg,#17243A,#101B2D);
    border:1px solid #34445D;
    border-radius:18px;
    padding:25px;
    margin-bottom:20px;
">

<div style="color:#E8C76A;font-size:14px;
font-weight:700;letter-spacing:3px;">
MARKET INTELLIGENCE PLATFORM
</div>

<div style="font-size:36px;font-weight:800;
color:#F2F5FA;margin-top:8px;">
GOLDSILVER <span style="color:#E8C76A;">AI</span>
</div>

<div style="font-size:15px;color:#AAB7CA;margin-top:8px;">
Advanced Market Analytics · Machine Learning · Trading Research
</div>

</div>
""", unsafe_allow_html=True)

st.caption(
    f"Latest historical observation: {latest['Timestamp_IST']}"
)

st.info(
    "Educational research platform. ML estimates are not guaranteed "
    "market outcomes or personalized financial advice."
)


# ==========================================
# LIVE MARKET OVERVIEW
# ==========================================

st.markdown("## 📊 Market Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🟡 GOLD / USD",
        f"${gold_live:,.2f}",
        "Live / latest available"
    )

with col2:
    st.metric(
        "⚪ SILVER / USD",
        f"${silver_live:,.2f}",
        "Live / latest available"
    )

with col3:
    correlation = df[
        ["Gold_Return", "Silver_Return"]
    ].corr().iloc[0, 1]

    st.metric(
        "🔗 CORRELATION",
        f"{correlation:.4f}"
    )

st.divider()


# ==========================================
# MACHINE LEARNING SECTION
# ==========================================

st.markdown("## 🤖 AI Intelligence & Predictions")

gold_col, silver_col = st.columns(2)

with gold_col:

    st.markdown("### 🟡 Gold Analysis")

    st.metric(
        "Latest Historical Price",
        f"${float(latest['Gold_Close']):,.2f}"
    )

    st.metric(
        "Estimated Next-Hour Price",
        f"${gold_price:,.2f}"
    )

    if gold_direction == "UP":
        st.success("Model Direction: UP ▲")
    else:
        st.error("Model Direction: DOWN ▼")


with silver_col:

    st.markdown("### ⚪ Silver Analysis")

    st.metric(
        "Latest Historical Price",
        f"${float(latest['Silver_Close']):,.2f}"
    )

    st.metric(
        "Estimated Next-Hour Price",
        f"${silver_price:,.2f}"
    )

    if silver_direction == "UP":
        st.success("Model Direction: UP ▲")
    else:
        st.error("Model Direction: DOWN ▼")


st.caption(
    "Predictions are generated using the trained machine learning models."
)

st.divider()


# ==========================================
# HISTORICAL PRICE ANALYSIS
# ==========================================

st.markdown("## 📈 Gold vs Silver Price History")

price_data = df[
    ["Timestamp_IST", "Gold_Close", "Silver_Close"]
].copy()

price_data["Timestamp_IST"] = pd.to_datetime(
    price_data["Timestamp_IST"],
    errors="coerce"
)

price_data = price_data.dropna(
    subset=["Timestamp_IST"]
).set_index("Timestamp_IST")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.markdown("### Gold Price")

    st.line_chart(
        price_data[["Gold_Close"]],
        color="#E8C76A",
        height=300
    )

with chart_col2:

    st.markdown("### Silver Price")

    st.line_chart(
        price_data[["Silver_Close"]],
        color="#AAB7CA",
        height=300
    )


# ==========================================
# VOLATILITY ANALYSIS
# ==========================================

st.markdown("## ⚡ Volatility Analysis")

volatility_data = df[
    ["Gold_Volatility", "Silver_Volatility"]
].copy()

st.line_chart(
    volatility_data,
    color=["#E8C76A", "#AAB7CA"],
    height=350
)


# ==========================================
# TRADING SESSION ANALYSIS
# ==========================================

st.markdown("## 🌍 Global Trading Sessions")

sessions = {
    "Asian": "Asian_Session",
    "London": "London_Session",
    "New York": "NewYork_Session",
    "London-New York Overlap": "Overlap_Session"
}

session_results = []

for name, column in sessions.items():

    temp = df[
        df[column].fillna(0).astype(bool)
    ]

    session_results.append({
        "Session": name,
        "Total Records": len(temp),
        "Gold Volatility": temp["Gold_Return"].std(),
        "Silver Volatility": temp["Silver_Return"].std()
    })

session_table = pd.DataFrame(session_results)

st.dataframe(
    session_table.round(4),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# HISTORICAL DATA
# ==========================================

with st.expander("📂 Explore Historical Dataset"):

    st.dataframe(
        df.tail(100),
        use_container_width=True
    )


# ==========================================
# ADVANCED TOOLS
# ==========================================

st.divider()

st.markdown("## 🛠️ Advanced Trading & Research Tools")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💼 Paper Trading",
    "📅 Economic Calendar",
    "📄 PDF Report",
    "🔮 Multi-Horizon",
    "📉 TradingView Chart"
])


# PAPER TRADING

with tab1:

    st.subheader("💼 Paper Trading Simulator")

    st.caption(
        "Virtual trading environment for educational research."
    )

    paper_trading.render(
        prices,
        signals
    )


# ECONOMIC CALENDAR

with tab2:

    st.subheader("📅 Economic Calendar")

    economic_calendar.render()


# PDF REPORT

with tab3:

    st.subheader("📄 Research Report")

    ev, _ = economic_calendar.load_events()

    upcoming = economic_calendar.get_upcoming(ev)

    chart = report.make_chart(
        price_data["Gold_Close"].tail(500),
        price_data["Silver_Close"].tail(500)
    )

    pdf_bytes = report.build_pdf(
        prices,
        preds,
        paper_trading.summary(prices),
        upcoming,
        chart
    )

    st.download_button(
        "⬇️ Download PDF Report",
        pdf_bytes,
        "goldsilver_report.pdf",
        "application/pdf",
        key="pdf_dl"
    )


# MULTI HORIZON

with tab4:

    st.subheader("🔮 Multi-Horizon Predictions")

    multi_horizon.render()


# TRADINGVIEW

with tab5:

    st.subheader("📉 Live TradingView Chart")

    tv_chart.render()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 📈 GoldSilver AI")

    st.caption("INTELLIGENCE TERMINAL")

    st.divider()

    st.markdown("### Market Status")

    st.success("Dashboard Loaded")

    st.metric("Gold", f"${gold_live:,.2f}")

    st.metric("Silver", f"${silver_live:,.2f}")

    st.divider()

    st.markdown("### Platform Modules")

    st.markdown("""
    - AI Predictions
    - Market Analytics
    - Correlation
    - Volatility
    - Paper Trading
    - Economic Calendar
    - Research Reports
    """)

    st.divider()

    st.caption("GoldSilver AI | Educational Research")

    st.caption("Machine Learning Powered")
