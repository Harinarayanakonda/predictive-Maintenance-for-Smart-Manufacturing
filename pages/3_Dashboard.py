import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime

# =================================================
# USER BACKGROUND IMAGE PATH
# =================================================
BACKGROUND_IMAGE = "images/6.png"

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

# =================================================
# LOAD BACKGROUND IMAGE (BASE64)
# =================================================
def load_bg_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if not os.path.exists(BACKGROUND_IMAGE):
    st.error("❌ Background image not found.")
    st.stop()

bg_base64 = load_bg_image_base64(BACKGROUND_IMAGE)

# =================================================
# GLOBAL STYLING (SAFE HEADER RESET)
# =================================================
st.markdown(f"""
<style>

/* ===== STREAMLIT HEADER (SAFE RESET) ===== */
header[data-testid="stHeader"] {{
    background: white !important;
}}

/* Header text only (Deploy, Settings, Rerun) */
header[data-testid="stHeader"] > div * {{
    color: black !important;
}}

/* Header icons (⋮ icon, refresh, etc.) */
header[data-testid="stHeader"] svg {{
    fill: black !important;
}}

/* DO NOT touch dropdown menus */
div[data-baseweb="menu"] * {{
    color: inherit !important;
}}

/* Hide sidebar */
[data-testid="stSidebar"] {{
    display: none;
}}

/* App background */
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}}

/* Divider line */
.section-line {{
    width: 100%;
    height: 2px;
    background: linear-gradient(to right, transparent, #DE802B, transparent);
    margin: 25px 0;
}}

/* Typography */
h1 {{ font-size: 48px !important; color: white; }}
h2 {{ font-size: 34px !important; color: white; }}
h3 {{ font-size: 26px !important; color: white; }}

p, span, label {{
    font-size: 18px !important;
    color: white !important;
}}

/* Metrics */
[data-testid="stMetricLabel"] {{
    font-size: 20px !important;
    color: white !important;
}}

[data-testid="stMetricValue"] {{
    font-size: 32px !important;
    color: #DE802B !important;
}}

/* Tabs */
.stTabs [role="tab"] {{
    font-size: 20px;
    padding: 12px;
}}

/* Buttons */
.stButton > button {{
    background-color: #DE802B;
    color: white;
    border-radius: 14px;
    font-weight: bold;
    padding: 0.6em 1.6em;
    border: none;
}}
.stButton > button:hover {{
    background-color: #ffb347;
}}

/* Charts */
svg, canvas {{
    background-color: white;
    border-radius: 12px;
    padding: 10px;
}}

/* Footer */
.footer {{
    text-align: center;
    opacity: 0.7;
    margin-top: 30px;
    font-size: 16px;
    color: white;
}}

</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD DATA (CACHED)
# =================================================
@st.cache_data
def load_data():
    dataset_path = os.path.join("datasets", sorted(os.listdir("datasets"))[-1])
    return pd.read_csv(dataset_path), dataset_path

if not os.path.exists("datasets") or not os.listdir("datasets"):
    st.error("❌ No dataset found.")
    st.stop()

df, dataset_path = load_data()

# =================================================
# CLEAN DATA (UNCHANGED)
# =================================================
id_like = [c for c in df.columns if c.lower() in ["udi", "id", "index"]]
df_clean = df.drop(columns=id_like, errors="ignore")
numeric_df = df_clean.select_dtypes(include=["int64", "float64"])

# =================================================
# HEADER
# =================================================
st.markdown(
    "<h1 style='text-align:center; font-size:89px;'>Predictive Maintenance Dashboard</h1>",
    unsafe_allow_html=True
)

st.caption("Operational overview, analytics & maintenance intelligence")
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

# =================================================
# TABS
# =================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Analytics", "Insights"])

# =================================================
# TAB 1: OVERVIEW
# =================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    failure_rate = df["Machine failure"].mean() * 100 if "Machine failure" in df.columns else None

    c1.metric("Total Records", len(df))
    c2.metric("Failure Rate (%)", round(failure_rate, 2))
    c3.metric("Avg Temperature", round(numeric_df.mean().iloc[0], 2))
    c4.metric("Avg Tool Wear", round(numeric_df.mean().iloc[-1], 2))

    if failure_rate and failure_rate > 5:
        st.error("🚨 Failure rate above acceptable threshold")
    else:
        st.success("✅ System operating within safe limits")

# =================================================
# TAB 2: ANALYTICS
# =================================================
with tab2:
    st.subheader("📈 Feature Distribution (Record Level)")
    feature = st.selectbox("Select Feature", numeric_df.columns)

    fig, ax = plt.subplots()
    sns.histplot(numeric_df[feature], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    if "Machine failure" in df.columns:
        st.subheader("⚖️ Feature vs Failure Comparison")
        fig, ax = plt.subplots()
        sns.boxplot(x=df["Machine failure"], y=df[feature], ax=ax)
        ax.set_xticklabels(["Healthy", "Failed"])
        st.pyplot(fig)

    st.subheader("📉 Feature Trend Across Records")
    fig, ax = plt.subplots()
    ax.plot(numeric_df[feature].values)
    ax.set_xlabel("Record Index")
    ax.set_ylabel(feature)
    st.pyplot(fig)

# =================================================
# TAB 3: INSIGHTS
# =================================================
with tab3:
    st.subheader("🛠️ Maintenance Recommendations")
    st.info("""
    - High tool wear → Replace tools  
    - High temperature → Inspect cooling  
    - High torque → Reduce load  
    - Frequent failures → Increase inspections  
    """)

    missing_pct = df.isnull().mean().mean() * 100
    last_updated = datetime.fromtimestamp(os.path.getmtime(dataset_path))

    st.subheader("📘 Data Trust Summary")
    st.write(f"• Missing data: **{missing_pct:.2f}%**")
    st.write(f"• Dataset updated: **{last_updated.strftime('%d %b %Y, %H:%M')}**")


# =================================================
# ✅ BOTTOM-RIGHT HOME BUTTON (VISIBLE NOW)
# =================================================
st.markdown('<div class="bottom-right">', unsafe_allow_html=True)

if st.button("🏠 Home"):
    st.switch_page("app.py")

st.markdown('</div>', unsafe_allow_html=True)
# =================================================
# FOOTER
# =================================================
st.markdown("""
<div class="footer">
© 2026 Predictive Maintenance System • Version 1.1 • Academic & Industrial Ready
</div>
""", unsafe_allow_html=True)