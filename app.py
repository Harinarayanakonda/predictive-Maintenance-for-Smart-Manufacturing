import streamlit as st
import base64
import time
from pathlib import Path

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# =================================================
# HIDE SIDEBAR
# =================================================
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0; }
</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD BACKGROUND IMAGE
# =================================================
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = Path(__file__).parent
bg_image = get_base64_image(BASE_DIR / "images" / "6.png")

# =================================================
# GLOBAL CSS
# =================================================
st.markdown(f"""
<style>

/* Background */
.stApp {{
    background-image: url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* Header */
.header {{
    padding: 110px 40px 40px 40px;
    text-align: center;
}}

.header h1 {{
    font-size: 68px;
    color: #FFFFFF;
    font-weight: 800;
    text-shadow: 0px 4px 10px rgba(0,0,0,0.6);
}}

.header h3 {{
    font-size: 34px;
    color: #FFFFFF;
    margin-top: 10px;
    text-shadow: 0px 3px 8px rgba(0,0,0,0.6);
}}

.header p {{
    font-size: 24px;
    color: #FFFFFF;
    margin-top: 14px;
    text-shadow: 0px 2px 6px rgba(0,0,0,0.6);
}}

/* Button row */
.button-wrapper {{
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 50px;
}}

/* Heartbeat animation */
@keyframes heartbeat {{
  0% {{ transform: scale(1); }}
  25% {{ transform: scale(1.12); }}
  40% {{ transform: scale(1); }}
  60% {{ transform: scale(1.12); }}
  100% {{ transform: scale(1); }}
}}

.pulse {{
    animation: heartbeat 1s ease-in-out infinite;
}}

/* Buttons */
.stButton > button {{
    background-color: #D1512D;
    color: #FFFFFF;
    padding: 18px 42px;
    border-radius: 16px;
    font-size: 20px;
    font-weight: 600;
    border: none;
}}

.stButton > button:hover {{
    background-color: #8C6A54;
}}

/* Loading text */
.loading-text {{
    color: #FFFFFF;
    font-size: 22px;
    text-align: center;
    margin-top: 20px;
}}

</style>
""", unsafe_allow_html=True)

# =================================================
# HEADER
# =================================================
st.markdown("""
<div class="header">
    <h1>Predictive Maintenance for Smart Manufacturing</h1>
    <h3>AI-Driven Machine Health Monitoring System</h3>
    <p>Predict failures before they happen and reduce downtime</p>
</div>
""", unsafe_allow_html=True)

# =================================================
# BUTTONS (GROUPED & CENTERED)
# =================================================

left, center, right = st.columns([2, 6, 2])

with center:
    btn1, btn2, btn3 = st.columns(3, gap="large")

    def animated_switch(label, msg, page):
        if st.button(label):
            st.markdown(
                "<script>"
                "document.querySelectorAll('button').forEach(btn => btn.classList.add('pulse'));"
                "</script>",
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="loading-text">{msg}</div>', unsafe_allow_html=True)
            time.sleep(2)
            st.switch_page(page)

    with btn1:
        animated_switch("Model Training",
                        "Preparing Model Training Module...",
                        "pages/1_Model_Training.py")

    with btn2:
        animated_switch("Prediction",
                        "Loading Prediction Engine...",
                        "pages/2_Inference.py")

    with btn3:
        animated_switch("Dashboard",
                        "Opening Analytics Dashboard...",
                        "pages/3_Dashboard.py")


st.markdown("</div>", unsafe_allow_html=True)
