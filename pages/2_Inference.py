import streamlit as st
import pandas as pd
import os
import joblib
import base64

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Machine Failure Prediction", layout="wide")

# =================================================
# BACKGROUND IMAGE
# =================================================
def set_background_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* -------- BACKGROUND (UNCHANGED) -------- */
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* -------- HIDE SIDEBAR -------- */
        [data-testid="stSidebar"] {{
            display: none;
        }}

        /* -------- TEXT COLOR -------- */
        h1, h2, h3, h4, h5, h6,
        label,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaption"] {{
            color: white !important;
        }}

        /* -------- INPUT BOXES -------- */
        input, textarea {{
            color: white !important;
            background-color: rgba(0,0,0,0.55) !important;
            border-radius: 6px !important;
        }}

        input::placeholder {{
            color: #d0d0d0 !important;
        }}

        /* -------- DROPDOWNS -------- */
        div[data-baseweb="select"] span {{
            color: white !important;
        }}

        div[role="listbox"] {{
            background-color: black !important;
            backdrop-filter: blur(14px);
            border-radius: 8px;
        }}

        div[role="option"] {{
            color: white !important;
            background-color: transparent !important;
        }}

        div[role="option"]:hover {{
            background-color: rgba(255,255,255,0.15) !important;
        }}

        /* -------- ERROR TEXT -------- */
        .error-text {{
            color: #ff6b6b;
            font-size: 13px;
        }}

        /* -------- RESULT BOXES -------- */
        .result-success {{
            background-color: rgba(18,59,26,0.85);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        .result-fail {{
            background-color: rgba(59,18,18,0.85);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        /* -------- BUTTONS -------- */
        .stButton > button {{
            background-color: #DF6D14 !important;
            color: black !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            padding: 10px 22px !important;
            border: none !important;
        }}

        /* -------- BOTTOM NAV (CENTERED AT BOTTOM OF CONTENT) -------- */
        .bottom-nav {{
            display: flex;
            gap: 14px;
            justify-content: center;
            margin-top: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =================================================
# SET BACKGROUND
# =================================================
set_background_image("images/6.png")

# =================================================
# TITLE
# =================================================
st.markdown("<h2 style='text-align:center;'>Machine Failure Prediction</h2>", unsafe_allow_html=True)

# =================================================
# LOAD MODELS
# =================================================
if not os.path.exists("models") or not os.listdir("models"):
    st.error("❌ No trained models found. Please train a model first.")
    st.stop()

model_files = sorted(os.listdir("models"), reverse=True)

st.subheader("Select Trained Model")

selected_model_file = st.selectbox(
    "Choose a trained model for prediction",
    model_files
)

model = joblib.load(os.path.join("models", selected_model_file))
st.caption(f"Using model: {selected_model_file}")

# =================================================
# EXTRACT FEATURE SCHEMA
# =================================================
preprocessor = model.named_steps["preprocessing"]

feature_names = []
numeric_cols = set()

for name, _, cols in preprocessor.transformers_:
    feature_names.extend(cols)
    if name == "num":
        numeric_cols.update(cols)

st.subheader("Enter Machine Parameters")

user_data = {}
errors = {}

# =================================================
# INPUT FORM
# =================================================
for col in feature_names:

    if col.lower() == "type":
        user_data[col] = st.selectbox(
            "Machine Type",
            ["Low", "Medium", "High"]
        )
        continue

    val = st.text_input(col, placeholder="Enter value")

    if not val.strip():
        errors[col] = "Required"
    else:
        if col in numeric_cols:
            try:
                user_data[col] = float(val)
            except:
                errors[col] = "Required"
        else:
            user_data[col] = val

    if col in errors:
        st.markdown("<div class='error-text'>Required</div>", unsafe_allow_html=True)

# =================================================
# PREDICTION
# =================================================
if st.button("🔍 Predict"):
    if errors:
        st.warning("⚠ Please fix errors before prediction.")
    else:
        input_df = pd.DataFrame([user_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-fail">
                    <h3>⚠ Failure Risk Detected</h3>
                    <p>Failure Probability: {probability:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-success">
                    <h3>✅ Machine Healthy</h3>
                    <p>Failure Probability: {probability:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =================================================
# BOTTOM NAVIGATION (SIDE BY SIDE, AT BOTTOM OF PAGE)
# =================================================
st.markdown('<div class="bottom-nav">', unsafe_allow_html=True)
if st.button("Home"):
    st.switch_page("app.py")
if st.button("Next"):
    st.switch_page("pages/3_Dashboard.py")
st.markdown('</div>', unsafe_allow_html=True)