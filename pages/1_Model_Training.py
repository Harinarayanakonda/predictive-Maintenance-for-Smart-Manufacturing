import streamlit as st
import pandas as pd
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import base64

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Model Training", layout="wide")

# =================================================
# BACKGROUND IMAGE (FILE PATH)
# =================================================
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* -------- HIDE NAV -------- */
        [data-testid="stSidebar"],
        [data-testid="stHeader"] {{
            display: none;
        }}

        /* -------- NORMAL TEXT = WHITE -------- */
        h1, h2, h3, h4, h5, h6,
        p, span, label, small,
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"],
        div[data-testid="stCaption"],
        div[data-testid="stDataFrame"] * {{
            color: white !important;
        }}

        /* -------- INPUT AREA TEXT = BLACK -------- */
        div[data-baseweb="select"] *,
        ul[data-baseweb="menu"] * {{
            color: black !important;
        }}

        /* -------- FILE UPLOADER PLACEHOLDER -------- */
        div[data-testid="stFileUploader"] section * {{
            color: black !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stFileUploader"] svg {{
            fill: black !important;
        }}

        /* -------- UPLOADED FILE ROW = ALL WHITE -------- */
        div[data-testid="stFileUploaderFile"],
        div[data-testid="stFileUploaderFileName"],
        div[data-testid="stFileUploaderFile"] small,
        div[data-testid="stFileUploaderFile"] svg,
        div[data-testid="stFileUploaderDeleteBtn"] button {{
            color: white !important;
            fill: white !important;
            font-weight: 600 !important;
        }}

        /* -------- ALL BUTTONS -------- */
        .stButton > button {{
            background-color: #DF6D14 !important;
            color: black !important;
            font-weight: 700 !important;
            border-radius: 6px !important;
            padding: 10px 18px !important;
            border: none !important;
        }}

        /* -------- BOTTOM RIGHT BUTTON CONTAINER -------- */
        .bottom-right {{
            position: fixed;
            bottom: 25px;
            right: 25px;
            display: flex;
            gap: 0px; /* ❌ NO SPACE BETWEEN BUTTONS */
            z-index: 999;
        }}

        /* Remove Streamlit column padding */
        .bottom-right > div {{
            padding: 0 !important;
            margin: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👉 BACKGROUND IMAGE PATH
set_background("images/6.png")

# =================================================
# TITLE
# =================================================
st.title("Model Training")

# =================================================
# FILE UPLOAD
# =================================================
file_type = st.selectbox(
    "Select Dataset File Type",
    ["csv", "json", "xlsx", "xls"]
)

uploaded_file = st.file_uploader(
    "Upload Dataset (Drag & Drop or Browse)",
    type=[file_type]
)

# =================================================
# MAIN LOGIC (UNCHANGED)
# =================================================
if uploaded_file:
    os.makedirs("datasets", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "json":
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    dataset_path = f"datasets/sensor_data_{timestamp}.csv"
    df.to_csv(dataset_path, index=False)

    st.success(f"Dataset saved successfully: {dataset_path}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Target Selection")
    target_column = st.selectbox("Select Target Variable", df.columns)

    st.subheader("Exclude Columns")
    excluded_columns = st.multiselect(
        "Select columns NOT to be used",
        options=[c for c in df.columns if c != target_column]
    )

    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Select Machine Learning Model",
        ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    )

    if st.button("Train Model"):
        X = df.drop(columns=[target_column] + excluded_columns)
        y = df[target_column]

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if y.nunique() == 2 else None
        )

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
            }[model_choice])
        ])

        pipeline.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_choice.replace(' ','_').lower()}_{timestamp}.joblib"
        joblib.dump(pipeline, model_path)

        st.success("Model trained successfully!")
        st.metric("Training Accuracy", f"{train_acc:.2%}")
        st.metric("Test Accuracy", f"{test_acc:.2%}")

        st.caption(f"Model saved at: {model_path}")

# =================================================
# BOTTOM RIGHT NAV (NO GAP)
# =================================================
st.markdown('<div class="bottom-right">', unsafe_allow_html=True)

if st.button("Home"):
    st.switch_page("app.py")
if st.button("Next"):
    st.switch_page("pages/2_Inference.py")

st.markdown('</div>', unsafe_allow_html=True)
