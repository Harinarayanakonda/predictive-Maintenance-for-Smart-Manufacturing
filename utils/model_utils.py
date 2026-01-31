import os, joblib
from datetime import datetime

def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"models/{name}_{ts}.joblib"
    joblib.dump(model, path)
    return path

def load_all_models():
    if not os.path.exists("models"):
        return []
    return [f for f in os.listdir("models") if f.endswith(".joblib")]
