import os
import pandas as pd
from datetime import datetime

def save_dataset(file):
    os.makedirs("datasets", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"datasets/dataset_{ts}.csv"
    file.to_csv(path, index=False)
    return path

def load_dataset(uploaded_file, file_type):
    if file_type == "CSV":
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)
