# src/run_preprocess.py
import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(proj_root, "data", "raw", "german.data")



if __name__ == "__main__":
    df = load_data(path=data_path)
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Risk"],
        random_state=42
    )
    df_train.to_csv("data/processed/train.csv", index=False)
    df_test.to_csv("data/processed/test.csv", index=False)
