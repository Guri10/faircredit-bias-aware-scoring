#!/usr/bin/env python3
import sys,os
import json


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(proj_root, "data", "raw", "german.data")
from src.preprocessing import load_data, split_and_preprocess


# 1) Load & split
df = load_data(data_path)
X_train, X_test, y_train, y_test, preproc = split_and_preprocess(df)

# 2) Build feature names
num_feats = df.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_feats = df.select_dtypes(include=["object"]).columns.drop("Risk")
ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
ohe_names = list(ohe.get_feature_names_out(cat_feats))
feature_names = num_feats + ohe_names

# 3) Grab first test row
values = X_test[0].tolist()

# 4) Create payloads
payload_split = {"columns": feature_names, "data": [values]}
payload_records = [dict(zip(feature_names, values))]
payload_csv = ",".join(feature_names) + "\n" + ",".join(str(v) for v in values)

# 5) Write to files
with open("payload_split.json","w") as f:
    json.dump(payload_split, f, indent=2)
with open("payload_records.json","w") as f:
    json.dump(payload_records, f, indent=2)
with open("payload.csv","w") as f:
    f.write(payload_csv)

print("Wrote: payload_split.json, payload_records.json, payload.csv")
