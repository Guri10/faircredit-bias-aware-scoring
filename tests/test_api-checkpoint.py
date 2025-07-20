# tests/test_api.py

import requests
import json

URL = "http://127.0.0.1:1234/invocations"

def test_api_records():
    # Load raw records (a list of dicts)
    with open("payload_records.json") as f:
        records = json.load(f)

    # Wrap under the 'dataframe_records' key
    payload = {"dataframe_records": records}

    resp = requests.post(URL, json=payload)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"

    preds = resp.json().get("predictions", resp.json())
    print("API returned:", preds)
    assert isinstance(preds, list)
