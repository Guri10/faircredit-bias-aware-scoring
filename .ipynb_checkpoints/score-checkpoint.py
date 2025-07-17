import json
import numpy as np
import pandas as pd
import mlflow.pyfunc
from azureml.contrib.services.aml_request import raw_http_response

def init():
    global model
    # Load the MLflow model
    model = mlflow.pyfunc.load_model("models:/faircredit_baseline/bdd9f70f60f747f1a159a16e8468d5c0")

def run(raw_data):
    # raw_data: JSON string or bytes
    data = json.loads(raw_data)
    df = pd.DataFrame(data["data"])
    preds = model.predict(df)
    return raw_http_response(json.dumps({"predictions": preds.tolist()}), status_code=200)
