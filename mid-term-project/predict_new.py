import pickle
import xgboost as xgb
import uvicorn

from fastapi import FastAPI
# from flask import Flask, request, jsonify 
from typing import Dict, Any

app = FastAPI(title = "subscribe")

with open('bank-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict_single(customer):
    sample = dv.transform([customer])
    feature_names = dv.get_feature_names_out().tolist()

    X =xgb.DMatrix(sample, feature_names=feature_names)

    y_pred = model.predict(X)

    result = (y_pred >= 0.5).astype(int)
    return int(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    score = predict_single(customer)
    if score == 0:
        result = 'Not subscribed'
    else:
        result = 'Subscribed'
    return {
        f"This user is {result}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)