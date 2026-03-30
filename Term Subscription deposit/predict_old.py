import pickle
import xgboost as xgb


# #### Load the model

with open('bank-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# customer

customer = {"age": 34,
 "job": "technician",
 "marital": "married",
 "education": "secondary",
 "default": "no",
 "balance": -346,
 "housing": "yes",
 "loan": "no",
 "contact": "unknown",
 "day": 3,
 "month": "jul",
 "duration": 115,
 "campaign": 4,
 "pdays": -1,
 "previous": 0,
 "poutcome": "unknown"}


sample = dv.transform([customer])
feature_names = dv.get_feature_names_out().tolist()

X =xgb.DMatrix(sample, feature_names=feature_names)


y_pred = model.predict(X)

result = (y_pred >= 0.5).astype(int)
print(result[0])
