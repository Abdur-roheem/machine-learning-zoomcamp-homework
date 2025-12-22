import requests

url = 'http://localhost:9696/predict'

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
 "poutcome": "unknown"
 }

response = requests.post(url, json=customer)
predictions = response.json()

print(predictions)
# if predictions['churn']:
#     print('customer is likely to churn, send promo')
# else:
#     print('customer is not likely to churn')