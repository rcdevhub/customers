# Customer Spend Prediction
This repo contains a demo of a machine learning model accessed by API, hosted either locally or in the Heroku cloud.

The cloud API is deployed at <a href="https://customerspendapidemo.herokuapp.com/docs/" target="_blank">https://customerspendapidemo.herokuapp.com/docs/</a>

It is possible to pass an example data record to the model and receive the predicted spend of the customer.

Example API calls are contained in `API_call.py` and `API_call_global.py`

The model deployment scipts use FastAPI and are `app1.py` (local) and `app2.py` (global).
The code to fit the XGBoost model is contained in `model1.py` We used the <a href="https://www.kaggle.com/imakash3011/customer-personality-analysis" target="_blank">Customer Personality Analysis</a> dataset. The records represent various information about customers, such as their birth year, income and number of children.
