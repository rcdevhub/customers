# -*- coding: utf-8 -*-
"""
Model deployment using FastAPI

Created on Tue Nov  2 14:54:54 2021

@author: rcpc4
"""

# Run this script from the command line to deploy API locally

import numpy as np
import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class Customer(BaseModel):
    year_birth:float
    income:float
    kidhome:float
    teenhome:float
    recency:float
    numdealspurchases:float
    numwebpurchases:float
    numcatalogpurchases:float
    numstorepurchases:float
    numwebvisitsmonth:float
    enrolment_lag:float
    
app = FastAPI()

with open ('std_scaler.sav','rb') as f:
    scaler = pickle.load(f)

with open('reg_model.sav','rb') as f:
    model = pickle.load(f)
    
@app.get('/')
def index():
    return {'message':'This is the homepage of the API'}

@app.post('/prediction')
def get_spend(data: Customer):
    
    received = data.dict()

    year_birth = received['year_birth']
    income = received['income']
    kidhome = received['kidhome']
    teenhome = received['teenhome']
    recency = received['recency']
    numdealspurchases = received['numdealspurchases']
    numwebpurchases = received['numwebpurchases']
    numcatalogpurchases = received['numcatalogpurchases']
    numstorepurchases = received['numstorepurchases']
    numwebvisitsmonth = received['numwebvisitsmonth']
    enrolment_lag = received['enrolment_lag']
    
    record = np.array([year_birth,
                       income,
                       kidhome,
                       teenhome,
                       recency,
                       numdealspurchases,
                       numwebpurchases,
                       numcatalogpurchases,
                       numstorepurchases,
                       numwebvisitsmonth,
                       enrolment_lag]).reshape(1,-1)
    
    record_std = scaler.transform(record)
    prediction = float(model.predict(record_std))
    
    return {"prediction": prediction}

# Deploy API locally
uvicorn.run(app,host='127.0.0.1',port=4000,debug=True)
