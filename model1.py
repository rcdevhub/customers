# -*- coding: utf-8 -*-
"""
Model to predict customer spending

Created on Wed Oct 27 21:50:00 2021

@author: rcpc4
"""

'''----------------------Setup----------------------------------''' 

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
from datetime import datetime
from datetime import timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import (train_test_split,KFold,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import mean_squared_error

# Set working directory
os.chdir('C://Code/Kaggle/customers')

'''----------------------Data prep functions------------------------''' 

def get_val_counts(data,variables):
    '''Return dict containing value counts.'''
    data_value_counts = {}
    for i in variables:
        data_value_counts[i] = data[i].value_counts()
        
    return data_value_counts

def add_one_hot(data,variables):
    ''''Return dataframe with variables converted to one-hot.'''
    for i in variables:
        enc = OneHotEncoder(categories='auto',drop=None)
        enc.fit(np.array(data[i]).reshape(-1,1))
        onehot = enc.transform(np.array(data[i]).reshape(-1,1)).toarray()
        onehot = pd.DataFrame(onehot,columns=np.squeeze(np.array(enc.categories_).astype(str)))
        data = pd.concat([data.drop(i,axis=1),
                          onehot],
                          axis=1)
        
    return data
    
'''----------------------Import data----------------------------------''' 

data_raw = pd.read_csv('data/marketing_campaign.csv',sep=None)

'''----------------------Prepare data----------------------------------''' 

# Find missing values
na_count = pd.Series([sum(data_raw[i].isna()) for i in data_raw.columns],
                    name='na_count',
                    index=data_raw.columns)
# Investigate cat variables
cat_vars = ['Year_Birth','Education','Marital_Status']
data_val_counts = get_val_counts(data_raw,cat_vars)
# Drop missing values (in income field)
data2 = data_raw.dropna().reset_index()
# Drop outlier years
data2 = data2.loc[data2['Year_Birth'] >= 1940].reset_index(drop=True)
# Convert unusual marital statuses to single
data2.loc[((data2['Marital_Status'] != 'Married') &
          (data2['Marital_Status'] != 'Together') &
          (data2['Marital_Status'] != 'Single') &
          (data2['Marital_Status'] != 'Divorced') &
          (data2['Marital_Status'] != 'Widow')),
          'Marital_Status'] = 'Single'
# Convert variables to one-hot
onehot_vars = ['Education','Marital_Status']
data2 = add_one_hot(data2,onehot_vars)
# Convert date of enrolment to days after earliest enrolment
data2['Dt_Customer'] = pd.to_datetime(data2['Dt_Customer'])
min_enrolment_date = min(data2['Dt_Customer'])
data2['Enrolment_lag'] = (data2['Dt_Customer'] - min_enrolment_date).dt.days
data2 = data2.drop('Dt_Customer',axis=1)
# Remove suspicious incomes
income_cut_off = 500000
data2 = data2[data2['Income'] <= income_cut_off]
# Add total amount purchased (target)
data2['MntTotal'] = (data2['MntWines']
                     + data2['MntFruits']
                     + data2['MntMeatProducts']
                     + data2['MntFishProducts']
                     + data2['MntSweetProducts']
                     + data2['MntGoldProds'])

# Split into train and test sets
train,test = train_test_split(data2,test_size=0.2,
                              random_state=456,shuffle=True)

# Reduce to necessary variables
# Removed marital status and education based on feature importance
model_vars = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
       'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
       'NumStorePurchases', 'NumWebVisitsMonth',
       #'2n Cycle', 'Basic','Graduation', 'Master', 'PhD',
       #'Divorced', 'Married', 'Single', 'Together', 'Widow',
       'Enrolment_lag']
X_train = train[model_vars].values
Y_train = train['MntTotal'].values
X_test = test[model_vars].values
Y_test = test['MntTotal'].values

# Normalise
scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

'''----------------------Explore data----------------------------'''

# Describe dataset
desc_train = train.describe()

# Generate histograms of numerical columns
num_cols = ['Income','Kidhome','Teenhome','Recency',
            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
           'MntSweetProducts', 'MntGoldProds', 'MntTotal', 'NumDealsPurchases',
           'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
           'NumWebVisitsMonth','Enrolment_lag']

for i in num_cols:
    plt.figure()
    train[i].plot.hist(bins=50)
    plt.title(i)

# Plot feature importance
# F-score
fs_f = SelectKBest(score_func=f_regression,k='all')
fs_f.fit(X_train,Y_train)
feat_score_f = fs_f.scores_
feat_rank_f = pd.DataFrame(feat_score_f,
                           columns=['score'],
                           index=model_vars).sort_values(
    by='score',ascending=False)
feat_rank_f.plot.bar()
plt.title('Feature importance by F-score')

# Mututal information
fs_m = SelectKBest(score_func=mutual_info_regression,k='all')
fs_m.fit(X_train,Y_train)
feat_score_m = fs_m.scores_
feat_rank_m = pd.DataFrame(feat_score_m,
                           columns=['score'],
                           index=model_vars).sort_values(
    by='score',ascending=False)
feat_rank_m.plot.bar()
plt.title('Feature importance by mutual information')

'''----------------------Fit model----------------------------'''

# Basic XGB model
reg1 = xgb.XGBRegressor(objective='reg:squarederror')

reg1.fit(X_train_norm,Y_train)

preds_train = reg1.predict(X_train_norm)

# XGB with cross validation for number of trees
# Other parameters untuned
params = reg1.get_xgb_params()
matrix = xgb.DMatrix(X_train_norm,label=Y_train)
crossval = xgb.cv(params=params,
                  dtrain=matrix,
                  num_boost_round=500,
                  early_stopping_rounds=50,
                  nfold=5,
                  metrics='rmse',
                  seed=8956)

best_num_trees = crossval.shape[0]

# Plot cross-validation results
crossval[['train-rmse-mean','test-rmse-mean']].plot()
crossval[['train-rmse-std','test-rmse-std']].plot()

# XGB with optimal number of trees
reg2 = xgb.XGBRegressor(objective='reg:squarederror',
                        n_estimators = best_num_trees)
reg2.fit(X_train_norm,Y_train)
preds_train = reg2.predict(X_train_norm)

'''----------------------Model diagnostics-----------------------'''

# Calculate MSE
mse_train = mean_squared_error(Y_train, preds_train)
rmse_train = np.sqrt(mse_train)

# Plot feature importance
reg1_feat_imp = pd.Series(reg1.feature_importances_,
                          index=model_vars)
reg1_feat_imp.sort_values(ascending=False).plot.bar()

# Calc residuals
resid_train = Y_train - preds_train

# Plot residuals
plt.figure()
sns.histplot(resid_train)
plt.axvline(x=0,linewidth=1,color='black')

# Plot predicted vs actual density
fig = plt.figure()
ax1 = fig.add_subplot()
plt.hist2d(x=Y_train,y=preds_train,bins=500,cmap=plt.cm.jet)
plt.plot(Y_train,Y_train,linestyle='-',linewidth=0.03,color='white')
plt.colorbar()
ax1.set_xlim([0,300])
ax1.set_ylim([0,300])
plt.title('Predicted vs Actual density')
plt.xlabel('Actual')
plt.ylabel('Predicted')