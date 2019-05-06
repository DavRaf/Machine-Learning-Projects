# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:16:54 2019

@author: David
"""
# load the packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# load the data
data = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

# split the data into features and targets sets
X = data.drop('quality', axis=1)
y = data.quality

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=123, stratify=y)

# perform feature scaling
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# tune model using cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

# refit on the entire training set
clf.refit

# predict a new set of data
y_pred = clf.predict(X_test)

# evaluate the model pipeline on test data
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')