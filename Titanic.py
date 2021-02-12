#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 10:49:48 2020

@author: elizczimm
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/elizczimm/Documents/Data_Science/Practice/Kaggle/Titanic/train.csv")

# EDA
df.isnull().sum()
df = df.drop(columns=['PassengerId','Ticket','Cabin','Name'],axis=1)
dummies_s = pd.get_dummies(df['Sex'])
dummies_e = pd.get_dummies(df['Embarked'])
df = pd.concat([df, dummies_s], axis=1)
df = pd.concat([df, dummies_e], axis=1)
df = df.drop(columns = ['Sex','Embarked'])

age = df['Age']
plt.hist(age[~np.isnan(age)])

# Address missing age 
df['Age'] = df['Age'].fillna(df['Age'].mean())


# Create pipeline to try different imputed age statistics
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler


# split into input and output elements
X = df.drop(columns='Survived',axis=1)
y = df['Survived']
X = X.values
y = y.values
# evaluate each strategy on the dataset
results = list()
strategies = ['mean', 'median', 'most_frequent', 'constant']
for s in strategies:
	# create the modeling pipeline
	pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())])
	# evaluate the model
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# store results
	results.append(scores)
	print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=strategies, showmeans=True)
pyplot.show()
# most frequent is the best candidate, showing the least variance

# split train, test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

# define pipeline
pipe = Pipeline(steps=[('i', SimpleImputer(strategy='most_frequent')),('scaler', StandardScaler()),('rf', RandomForestClassifier())])

# set parameters for grid search
params = {
        'rf__n_estimators': [10, 25, 50, 100, 250, 500],
        'rf__max_depth': [10, 25, 50],
        'rf__min_samples_split': [2,3],
        'rf__min_samples_leaf': [3,5, 10],
        'rf__criterion': ['gini','entropy']
        }

# create grid search for parameters using 'most frequent' imputation for age
RF_gs = GridSearchCV(pipe, param_grid=params, scoring='roc_auc', cv=3)
RF_gs.fit(X_train, y_train)
best_parameters = RF_gs.best_params_

RF_gs.score(X_train, y_train)

RF_gs.score(X_test, y_test)


test = pd.read_csv("/Users/elizczimm/Documents/Data_Science/Practice/Kaggle/Titanic/test.csv")
pass_id = test['PassengerId']
pass_id = pass_id.values
test = test.drop(columns=['PassengerId','Ticket','Cabin','Name'],axis=1)
dummies_s = pd.get_dummies(test['Sex'])
dummies_e = pd.get_dummies(test['Embarked'])
test = pd.concat([test, dummies_s], axis=1)
test = pd.concat([test, dummies_e], axis=1)
test = test.drop(columns = ['Sex','Embarked'])

X_pred = test.values

pred = RF_gs.predict(X_pred)
final = pd.DataFrame(pass_id)
final.columns=['PassengerId']
final['Survived'] = pred
final.to_csv("/Users/elizczimm/Documents/Data_Science/Practice/Kaggle/Titanic/submission.csv",index=False)
