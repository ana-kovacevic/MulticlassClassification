# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:18:56 2018

@author: Ana Kovacevic
"""
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

np.set_printoptions(suppress=True)

# read data
data = pd.read_csv("Data-Classification.txt")

# quick audit of the data
data.head()
data.isnull().any().any()
data.dtypes
data.describe()

# destribution of the classes
data['grp'].value_counts()
data['grp'].value_counts().plot(kind='bar')


#Creating the dependent variable class
factor = pd.factorize(data['grp'])
data.grp = factor[0]
definitions = factor[1]
print(data.grp.head())
print(definitions)

# separate target and estimators
data.shape
y = data['grp']
X = data.iloc[:,1:]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
print('X Train:', X_train.shape, '; y Train:' , y_train.shape, ';\nX Test:', X_test.shape,';y test:', y_test.shape )

y_train.value_counts()
y_test.value_counts()

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##################################### learn Logistic Regression Model
modelLR = LogisticRegression()
modelLR.fit(X_train, y_train)

# predict
LR_predTrain =np.array( modelLR.predict(X_train))
LR_predTest =np.array( modelLR.predict(X_test))
# predict probability
LR_probasTrain = modelLR.predict_proba(X_train)
LR_probasTest = modelLR.predict_proba(X_test)

# evaluate model
LR_accTrain = accuracy_score(y_train, LR_predTrain)
LR_accTest = accuracy_score(y_test, LR_predTest)
print('LR Train Accuracy:', LR_accTrain, '\nLR Test Accuracy:', LR_accTest)

reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
LR_predTest = np.vectorize(reversefactor.get)(LR_predTest)
pd.crosstab(y_test, LR_predTest, rownames=['Actual Classes'], colnames=['Predicted Classes'])

####################################### Learn Random forest

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

<<<<<<< HEAD
<<<<<<< HEAD
# Predicting the Test set results
RF_predTrain = classifier.predict(X_train)
RF_predTest = classifier.predict(X_test)
=======
# Exploratory analysis
>>>>>>> origin/master
=======
# Exploratory analysis
>>>>>>> origin/master


RF_accTrain = accuracy_score(y_train, RF_predTrain )
RF_accTest = accuracy_score(y_test, RF_predTest)
print('RF Train Accuracy:', RF_accTrain, '\nLR Test Accuracy:', RF_accTest)


reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
RF_predTest = np.vectorize(reversefactor.get)(RF_predTest)
pd.crosstab(y_test, RF_predTest, rownames=['Actual Classes'], colnames=['Predicted Classes'])
