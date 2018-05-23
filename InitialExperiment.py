# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:18:56 2018

@author: Ana Kovacevic
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

# read data
data = pd.read_csv("Data-Classification.txt")

# quick audit of the data
data.head()
data.describe()

# destribution of the classes
data['grp'].value_counts()
data['grp'].value_counts().plot(kind='bar')

# separate target and estimators
data.shape
y = data['grp']
X = data.iloc[:,1:]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
print('X Train:', X_train.shape, '; y Train:' , y_train.shape, ';\nX Test:', X_test.shape,';y test:', y_test.shape )

y_train.value_counts()
y_test.value_counts()


# learn the model
modelLR = LogisticRegression()
modelLR.fit(X_train, y_train)

# predict
predTrain =np.array( modelLR.predict(X_train))
predTest =np.array( modelLR.predict(X_test))
# predict probability
probasTrain = modelLR.predict_proba(X_train)
probasTest = modelLR.predict_proba(X_test)

# evaluate model
accTrain = accuracy_score(y_train, predTrain)
accTest = accuracy_score(y_test, predTest)

print('Train Accuracy:', accTrain, '\nTest Accuracy:', accTest)




