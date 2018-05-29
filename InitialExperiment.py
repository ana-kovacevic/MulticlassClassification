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
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


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

data['grpOrig'] = data['grp']
#Creating the dependent variable class

factor = pd.factorize(data['grp'])
data.grp = factor[0]
definitions = factor[1]
print(data.grp.head())
print(definitions)
# separate target and estimators
data.shape
y = data['grp']
X = data.iloc[:,1:31]

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


####################################### Learn Random forest

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
RF_predTrain = classifier.predict(X_train)
RF_predTest = classifier.predict(X_test)

RF_accTrain = accuracy_score(y_train, RF_predTrain )
RF_accTest = accuracy_score(y_test, RF_predTest)
print('RF Train Accuracy:', RF_accTrain, '\nRF Test Accuracy:', RF_accTest)

######## confusion matrix
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
RF_predTest = np.vectorize(reversefactor.get)(RF_predTest)
pd.crosstab(y_test, RF_predTest, rownames=['Actual Classes'], colnames=['Predicted Classes'])

LR_predTest = np.vectorize(reversefactor.get)(LR_predTest)
pd.crosstab(y_test, LR_predTest, rownames=['Actual Classes'], colnames=['Predicted Classes'])


#################### one hot encoded

# define example
#data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
from numpy import argmax
def hot_encoded(multiClassVec):
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    
    values = np.array(multiClassVec)
    #print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    return(onehot_encoded)

###### ROC curve
# Compute ROC curve and ROC area for each class
encodedLR_predTest = hot_encoded(LR_predTest)
encodedLR_predTest = hot_encoded(LR_predTest)


encodedYtrain = hot_encoded(y_train)
encodedYtest = hot_encoded(y_test)


modelLR = LogisticRegression()
modelLR.fit(X_train, encodedYtrain[:, 0])

# predict
LR_predTrain =np.array( modelLR.predict(X_train))
LR_predTest =np.array( modelLR.predict(X_test))


# evaluate model
LR_accTrain = accuracy_score(encodedYtrain[:,0], LR_predTrain)
LR_accTest = accuracy_score(encodedYtest[:,0], LR_predTest)
print('LR Train Accuracy:', LR_accTrain, '\nLR Test Accuracy:', LR_accTest)

fpr, tpr, _ = roc_curve(encodedYtest[:, 0], encodedLR_predTest[:, 0])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(encodedYtest[:, i], encodedLR_predTest[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(encodedYtest.ravel(), encodedLR_predTest.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 3

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
#plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)

#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()