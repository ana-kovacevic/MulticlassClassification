# -*- coding: utf-8 -*-
"""
Created on Sat May 26 23:31:40 2018

@author: Ana Kovacevic
"""
#import pyplot as plt

##### Example on pipe
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline


data = pd.read_csv("Data-Classification.txt")

# quick audit of the data
data.head()
data.isnull().any().any()
data.dtypes
data.describe()

# destribution of the classes
data['grp'].value_counts()
data['grp'].value_counts().plot(kind='bar')

# separate target and estimators
data.shape
y = data['grp']
X = data.iloc[:,1:]
#X1, y1 = samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42)
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
print('X Train:', X_train.shape, '; y Train:' , y_train.shape, ';\nX Test:', X_test.shape,';y test:', y_test.shape )

############################################################################
anova_filter = SelectKBest(f_regression, k = 5)
clf = svm.SVC(kernel = 'linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X,y)
