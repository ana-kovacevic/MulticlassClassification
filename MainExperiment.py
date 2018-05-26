# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:10:46 2018

@author: Ana Kovacevic
"""
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Data-Classification.txt")
factor = pd.factorize(data['grp'])
data.grp = factor[0]
definitions = factor[1]
print(data.grp.head())
print(definitions)

y = data['grp']
X = data.iloc[:,1:]
#X1, y1 = samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42)
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
print('X Train:', X_train.shape, '; y Train:' , y_train.shape, ';\nX Test:', X_test.shape,';y test:', y_test.shape )

#################################################################################
######                           SELECT BEST MODEL     ##########################
#################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    #MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=9, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
cv_df.groupby('model_name').accuracy.mean()


conf_mat = confusion_matrix(y_test, LR_predTest)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=data.grp.values, yticklabels=data.grp.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(metrics.classification_report(y_test, LR_predTest, target_names=['A', 'B', 'C']))
