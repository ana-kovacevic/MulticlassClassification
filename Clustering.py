# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:51:34 2018

@author: Ana Kovacevic
"""

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X_norm=scaler.fit_transform(X)


k_means_model= KMeans(n_clusters=2, init='k-means++', random_state=0, max_iter=100).fit(X_norm)
clustering=k_means_model.predict(X_norm)

X_norm = pd.DataFrame(X_norm)
X_norm.head()
X_norm['clusterLbl'] = clustering

X_norm 
###############################################################################
###############################################################################
evalDict = {'1eval measure':{2:161, 3:95, 4:67}, '1eval measure':{2:0.9, 3:0.09, 4:0.65}}
