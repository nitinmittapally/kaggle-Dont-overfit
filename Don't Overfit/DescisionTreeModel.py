# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:29:21 2019

@author: nmittapa
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


train_data_path = './data/train.csv'
test_data_path  = './data/test.csv'
preds_file_path = './predictions/'


train_data = pd.read_csv(train_data_path)

X = train_data.iloc[:,2:]
y = train_data.iloc[:,1]

#use DecisionTreeClassifier tree to find important variables

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=10)
model.fit(X,y)

(imp_features) = np.where((np.array(model.feature_importances_) > 0.05))

imp_features[0]


x_train = X.iloc[:, imp_features[0]]

xcorr = train_data.iloc[:, np.append(imp_features[0], [2])].corr()
g = sns.heatmap(xcorr)

model = tree.DecisionTreeClassifier(max_depth=10)
model.fit(x_train,y)

(imp_features) = np.where((np.array(model.feature_importances_) > 0.05) == True)


model = tree.DecisionTreeClassifier(max_depth=5)
model.fit(x_train,y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=1)
model.fit(x_train,y)

test_data = pd.read_csv(test_data_path)

X_test = test_data.iloc[:,1:]
new_index = pd.Index(test_data.iloc[:,0])

x_test = X_test.iloc[:, imp_features[0]]
y_pred = pd.DataFrame(model.predict(x_test))
y_pred  = y_pred.set_index(new_index)

y_pred.to_csv(preds_file_path + 'DescisionTreeModel.csv')


from scipy import interpolate

interpolate.interpn()