# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:16:05 2019

@author: nmittapa
"""
import pandas as pd

train_file_path = './train.csv'
data = pd.read_csv(train_file_path)

y = data.target
x = data.iloc[:, 2:]

#fea

from sklearn.ensemble import RandomForestClassifier

model =  RandomForestClassifier(n_estimators=50, max_leaf_nodes=2, random_state=0)
model.fit(x, y)

#get the test data
test_file_path='./test.csv'

test_data = pd.read_csv(test_file_path)

x_test = test_data.iloc[:,1:]
y_pred = model.predict(x_test)





