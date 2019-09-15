# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:22:28 2019

@author: nmittapa
"""

import pandas as pd

train_file_path = './data/train.csv'
sub_file_path = './data/sample_submission.csv'
data = pd.read_csv(train_file_path)
sub = pd.read_csv(sub_file_path)

sub['target'] = data_test[str(33)]

x_train = data.iloc[:, 2:]
y_train = data.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train))


'''
from sklearn.decomposition import PCA
pca_analysis = PCA(n_components= None)
x_pca = pca_analysis.fit_transform(x_train)
explained_variance = pca_analysis.explained_variance_ratio_

pca_model = PCA(n_components=130)
x_pca = pca_model.fit_transform(x_train)
'''


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)
model.fit(x_train,y_train)

data_test = pd.read_csv('./data/test.csv')


x_test = data_test.iloc[:, 1:]

x_test = pd.DataFrame(scaler.transform(x_test))
predictions = pd.DataFrame( model.predict(x_test) )

p = predictions

p = p.set_index(pd.Index(data_test.id.values))

p.iloc[:,1] = p.iloc[:,1].rename_axis('target')

p.to_csv('./predictions/Log1.csv')

