# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:43:56 2019

@author: nmittapa
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


data_path = "../data/train.csv"
data = pd.read_csv(data_path)

data.isnull()

corrMat = data.drop(['Id', 'SalePrice'], axis=1).corr()

# =============================================================================
# Plot the Correlation HeatMap to analyse the data
# =============================================================================
