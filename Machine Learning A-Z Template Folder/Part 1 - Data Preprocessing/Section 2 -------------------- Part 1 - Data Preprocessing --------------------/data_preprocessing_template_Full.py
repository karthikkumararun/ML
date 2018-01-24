# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data.
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = "NaN", strategy="mean",axis=0)
imputer = imputer.fit(x[: , 1:3]) # left index is lowerbound and right is upperbound.
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,0] = labelEncoder_x.fit_transform(x[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
x = oneHotEncoder.fit_transform(x).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)