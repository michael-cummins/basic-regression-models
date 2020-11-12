# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#selcting index for categorical data '3'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the multipplelinear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
#rehape changes from vertical to horizonal
#changes from nxm to mxn
#axis = "1"
print(np.concatenate((y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1)),1))

#making a single prediction with r&d = 160000, admin spend = 130000,marketing spend = 300000 and state = california
print(regressor.predict([[1, 0, 0, 160000,130000,300000]])) 

#getting final equation      
print(regressor.coef_)
print(regressor.intercept_)
#manually predicting using coefficents and intercept
pred = [1, 0 , 0, 160000, 130000, 300000]
sum = regressor.intercept_
for  x in range(0,6):
    sum = sum+regressor.coef_[x]*pred[x]
print(sum)