# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
#changing x because we dont want first column
#only interested with index 1
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#dont use train test split to maximise return from small data set
#train linear regression model
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(X, y)

#train polynomial class
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
#will only go up to polynomial degree 2
X_poly = poly_reg.fit_transform(X)
#x_poly is a new matrix of features which you can now just plug into a simple linear regresssion model
Lin_reg_2 = LinearRegression()
Lin_reg_2.fit(X_poly, y) 

#visualising linear regression results
plt.scatter(X, y, color = 'red')
#plots regression line
plt.plot(X, Lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff Linear regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#higher resolution
X_grid = np.arange(min(X), max(X), .05)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
#plots regression line
plt.plot(X, Lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or bluff Linear regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#predictiing using models
#double square bracket for 2 dimensions
print(Lin_reg.predict([[6.5]]))
#add in all features
list1 = [0,0,0,0,0,0,0]
for x in range(1,7): 
    list1[x] = 6.5**x
list1.reshape(-1,1)
print(Lin_reg_2.predict([list1[1:7]]))