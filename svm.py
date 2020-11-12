# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
#changing x because we dont want first column
#only interested with index 1
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#feature scaling
#reshape y from a 1d vector into a 2d array.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1)),1)
y  = y.reshape(len(y),1)
from sklearn.preprocessing import StandardScaler
sc_y  = StandardScaler()
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
#training svr module
from  sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)
#predicting salary
y_pred = regressor.predict(sc_x.transform([[6.5]]))
print(sc_y.inverse_transform(y_pred))

#x = sc_x.inverse_transform(x)
#y = sc_y.inverse_transform(y)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), .1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
#plots regression line
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Truth or bluff Support vector regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
