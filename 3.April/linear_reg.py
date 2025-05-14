# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:29:32 2025

@author: Deepti Jaiswar
"""

import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=0)

#reshape

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

import matplotlib.pyplot as plt

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Expectation (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#i want to predit the future y = mx + c
m_slope = regressor.coef_ #slope and coffiecient
print(m_slope)

c_intercept = regressor.intercept_  # contant and intercept
print(c_intercept)

pred_12yr_emp_salary = m_slope* 12 + c_intercept
print(pred_12yr_emp_salary)

pred_20yr_emp_salary = m_slope* 20 + c_intercept
print(pred_20yr_emp_salary)