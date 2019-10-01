# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:05:58 2019

@author: ankit
"""
# =============================================================================
# https://github.com/codebasics/py/blob/master/ML/1_linear_reg/1_linear_regression.ipynb
# https://www.youtube.com/watch?v=8jazNUpO3lQ
# =============================================================================

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\ankit\OneDrive\Data Sceince\Datasets\LinearRegression\HomePrices.txt')
# =============================================================================
# Out[3]: 
#    area   price
# 0  2600  550000
# 1  3000  565000
# 2  3200  610000
# 3  3600  680000
# 4  4000  725000
# =============================================================================
#PLot graph now

%matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')

#Drop the price column
new_df = df.drop('price',axis='columns')
new_df
#Create one set of only price
price = df.price
price

# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)

#Predict price of a home with area = 3300 sqr ft
reg.predict([[3300]])
#Out[8]: array([628715.75342466])

#Y = m * X + b (m is coefficient and b is intercept)
m = reg.coef_
#value of m is array([135.78767123])
x = reg.predict([[3300]])
b = reg.intercept_
#####################NOw use y = mx+b
y = m*x + b
#value of Y is Out[18]: array([85552464.46331395])

--------------------------------------------
reg.predict([[5000]])
# Now Reg.predict is the function we have created and we will use it on a diffrent data set
# i'll use a new file with area info in it, and it will generate its prices

area_df = pd.read_csv(r'C:\Users\ankit\OneDrive\Data Sceince\Datasets\LinearRegression\Area.txt')
# Now using our function reg.predict, we will save all the values in new varaible p
p = reg.predict(area_df)
# now save the value of P as a new column in area_df

area_df['Prices']=p
