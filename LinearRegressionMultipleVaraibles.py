# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:33:47 2019

@author: ankit
"""
#To download SCV
#https://github.com/codebasics/py

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv(r'C:\Users\ankit\OneDrive\Data Sceince\Datasets\LinearRegressionMultp_Variables\HomePrices.txt')
# =============================================================================
# Out[5]: 
#    area  bedrooms  age   price
# 0  2600       3.0   20  550000
# 1  3000       4.0   15  565000
# 2  3200       NaN   18  610000
# 3  3600       3.0   30  610000
# 4  4000       5.0    8  595000
# =============================================================================
# replace NaN with Median value, to make data acceptable
df.bedrooms.median()
#Out[6]: 3.5
import math
median_bedroom = math.floor(df.bedrooms.median())
# we used above lines to keep it integer

# now use fillna function to replace nulls i.e. Nan with median

df.bedrooms.fillna(median_bedroom)
# now bedroom column has been rewritten, copy it over the df dataset

df.bedrooms = df.bedrooms.fillna(median_bedroom)
# =============================================================================
# Out[13]: 
#    area  bedrooms  age   price
# 0  2600       3.0   20  550000
# 1  3000       4.0   15  565000
# 2  3200       3.0   18  610000
# 3  3600       3.0   30  610000
# 4  4000       5.0    8  595000
# =============================================================================
compare with above

#Now create a Regression class object
reg = linear_model.LinearRegression()
#Now call the fit method to train your model, below three values,
# area,room and age define prices, so the three are together, and price is target
reg.fit(df[['area','bedrooms','age']],df.price)
# Model is ready now, its a good idea to look at the coefficient value
# just like y=mx+b in linear regression, in multiple values we use, y=m1x+m2x+m3x+b, 
# m1, m2, m3 are coefficient
reg.coef_
#Out[16]: array([    70.875, -41287.5  ,  -1987.5  ])

#B that is intercept is derived as : reg.intercept_
reg.intercept_
#Out[17]: 538337.4999999997

#################### Now to predict prices

reg.predict([[50000,13,2]])




