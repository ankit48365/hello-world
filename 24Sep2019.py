# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:08:49 2019

@author: ankit
"""
#6.2 CLASSIFICATION AND REGRESSION TREES
#6.2.1 How to Build CART Decision Trees Using Python

import pandas as pd
import numpy as np
import statsmodels.tools.tools as stattools
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#Load Adult tr DS
adult_tr = pd.read_csv(r"C:\Users\ankit\OneDrive\Data Sceince\Datasets\DSPR_Data_Sets\Website Data Sets\adult_ch6_training")
adult_tr.shape
adult_tr.head(3)
# =============================================================================
#   Marital status Income  Cap_Gains_Losses
# 0  Never-married  <=50K           0.02174
# 1       Divorced  <=50K           0.00000
# 2        Married  <=50K           0.00000
# 3        Married  <=50K           0.00000
# 4        Married  <=50K           0.00000
# =============================================================================
# =============================================================================
# For simplicity, we save the Income variable as y.
# y = adult_tr[['Income']]
y = adult_tr[['Income']]
# y was created with only one column, [18761 rows x 1 columns]
# We have a categorical variable, Marital status, among our predictors. 
# The CART model implemented in the sklearn package needs categorical variables converted to a dummy variable form. 
# Thus, we will make a series of dummy variables for Marital status using the categorical() command.
# =============================================================================
mar_np = np.array(adult_tr['Marital status'])
#mar_np created - We turn the variable Marital status into an array using array(), 
mar_cat = sm.categorical(mar_np, drop=True)
mar_cat_dict = stattools.categorical(mar_np, dictnames = True)
#Now, we need to add the newly made dummy variables back into the X variables.
mar_cat_pd = pd.DataFrame(mar_cat)
#we converted the mar_cat matrix into a data frame using the DataFrame() command
X = pd.concat((adult_tr[['Cap_Gains_Losses']], mar_cat_pd), axis = 1)
# =============================================================================
# We then use the concat() command to attach the predictor variable Cap_Gains_Losses to 
# the data frame of dummy variables that represent marital status. We save the result as X.
# =============================================================================
# =============================================================================
# Data is like this
# 18749          0.000000  0.0  1.0  0.0  0.0  0.0
# 18750          0.010550  0.0  0.0  1.0  0.0  0.0
# 18751          1.000000  0.0  1.0  0.0  0.0  0.0
# 18752          0.362489  0.0  1.0  0.0  0.0  0.0
# 
# =============================================================================
# Column names on top of X ---> X Names 
X_names = ["Cap_Gains_Losses", "Divorced", "Married", "Never-married","Separated", "Widowed"]
#and Y names aswell
y_names = ["<=50K", ">50K"]

# =============================================================================
# Now, we are ready to run the CART algorithm!
# =============================================================================
cart01 = DecisionTreeClassifier(criterion = "gini", max_leaf_nodes=5).fit(X,y)
# ============================================================================= , fit is using X and y we created before
# The DecisionTreeClassifier() command sets up the various parameters for the decision tree. 
# For example, the criterion = “gini” input specifies that we are using a CART model which utilizes the Gini criterion, 
# and the max_leaf_nodes input trims the CART tree to have at most the specified number of leaf nodes. For this example, 
# we have limited our tree to five leaf nodes. The fit() command tells Python to fit the decision tree that was 
# previously specified to the data. The predictor variables are given first, followed by the target variable. 
# Thus, the two inputs to fit() are the X and y objects we created. We save the decision tree as cart01.
# =============================================================================
# =============================================================================
# Output of Cart01 is 
# Out[53]: 
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#                        max_features=None, max_leaf_nodes=5,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, presort=False,
#                        random_state=None, splitter='best')
# =============================================================================
#Finally, to obtain the tree structure, we use the export_graphviz() command.
export_graphviz(cart01, out_file = "C:/Users/ankit/OneDrive/Data Sceince/Datasets/cart01.dot", feature_names=X_names, class_names=y_names)
# after this, the book only says this - 
# =============================================================================
# The out_file input will save the tree structure to the specified location and name the file cart01.dot. 
# Run the contents of the file through the graphviz package to display the CART model. 
# Specifying feature_names = X_names and class_names = y_names add the predictor variable names and the target variable values to the cart01.dot file, 
# greatly increasing its readability.
# To obtain the classifications of the Income variable for every variable in the training data set, use the predict() command.
# =============================================================================
predIncomeCART = cart01.predict(X)
# =============================================================================
# Using the predict() command on cart01 says that we want to use our CART model to make the classifications. 
# Including the predictor variables X as input specifies that we want predictions for those records in particular. 
# The result is the classification, according to our CART model, for every record in the training data set. We save the predictions as predIncomeCART.
# =============================================================================

# below i am doing from here.....https://www.youtube.com/watch?v=PHxYNGo8NcI
from sklearn import tree
model = tree.DecisionTreeClassifier()
#see line 64, cart 01 =..... it has fit x,y
model.fit(X,y)
# compare this to, out[53] above=============================================================================
# Out[61]: 
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#                        max_features=None, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, presort=False,
#                        random_state=None, splitter='best')
# =============================================================================
model.score(X,y)
#Out[62]: 0.8309258568306593
#output of one is perfact, the more challenging the data is, more away it gets, less then one it gets
#model.predict([[0]])

c50_01 = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=5).fit(X,y)
export_graphviz(c50_01, out_file = "C:/Users/ankit/OneDrive/Data Sceince/Datasets/c50_01.dot", feature_names=X_names, class_names=y_names)
c50_01.predict(X)














































