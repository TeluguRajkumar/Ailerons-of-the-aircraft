#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:07:08 2022

@author: rajkumar
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/rajkumar/Desktop/internshala feb 20/new as/aileronstrain.csv")
df.head()
df.describe()
df.info()
X = df[:, 1:(end - 1)]
y = df.goal
(train_X, train_y), (test_X, test_y) = IAI.split_data(:regression, X, y, seed=1)

#Model fitting
#We will use a GridSearch to fit an OptimalFeatureSelectionRegressor:
grid = IAI.GridSearch(
    IAI.OptimalFeatureSelectionRegressor(
        random_seed=1,
    ),
    sparsity=1:10,
)
IAI.fit!(grid, train_X, train_y)

using Plots
plot(grid, type=:validation)
IAI.variable_importance(IAI.get_learner(grid))
plot(grid, type=:importance)
IAI.predict(grid, test_X)
IAI.score(grid, train_X, train_y, criterion=:mse) 

##Optimal Feature Selection Visualization
#Validation Score Against Sparsity
using Plots
plot(grid, type=:validation)

#Variable Importance Heatmap
plot(grid, type=:importance)

#Combining Multiple Plots
plot(grid, type=[:importance :validation])
plot(grid, type=[:validation; :importance])

#Using Optimal Feature Selection with Missing Data
df = pd.read_csv("/Users/rajkumar/Desktop/internshala feb 20/new as/aileronstrain.csv")
X = df[:, 1:(end - 1)]
y = df[:, end]
X

#Let's first split the data into training and testing:
(X_train, y_train), (X_test, y_test) = IAI.split_data(:classification, X, y, seed=4)

#Approach 1: Optimal Imputation
imputer = IAI.ImputationLearner(:opt_knn, random_seed=1)
X_train_imputed = IAI.fit_transform!(imputer, X_train)
X_test_imputed = IAI.transform(imputer, X_test)

grid = IAI.GridSearch(
    IAI.OptimalFeatureSelectionClassifier(random_seed=1),
    sparsity=1:size(X, 2),
)
IAI.fit!(grid, X_train_imputed, y_train)
IAI.score(grid, X_test_imputed, y_test, criterion=:auc)
#Warning: For full sparsity, use ridge regression for faster performance.
#0.8872767857142855

#Approach 2: Engineer features to encode missingness pattern
imputer = IAI.ImputationLearner(:zero, normalize_X=false)
X_train_finite = IAI.fit_and_expand!(imputer, X_train, type=:finite)
X_test_finite = IAI.transform_and_expand(imputer, X_test, type=:finite)
names(X_train_finite)

grid = IAI.GridSearch(
    IAI.OptimalFeatureSelectionClassifier(random_seed=1),
    sparsity=1:size(X_train_finite, 2),
)
IAI.fit!(grid, X_train_finite, y_train)
IAI.get_learner(grid)

#We can evaluate the model on the transformed and expanded test set:

IAI.score(grid, X_test_finite, y_test, criterion=:auc)
#0.8775111607142859

imputer = IAI.ImputationLearner(:zero, normalize_X=false)
X_train_affine = IAI.fit_and_expand!(imputer, X_train, type=:affine)
X_test_affine = IAI.transform_and_expand(imputer, X_test, type=:affine)
names(X_train_affine)

grid = IAI.GridSearch(
    IAI.OptimalFeatureSelectionClassifier(random_seed=1),
    sparsity=1:size(X_train_affine, 2),
)
IAI.fit!(grid, X_train_affine, y_train)
IAI.get_learner(grid)

IAI.score(grid, X_test_affine, y_test, criterion=:auc)
#0.8962053571428573


