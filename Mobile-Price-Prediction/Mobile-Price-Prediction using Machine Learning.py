#!/usr/bin/env python
# coding: utf-8

# # Mobile-Price-Prediction using Machine Learning

# ## Algorithms used:
# 1. Decision Tree 
# 2. Logistic Regression 
# 3. KNN (K-Nearest Neighbor)
# 4. SVM (Support Vector Machine)


#importing Dependencies
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mobile_datatt = pd.read_csv('~/Desktop/Datasets/mobile-train.csv')
mobile_datate = pd.read_csv('~/Desktop/Datasets/mobile-test.csv')


mobile_datatt.head(2)
mobile_datate.head(2)

## We are going to use only mobile-train dataset only
mobile = mobile_datatt
mobile

## Explortory Data Analysis

mobile.info()
mobile.isnull().sum()
mobile.describe()
mobile.shape
mobile.sum()
mobile['clock_speed'].unique()
mobile['battery_power'].unique()
mobile['wifi'].unique()
mobile['price_range'].unique()
mobile['blue'].unique()


## Data Visualisation

plt.figure(figsize=(8,6))
sns.countplot(mobile['price_range'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(mobile['wifi'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(mobile['battery_power'])
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(mobile['ram'])
plt.show()

mobile.plot(x='price_range',y='ram',kind='scatter')
plt.show()
mobile.plot(x='price_range',y='battery_power',kind='scatter')
plt.show()
mobile.plot(x='price_range',y='px_width',kind='scatter')
plt.show()
mobile.plot(x='price_range',y='four_g',kind='scatter')
plt.show()
mobile.plot(x='price_range',y='n_cores',kind='scatter')
plt.show()

## Check for outliers

mobile.plot(kind='box',figsize=(22,12))
plt.show()
mobile
mobile.describe()

X = mobile.drop('price_range',axis=1)
# Removing price_range columns from X
X

Y = mobile['price_range']
# Now Y contains only price_range column
Y

#Splitting Data into Train and Test Data

from sklearn.model_selection import train_test_split

X_train_mobile, X_test_mobile, Y_train_mobile, Y_test_mobile = train_test_split(X,Y,test_size=0.2,random_state=1)
X_train_mobile
Y_train_mobile
X_test_mobile
Y_test_mobile


# KNN - K Nearest Neighbor 

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_std_train = std.fit_transform(X_train_mobile)
std_X_test = std.transform(X_test_mobile)
X_std_train
td_X_test

near.fit(X_std_train,Y_train_mobile)
near_predict = near.predict(std_X_test)
near_predict
Y_test_mobile
knn_accuracy = accuracy_score(Y_test_mobile,near_predict)
knn_accuracy*100


# Decision Tree

dtree = DecisionTreeClassifier()
dtree.fit(X_train_mobile,Y_train_mobile)
dtree_predict = dtree.predict(X_test_mobile)
dtree_predict
Y_test_mobile
dtree_accuracy = accuracy_score(Y_test_mobile,dtree_predict)
dtree_accuracy*100


# Logistic Regression

lr_model = LogisticRegression()
lr_model.fit(X_std_train,Y_train_mobile)
lr_predict = lr_model.predict(std_X_test)
lr_predict
Y_test_mobile
lr_accuracy = accuracy_score(Y_test_mobile,lr_predict)
lr_accuracy*100


#SVM

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x = mobile
x = sc.fit_transform(x)
svc_model = SVC()
svc_model.fit(X_train_mobile,Y_train_mobile)
svc_predict = svc_model.predict(X_test_mobile)
svc_predict
Y_test_mobile
svc_accuracy = accuracy_score(Y_test_mobile,svc_predict)
svc_accuracy*100


plt.bar(x=['knn_accuracy','dtree_accuracy','lr_accuracy','svc_accuracy'],height=[knn_accuracy,dtree_accuracy,lr_accuracy,svc_accuracy])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()

# Trying SVM with different Kernels

# kernel = " linear "
svc = SVC(kernel='linear')
svc.fit(X_train_mobile,Y_train_mobile)
svc_predict_linear = svc.predict(X_test_mobile)
svc_predict_linear
svc_linear_accuracy = accuracy_score(Y_test_mobile,svc_predict_linear)
svc_linear_accuracy*100


#  kernel = " rbf "
svc=SVC(kernel='rbf')
svc.fit(X_train_mobile,Y_train_mobile)
svc_predict_rbf = svc.predict(X_test_mobile)
svc_predict_rbf
svc_rbf_accuracy = accuracy_score(Y_test_mobile,svc_predict_rbf)
svc_rbf_accuracy*100


# kernel = " poly "
svc=SVC(kernel='poly')
svc.fit(X_train_mobile,Y_train_mobile)
svc_predict_poly = svc.predict(X_test_mobile)
svc_predict_poly
svc_poly_accuracy = accuracy_score(Y_test_mobile,svc_predict_poly)
svc_poly_accuracy*100

plt.bar(x=['svc_accuracy','svc_linear_accuracy','svc_rbf_accuracy','svc_poly_accuracy'],height=[svc_accuracy,svc_linear_accuracy,svc_rbf_accuracy,svc_poly_accuracy])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()


# ## Logistic Regression gives us the Highest Accuracy Score - 96%
# ## But SVM with linear Kernel gives the Best Accuracy Score - 96.25%
