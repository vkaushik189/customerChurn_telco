# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 03:09:43 2018

@author: kaushik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import lime
import lime.lime_tabular


dataset = pd.read_csv("C:\\Users\\kaushik\\rcodes\\TelcoCustomerChurn.csv")

print(dataset.isnull().sum())

dataset['Churn'] = dataset['Churn'].map({'Yes': 1, 'No': 0})

        

x = dataset.iloc[:,1:20]
y = dataset.iloc[:,20]

#converting total charges to float
x['TotalCharges'] = pd.to_numeric(x['TotalCharges'], errors='coerce')
x['TotalCharges'].isnull().sum()
#imputing missing values with mean
x['TotalCharges'].fillna((x['TotalCharges'].mean()), inplace=True)


#one hot encoding of categorical features
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
x1 = pd.get_dummies(x, prefix = categorical_features, columns = categorical_features)


X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Initializing Neural Network
classifier = Sequential()


# Adding the input layer and the first hidden layer
#output_dim = number of nodes you want to add to each layer
#init = init of stochastic gradient descent, At the time of initialization, weights should be close to 0 and we will randomly initialize weights using uniform function
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 46))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

##################################################################
"""
Lime
"""

feature_names = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

labels = dataset.iloc[:,20]
le= LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_