# -*- coding: utf-8 -*-
""" Importing the libraries """
import numpy as np
import pandas as pd
import tensorflow as tf
import socket
"""Reading data into dependant and independant variables"""
dataset = pd.read_csv("Imaginary.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


"""For encoding the data set (if it has strings in the data)"""

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label = LabelEncoder()
y = label.fit_transform(y)
y =y.reshape(-1,1)
ohe = OneHotEncoder()
y= ohe.fit_transform(y).toarray()
#y = y[:,1:]
"""Splitting data into training and test set"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

"""scalling the data"""

from sklearn.preprocessing import StandardScaler
sctr = StandardScaler()
X_train = sctr.fit_transform(X_train)
scte = StandardScaler()
X_test = scte.fit_transform(X_test)
"""
#training and prediction
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)
"""
"""
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
"""
"""
# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train[:,1])
"""
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 3)
classifier.fit(X_train, y_train)
"""
"""
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
"""
"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
"""
classifier = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Adding the second hidden layer
classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))


# Adding the output layer
classifier.add(tf.keras.layers.Dense(units=4, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 100)
y_pred = classifier.predict(X_test)
correct = 0
wrong = 0
for z in range(len(y_test)):
    a = (y_test[z] == y_pred[z])
    if (np.all(a)):
        correct = correct + 1
    else:
        
        wrong = wrong +1
print(correct+wrong)

test = pd.read_csv("test.csv")
lol = test.iloc[:,:-1].values
loly = test.iloc[:,-1].values
labelt = LabelEncoder()
loly = labelt.fit_transform(loly)
loly =loly.reshape(-1,1)
ohet = OneHotEncoder()
loly= ohet.fit_transform(loly).toarray()
#loly = loly[:,1:]
lol = sctr.transform(lol)
hehe =  classifier.predict(lol)
correctt = 0
wrongt = 0
for n in range(len(loly)):
    x = (loly[n] == hehe[n])
    if (np.all(x)):
        correctt = correctt + 1
    else:
        
        wrongt = wrongt +1

test_accuracy  = (correctt/len(loly))*100
training_accuracy  = (correct/len(y_test))*100

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
HOST = '192.168.0.30'  # The server's hostname or IP address
PORT = 2345       # The port used by the server

s.connect((HOST, PORT))
x = ''
while (x != 'stop'):
    x = input('> ')
    s.send(bytes(x,'utf-8'))
    x = ''
        
