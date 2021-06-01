import pynput
import tensorflow as tf
import keypress
import numpy as np
import pandas as pd


dataset = pd.read_csv("dataset04.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label = LabelEncoder()
y = label.fit_transform(y)
y =y.reshape(-1,1)
ohe = OneHotEncoder()
y= ohe.fit_transform(y).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from sklearn.preprocessing import StandardScaler
sctr = StandardScaler()
X_train = sctr.fit_transform(X_train)
scte = StandardScaler()
X_test = sctr.transform(X_test)


classifier = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Adding the second hidden layer
classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))


# Adding the output layer
classifier.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# Part 3 - Training the ANN

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 50)
y_pred = classifier.predict(X_test)


'''
y_lmao = y_pred
for x in range(len(y_pred[:,1])):
   ree = np.argmax(y_lmao[x])
   y_lmao[x] = [0,0,0,0]
   y_lmao[x,ree] = 1


print(y_lmao)
print("*****************************")
print(y_test)
correct = 0
total = len(y_lmao[:,1])
for x in range(len(y_pred[:,1])):
  hehe = 0
  for y in range(0,4):
    if y_lmao[x,y] == y_test[x,y]:
      hehe = hehe + 1
  if hehe == 4:
    correct = correct+ 1

#print(correct)
print((correct/total)*100)
'''

test = pd.read_csv("test04.csv")
lol = test.iloc[:,:-1].values
actual = test.iloc[:, -1].values
labelt = LabelEncoder()
actual = labelt.fit_transform(actual)
actual = actual.reshape(-1, 1)
ohet = OneHotEncoder()
actual = ohet.fit_transform(actual).toarray()
#loly = loly[:,1:]
lol = sctr.transform(lol)
prediction = classifier.predict(lol)


print(prediction)


import time
for x in range(len(prediction[:,1])):
    ree = np.argmax(prediction[x])
    prediction[x] = [0,0,0,0]
    prediction[x,ree] = 1
    time.sleep(1.5)
    if ree == 0:
        keypress.press('d')
    elif ree == 1:
        keypress.press('a')
    elif ree == 2:
        keypress.press('w')
    elif ree == 3:
        keypress.press('s')
    time.sleep(1.5)

