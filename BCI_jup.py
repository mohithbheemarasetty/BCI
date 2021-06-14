import tensorflow as tf
import keypress
import numpy as np
import pandas as pd
import time
import csv
from pylsl import StreamInlet, resolve_stream


dataset = pd.read_csv("MI_data0.csv")
X = dataset.iloc[:, :].values
y = dataset.iloc[:, :].values


# In[4]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label = LabelEncoder()
y = label.fit_transform(y)
y =y.reshape(-1,1)
ohe = OneHotEncoder()
y= ohe.fit_transform(y).toarray()


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[23]:


from sklearn.preprocessing import StandardScaler
sctr = StandardScaler()
X_train = sctr.fit_transform(X_train)
scte = StandardScaler()
X_test = sctr.transform(X_test)


# In[24]:


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


'''y_lmao = y_pred
for x in range(len(y_pred[:,1])):
   ree = np.argmax(y_lmao[x])
   y_lmao[x] = [0,0,0,0]
   y_lmao[x,ree] = 1
'''

'''
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
print("starting...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
duration = 0.5


def predict():
    def sample_extract():
        start = time.time()
        with open('openbci.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            while time.time() <= start + duration:
                samples, timestamp = inlet.pull_chunk()
                if samples:
                    writer.writerows(samples)

    def processing_data():
        with open('prediction_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for sam in range(0, 21):  # increase the range to get more recordings
                sample_extract()
                if sam == 0:
                    continue
                raw_dataset = pd.read_csv("openbci.csv")
                raw = raw_dataset.iloc[:, :].values
                print(len(raw))
                p = []
                for r in range(len(raw[0])):
                    p.append(np.average(raw[:, r]))
                print(p)
                writer.writerow([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]])

    processing_data()

    test = pd.read_csv("prediction_data.csv")
    lol = test.iloc[:, :].values
    lol = sctr.transform(lol)
    prediction = classifier.predict(lol)
    ree = np.argmax(prediction[0])
    prediction[0] = [0, 0, 0, 0]
    prediction[0, ree] = 1
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


while 1:
    predict()

