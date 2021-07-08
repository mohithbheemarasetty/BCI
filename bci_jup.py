import tensorflow as tf
import keypress
import numpy as np
import pandas as pd
import time
import csv
from pylsl import StreamInlet, resolve_stream
import statistics
import sklearn.metrics as met
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
# Reading the dataset
dataset = pd.read_csv("MI_data_6.csv")
X = dataset.iloc[:, :24].values
y = dataset.iloc[:, -1].values
print(y[400], y[3000], y[5000])
# encoding the labels
label = LabelEncoder()
y = label.fit_transform(y)
y = y.reshape(-1, 1)
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print(y[400], y[3000], y[5000])
# hold _out validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = tf.keras.models.Sequential()
# hidden layers
classifier.add(tf.keras.layers.Dense(units=29, activation='sigmoid'))

classifier.add(tf.keras.layers.Dense(units=29, activation='sigmoid'))

classifier.add(tf.keras.layers.Dense(units=29, activation='sigmoid'))

classifier.add(tf.keras.layers.Dense(units=29, activation='sigmoid'))

classifier.add(tf.keras.layers.Dense(units=29, activation='sigmoid'))

classifier.add(tf.keras.layers.Dense(units=3, activation='softmax'))

classifier.compile(optimizer='RMSprop', loss='hinge', metrics=['accuracy'])
# training
classifier.fit(X_train, y_train, batch_size=50, epochs=200)
# using hold out validation to test the performance
y_pred = classifier.predict(X_test)
final = []
tes = []
con = 0
for d in range(len(y_pred)):
    final.append(np.argmax(y_pred[d]))
for d in range(len(y_test)):
    tes.append(np.argmax(y_test[d]))
for d in range(len(final)):
    if final[d] == tes[d]:
        con += 1
print("test accuracy is", (con*100)/len(final))
# using an different dataset to test the true performance
test = pd.read_csv("MI_data_5.csv")
lol = test.iloc[:, :24].values
y2 = test.iloc[:, -1].values
label2 = LabelEncoder()
y2 = label2.fit_transform(y2)
y2 = y2.reshape(-1, 1)
ohe2 = OneHotEncoder()
y2 = ohe2.fit_transform(y2).toarray()
prediction = classifier.predict(lol)
final = []
tes = []
con = 0
for d in range(len(prediction)):
    final.append(np.argmax(prediction[d]))
for d in range(len(y2)):
    tes.append(np.argmax(y2[d]))
for d in range(len(final)):
    if final[d] == tes[d]:
        con += 1
print("test accuracy is", (con*100)/len(final))
matrix = met.confusion_matrix(prediction.argmax(axis=1), y2.argmax(axis=1))
print(matrix)
# code for character control
time.sleep(5)
print("starting...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
duration = 0.1


def predictit():
    print("relax")
    time.sleep(3)

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
            for sam in range(0, 51):  # increase the range to get more recordings
                sample_extract()
                if sam == 0:
                    continue
                raw_dataset = pd.read_csv("openbci.csv")
                raw = raw_dataset.iloc[:, :].values
                print(len(raw))
                p = []
                q = []
                av = []
                for r in range(len(raw[0])):
                    av.append(np.average(raw[:, r]))
                    p.append(max(raw[:, r]))
                    q.append(min(raw[:, r]))
                print(p)
                writer.writerow([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                                 q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7],
                                 av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7]])

    processing_data()

    test = pd.read_csv("prediction_data.csv")
    lol = test.iloc[:, :].values
    # lol = sctt.fit_transform(lol)
    prediction = classifier.predict(lol)
    final_sim = []
    for d in range(len(prediction)):
        final_sim.append(np.argmax(prediction[d]))
    ree = statistics.mode(final_sim)
    print(final_sim)
    if ree == 0:
        print("Idle, d")
        keypress.pres('d')
    elif ree == 1:
        print("hand, a")
        keypress.pres('a')  # needs to be changed to do nothing
    elif ree == 2:
        print("Leg, w")
        keypress.pres('w')


while 1:
    predictit()
    cont = input("enter 1 to continue or any other number to stop")
    if cont != "1":
        break
