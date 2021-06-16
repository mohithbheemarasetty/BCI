import tensorflow as tf
import keypress
import numpy as np
import pandas as pd
import time
import csv
from pylsl import StreamInlet, resolve_stream
import statistics

dataset = pd.read_csv("MI_data.csv")
X = dataset.iloc[:, :8].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label = LabelEncoder()
y = label.fit_transform(y)
y =y.reshape(-1,1)
ohe = OneHotEncoder()
y= ohe.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.preprocessing import StandardScaler
sctr = StandardScaler()
X_train = sctr.fit_transform(X_train)
scte = StandardScaler()
X_test = scte.fit_transform(X_test)

classifier = tf.keras.models.Sequential()

classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))

classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))

classifier.add(tf.keras.layers.Dense(units=2, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=50, epochs=25)
y_pred = classifier.predict(X_test)

print("starting...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
duration = 0.5

def predictit():
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
    sctt = StandardScaler()
    lol = sctt.fit_transform(lol)
    prediction = classifier.predict(lol)
    final = []
    for d in range(len(prediction)):
        final.append(np.argmax(prediction[d]))
    ree = statistics.mode(final)
    if ree == 0:
        keypress.press('d')
    elif ree == 1:
        keypress.press('a')
    elif ree == 2:
        keypress.press('w')
    elif ree == 3:
        keypress.press('s')


while 1:
    predictit()

