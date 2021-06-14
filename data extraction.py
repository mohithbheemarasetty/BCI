import time
from pylsl import StreamInlet, resolve_stream
import csv
import pandas as pd
import numpy as np
print("starting...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
duration = 0.5


def testLSLSamplingRate():
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
        for sam in range(0, 20):
            testLSLSamplingRate()
            dataset = pd.read_csv("openbci.csv")  # change the name, directory of the file to include the bci file
            raw = dataset.iloc[:, :].values
            print(len(raw))
            X = []
            for y in range(len(raw[0])):
                X.append(np.average(raw[:, y]))
            print(X)
            writer.writerow([X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]])


while 1:
    print("imagine ")
    processing_data()
    print("relax")
    time.sleep(3)
