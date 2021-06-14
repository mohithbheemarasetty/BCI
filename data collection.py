import time
from pylsl import StreamInlet, resolve_stream
import csv
import pandas as pd
import numpy as np
print("starting...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
duration = 0.1
count = 0


def testLSLSamplingRate(b, d):
    start = time.time()
    a = 'openbci'
    c = '.csv'
    global file
    file = a+str(b)+'_'+str(d)+c
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        while time.time() <= start + duration:
            samples, timestamp = inlet.pull_chunk()
            if samples:
                writer.writerows(samples)


def processing_data(b):
    a = 'MI_data'
    c = '.csv'
    with open(a+str(b)+c, 'w', newline='') as csvfile:  # the name for the final data set
        writer = csv.writer(csvfile, delimiter=',')
        for sam in range(0, 6):
            testLSLSamplingRate(b, sam)
            if sam == 0:
                continue
            dataset = pd.read_csv(file)  # change the name, directory of the file to include the bci file
            raw = dataset.iloc[:, :].values
            print(len(raw))
            X = []
            for y in range(len(raw[0])):
                X.append(np.average(raw[:, y]))
            print(X)
            writer.writerow([X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]])


while 1:
    print('data set no :', count)
    print("relax for 2 seconds")
    time.sleep(2)
    print("imagine moving ur right arm")
    processing_data(count)
    con = int(input("do want to do it again? if so press 1 else press anything else"))
    if con != 1:
        break
    count += 1

print("end of dataset collection")
