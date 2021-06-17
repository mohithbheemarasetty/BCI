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


def sample_extract(b, d):
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


def processing_data(b, z):
    a = 'MI_data'
    c = '.csv'
    with open(a+c, 'a', newline='') as csvfile:  # the name for the final data set
        writer = csv.writer(csvfile, delimiter=',')
        for sam in range(0, 51):
            sample_extract(b, sam)
            if sam == 0:
                continue
            dataset = pd.read_csv(file)  # change the name, directory of the file to include the bci file
            raw = dataset.iloc[:, :].values
            print(len(raw))
            X = []
            for y in range(len(raw[0])):
                X.append(np.average(raw[:, y]))
            print(X)
            writer.writerow([X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], z])


cur = time.time()
leg = 'L'
hand = 'H'
idle = 'I'
while 1:
    print("relax for 2 seconds")
    time.sleep(4)

    if time.time() <= cur + 400:
        print("imagine moving ur legs")
        time.sleep(1)
        processing_data(count, leg)
    if cur + 800 >= time.time() > cur + 400:
        print("imagine moving ur hand")
        time.sleep(1)
        processing_data(count, hand)
    if cur + 1200 >= time.time() > cur + 800:
        print("do nothing")
        time.sleep(1)
        processing_data(count, idle)
    if time.time() > cur + 1200:
        print("end of data collection")
        break
    count += 1
