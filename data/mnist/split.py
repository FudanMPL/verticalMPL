import csv
import numpy as np
import random

def split():
    with open("mnist_train.csv", 'r') as f, \
            open("mnist_train_party0.csv", "w", newline='') as p1,\
            open("mnist_train_party1.csv", "w", newline='') as p2:
        reader = csv.reader(f)
        writer1 = csv.writer(p1)
        writer2 = csv.writer(p2)
        for row in reader:
            row1=row[1:401]
            row2=row[401:785]
            row2.append(row[0])
            writer1.writerow(row1)
            writer2.writerow(row2)

split()