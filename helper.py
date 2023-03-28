import numpy as np
import matplotlib.pyplot as plt

def indexing(dataset):
    data_entry = np.empty(len(dataset), dtype=object)
    for i in range(len(dataset)):
        data = {}
        for word in dataset[i]:
            data[word] = 1 if word not in data else data[word] + 1
        data_entry[i] = data
    return data_entry


def preprocess(dataset):
    data = {}
    for entry in dataset:
        for word in entry:
            data[word] = 1 if word not in data else data[word] + 1
    return data

def plot(x, y, title, x_label, y_label, fileName):
    plt.xscale("log")
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fileName)
    plt.show()
    plt.close()