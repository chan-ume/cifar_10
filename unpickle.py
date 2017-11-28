#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_file = unpickle("cifar10/batches.meta")
label_names = label_file[b'label_names']
d = unpickle("cifar10/data_batch_1")

data = d[b"data"]
labels = np.array(d[b"labels"])
nsamples = len(data)

print (label_names)

nclasses = 10
pos = 1
for i in range(nclasses):
    targets = np.where(labels == i)[0]
    np.random.shuffle(targets)
    
    for idx in targets[:10]:
        plt.subplot(10, 10, pos)
        img = data[idx]
        # (channel, row, column) => (row, column, channel)
        plt.imshow(img.reshape(3, 32, 32).transpose(1, 2, 0))
        plt.axis('off')
        label = label_names[i]
        pos += 1
plt.show()
