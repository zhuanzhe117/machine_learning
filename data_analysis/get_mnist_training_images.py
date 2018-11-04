#encoding=utf-8

import os
import struct
import numpy as np

# def load_mnist():
#     labels_path = "data/train-labels-idx1-ubyte.gz"
#     images_path = "data/train-images-idx3-ubyte.gz"
#     # labels_path = os.path.join(path, 'train-labels-idx1-ubyte')
#     # images_path = os.path.join(path, 'train-images-idx3-ubyte')
#     with open(labels_path, 'rb') as lbpath: #rb表示以二进制方式读取
#         magic, n = struct.unpack('>II', lbpath.read(8))
#         labels = np.fromfile(lbpath, dtype=np.uint8)
#         print len(labels)
#     with open(images_path, 'rb') as imgpath:
#         magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
#         timages = np.fromfile(imgpath, dtype=np.uint8)
#         print len(timages)
#         images = timages.reshape(len(labels), 784)
#
#     return images, labels

# import matplotlib.pyplot as plt
# X_train ,y_train = load_mnist()
# fig, ax = plt.subplots(
#     nrows=2,
#     ncols=5,
#     sharex=True,
#     sharey=True, )
#
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X,Y=mnist["data"],mnist["target"]

print(X.shape,Y.shape)