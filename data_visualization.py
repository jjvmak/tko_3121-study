from keras.utils import to_categorical
from scipy.stats import stats
from sklearn.decomposition import PCA
import numpy as np
import tensorflow
import pickle
import os
import matplotlib.pyplot as plt

def make_dumps():
    print('making dumps')
    (x_train, y_train), (x_test, y_test) = \
    tensorflow.keras.datasets.cifar10.load_data()
    pickle.dump(x_test, open("x_test.p", "wb"))
    pickle.dump(x_train, open("x_train.p", "wb"))
    pickle.dump(y_test, open("y_test.p", "wb"))
    pickle.dump(y_train, open("y_train.p", "wb"))


def load_dump(dump_name):
    print('loading dump: ' + dump_name)
    return pickle.load(open(dump_name, "rb"))


def PCA_scatter_plot(x, y):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 12)
    ax.grid(True)

    pca = PCA(n_components=6)
    x_reshape = x.reshape(-1, 3072)
    pca.fit(x_reshape)
    pca_x = pca.transform(x_reshape)


    ax.scatter(pca_x[:, 0], pca_x[:, 1], c=y.reshape(50000), cmap='Dark2', alpha=0.3)
    ax.legend(loc=4)
    plt.show()


print('looking for dumps')
if not os.path.isfile('./x_test.p'):
    print('no such file exists')
    make_dumps()

x_train = load_dump('x_train.p')
y_train = load_dump('y_train.p')
x_test = load_dump('x_test.p')
y_test = load_dump('y_test.p')
y_train_cat = to_categorical(y_train)
x_train = x_train.astype('float32')
x_train /= 255

PCA_scatter_plot(x_train, y_train)