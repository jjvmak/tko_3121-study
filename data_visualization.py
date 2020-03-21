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


def extract_rgb_from_single_image(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    #r_var = np.var(r)
    #g_var = np.var(g)
    #b_var = np.var(b)

    return r_mean, g_mean, b_mean

def compute_channel_mean_values(x):
    r_total = []
    g_total = []
    b_total = []
    for i in range(len(x)):
        r_mean, g_mean, b_mean = extract_rgb_from_single_image(x[i])
        r_total = np.append(r_total, r_mean)
        g_total = np.append(g_total, g_mean)
        b_total = np.append(b_total, b_mean)
    r_total_mean = np.mean(r_total)
    g_total_mean = np.mean(g_total)
    b_total_mean = np.mean(b_total)

    print(r_total_mean)
    print(g_total_mean)
    print(b_total_mean)

    fig, ax = plt.subplots()
    labels = ['R', 'G', 'B']
    locations = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(locations, [r_total_mean, g_total_mean, b_total_mean], width)
    ax.set_ylabel('Mean value')
    ax.set_title('RGB values')
    ax.set_xticks(locations)
    ax.set_xticklabels(labels)
    fig.tight_layout()

    plt.show()

def histogram_over_data(x):
    # _ = plt.hist(x[:, :, :, :].ravel(), bins=256, color='orange', )
    _ = plt.hist(x[:, :, :, 0].ravel(), bins=256, color='red', alpha=0.5)
    _ = plt.hist(x[:, :, :, 1].ravel(), bins=256, color='Green', alpha=0.3)
    _ = plt.hist(x[:, :, :, 2].ravel(), bins=256, color='Blue', alpha=0.2)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.show()


def PCA_scatter_plot_for_rgb(x, y):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 12)
    ax.grid(True)

    pca = PCA(n_components=2)
    x_reshape = x.reshape(-1, 3072)
    pca.fit(x_reshape)
    pca_x = pca.transform(x_reshape)


    scatter = ax.scatter(pca_x[:, 0], pca_x[:, 1], c=y.reshape(15000), cmap='viridis_r', alpha=0.3)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.yticks(np.arange(-8000, 8000, 1000))
    plt.xticks(np.arange(-8000, 8000, 1000))
    plt.show()


def PCA_scatter_plot_for_dwt2(x, y):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 12)
    ax.grid(True)

    pca = PCA(n_components=2)

    pca.fit(x)
    pca_x = pca.transform(x)


    scatter = ax.scatter(pca_x[:, 0], pca_x[:, 1], c=y.reshape(15000), cmap='viridis_r', alpha=0.3)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.yticks(np.arange(-8000, 8000, 1000))
    plt.xticks(np.arange(-8000, 8000, 1000))
    plt.show()


print('looking for dumps')
if not os.path.isfile('./x_test.p'):
    print('no such file exists')
    make_dumps()

x_train = load_dump('x_train.p')
y_train = load_dump('y_train.p')
x_test = load_dump('x_test.p')
y_test = load_dump('y_test.p')
dwt2_features = load_dump('dwt2_features.p')
y_train_cat = to_categorical(y_train)
x_train = x_train.astype('float32')
#x_train /= 255

# get the indexes
y_0 = np.where(y_train[:,0] == 0)
y_1 = np.where(y_train[:,0] == 1)
y_2 = np.where(y_train[:,0] == 2)
y_3 = np.where(y_train[:,0] == 3)
y_4 = np.where(y_train[:,0] == 4)
y_5 = np.where(y_train[:,0] == 5)
y_6 = np.where(y_train[:,0] == 6)
y_7 = np.where(y_train[:,0] == 7)
y_8 = np.where(y_train[:,0] == 8)
y_9 = np.where(y_train[:,0] == 9)

train_x_0 = np.take(x_train, y_0, axis=0)
train_x_0 = train_x_0[0]
train_y_0 = np.take(y_train, y_0)

train_x_2 = np.take(x_train, y_2, axis=0)
train_x_2 = train_x_2[0]
train_y_2 = np.take(y_train, y_2)

train_x_5 = np.take(x_train, y_5, axis=0)
train_x_5 = train_x_5[0]
train_y_5 = np.take(y_train, y_5)

dwt2_x_0 = np.take(dwt2_features, y_0, axis=0)
dwt2_x_0 = dwt2_x_0[0]

dwt2_x_2 = np.take(dwt2_features, y_2, axis=0)
dwt2_x_2 = dwt2_x_2[0]

dwt2_x_5 = np.take(dwt2_features, y_5, axis=0)
dwt2_x_5 = dwt2_x_5[0]


train_x_two_class = np.concatenate((train_x_0, train_x_5))
train_y_two_class = np.concatenate((train_y_0, train_y_5))

train_x_three_class = np.concatenate((train_x_two_class, train_x_2))
train_y_three_class = np.concatenate((train_y_two_class, train_y_2))

dwt2_x_two_class = np.concatenate((dwt2_x_0, dwt2_x_5))
dwt2_x_three_class = np.concatenate((dwt2_x_two_class, dwt2_x_2))



PCA_scatter_plot_for_dwt2(dwt2_x_three_class, train_y_three_class)
#PCA_scatter_plot_for_rgb(train_x_three_class, train_y_three_class)

#compute_channel_mean_values(x_train)

#histogram_over_data(train_x_two_class)

