import datetime
import pywt
import tensorflow
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

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


def features(x):
    features_per_image = []
    for i in range(len(x)):
        print(i)
        gray = cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY)
        coeffs2 = pywt.dwt2(gray, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2

        LL = np.reshape(LL, (324))
        LH = np.reshape(LH, (324))
        HL = np.reshape(HL, (324))
        HH = np.reshape(HH, (324))

        conc = np.concatenate((LL, LH), axis=0)
        conc = np.concatenate((conc, HL), axis=0)
        conc = np.concatenate((conc, HH), axis=0)

        if i == 0:
            features_per_image = conc
        else:
            features_per_image = np.vstack((features_per_image, conc))
    pickle.dump(features_per_image, open("dwt2_features_test.p", "wb"))
    #return features_per_image

def wat(x):
    for i in range(len(x)):
        print(x)

print('looking for dumps')
if not os.path.isfile('./x_test.p'):
    print('no such file exists')
    make_dumps()

x_train = load_dump('x_train.p')
y_train = load_dump('y_train.p')
x_test = load_dump('x_test.p')
y_test = load_dump('y_test.p')
dwt2_features = load_dump('dwt2_features.p')
dwt2_test = load_dump('dwt2_features_test.p')
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 1
x_test /= 1

print('dumps loaded')

input_dim = dwt2_features.shape[1]
print(input_dim)

# tensorboard callback
#log_dir = './logs/fc_model'
#tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = Sequential()
model.add(Dense(1296, activation='relu', input_dim=input_dim))
model.add(Dense(1296, activation='relu', name='fc_1'))
model.add(Dense(1296, activation='relu', name='fc_2'))
model.add(Dense(1296, activation='relu', name='fc_3'))
model.add(Dense(512, activation='relu', name='fc_4'))
model.add(Dense(10, activation='softmax', name='output'))

opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print('model built')

print(model.summary())

print('training model')
model.fit(batch_size=128, x=dwt2_features, y=y_train_cat, epochs=3, verbose=1, validation_split=0.2)
print('model trained')

print('saving model')
json_string = model.to_json()
open('./fc_model_architecture.json', 'w').write(json_string)
model.save_weights('fc_model_weights.h5')
print('model saved')

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(dwt2_test, y_test_cat, batch_size=128)
print('test loss, test acc:', results)