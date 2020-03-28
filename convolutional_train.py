import datetime

import tensorflow
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os
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


print('looking for dumps')
if not os.path.isfile('./x_test.p'):
    print('no such file exists')
    make_dumps()

# tensorboard callback
log_dir = './logs/convolution_model'
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Stop when val_accuracy starts to decrease -> mitigate overfitting
early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01,
                                            patience=10, verbose=1, mode='max', restore_best_weights=True)

x_train = load_dump('x_train.p')
y_train = load_dump('y_train.p')
x_test = load_dump('x_test.p')
y_test = load_dump('y_test.p')
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#std
std = np.std(x_train)
# rgb
# 125.306918046875
# 122.950394140625x_train
# 113.86538318359375
# Subtract channel mean values
x_train[:,:,:,0] -= 125.306918046875
x_train[:,:,:,1] -= 122.950394140625
x_train[:,:,:,2] -= 113.86538318359375
# divide by std
x_train /= std
# Normalised to [0,1]
x_train= (x_train - np.min(x_train))/np.ptp(x_train)

# Subtract channel mean values
x_test[:,:,:,0] -= 125.306918046875
x_test[:,:,:,1] -= 122.950394140625
x_test[:,:,:,2] -= 113.86538318359375
# divide by std
x_test /= std
# Normalised to [0,1]
x_test= (x_test - np.min(x_test))/np.ptp(x_test)




print('dumps loaded')

print('building model')
model = Sequential()

# block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(Dropout(0.5))
# output
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print('model built')

print(model.summary())

print('training model')
model.fit(x=x_train, y=y_train_cat, epochs=50, verbose=1, validation_split=0.3, callbacks=[tensorboard_callback, early_stop])
print('model trained')

print('saving model')
json_string = model.to_json()
open('./model_architecture.json', 'w').write(json_string)
model.save_weights('convolution_model_weights.h5')
print('model saved')

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test_cat)
print('test loss, test acc:', results)