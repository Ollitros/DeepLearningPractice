import numpy as np
import matplotlib.pyplot as plt
from keras import layers,  backend
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras import models, utils, optimizers, datasets

from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

img_rows, img_cols = 28, 28
num_classes = 10

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    x_train = np.reshape(x_train, (60000, 784))
    x_test = np.reshape(x_test, (10000, 784))
x_train = MinMaxScaler().fit_transform(x_train)
x_test = MinMaxScaler().fit_transform(x_test)

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

inputs = layers.Input(shape=input_shape)
conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs) #28 x 28 x 32
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

#decoder
conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
up1 = layers.UpSampling2D((2,2))(conv4) # 14 x 14 x 128
conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
up2 = layers.UpSampling2D((2,2))(conv5) # 28 x 28 x 64
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1

model = models.Model(input=inputs, output=decoded)
model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

with tf.device('/gpu:0'):
    model.fit(x_train, x_train, batch_size=100, epochs=10, shuffle=True, validation_data=(x_test, x_test))
prediction = model.predict(x_train)
plt.imshow(np.reshape(prediction[0], (28, 28)))
plt.show()
plt.imshow(np.reshape(x_train[0], (28, 28)))
plt.show()

fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks': [], 'yticks': []},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(np.reshape(prediction[i], (28, 28)), cmap='binary_r')
    ax[1, i].imshow(np.reshape(x_train[i], (28, 28)), cmap='binary_r')

plt.show()