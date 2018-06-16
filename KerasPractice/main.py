import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, utils, backend
import tensorflow as tf
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras import models, utils, optimizers, datasets, losses, applications, preprocessing


# Load and transform data for model
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

img_rows, img_cols = 32, 32
num_classes = 10

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)


img_path = 'data/children/child.jpg'
img = preprocessing.image.load_img(img_path, target_size=(2048, 2048))
x = preprocessing.image.img_to_array(img)



# model = applications.ResNet50(weights=None)
# print(model.summary())
#
# inputs = layers.Input(shape=input_shape)
# flatten = Flatten()(inputs)
# x = Dense(5000, activation='relu')(flatten)
# x = Dense(10, activation='relu')(x)
# final_model = models.Model(input=inputs, output=x)
# final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# final_model.fit(x_train, y_train, batch_size=1, epochs=1)



# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=100, epochs=1)