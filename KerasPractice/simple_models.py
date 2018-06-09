import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, utils, backend, optimizers, losses, datasets
import tensorflow as tf


# Load and transform data for model
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

# Create convolve model
model = models.Sequential([layers.Conv2D(32, input_shape=input_shape, kernel_size=(3, 3),
                                         padding='same', activation='relu'),
                           layers.MaxPool2D(),
                           layers.Dropout(0.25),
                           layers.Flatten(),
                           layers.Dense(128, activation='relu'),
                           layers.Dropout(0.25),
                           layers.Dense(10, activation='softmax')])
model.compile(optimizer=optimizers.SGD(lr=0.01),
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=10)

# Evaluate model
score_train = model.evaluate(x_train, y_train)
score_test = model.evaluate(x_test, y_test)
print("Test score for origin model - ", score_test, "Train score for origin model - ", score_train)

# Save weights and model
model.save_weights("data/weights.h5")
model.save("data/my_model.h5")
print("Saved model to disk")


# Load model and evaluate
loaded_model = models.load_model('data/my_model.h5')
loaded_score_test = loaded_model.evaluate(x_test, y_test)
print("Test score for loaded model - ", loaded_score_test)


# Create new model and load weights for it
model_w = models.Sequential([layers.Conv2D(32, input_shape=input_shape, kernel_size=(3, 3),
                                           padding='same', activation='relu'),
                             layers.MaxPool2D(),
                             layers.Dropout(0.20),
                             layers.Flatten(),
                             layers.Dense(128, activation='relu'),
                             layers.Dropout(0.20),
                             layers.Dense(10, activation='softmax')])
model_w.load_weights('data/weights.h5')
model_w.compile(optimizer=optimizers.Adadelta(),
                loss=losses.mean_squared_error,
                metrics=['accuracy'])
w_score_test = model_w.evaluate(x_test, y_test)
print("Test score for model with loaded weights - ", w_score_test)

# Predict class, must be 4-d vector
print(model.predict_classes(x_test[:1]))