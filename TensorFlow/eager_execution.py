import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from keras.utils import to_categorical
sns.set(style="darkgrid")


tf.enable_eager_execution()


class Model():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_conv = tfe.Variable(tf.truncated_normal((5, 5, 1, 32), stddev=0.1))
        self.b_conv = tfe.Variable(tf.constant(0.1, shape=[32]))

        self.W = tfe.Variable(tf.random_normal([14*14*32, self.output_shape]))
        self.B = tfe.Variable(tf.random_normal([self.output_shape]))

        self.variables = [self.W, self.B, self.w_conv, self.b_conv]

    def predict(self, x_train):
        convolve = tf.nn.conv2d(x_train, self.w_conv, strides=[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(convolve)
        pooling = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        out = tf.nn.softmax(tf.matmul(tf.reshape(pooling, [-1, 14*14*32]), self.W) + self.B)
        return out

    def loss(self, predicted_y, desired_y):
        return tf.reduce_mean(tf.square(predicted_y - desired_y))

    def fit(self, x_train, y_train, epochs, batch_size):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        for epoch in range(epochs):
            for i in range(x_train.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    predicted = model.predict(x_train[(i * batch_size): (i+1) * batch_size])
                    curr_loss = self.loss(predicted, y_train[(i * batch_size): (i+1) * batch_size])
                grads = tape.gradient(curr_loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

            print("Loss at step {:d}: {:.3f}".format(epoch + 1, self.loss(model.predict(x_train[(i * batch_size): (i+1) * batch_size]),
                                                                      y_train[(i * batch_size): (i+1) * batch_size])))


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

x_train = tf.constant(np.reshape(x_train, [-1, 28, 28, 1]), dtype=tf.float32)
x_test = tf.constant(np.reshape(x_test, [-1, 28, 28, 1]), dtype=tf.float32)


model = Model(input_shape=x_train.shape[1], output_shape=10)
model.fit(x_train, y_train, epochs=10, batch_size=1000)

prediction = model.predict(x_test)
print(y_test[0:3])
print(prediction[0:3])