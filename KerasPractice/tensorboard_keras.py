import tensorflow as tf


# Load and transform data for model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

img_rows, img_cols = 28, 28
num_classes = 10

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


###
# python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]
# [PATH_TO_LOGDIR] - without ' or ", + add '/'
# FOR EXAMPLE: "python -m tensorboard.main --logdir=logs/"
# where 'logs/' - is your directory where your log file placed from writer or smth else
###

logs = tf.keras.callbacks.TensorBoard(log_dir="logs/")

# Create convolve model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, input_shape=input_shape, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=1, callbacks=[logs])

print(model.inputs)
print(model.outputs)
print(model.get_config())
print(model.summary())
print(model.layers)

