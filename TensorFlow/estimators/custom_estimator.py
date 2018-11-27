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

estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model)

print(model.input_names)  # 'conv2d_input'
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'conv2d_input': x_train}, y=y_train, num_epochs=100, shuffle=False)
evaluation_input_fn = tf.estimator.inputs.numpy_input_fn(x={'conv2d_input': x_test}, y=y_test, shuffle=False)

estimator_model.train(input_fn=train_input_fn, steps=100)
score = estimator_model.evaluate(input_fn=evaluation_input_fn)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**score))