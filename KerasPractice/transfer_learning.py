import numpy as np
import random
import os
import urllib
import matplotlib.pyplot as plt
from keras import models, layers, utils, backend, optimizers, losses, datasets, applications
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


def how_transfer_works():
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

    # Create model
    model = models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), name='block1_conv1'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
    model.add(Dropout(0.25, name='block1_dropout1'))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='block2_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool1'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool2'))
    model.add(Dropout(0.25, name='block2_dropout1'))

    model.add(Flatten(name='block3_flatten'))
    model.add(Dense(1024, activation='relu', name='block3_dense1'))
    model.add(Dropout(0.5, name='block3_dropout1'))
    model.add(Dense(10, activation='softmax', name='block3_dense2'))

    # Train model on GPU
    with tf.device('/gpu:0'):
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=0.0001, decay=1e-6),
                      metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=100, epochs=5)

        # Evaluate model
        score_train = model.evaluate(x_train, y_train)
        score_test = model.evaluate(x_test, y_test)
        print("Train score for origin model - ", score_train, "Test score for origin model - ", score_test)

    # Save model
    model.save_weights("data/weights.h5")
    model.save("data/my_model.h5")
    print("Saved model to disk")

    # Load previous model
    loaded_model = models.load_model('data/my_model.h5')

    # Freeze first 9 layers
    for layer in loaded_model.layers[:9]:
        layer.trainable = False

    # Create custom layers
    x = loaded_model.output
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(10, activation="softmax")(x)

    # Creating the final model
    model_final = models.Model(input=loaded_model.input, output=predictions)
    model_final.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.Adam(lr=0.0001, decay=1e-6),
                        metrics=['accuracy'])

    # Train the final model on GPU
    with tf.device('/gpu:0'):
        model_final.fit(x_train, y_train, batch_size=100, epochs=5)

    # Evaluate the final model
    score_train = model_final.evaluate(x_train, y_train)
    score_test = model_final.evaluate(x_test, y_test)
    print("Train score for origin model - ", score_train, "Test score for origin model - ", score_test)

    # Shows summaries for first and transfer models
    print(model.summary())
    print(model_final.summary())


if __name__ == '__main__':
    # how_transfer_works()

    # Download data from urls
    def download_from_url():

        def downloader(image_url, myPath):
            file_name = random.randrange(1, 10000)
            full_file_name = str(file_name) + '.jpg'
            fullfilename = os.path.join(myPath, full_file_name)
            urllib.request.urlretrieve(image_url, fullfilename)

            with open('data/children_path.txt', 'a') as file:
                name = str(fullfilename) + '\n'
                file.write(name)

        urls = list()
        with open("data/child_urls.txt", "r") as file:
            urls = file.readlines()

        myPath = 'data/child_images/'
        for image_url in urls:
            downloader(image_url, myPath)

    # Prepare data for training
    names = list()
    x_train = np.array([])
    with open("data/children_path.txt", "r") as file:
        names = file.readlines()

    x_train = list()
    for i in names:
        img_path = i.rstrip()
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = list(x)
        sub = list()
        sub.append(x)
        x_train.append(sub)
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, [65, 224, 224, 3])
    y_train = np.ones(shape=x_train.shape[0])
    y_train = utils.to_categorical(y_train, num_classes=2)

    # Dowload pretrained model
    vgg16 = applications.VGG16(input_shape=(224, 224, 3))
    print(vgg16.summary())

    # Transform images
    img_path = 'data/child.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    test1 = x

    img_path = 'data/dog.jpeg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    test2 = x
    preds = vgg16.predict(x)


    preds = vgg16.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])

    # Retrain
    for layer in vgg16.layers:
        layer.trainable = False

    output = vgg16.output
    x = Dense(100, activation='softmax')(output)
    prediction = Dense(2, activation='softmax')(x)

    final_model = models.Model(input=vgg16.input, output=prediction)
    final_model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.Adam(lr=0.0001, decay=1e-6),
                        metrics=['accuracy'])

    with tf.device('/gpu:0'):
        final_model.fit(x_train, y_train, batch_size=10, epochs=100)
        prediction1 = final_model.predict(test1)
        prediction2 = final_model.predict(test2)
        print(prediction1, prediction2)
    # input_new = layers.Input(shape=(arr.shape), name='image_input')
    # output_vgg16 = vgg16(input_new)
    # x = Dense(256, activation='relu', name='fc1')(output_vgg16)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # creating the final model
    # model_final = models.Model(input=input_new, output=output_vgg16)
    # model_final.compile(loss='categorical_crossentropy',
    #                     optimizer=optimizers.Adam(lr=0.0001, decay=1e-6),
    #                     metrics=['accuracy'])














