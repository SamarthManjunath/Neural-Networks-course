# Manjunath, Samarth
# 1001-522-809
# 2019-12-02
# Assignment-04-02

import pytest
import numpy as np
from cnn import CNN
import os

def test_evaluate():
    from tensorflow.keras.datasets import mnist
    import tensorflow.keras as keras

    bsize_entity = 128
    no_of_cls = 10
    epochs = 3
    img_row, img_col = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, no_of_cls)
    y_test = keras.utils.to_categorical(y_test, no_of_cls)

    model = CNN()
    model.add_input_layer(shape=(28, 28, 1), name="Input")
    model.append_conv2d_layer(32, kernel_size=(3, 3))
    model.append_conv2d_layer(64, kernel_size=(3, 3))
    model.append_maxpooling2d_layer(pool_size=(2, 2))
    model.append_flatten_layer()
    model.append_dense_layer(128, activation="relu")
    model.append_dense_layer(no_of_cls, activation="softmax")
    model.set_loss_function("categorical_crossentropy")
    model.set_metric("accuracy")
    model.set_optimizer("Adagrad")
    model.train(x_train, y_train, batch_size=bsize_entity, num_epochs=epochs)

    mk = model.evaluate(x_test, y_test)
    correct = np.array([0.05093684684933396, 0.9907])

    np.testing.assert_almost_equal(correct[0], mk[0], decimal=2)
    np.testing.assert_almost_equal(correct[1], mk[1], decimal=2)

def test_train():
    from tensorflow.keras.datasets import mnist
    import tensorflow.keras as keras

    bsize_entity = 128
    no_of_cls = 10
    epochs = 1
    img_row, img_col = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, no_of_cls)
    y_test = keras.utils.to_categorical(y_test, no_of_cls)

    model = CNN()
    model.add_input_layer(shape=(28, 28, 1), name="Input")
    model.append_conv2d_layer(32, kernel_size=(3, 3))
    model.append_conv2d_layer(64, kernel_size=(3, 3))
    model.append_maxpooling2d_layer(pool_size=(2, 2))
    model.append_flatten_layer()
    model.append_dense_layer(128, activation="relu")
    model.append_dense_layer(no_of_cls, activation="softmax")
    model.set_loss_function("categorical_crossentropy")
    model.set_metric("accuracy")
    model.set_optimizer("Adagrad")
    model.train(x_train, y_train, batch_size=bsize_entity, num_epochs=epochs)

    mk = model.evaluate(x_test, y_test)
    correct = np.array([0.06997422293154523, 0.9907])

    np.testing.assert_almost_equal(correct[0], mk[0], decimal=2)
    np.testing.assert_almost_equal(correct[1], mk[1], decimal=2)