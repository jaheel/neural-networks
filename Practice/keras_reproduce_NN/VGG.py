import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10

import numpy as np


#VGG16
def VGG16():

    model = Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


#VGG19
def VGG19():

    model = Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

# std
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(y_train.shape)
print(x_train.shape)

model = VGG16()
model.fit(x_train, y_train, batch_size=32, epochs=10)
