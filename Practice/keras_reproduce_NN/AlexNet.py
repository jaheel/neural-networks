import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import cv2
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# std
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = np.array([cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in x_train])
x_test = np.array([cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in x_test])

x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

AlexNet = Sequential([layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
            layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            layers.Flatten(),
            layers.Dense(units=4096, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(units=4096, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(units=10, activation='softmax')
])

AlexNet.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])



AlexNet.fit(x_train, y_train, batch_size=32, epochs=10)