import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential

print(tf.__version__)
print(tf.keras.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (n, rows, cols) change to (n, rows, cols, channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# std
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Init model LeNet-5

# model architecture
model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='sigmoid'))
model.add(layers.Dense(units=84, activation='sigmoid'))
model.add(layers.Dense(units=10, activation='softmax'))
# model compile
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model train
model.fit(x_train, y_train, batch_size=32, epochs=10)