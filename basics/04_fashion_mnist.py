# https://github.com/zalandoresearch/fashion-mnist
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide those Keras messages

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

model_file_name = '04_fashion_mnist.h5'

num_classes = 10;
batch_size = 1;  # 128
epochs = 1;  # 24

img_rows, img_cols = 28, 28;

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

input_shape = 0 #any number, it will be updated later


# This is important because different backends work in different ways (TensorFlow, Theanos, etc.)
# IF -> output: (60000, 28, 28)
# ELSE -> output (60000, 28, 28, 1)
if K.image_data_format() == 'channel_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# x_train.shape -> 6000 x 28 x 28 x 1
# input_shape -> 28 x 28 x 1



# Scale the pixel intensity - To make the values range from 0 to 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255


# Convert the classes into an array with only 1 position -> 3 to 0 0 1 0.... or 2 to 0 1 0 0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Define the model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))  # 32 filter of 3x3
model.add(MaxPooling2D(pool_size=(2, 2)))  # 1D: time serie, 3D: Video sequences
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))  # 32 filter of 3x3
model.add(MaxPooling2D(pool_size=(2, 2)))  # 1D: time serie, 3D: Video sequences

model.add(Flatten())  # from 2D to 1D

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# print(model.summary())
# model = load_model(model_file_name)

# Define the compiler to minimize categorical loss
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Train the model, and test/validate it with the test data after each cycle (epoch)
# Return history of loss and accuracy for each epic

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])



model.save('asd.h5')


import matplotlib.pyplot as plt
epoch_list = list(range(1, len(hist.history['accuracy']) + 1))
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()

