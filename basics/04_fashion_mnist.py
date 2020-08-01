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
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import random
import numpy as np
import matplotlib.pyplot as plt



num_classes = 10;
batch_size = 128;  # 128
epochs = 50;  # 50
dropout = 0.8
img_rows, img_cols = 28, 28;

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Shrinking the dataset
x_train = x_train[:5000]  # 5000
y_train = y_train[:5000]  # 5000
x_test = x_test[:500]  # 500
y_test = y_test[:500]  # 500

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


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
model.add(Dropout(dropout))
model.add(Dense(num_classes, activation='softmax'))

# print(model.summary())

# Define the compiler to minimize categorical loss
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Train the model, and test/validate it with the test data after each cycle (epoch)
# Return history of loss and accuracy for each epic

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model
score_train = np.round(model.evaluate(x_test, y_test, verbose=0), 2)
score_test = np.round(model.evaluate(x_test, y_test, verbose=0), 2)
print('Test loss: ', score_test[0])
print('Test accuracy: ', score_test[1])




# Predictions

random = int(random.random() * len(x_test))

x_toPredict = x_test[random]
y_toPredict = y_test[random]


x_toPredict = np.array([x_toPredict])  # shape (1,28,28,1)

prediction = model.predict(x=x_toPredict, verbose=0)  # shape (1,10)
prediction = prediction[0,:]  # shape (10,)
prediction = np.round(prediction.astype(np.float), 2)

print("Real class: " + str(y_toPredict))
print("Classification: " + str(prediction))



# Print the model
epoch_list = list(range(1, len(hist.history['accuracy']) + 1))
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy: ' +  str(score_train[1]), 'Validation Accuracy: ' + str(score_test[1])))

plt.savefig('04_model_chart.png')

plt.show()

# Save the Model
model.save('04_model.h5')
# plot_model(model, to_file='04_model_arch.png', show_shapes=True, show_layer_names=True)
