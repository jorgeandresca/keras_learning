#https://www.pluralsight.com/guides/deep-learning-model-add

import sys
sys.path.append("modules")
import clearterminal

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset from file
df = pd.read_csv('./data/multiplication.csv')
X = df.values[:,0:2]
Y = df.values[:,2]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

"""
# Creating dataset without .csv file
X_train = np.array([[1.0,1.0]])
Y_train = np.array([2.0])
for i in range(3,10000,2):
    X_train= np.append(X_train,[[i,i]],axis=0)
    Y_train= np.append(Y_train,[i+i])

X_test = np.array([[2.0,2.0]])
Y_test = np.array([4.0])
for i in range(4,8000,4):
    X_test = np.append(X_test,[[i,i]],axis=0)
    Y_test = np.append(Y_test,[i+i])
"""

# Model - Architecture
model = Sequential([
    Dense(20, activation="relu"),
    Dense(20, activation="relu"),
    Dense(1)
])

# Model - Setup
model.compile(optimizer=Adam(0.001),
              loss='mse',
              metrics=['mae'])

# Model - Fit data
hist = model.fit(X_train, Y_train, epochs=100, batch_size=5)

# Model - Evaluation
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
a = np.array([[3,2], [4,5], [2,1], [15,3]])

print(model.predict(a))

plt.plot(hist.history['mse'])
plt.plot(hist.history['mae'])
plt.title('Model Mean Square Error')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()