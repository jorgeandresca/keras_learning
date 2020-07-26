import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("Modules")
import clearterminal

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.models import load_model

def plot_data(pl, X, y):
    pl.plot(X[y == 0, 0], X[y == 0, 1], 'ob', alpha=0.5)
    pl.plot(X[y == 1, 0], X[y == 1, 1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl


def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))

    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)

    plot_data(plt, X, y)

    return plt


X, y = make_circles(n_samples=1000, factor=.6, noise=0.1, random_state=42)

#pl = plot_data(plt, X, y)
#pl.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation="tanh", name="hidden_1"))
model.add(Dense(4, activation="tanh", name="hidden_2"))
model.add(Dense(4, activation="tanh", name="hidden_3"))
model.add(Dense(1, activation="sigmoid", name="output"))

model.compile(Adam(lr=0.05), "binary_crossentropy", metrics=["accuracy"])

earlystopping_callback = [EarlyStopping(monitor='val_accuracy', patience=15, mode=max)]
model.fit(X_train, y_train, epochs=10, verbose=1, callbacks=earlystopping_callback, validation_data=(X_test, y_test))

print("-------------------")

eval_result = model.evaluate(X_test, y_test)
print("\n\n Round 1 Test loss: ", eval_result[0], " - Test accuracy: ", eval_result[1])

print("-------------------")

model.fit(X_train, y_train, epochs=10, verbose=1, callbacks=earlystopping_callback, validation_data=(X_test, y_test))

eval_result = model.evaluate(X_test, y_test)
print("\n\n Round 2 Test loss: ", eval_result[0], " - Test accuracy: ", eval_result[1])


"""
#plot_decision_boundary(model, X, y).show()


#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#print(model.summary())

"""

