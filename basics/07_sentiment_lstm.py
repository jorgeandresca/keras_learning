import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide those Keras messages

"""
    This data has already preprocessed the reviews. This preprocessing replaces he actual work with the encoding. 
    So, the second most popular word is replaced by 2, third most popular by 3, etc
"""

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb


num_words = 6000  # the top most n frequent words to consider
skip_top = 0  # skip the top most words that are likely -> the, and, a
max_review_len = 400  # max number of words from a review

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words, skip_top=skip_top)


# Reviews must have all the same length
# 1 row for each review
x_train = sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_len)


# Model
model = Sequential()
model.add(Embedding(num_words, 64))
model.add(LSTM(128))  # 128 outputs
model.add(Dense(1, activation='sigmoid'))


# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit
batch_size = 24
epoch = 5
callback_early_stopping = EarlyStopping(monitor='val_accuracy', mode='max')

model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch,
          validation_data=(x_test, y_test),
          callbacks=[callback_early_stopping])

