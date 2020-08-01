import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide those Keras messages

from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.imagenet_utils import decode_predictions

import numpy as np

model = InceptionV3()  # Input (None, 299,299,3 / print(model.input_shape)


image = Image.open('data/test/hare.jpg')

image = image.resize((299,299))

image = np.array(image)  # Image -> Array
image = np.expand_dims(image, axis=0)  # (299,299,3) -> (1, 299,299,3)

image = preprocess_input(image)

prediction = model.predict(image, verbose=1)

print(decode_predictions(prediction))