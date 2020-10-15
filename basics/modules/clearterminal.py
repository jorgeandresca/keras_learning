import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # This will hide those Keras messages

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
    