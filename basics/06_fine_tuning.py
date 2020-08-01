from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide those Keras messages


"""
    InvceptionV3 has input (299, 299, 3) ((in case the environment is configured to have the channel at the end)
"""


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import numpy as np
import matplotlib.pyplot as plt

# Get number of files in folder and subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


# Get number of subfolders directly below the folder in path
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])


# Created variations of the images by rotating, shift up, down left, right, sheared, zoom in,
#   or flipped horizontally on vertical axis
def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.9  # This will split the dataset in Training and Validation subsets.
    )


# Main Code
image_width, image_height = 299, 299;
num_epochs = 1
batch_size = 32

train_dir = 'data/dogs_vs_cats/train'
validate_dir = 'data/dogs_vs_cats/validate'

real_num_train_samples = get_num_files(train_dir)
real_num_validate_samples = get_num_files(validate_dir)
num_classes = get_num_subfolders(train_dir)

print("real_num_train_samples: " + str(real_num_train_samples))
print("real_num_validate_samples: " + str(real_num_validate_samples))
print("num_classes: " + str(num_classes))


# Image generation (data augmentation)
#    Each new batch of data is randomly adjusted according to the parameters supplied to ImageDataGenerator
#   that's why we need to use. fit_generator later and not .fit
#   However ->>> .fit_generator is deprecated, from now we must use .fit only

train_generator = create_img_generator().flow_from_directory(
    train_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    seed=42,
    subset='training'  # 2 options: Training / Validation. This works if ImageDataGenerator.validation_split is set.
)


validation_generator = create_img_generator().flow_from_directory(
    validate_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    seed=42
)


num_train_samples = len(train_generator.filepaths)
num_validate_samples = len(validation_generator.filepaths)

print("")
print("model num_train_samples: " + str(num_train_samples))
print("model num_validate_samples: " + str(num_validate_samples))
print("")


# Load the pretrained model
#   Exclude the final fully connected layer (include_top=false)
inceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)


# Define a new classifier to attach to the pretrained model
x = inceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)


# Merge the pretrained model and the new FC classifier
model = Model(inputs=inceptionV3_base_model.input, outputs=output)


# Freeze all layers in the pretrained model
for layer in inceptionV3_base_model.layers:
    layer.trainable = False

# Compile
#   We user categorical_crossentropy since our model is trying to classify categorical result
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit
hist = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=num_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=num_validate_samples // batch_size
    #class_weight='auto'
)

# Evaluate the model
score_train = np.round(model.evaluate(train_generator, verbose=0), 2)
score_test = np.round(model.evaluate(validation_generator, verbose=0), 2)
print('Test loss: ', score_test[0])
print('Test accuracy: ', score_test[1])




#  --- Fine Tuning ---
layers_to_freeze = 172
for layer in model.layers[:layers_to_freeze]:
    layer.trainable = False
for layer in model.layers[layers_to_freeze:]:
    layer.trainable = True



# Compile
#   We user categorical_crossentropy since our model is trying to classify categorical result
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# Fit
hist = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=num_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=num_validate_samples // batch_size
    #class_weight='auto'
)

# Evaluate the model
score_train = np.round(model.evaluate(train_generator, verbose=0), 2)
score_test = np.round(model.evaluate(validation_generator, verbose=0), 2)
print('Test loss: ', score_test[0])
print('Test accuracy: ', score_test[1])


model.save('06_model.h5')


