"""Transfer learning a deep CNN model on the clean data with augmentation"""

import keras
from keras.layers import *
from keras.applications import Xception
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from project_paths import learn_data_paths, model_paths


# Learn parameters
img_cols = 150
img_rows = 150
n_channels = 3
epochs = 5
batch_size = 16
train_samples = 2000
validation_samples = 800

# Model
base_model = Xception(input_shape=(img_cols, img_rows, n_channels), weights='imagenet', include_top=False)

out = base_model.output
out = GlobalAveragePooling2D()(out)
predictions = Dense(2, activation='softmax')(out) 

for layer in base_model.layers:
    layer.trainable = False

model = Model(base_model.input, predictions)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Learning
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    directory=learn_data_paths['Train'],
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_gen = validation_datagen.flow_from_directory(
    directory=learn_data_paths['Validation'],
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_samples/batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint(model_paths["Deep_cnn"], monitor="val_acc", verbose=1, save_best_only=True)],
    validation_data=validation_gen,
    validation_steps=validation_samples/batch_size
)

# Evaluating
model.load_weights(model_paths["Deep_cnn"])
score = model.evaluate_generator(
    generator=validation_gen,
    steps=validation_samples/batch_size
)

print("Best loss: {}".format(score[0]))
print("Best accuracy: {}".format(score[1]))
