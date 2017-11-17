"""Learning a simple CNN model"""

import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from project_paths import learn_data_paths, model_paths

# Learning parameters
img_cols = 150
img_rows = 150
n_channels = 3
epochs = 40
batch_size = 16
train_samples = 2000
validation_samples = 800

# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_cols, img_rows, n_channels)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.rmsprop(lr=0.001),
              metrics=['accuracy'])

# Learning
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=learn_data_paths['Train'],
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='binary')

validation_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=learn_data_paths['Validation'],
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_samples/batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint(model_paths["Simple_cnn"], monitor="val_acc", verbose=1, save_best_only=True)],
    validation_data=validation_gen,
    validation_steps=validation_samples/batch_size
)

# Evaluating
model.load_weights(model_paths["Simple_cnn"])
score = model.evaluate_generator(
    generator=validation_gen,
    steps=validation_samples/batch_size
)

print("Best loss: {}".format(score[0]))
print("Best accuracy: {}".format(score[1]))
