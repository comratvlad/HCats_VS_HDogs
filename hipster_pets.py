"""Providing comfortable access to hipster animals dataset"""

import os
import pickle
from numpy import random, array
from skimage import io, color, transform

from project_paths import raw_data_paths, binary_data_path, learn_data_paths

import warnings
warnings.filterwarnings('ignore')


# Not used anymore.
def create_binary_dataset(size=1400):
    """ Creating binary dataset file for quick load.

    :param size: int, optional (default=1400)
        Samples count for each classes.
    """
    assert type(size) == int and size > 0, 'Wrong format of size, it must be positive integer.'
    tags = raw_data_paths.keys()
    data = []
    for i, tag in enumerate(tags):
        assert len(
            os.listdir(raw_data_paths[tag])) >= size, 'Size is too large, try to reload data with bigger min_size.'
        tag_data = [[io.imread(raw_data_paths[tag] + filename), i] for filename in
                    os.listdir(raw_data_paths[tag])[:size]]
        data += tag_data

    with open(binary_data_path, "wb") as file:
        pickle.dump(array(data), file)


# Not used anymore.
def get_data(count_per_tag=1400, shuffle=True, train_count=1000, size=(150, 150, 3)):

    assert type(count_per_tag) == type(train_count) == int, 'Wrong count format.'
    assert size[2] in {1, 3}, "Wrong number of channels, it must be 1 or 3"

    n_channels = size[2]

    with open(binary_data_path, "rb") as file:
        data = pickle.load(file)

    if n_channels == 1:
        for i in range(len(data)):
            data[i][0] = color.rgb2gray(data[i][0])

    if shuffle:
        random.shuffle(data)

    x_train = array([transform.resize(data[i][0], size) for i in range(train_count * 2)])
    y_train = array([[data[i][1]] for i in range(train_count * 2)])

    x_test = array([transform.resize(data[i][0], size) for i in range(train_count * 2, count_per_tag * 2)])
    y_test = array([[data[i][1]] for i in range(train_count * 2, count_per_tag * 2)])

    return x_train, y_train, x_test, y_test


def create_learning_directory(count_per_tag=1400, train_count=1000, size=(150, 150, 3)):
    """Creating learning dataset directory to work with keras.preprocessing.image.ImageDataGenerator

    :param count_per_tag: int, optional (default=1400)
        Samples count for each classes.
    :param train_count: int, optional (default=1000)
        Samples count of learning samples for each classes.
    :param size: tuple, optional (default=(150, 150, 3))
        Format of pictures.
    """
    assert size[2] in {1, 3}, "Wrong number of channels, it must be 1 or 3"
    tags = raw_data_paths.keys()
    for i, tag in enumerate(tags):
        for filename in os.listdir(raw_data_paths[tag])[:train_count]:
            img = io.imread(raw_data_paths[tag] + filename)
            img = transform.resize(img, size)
            io.imsave('{path}/{tag}/{file}'.format(path=learn_data_paths['Train'], tag=tag, file=filename), img)
        for filename in os.listdir(raw_data_paths[tag])[train_count:count_per_tag]:
            img = io.imread(raw_data_paths[tag] + filename)
            img = transform.resize(img, size)
            io.imsave('{path}/{tag}/{file}'.format(path=learn_data_paths['Validation'], tag=tag, file=filename), img)


if __name__ == "__main__":
    create_learning_directory()
