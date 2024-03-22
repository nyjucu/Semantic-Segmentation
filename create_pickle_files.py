import pickle as pkl
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


def fetch_train_images() -> (np.array, np.array):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), np.uint8)
    y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), np.bool_)

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_

        img_raw = tf.io.read_file(path + '/images/' + id_ + '.png')
        img = tf.image.decode_png(img_raw, channels=3)
        img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))

        X_train[n] = img

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_raw = tf.io.read_file(path + '/masks/' + mask_file)
            mask_ = tf.image.decode_png(mask_raw, channels=1)
            mask_ = tf.image.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask = tf.maximum(mask, mask_)

        y_train[n] = mask

    return X_train, y_train


def fetch_test_images() -> np.array:
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), np.uint8)

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_

        img_raw = tf.io.read_file(path + '/images/' + id_ + '.png')
        img = tf.image.decode_png(img_raw, channels=3)
        img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))

        X_test[n] = img

    return X_test


def save_pickle(X_train_, y_train_, X_test_) -> None:
    with open('X_train.pkl', 'wb') as f:
        pkl.dump(X_train_, f)

    with open('y_train.pkl', 'wb') as f:
        pkl.dump(y_train_, f)

    with open('X_test.pkl', 'wb') as f:
        pkl.dump(X_test_, f)


if __name__ == "__main__":
    X_train, y_train = fetch_train_images()
    X_test = fetch_test_images()

    save_pickle(X_train, y_train, X_test)
