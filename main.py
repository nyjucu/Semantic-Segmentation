import pickle as pkl
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Conv2DTranspose, Input, Dropout, Concatenate, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)


def load_pickle() -> (np.array, np.array, np.array):
    with open('X_train.pkl', 'rb') as f:
        X_train_loaded = pkl.load(f)

    with open('y_train.pkl', 'rb') as f:
        y_train_loaded = pkl.load(f)

    with open('X_test.pkl', 'rb') as f:
        X_test_loaded = pkl.load(f)

    return X_train_loaded, y_train_loaded, X_test_loaded


def encoder_mini_block(input_layer, n_filters=32, dropout_prob=0.2, max_pooling=True) -> (tf.keras.layers, tf.keras.layers):
    conv = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(input_layer)
    conv = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(conv)

    conv = BatchNormalization()(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPool2D((2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv

    return skip_connection, next_layer


def decoder_mini_block(input_layer, skip_layer, n_filters=32) -> tf.keras.layers:
    up = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(input_layer)

    merge = Concatenate()([up, skip_layer])

    conv = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(merge)
    conv = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(conv)

    return conv


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    dice_coefficient = numerator / (denominator + tf.keras.backend.epsilon())
    return 1 - dice_coefficient


if __name__ == "__main__":
    X_train, y_train, X_test = load_pickle()

    inputs = Input(shape=INPUT_SHAPE)
    normed_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Encoder
    conv1, pool1 = encoder_mini_block(normed_inputs, 32, 0.1)
    conv2, pool2 = encoder_mini_block(pool1, 64, 0.1)
    conv3, pool3 = encoder_mini_block(pool2, 128, 0.2)
    conv4, pool4 = encoder_mini_block(pool3, 256, 0.2)
    conv5 = encoder_mini_block(pool4, 512, 0.3, False)[0]

    # Decoder
    conv6 = decoder_mini_block(conv5, conv4, 256)
    conv7 = decoder_mini_block(conv6, conv3, 128)
    conv8 = decoder_mini_block(conv7, conv2, 64)
    conv9 = decoder_mini_block(conv8, conv1, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

    callbacks = [
        EarlyStopping(patience=2, monitor='val_loss'),
        TensorBoard(log_dir='logs'),
        checkpointer
    ]

    datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True
    )

    history = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=100, verbose=1, callbacks=callbacks)

    model.save("segmentation.keras")
