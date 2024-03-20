import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imshow
from main import load_pickle
from tensorflow.keras.models import load_model


def show_example() -> None:
    image_x = random.randint(0, len(X_train))
    imshow(X_train[image_x])
    plt.show()
    imshow(np.squeeze(y_train[image_x]))
    plt.show()


def sanity_check_train() -> None:
    image_x = random.randint(0, len(preds_train))
    imshow(X_train[image_x])
    plt.show()
    imshow(np.squeeze(y_train[image_x]))
    plt.show()
    imshow(np.squeeze(preds_train[image_x]))
    plt.show()


def sanity_check_test() -> None:
    image_x = random.randint(0, len(preds_test))
    imshow(X_test[image_x])
    plt.show()
    imshow(np.squeeze(preds_test[image_x]))
    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_test = load_pickle()

    model = load_model("segmentation.keras")
    model.summary()

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    preds_train = (preds_train > 0.5).astype(np.uint8)
    preds_test = (preds_test > 0.5).astype(np.uint8)

    N_TESTS = 10

    for test in range(N_TESTS):
        sanity_check_train()
        sanity_check_test()
