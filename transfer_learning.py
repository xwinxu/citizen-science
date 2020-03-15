from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
# tf 1.14
import tensorflow as tf
# "scipy==1.1.0", "pillow<7"

from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image
import os
from types import MethodType
import pandas as pd
import scipy
import skimage.transform as transform

import matplotlib.pyplot as plt
from IPython import display
from matplotlib.pyplot import imshow
plt.style.use('dark_background')

######## GLOBAL VARIABLES #########
IMAGE_SHAPE = (224, 224)
SOURCE_IMAGE_SHAPE = [300, 200, 3]
TARGET_IMAGE_SHAPE = (224, 224, 3)
height_diff = TARGET_IMAGE_SHAPE[1] - SOURCE_IMAGE_SHAPE[1]
PAD = ((0,  0), (height_diff//2, height_diff // 2), (0, 0))
DATA_ROOT = '/'
OUTPUT_DIR = '/'
EPOCHS = 100

######## IMAGE UTILITY FUNCTIONS #########


def load_model():
    classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

    cls = tf.keras.applications.MobileNetV2(
        input_shape=TARGET_IMAGE_SHAPE,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    cls2 = tf.keras.applications.MobileNetV2(
        input_shape=TARGET_IMAGE_SHAPE,
        include_top=True,
        weights='imagenet'
    )

    return cls, cls2


def data_preprocess(img):
    img = img.squeeze()
    padded = np.pad(img, PAD, mode='reflect')
    resized = transform.resize(padded, TARGET_IMAGE_SHAPE)
    resized = np.squeeze(resized)

    out = np.expand_dims(resized, 0)
    return out


def get_data():
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
    data_root = DATA_ROOT
    DATA_FORMAT = 'channels_last'
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255, data_format=DATA_FORMAT)
    image_data = image_generator.flow_from_directory(str(data_root), target_size=(
        SOURCE_IMAGE_SHAPE[0], SOURCE_IMAGE_SHAPE[1]), batch_size=1)
    # manually get from the generator the inputs and targets
    xs = []
    ys = []
    i = 0
    for x, y in image_data:
        if i == 125:
            break
        xs.extend(data_preprocess(x))
        ys.extend(y)
        i += 1

    xs = np.array(xs)
    ys = np.array(ys)
    ys2 = np.zeros((xs.shape[0], 1000))
    ys2[:, 335] = 1  # squirrel

    return xs, ys, ys2


def plot_results(model, xs):
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)

    class_, color = model.predict(xs[115:, ...])
    # select rows where argmax == 335
    mask = np.argmax(class_, 1)
    # this is a numpy array
    _image = xs[115:, ...][mask == 335]
    _color = np.argmax(color, 1)[mask == 335]
    _color = [1, 2, 0, 1, 2, 1]
    plt.subplots(figsize=(30, 30))

    for i in range(2):
        for j in range(3):
            ax = plt.subplot(2, 3, (i + j * 2) + 1)
            plt.imshow(_image[i + j * 2].reshape(224, 224, 3))
            c = _color[i+j*2]
            if c == 0:
                t = 'other'
            elif c == 1:
                t = 'grey'
            else:
                t = 'brown'
            ax.set_title(t, fontsize=50, fontweight=10)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig(OUTPUT_DIR)

######## Custom Transfer Learning #########


class CustomModel(tf.keras.Model):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        cls, cls2 = load_model()
        self.before_logits = cls
        for layer in cls.layers:
            layer.trainable = False
        self.mobile_last = cls2.get_layer(index=-1)
        for layer in cls2.layers:
            layer.trainable = False
        self.our_last = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        before_logs = self.before_logits(inputs, training=training)
        mobile_last = self.mobile_last(before_logs)
        our_last = self.our_last(before_logs)
        return [mobile_last, our_last]


class CustomCatLoss(tf.keras.losses.CategoricalCrossentropy):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.concat([y_pred[:, :-3], y_true], 1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

######### Main #########


def main():
    model = CustomModel()

    callback = tf.keras.callbacks.EarlyStopping(
        patience=5, mode='min', restore_best_weights=True)

    model.compile(optimizer='adam',
                  loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                        tf.keras.losses.CategoricalCrossentropy(from_logits=False)])

    xs, ys, ys2 = get_data()

    # train model
    model.fit(xs[:115, ...], (ys2[:115, ...], ys[:115, ...]), epochs=EPOCHS,
              shuffle=True, callbacks=[callback], validation_split=0.15)

    # evaluate
    model.evaluate(xs[115:, ...], (ys2[115:, ...], ys[115:, ...]))
    class_, color = model.predict(xs[115:, ...])

    # plot results
    plot_results(model, xs)


if __name__ == '__main__':
    main()
