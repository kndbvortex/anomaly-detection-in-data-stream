import os
import sys

import six

import h5py
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Activation, Conv2D, Dense, Dropout,
                          Flatten, GlobalAveragePooling1D,
                          GlobalAveragePooling2D, Input, MaxPooling2D, Permute,
                          Reshape, TimeDistributed, UpSampling1D, concatenate,
                          merge)
from keras.layers.convolutional import AveragePooling1D, Conv1D, MaxPooling1D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.utils import np_utils
# from keras.utils.training_utils import multi_gpu_model
from scipy.stats import zscore

CHANNELS_FIRST = 'channels_first'


## Simple Resnet: Adapted from Time Series Classification with deep neural networks: A strong baseline
def build_resnet_base(x, input_shape, n_feature_maps):
    fmt = 'channels_last'
    n_input_series = 1
    print('build conv_x')
    conv_x = keras.layers.normalization.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, (8, 1),
                                 padding='same',
                                 data_format=fmt)(conv_x)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, (5, 1),
                                 padding='same',
                                 data_format=fmt)(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, (3, 1),
                                 padding='same',
                                 data_format=fmt)(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, (1, 1),
                                         padding='same',
                                         data_format=fmt)(x)
        shortcut_y = keras.layers.normalization.BatchNormalization()(
            shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps * 2, (8, 1),
                                 padding='same',
                                 data_format=fmt)(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps * 2, (5, 1),
                                 padding='same',
                                 data_format=fmt)(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps * 2, (3, 1),
                                 padding='same',
                                 data_format=fmt)(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps * 2, (1, 1),
                                         padding='same',
                                         data_format=fmt)(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(
            shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps * 2, (8, 1),
                                 padding='same',
                                 data_format=fmt)(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps * 2, (5, 1),
                                 padding='same',
                                 data_format=fmt)(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps * 2, (3, 1),
                                 padding='same',
                                 data_format=fmt)(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps * 2, (1, 1),
                                         padding='same',
                                         data_format=fmt)(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(
            shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)

    print(y.shape)
    full = keras.layers.pooling.GlobalAveragePooling2D(
        name='GlobalAvgPoolFinal', data_format=fmt)(y)
    return full


def build_resnet(input_shape, n_feature_maps, num_outputs):
    ip = Input(shape=input_shape)
    x = build_resnet_base(ip, input_shape, n_feature_maps)
    out = Dense(num_outputs, activation='sigmoid')(x)
    out = Reshape((num_outputs, 1))(out)
    return Model(ip, out)
