import logger as log
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
import os
from src.cnn.convBlock2D import ConvBlock2D
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from src.parameters import *


def separated_3l_convolution_net(inputs):
    # 3 * 3 * 64 convolution stride 1
    x = layers.Conv2D(filters=32,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(inputs)
    #x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    #x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=64,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 64 convolution stride 1
    x = layers.Conv2D(filters=32,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    #x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    #x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=64,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 128 convolution stride 1
    x = layers.Conv2D(filters=64,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=64,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=128,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 128 convolution stride 1
    x = layers.Conv2D(filters=64,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=64,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=128,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 256 convolution stride 1
    x = layers.Conv2D(filters=128,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 256 convolution stride 1
    x = layers.Conv2D(filters=128,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 256 convolution stride 1
    x = layers.Conv2D(filters=128,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=512,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=512,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=512,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=512,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=512,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.Conv2D(filters=256,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    # x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    x = layers.Conv2D(filters=512,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      use_bias=USE_BIAS,
                      kernel_initializer='glorot_uniform',  # Xavier initializer
                      bias_initializer='zeros'
                      )(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    output = layers.Dense(N_CLASSES,
                           activation        = 'softmax',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    return output


# TODO KERNEL SIZE -> 3*1 1*3
def separated_2l_convolution_net(inputs):
    # 3 * 3 * 64 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=64, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(inputs)
    # 3 * 3 * 64 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=64, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 128 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=128, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 126 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=128, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 256 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 256 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 256 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    output = layers.Dense(N_CLASSES,
                           activation        = 'softmax',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    return output


def normal_convolution_net(inputs):
    # 3 * 3 * 64 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(inputs)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)

    # 3 * 3 * 64 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 128 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 128 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 256 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 256 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 256 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    output = layers.Dense(N_CLASSES,
                           activation        = 'softmax',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    return output


"""def bayesian_cnn(inputs):
    # 3 * 3 * 64 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=64, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(inputs)
    # 3 * 3 * 64 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=64, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 128 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=128, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 126 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=128, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 256 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 256 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 256 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 3 * 3 * 512 convolution stride 1
    x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=DROP_RATE, strides=1, use_bias=USE_BIAS,
                          separable_convolution=SEPARABLE_CONVOLUTION)(x)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=DROP_RATE)(x)
    output = layers.Dense(N_CLASSES,
                           activation        = 'softmax',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    return output"""


def bayesian_cnn(inputs):
    if SEPARABLE_CONVOLUTION :
        return separated_3l_convolution_net(inputs)
    else :
        return normal_convolution_net(inputs)


def compile_model(x_train, y_train, x_test, y_test):
    input_tensor = tf.keras.Input(shape=x_train[0].shape)
    bayesian_model = models.Model(input_tensor, bayesian_cnn(inputs = input_tensor))
    opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, decay = DECAY)
    bayesian_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    model_name = str(x_train[0].shape[0]) + "_" + str(N_EPOCH) + "_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) \
                 + "_" + str(DECAY) + "_" + str(DROP_RATE) + "_" + str(USE_BIAS) + "_" + str(DENSE_SIZE) + "_" \
                 + str(SEPARABLE_CONVOLUTION)
    bayesian_model.summary()
    # Save model skeleton
    if not os.path.isdir(SUMMARY_FOLDER):
        os.makedirs(SUMMARY_FOLDER)
    summary_path = os.path.join(SUMMARY_FOLDER, model_name+".txt")
    with open(summary_path, 'w') as f:
        with redirect_stdout(f):
            bayesian_model.summary()

    bayesian_train = bayesian_model.fit( x_train, y_train,
                        batch_size     = BATCH_SIZE,
                        epochs         = N_EPOCH,
                        validation_data= (x_test, y_test),
                        shuffle        = True )
    # Save model and weights
    if not os.path.isdir(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    model_path = os.path.join(MODEL_FOLDER, model_name)
    bayesian_model.save(model_path)
    log.print_info('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = bayesian_model.evaluate(x_test, y_test, verbose=1)
    log.print_info('Test loss : ' + str(scores[0]))
    log.print_info('Test accuracy : ' + str(scores[1]))

    loss = bayesian_train.history['loss']
    val_loss = bayesian_train.history['val_loss']
    epochs = range(N_EPOCH)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
