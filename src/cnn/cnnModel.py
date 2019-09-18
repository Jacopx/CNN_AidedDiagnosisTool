import datetime
import os
import tensorflow as tf
import numpy as np
from src.cnn.convBlock2D import ConvBlock2D


class CnnModel:
    def __init__(self,
                 data_provider,
                 n_epochs,
                 batch_size,
                 input_shape,
                 learning_rate=0.1,
                 drop_rate=0.2,
                 use_bias=False,
                 dense_size=1024,
                 logdir=None,
                 separable_convolution=False
                 ):
        self.data_provider=data_provider
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.use_bias=use_bias
        self.dense_size=dense_size
        if logdir is None:
            self.logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.logdir = os.path.join("logs", logdir)
        self.separable_convolution = separable_convolution
        self._build_model()

    def _build_model(self):
        """
        self.model = tf.keras.models.Sequential()
        self.model.add(ConvBlock2D(kernel_size=3, filters=64, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=64, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
        self.model.add(ConvBlock2D(kernel_size=3, filters=128, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=128, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
        self.model.add(ConvBlock2D(kernel_size=3, filters=256, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=256, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=256, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
        self.model.add(ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
        self.model.add(ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                                   separable_convolution=self.separable_convolution))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
        self.model.add(tf.keras.layers.Dense(self.dense_size,
                                             activation='relu',
                                             use_bias=self.use_bias,
                                             kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
        self.model.add(tf.keras.layers.Dropout(rate=self.drop_rate))
        self.model.add(tf.keras.layers.Dense(self.dense_size,
                                             activation='relu',
                                             use_bias=self.use_bias,
                                             kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
        self.model.add(tf.keras.layers.Dropout(rate=self.drop_rate))
        self.model.add(tf.keras.layers.Dense(self.dense_size,
                                             activation='relu',
                                             use_bias=self.use_bias,
                                             kernel_initializer='glorot_uniform',
                                             bias_initializer='zeros'))
        """
        inputs = tf.keras.Input(shape=(10,))
        x = ConvBlock2D(kernel_size=3, filters=64, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(inputs)
        x = ConvBlock2D(kernel_size=3, filters=64, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
        x = ConvBlock2D(kernel_size=3, filters=128, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = ConvBlock2D(kernel_size=3, filters=128, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
        x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = ConvBlock2D(kernel_size=3, filters=256, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
        x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
        x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = ConvBlock2D(kernel_size=3, filters=512, drop_rate=self.drop_rate, strides=1, use_bias=self.use_bias,
                        separable_convolution=self.separable_convolution)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Dense(3)(x)
        outputs = tf.keras.layers.Dropout(0.5)(x, training=True)
