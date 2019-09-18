from tensorflow.keras import layers


class ConvBlock2D(layers.Layer):

    def __init__(self,
                 kernel_size,  # K
                 filters,  # N = Number of kernels
                 drop_rate,
                 strides=1,  # S
                 use_bias=False,
                 separable_convolution=False
                 ):
        self.kernel_size = kernel_size
        self.filters = filters
        self.drop_rate = drop_rate
        self.strides = strides
        self.use_bias=use_bias
        self.separable_convolution = separable_convolution

    def call(self, input_tensor):
        if not self.separable_convolution:
            x = layers.ZeroPadding2D(padding=1)(input_tensor)
            x = layers.Conv2D(filters=self.filters,
                              kernel_size=self.kernel_size,
                              strides=1,
                              padding='valid',
                              activation='relu',
                              use_bias=self.use_bias,
                              kernel_initializer='glorot_uniform',  # Xavier initializer
                              bias_initializer='zeros'
                              )(x)
            return layers.Dropout(rate=self.drop_rate)(x, training=True)
        else:
            x = layers.Conv2D(filters=self.filters/2,
                              kernel_size=1,
                              strides=1,
                              padding='valid',
                              activation='relu',
                              use_bias=self.use_bias,
                              kernel_initializer='glorot_uniform',  # Xavier initializer
                              bias_initializer='zeros'
                              )(input_tensor)
            x = layers.Dropout(rate=self.drop_rate)(x, training=True)
            x = layers.Conv2D(filters=self.filters/2,
                              kernel_size=self.kernel_size,
                              strides=1,
                              padding='valid',
                              activation='relu',
                              use_bias=self.use_bias,
                              kernel_initializer='glorot_uniform',  # Xavier initializer
                              bias_initializer='zeros'
                              )(x)
            x = layers.ZeroPadding2D(padding=1)(x)
            x = layers.Dropout(rate=self.drop_rate)(x, training=True)
            x = layers.Conv2D(filters=self.filters,
                              kernel_size=1,
                              strides=1,
                              padding='valid',
                              activation='relu',
                              use_bias=self.use_bias,
                              kernel_initializer='glorot_uniform',  # Xavier initializer
                              bias_initializer='zeros'
                              )(x)
            return layers.Dropout(rate=self.drop_rate)(x, training=True)


