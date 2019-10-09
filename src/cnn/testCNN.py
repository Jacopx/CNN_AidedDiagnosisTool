import logger as log
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
import os
from src.cnn.convBlock2D import ConvBlock2D
import matplotlib.pyplot as plt

INPUT_SHAPE = (int(224), int(224), int(3))
N_CLASSES = 3
N_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DECAY = 1e-6
DROP_RATE = 0.05
USE_BIAS = False
DENSE_SIZE = 1024
SEPARABLE_CONVOLUTION = True
MODEL_FOLDER = os.path.join("resources", "models")


def bayesian_cnn(inputs):
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


def compile_model(x_train, y_train, x_test, y_test):
    input_tensor = tf.keras.Input(shape=INPUT_SHAPE)
    bayesian_model = models.Model(input_tensor, bayesian_cnn(inputs = input_tensor))
    opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, decay = DECAY)
    bayesian_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    bayesian_model.summary()
    bayesian_train = bayesian_model.fit(  x_train, y_train,
                                          batch_size     = BATCH_SIZE,
                                          epochs         = N_EPOCH,
                                          validation_data= (x_test, y_test),
                                          shuffle        = True)

    model_name = str(N_EPOCH) + "_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) + "_" + str(DECAY) + "_" + str(
        DROP_RATE) + "_" + str(USE_BIAS) + "_" + str(DENSE_SIZE) + "_" + str(SEPARABLE_CONVOLUTION)
    # Save model and weights
    if not os.path.isdir(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    model_path = os.path.join(MODEL_FOLDER, model_name)
    bayesian_train.save(model_path)
    log.print_info('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = bayesian_train.evaluate(x_test, y_test, verbose=1)
    log.print_info('Test loss:', scores[0])
    log.print_info('Test accuracy:', scores[1])

    loss = bayesian_train.history['loss']
    val_loss = bayesian_train.history['val_loss']
    epochs = range(N_EPOCH)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
