import logger as log
from tensorflow.keras import layers
from tensorflow.keras import models
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from contextlib import redirect_stdout
from src.parameters import *


def normal_convolution_net(inputs, drop_rate):
    x = layers.BatchNormalization
    # 3 * 3 * 64 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(inputs)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)

    # 3 * 3 * 64 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 128 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 3 * 3 * 128 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 256 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 3 * 3 * 256 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 3 * 3 * 256 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 3 * 3 * 512 convolution stride 1
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='valid', activation='relu',
                      use_bias=USE_BIAS, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    # 2 * 2 max pooling stride 2
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    x = layers.Dense(DENSE_SIZE,
                           activation        = 'relu',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    x = layers.Dropout(rate=drop_rate)(x, training=True)
    output = layers.Dense(N_CLASSES,
                           activation        = 'softmax',
                           use_bias          = USE_BIAS,
                           kernel_initializer= 'glorot_uniform',
                           bias_initializer  = 'zeros')(x)
    return output


def bayesian_cnn(inputs, drop_rate):
    return normal_convolution_net(inputs, drop_rate)


def compile_model(x_train, y_train, x_test, y_test, drop_rate):

    if tf.test.is_built_with_cuda:
        if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
            log.print_error("MAYBE GPU IS USED")
        else:
            log.print_warning("NO GPU IS USED")
    else:
        log.print_warning("THIS VERSION OF TENSORFLOW DOES NOT USES CUDA")
    input_tensor = tf.keras.Input(shape=x_train[0].shape)
    bayesian_model = models.Model(input_tensor, bayesian_cnn(inputs = input_tensor, drop_rate=drop_rate))
    opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, decay = DECAY)
    bayesian_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    model_name = str(x_train[0].shape[0]) + "_" + str(N_EPOCH) + "_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) \
                 + "_" + str(DECAY) + "_" + str(drop_rate) + "_" + str(USE_BIAS) + "_" + str(DENSE_SIZE) + "_" \
                 + str(SEPARABLE_CONVOLUTION) + "_local.h5"
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
    bayesian_model.save_weights(model_path)
    log.print_info('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = bayesian_model.evaluate(x_test, y_test, verbose=1)
    log.print_info('Test loss : ' + str(scores[0]))
    log.print_info('Test accuracy : ' + str(scores[1]))

    """loss = bayesian_train.history['loss']
    val_loss = bayesian_train.history['val_loss']
    epochs = range(N_EPOCH)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()"""


def predict_from_model(batch_to_predict, drop_rate):
    model_name = "224_10_32_0.0001_1e-06_0.1_True_2048_False_local.h5"
    model_path = os.path.join(MODEL_FOLDER, model_name)
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    bayesian_model = models.Model(input_tensor, bayesian_cnn(inputs=input_tensor, drop_rate=drop_rate))
    bayesian_model.load_weights(model_path, by_name=False)
    log.print_info('Model loaded')
    print(str(batch_to_predict.shape))
    #return bayesian_model.predict(batch_to_predict, batch_to_predict.shape[0]);
    prediction_list = []
    i = 0
    j = 0
    k = 0
    div = min(batch_to_predict.shape[0],40)
    for i in range(0, batch_to_predict.shape[0]):
        prediction_list.append(bayesian_model.predict(batch_to_predict[i:i+1], 1))
        temp = int(batch_to_predict.shape[0]/div)
        if (i % temp) == 0:
            print("", end="\r")
            print("Prediction status: ", end = '')
            for j in range (0,int(i*div/batch_to_predict.shape[0])):
                print("#", end = '')
            for k in range(j, div):
                print("=", end = '')
            perc = int(i * 100 / batch_to_predict.shape[0])
            print(" "+str(perc)+" %", end = '')
    print("", end="\r")
    print("Prediction status: ", end='')
    for k in range(0, div):
        print("#", end='')
    print(" 100%")
    return prediction_list


def load_model(model_name, drop_rate):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    bayesian_model = models.Model(input_tensor, bayesian_cnn(inputs=input_tensor, drop_rate=drop_rate))
    bayesian_model.load_weights(model_path, by_name=False)
    log.print_info("Loaded model "+model_name)
    return bayesian_model


def predict_from_model_multithread(batch_to_predict, drop_rate):
    bayesian_model = load_model("224_10_32_0.0001_1e-06_"+str(drop_rate)+"_True_2048_False_local.h5", drop_rate)
    predictions = bayesian_model.predict(batch_to_predict, batch_size=BATCH_SIZE, verbose=1, workers=100, use_multiprocessing=True)
    del bayesian_model
    return predictions
