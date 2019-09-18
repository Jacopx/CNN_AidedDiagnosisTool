import os
import json
import time
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from datetime import timedelta


class CnnV1:
    """
    Abstract class for CNNs in tensorflow.
    """

    def __init__(self,
                 data_provider,
                 epochs,
                 batch_size,
                 initial_learning_rate=0.1,
                 optimizer_name="SGD-Momentum",
                 computing_device=None,
                 root_dirpath_logs_save="tmp"):

        # Mandatory arguments
        self.data_provider = data_provider
        self.epochs = epochs
        self.batch_size = batch_size

        self.initial_learning_rate = initial_learning_rate
        self.optimizer_name = optimizer_name

        # Internal use (not modified by subclass)
        self._data_shape = self.data_provider.data_shape
        self._n_classes = self.data_provider.n_classes
        self._batches_step = 0
        self._epoch_counter = 0
        self._merged_summaries_op = None
        self._total_training_steps = 0

        # Arguments set by subclass in _build_graph method, called in subclass _init()
        self.optimizer = None
        self.l2_regularization = None
        self.train_operation = None
        self.loss = None
        self.accuracy = None
        self.predictions = None
        self.last_fc_before_softmax = None
        self.show_weight_posteriors = False  # For bayesian CNN

        # Parameters optionally overwritten by child class
        self.learning_rate_patience = 10
        self.learning_strategy = "fixed"
        self.learning_rate_reduction_factor = 0.1
        self.nesterov_momentum = 0.9
        self.weight_decay = 1e-4
        self.adam_epsilon = 1
        self.num_inter_threads = 1
        self.num_intra_threads = 1
        self.computing_device = computing_device
        self.beta = 1
        self.rate = 0.5

        self.logits = None

        # Outputs related
        self.dirpath_logs = root_dirpath_logs_save
        self.dirpath_save = root_dirpath_logs_save
        self.dirpath_logs = os.path.join(root_dirpath_logs_save, 'logs')
        self.dirpath_save = os.path.join(root_dirpath_logs_save, 'saves')

        self.training_stats = dict()

        self._print_parameter_value("Training epochs", self.epochs)
        self._print_parameter_value("Batch size", self.batch_size)
        self._print_parameter_value("Optimizer", self.optimizer_name)
        self._print_parameter_value("Initial learning rate", self.initial_learning_rate)

    @staticmethod
    def _print_parameter_value(parameter_name, parameter_value):
        string = "{} : {}".format(parameter_name, parameter_value)
        print('-'*len(string)+"\n{}".format(string)+'\n'+'-'*len(string))

    @staticmethod
    def _setter(attribute_value, keyword_parameter):
        if keyword_parameter:
            return keyword_parameter
        else:
            return attribute_value

    def init(self, *args, **kwargs):
        self._init(*args, **kwargs)

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

    def _save_params(self):
        file_path = os.path.join(self.dirpath_logs, "params.json")
        d = dict()
        d['learning_rate'] = self.initial_learning_rate
        d['optimizer_name'] = self.optimizer_name
        d['learning_strategy'] = self.learning_strategy
        d['learning_rate_patience'] = self.learning_rate_patience
        d['learning_rate_reduction_factor'] = self.learning_rate_reduction_factor
        d['_n_classes'] = self._n_classes
        d['batch_size'] = self.batch_size
        d['epochs'] = self.epochs

        with open(file_path, 'w') as f:
            json.dump(d, f)

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        if self.computing_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.computing_device

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        # Specify the CPU inter and Intra threads used by MKL
        config.intra_op_parallelism_threads = self.num_intra_threads
        config.inter_op_parallelism_threads = self.num_inter_threads

        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init_operation = tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer())
        self.sess.run(init_operation)
        logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_train_writer = logswriter(os.path.join(self.logs_path, "training"))
        self.summary_validation_writer = logswriter(os.path.join(self.logs_path, "validation"))

    def _define_inputs(self):
        shape = [None]
        shape.extend(self._data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self._n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def _define_loss_and_optimizer(self):

        prediction = tf.nn.softmax(self.logits)

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.labels))
        self.loss = cross_entropy
        self.l2_regularization = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # Optimizer
        self.optimizer = self.get_optimizer(self.optimizer_name)
        if self.weight_decay:
            self.train_operation = self.optimizer.minimize(
                self.loss + self.l2_regularization * self.weight_decay)
        else:
            self.train_operation = self.optimizer.minimize(
                self.loss + self.l2_regularization)

        # Evaluation metrics
        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.predictions = tf.cast(prediction, tf.float32)

    def _define_summaries(self):
        with tf.name_scope("Performances"):
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("learning_rate", self.learning_rate)

        with tf.name_scope("Weights"):
            str_print = "Trainable parameters:"
            print('-'*len(str_print))
            print(str_print)
            print('-'*len(str_print))
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for grad, var in grads_and_vars:
                print(var.op.name)
                if "kernel" in var.op.name:
                    mean = tf.reduce_mean(tf.abs(grad))
                    tf.summary.scalar("mean_{}".format(var.op.name), mean)
                    tf.summary.histogram("histogram_{}".format(var.op.name), grad)
                    tf.summary.histogram("hist_weights_{}".format(var.op.name), grad)

        self._merged_summaries_op = tf.summary.merge_all()

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        self.total_parameters = total_parameters
        string = "Total training params: {:.3}M".format(total_parameters / 1e6)
        print('-'*len(string)+"\n{}".format(string)+'\n'+'-'*len(string))

    @property
    def save_path(self):
        os.makedirs(self.dirpath_save, exist_ok=True)
        save_path = os.path.join(self.dirpath_save, 'model.chkpt')
        return save_path

    @property
    def logs_path(self):
        os.makedirs(self.dirpath_logs, exist_ok=True)
        return self.dirpath_logs

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    @staticmethod
    def log_loss_accuracy(self, loss, accuracy, *kargs, **kwargs):
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))

    @staticmethod
    def weight_variable_msra(shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    @staticmethod
    def weight_variable_xavier(shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def train_all_epochs(self):

        lr_reduction_metrics = list()

        self._save_params()

        training_stats = dict()
        training_stats['acc'] = list()
        training_stats['epochs'] = list()
        training_stats['loss'] = list()
        training_stats['training_params'] = self.total_parameters
        training_stats['per_epoch_time'] = list()

        total_start_time = time.time()

        if self.data_provider.train is None:
            raise ValueError("Warning!! Train is None")

        learning_rate = self.initial_learning_rate

        for epoch in range(1, self.epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()

            learning_rate = self._learning_rate_management(epoch,
                                                           learning_rate,
                                                           lr_reduction_metrics)
            print("Training...")

            loss, acc = self.train_one_epoch(
                self.data_provider.train, epoch, self.batch_size, learning_rate)
            self.log_loss_accuracy(loss, acc)

            if self.data_provider.validation:
                print("Validation...")
                loss, acc, _, _ = self.test(self.data_provider.validation, self.batch_size, epoch)
                self.log_loss_accuracy(loss, acc)
                lr_reduction_metrics.append(acc)

            time_per_epoch = time.time() - start_time
            seconds_left = int((self.epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            training_stats['epochs'].append(epoch)
            training_stats['acc'].append(acc)
            training_stats['loss'].append(loss)
            training_stats['per_epoch_time'].append(time_per_epoch)

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))
        print("Saving model in {}".format(self.save_path))
        self.save_model()
        return training_stats

    def train_one_epoch(self, data, epoch, batch_size, learning_rate):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        summaries = None
        for i in range(num_examples // batch_size):

            self._total_training_steps += 1

            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_operation, self.loss, self.accuracy, self._merged_summaries_op]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy, summaries = result

            total_loss.append(loss)
            total_accuracy.append(accuracy)

        if summaries is not None:
            self.summary_train_writer.add_summary(summaries, epoch)
        else:
            raise ValueError("None summaries in train_one_epoch")

        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size, epoch=None):

        num_examples = data.num_examples
        total_loss = list()
        total_accuracy = list()
        prediction = list()
        activation_before_softmax = list()

        for i in range(int(np.ceil(num_examples / batch_size))):
            batch = data.next_batch(batch_size)

            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }

            fetches = [self.loss, self.accuracy, self.predictions, self.last_fc_before_softmax]
            loss, accuracy, prediction_one_hot, act_before_softmax = self.sess.run(fetches, feed_dict=feed_dict)
            prediction += prediction_one_hot.tolist()
            activation_before_softmax += act_before_softmax.tolist()

            total_loss.append(loss)
            total_accuracy.append(accuracy)

        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)

        if epoch:
            with tf.name_scope("Validation_performances"):
                summary = tf.Summary()
                summary.value.add(tag="accuracy val", simple_value=mean_accuracy)
                summary.value.add(tag="loss val", simple_value=mean_loss)

                self.summary_validation_writer.add_summary(summary, epoch)

        return mean_loss, mean_accuracy, prediction, activation_before_softmax

    def get_optimizer(self, optimizer_name):
        return {
            "Adam": tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           name="Adam"),
            "SGD-Momentum": tf.train.MomentumOptimizer(self.learning_rate,
                                                       self.nesterov_momentum,
                                                       use_nesterov=True,
                                                       name="SGD-Momentum")
        }.get(optimizer_name)

    def _learning_rate_management(self,
                                  epoch,
                                  current_lr,
                                  metric=None):

        if self.learning_strategy == "two_times":
            reduce_lr_epoch_1 = int(self.epochs / 2)
            reduce_lr_epoch_2 = int(self.epochs * 0.75)
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                current_lr = current_lr * 0.1
                print("Decrease learning rate, new lr = %f" % current_lr)

        elif self.learning_strategy == "acc_plateau" and metric is not None:
            current_lr = self._reduce_lr_on_plateau(metric, current_lr, self.learning_rate_patience)

        return current_lr

    def _reduce_lr_on_plateau(self,
                              metric,
                              current_lr,
                              patience):
        min_delta = 0.01
        self._epoch_counter += 1
        new_lr = current_lr
        if self._epoch_counter > patience:
            last_value = metric[-1]
            last_n_values = metric[-patience:-1]
            reduce_lr = True

            for value in last_n_values:
                if last_value > value + min_delta:
                    reduce_lr = False

            if reduce_lr:
                print("Decrease learning rate due to val acc plateau, new lr = {}".format(
                    current_lr * self.learning_rate_reduction_factor))
                self._epoch_counter = 0

                new_lr = current_lr * self.learning_rate_reduction_factor

        return new_lr
