import os
import json
import time
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from datetime import timedelta


class ShallowNetMcd(CnnV1):

    def _init(self, *args, **kwargs):

        # Peculiar attributes
        # >>>>>>>>>>>>>>>>>>>>
        self.model = None
        self.draws = self._setter(1, kwargs.get("draws"))
        self.weights_init = self._setter(False, kwargs.get("weights_init"))

        # Learing rate related
        self.learning_rate_patience = self._setter(self.learning_rate_patience, kwargs.get("learing_rate_patience"))
        self.learning_strategy = self._setter(self.learning_strategy, kwargs.get("learning_strategy"))
        self.learning_rate_reduction_factor = self._setter(self.learning_rate_reduction_factor,
                                                           kwargs.get("learning_rate_reduction_factor"))

        # Optimizer and weigths related
        self.nesterov_momentum = self._setter(self.nesterov_momentum, kwargs.get("nesterov_momentum"))
        self.adam_epsilon = self._setter(self.adam_epsilon, kwargs.get("adam_epsilon"))

        # Others
        self.num_inter_threads = self._setter(self.num_inter_threads, kwargs.get("num_inter_threads"))
        self.num_intra_threads = self._setter(self.num_intra_threads, kwargs.get("num_intra_threads"))
        self.computing_device = self._setter(self.computing_device, kwargs.get("computing_device"))
        self.rate = self._setter(self.rate, kwargs.get("rate"))

        # Init methods
        self._define_inputs()  # Definito in MyNet
        self._build_graph()  # Class specific
        self._define_loss_and_optimizer()  # Class specific
        self._define_summaries()  # Definito in MyNet
        self._initialize_session()  # Definito in MyNet
        self._count_trainable_params()  # Definito in MyNet

    def _build_graph(self):
        """
        For the normal distribution, the location and scale parameters correspond
        to the mean and standard deviation, respectively.
        However, this is not necessarily true for other distributions.
        """

        inputs = tf.keras.Input(shape=(self._data_shape[0], self._data_shape[1], self._data_shape[2]))

        ...  # Dropout model

        neural_net = tf.keras.Model(inputs=inputs, outputs=outputs)

        logits = neural_net(self.images)

        self.model = neural_net
        self.logits = logits
        self.last_fc_before_softmax = self.logits  # Exists for compatibility

    def compute_predictions_probs(self,
                                  data_provider,
                                  batch_size,
                                  draws):

        num_examples = data_provider.num_examples
        predicted_class_variances_list = list()
        predictions_list = list()
        aleatoric_list = list()
        epistemic_list = list()
        predictive_variances_list = list()
        labels_list = list()
        images_list = list()

        for i in range(int(np.ceil(num_examples / batch_size))):
            batch = data_provider.next_batch(batch_size)
            images, labels = batch
            predicted_class_variances, predictions, aleatoric, epistemic, predictive_variances = \
                self._compute_predictions_probs(images,
                                                labels,
                                                draws)

            predicted_class_variances_list += predicted_class_variances.tolist()
            aleatoric_list += aleatoric.tolist()
            epistemic_list += epistemic.tolist()
            predictive_variances_list += predictive_variances.tolist()
            predictions_list += predictions.tolist()
            labels_list += labels.tolist()
            images_list += images.tolist()

        output = dict()
        output['predicted_class_variances'] = np.asarray(predicted_class_variances_list)
        output['labels'] = np.asarray(labels_list)
        output['epistemic'] = np.asarray(epistemic_list)
        output['aleatoric'] = np.asarray(aleatoric_list)
        output['predictive_variances'] = np.asarray(predictive_variances_list)
        output['predictions'] = np.asarray(predictions_list)
        output['images'] = np.asarray(images_list)

        return output

    def _compute_predictions_probs(self,
                                   image,
                                   label,
                                   draws):

        p_hat = np.asarray([self.sess.run(self.predictions,
                                          feed_dict={self.images: image,
                                                     self.labels: label,
                                                     self.is_training: False})
                            for _ in range(draws)])

        mean_probs_over_draw = np.mean(p_hat, axis=0)  # Media sui Montecarlo samples
        predictions = np.argmax(mean_probs_over_draw, axis=1)

        aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
        epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
        predictive_variances = np.var(p_hat, axis=0)

        uncertainties_among_labels = epistemic + aleatoric
        predicted_class_variances = np.asarray([uncertainty[prediction] for prediction, uncertainty in
                                               zip(predictions, uncertainties_among_labels)])

        return predicted_class_variances, predictions, aleatoric, epistemic, predictive_variances
