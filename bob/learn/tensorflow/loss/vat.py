# Adapted from https://github.com/takerum/vat_tf Its license:
#
# MIT License
#
# Copyright (c) 2017 Takeru Miyato
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
from functools import partial


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), list(range(1, len(d.get_shape()))), keepdims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), list(range(1, len(d.get_shape()))), keepdims=True))
    return d


def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keepdims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keepdims=True))
    return lsm


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))


class VATLoss:
    """A class to hold parameters for Virtual Adversarial Training (VAT) Loss
    and perform it.

    Attributes
    ----------
    epsilon : float
        norm length for (virtual) adversarial training
    method : str
        The method for calculating the loss: ``vatent`` for VAT loss + entropy
        and ``vat`` for only VAT loss.
    num_power_iterations : int
        the number of power iterations
    xi : float
        small constant for finite difference
    """

    def __init__(self, epsilon=8.0, xi=1e-6, num_power_iterations=1, method='vatent', **kwargs):
        super(VATLoss, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.xi = xi
        self.num_power_iterations = num_power_iterations
        self.method = method

    def __call__(self, features, logits, architecture, mode):
        """Computes the VAT loss for unlabeled features.
        If you are doing semi-supervised learning, only pass the unlabeled
        features and their logits here.

        Parameters
        ----------
        features : object
            Tensor representing the (unlabeled) features
        logits : object
            Tensor representing the logits of (unlabeled) features.
        architecture : object
            A callable that constructs the model. It should accept ``mode`` and
            ``reuse`` as keyword arguments. The features will be given as the
            first input.
        mode : str
            One of tf.estimator.ModeKeys.{TRAIN,EVAL} strings.

        Returns
        -------
        object
            The loss.

        Raises
        ------
        NotImplementedError
            If self.method is not ``vat`` or ``vatent``.
        """
        if mode != tf.estimator.ModeKeys.TRAIN:
            return 0.
        architecture = partial(architecture, reuse=True)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            vat_loss = self.virtual_adversarial_loss(features, logits, architecture, mode)
            tf.summary.scalar("loss_VAT", vat_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, vat_loss)
            if self.method == 'vat':
                loss = vat_loss
            elif self.method == 'vatent':
                ent_loss = entropy_y_x(logits)
                tf.summary.scalar("loss_entropy", ent_loss)
                tf.add_to_collection(tf.GraphKeys.LOSSES, ent_loss)
                loss = vat_loss + ent_loss
            else:
                raise ValueError
            return loss

    def virtual_adversarial_loss(self, features, logits, architecture, mode, name="vat_loss_op"):
        r_vadv = self.generate_virtual_adversarial_perturbation(features, logits, architecture, mode)
        logit_p = tf.stop_gradient(logits)
        adversarial_input = features + r_vadv
        tf.summary.image("Adversarial_Image", adversarial_input)
        logit_m = architecture(adversarial_input, mode=mode)[0]
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return tf.identity(loss, name=name)

    def generate_virtual_adversarial_perturbation(self, features, logits, architecture, mode):
        d = tf.random_normal(shape=tf.shape(features))

        for _ in range(self.num_power_iterations):
            d = self.xi * get_normalized_vector(d)
            logit_p = logits
            logit_m = architecture(features + d, mode=mode)[0]
            dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)

        return self.epsilon * get_normalized_vector(d)
