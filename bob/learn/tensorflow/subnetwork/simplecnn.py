import adanet
import functools
import tensorflow as tf

slim = tf.contrib.slim


_FEATURES_KEY = "data"
_N_CONVS_KEY = "num_convs"


class _SimpleCNNBuilder(adanet.subnetwork.Builder):
    """Builds a CNN subnetwork for AdaNet."""

    def __init__(self, optimizer, num_convs, learn_mixture_weights, seed):
        """Initializes a `_CNNBuilder`.

        Parameters
        ----------
        optimizer
            An `Optimizer` instance for training both the subnetwork and
            the mixture weights.
        num_convs
            The number of hidden layers.
        learn_mixture_weights
            Whether to solve a learning problem to find the best mixture
            weights, or use their default value according to the mixture weight
            type. When `False`, the subnetworks will return a no_op for the
            mixture weight train op.
        seed
            A random seed.
        """

        self._optimizer = optimizer
        self._n_convs = num_convs
        self._learn_mixture_weights = learn_mixture_weights
        self._seed = seed

    def build_subnetwork(
        self,
        features,
        logits_dimension,
        training,
        iteration_step,
        summary,
        previous_ensemble=None,
    ):
        """See `adanet.subnetwork.Builder`."""

        # input layer
        graph = tf.to_float(features[_FEATURES_KEY])
        # kernel_initializer = tf.glorot_uniform_initializer(seed=self._seed)
        # initializer = tf.contrib.layers.xavier_initializer(seed=self._seed)

        batch_norm_params = {
            # Decay for the moving averages.
            "decay": 0.995,
            # epsilon to prevent 0s in variance.
            "epsilon": 0.001,
            # force in-place updates of mean and variance estimates
            "updates_collections": None,
        }

        weight_decay = 5e-5
        end_points = {}

        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
        ), tf.variable_scope("SimpleCNN", reuse=False), slim.arg_scope(
            [slim.batch_norm, slim.dropout], is_training=training
        ):

            for i in range(self._n_convs):

                name = "conv{}".format(i)
                graph = slim.conv2d(
                    graph, 32, (5, 5), activation_fn=tf.nn.relu, stride=1, scope=name
                )
                end_points[name] = graph

                name = "pool{}".format(i)
                graph = slim.max_pool2d(graph, [2, 2], scope=name)
                end_points[name] = graph

            graph = slim.flatten(graph, scope="flatten")
            end_points["flatten"] = graph

            name = "dense1"
            graph = slim.fully_connected(
                graph, 128, activation_fn=tf.nn.relu, scope=name
            )
            end_points[name] = graph

            name = "dense2"
            graph = slim.fully_connected(
                graph, 32, activation_fn=tf.nn.relu, scope=name
            )
            end_points[name] = graph

            name = "dropout"
            graph = slim.dropout(graph, 0.6, scope="Dropout")
            end_points[name] = graph

        name = "logits"
        logits = slim.fully_connected(
            graph,
            logits_dimension,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=None,
            scope=name,
        )
        end_points[name] = logits

        shared = {_N_CONVS_KEY: self._n_convs}
        return adanet.Subnetwork(
            last_layer=graph,
            logits=logits,
            complexity=self._measure_complexity(),
            shared=shared,
        )

    def _measure_complexity(self):
        """Approximates Rademacher complexity as the square-root of the
        depth."""
        return tf.sqrt(tf.to_float(self._n_convs))

    def build_subnetwork_train_op(
        self,
        subnetwork,
        loss,
        var_list,
        labels,
        iteration_step,
        summary,
        previous_ensemble,
    ):
        """See `adanet.subnetwork.Builder`."""
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(
        self, loss, var_list, logits, labels, iteration_step, summary
    ):
        """See `adanet.subnetwork.Builder`."""

        if not self._learn_mixture_weights:
            return tf.no_op()
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""

        if self._n_convs == 0:
            # A CNN with no hidden layers is a linear model.
            return "dnn"
        return "{}_layer_cnn".format(self._n_convs)


class SimpleCNNGenerator(adanet.subnetwork.Generator):
    """Generates a two CNN subnetworks at each iteration.

    The first CNN has an identical shape to the most recently added subnetwork
    in `previous_ensemble`. The second has the same shape plus one more dense
    layer on top. This is similar to the adaptive network presented in Figure 2
    of [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
    connections to hidden layers of networks from previous iterations.
    """

    def __init__(self, optimizer, learn_mixture_weights=False, seed=None):
        """Initializes a CNN `Generator`.

        Parameters
        ----------
        optimizer
            An `Optimizer` instance for training both the subnetwork and
            the mixture weights.
        learn_mixture_weights
            Whether to solve a learning problem to find the best mixture
            weights, or use their default value according to the mixture weight
            type. When `False`, the subnetworks will return a no_op for the
            mixture weight train op.
        seed
            A random seed.
        """

        self._seed = seed
        self._cnn_builder_fn = functools.partial(
            _SimpleCNNBuilder,
            optimizer=optimizer,
            learn_mixture_weights=learn_mixture_weights,
        )

    def generate_candidates(
        self,
        previous_ensemble,
        iteration_number,
        previous_ensemble_reports,
        all_reports,
    ):
        """See `adanet.subnetwork.Generator`."""

        num_convs = 1
        seed = self._seed

        if previous_ensemble:
            num_convs = previous_ensemble.weighted_subnetworks[-1].subnetwork.shared[
                _N_CONVS_KEY
            ]
            print(f"Found a previous example with {num_convs} samples")

        if seed is not None:
            seed += iteration_number

        return [
            self._cnn_builder_fn(num_convs=num_convs, seed=seed),
            self._cnn_builder_fn(num_convs=num_convs + 1, seed=seed),
        ]
