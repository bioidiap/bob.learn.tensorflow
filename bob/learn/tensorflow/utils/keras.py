import tensorflow.keras.backend as K
from .network import is_trainable
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def keras_channels_index():
    return -3 if K.image_data_format() == "channels_first" else -1


def keras_model_weights_as_initializers_for_variables(model):
    """Changes the initialization operations of variables in the model to take the
    current value as the initial values.
    This is useful when you want to restore a pre-trained Keras model inside the
    model_fn of an estimator.

    Parameters
    ----------
    model : object
        A Keras model.
    """
    sess = K.get_session()
    n = len(model.variables)
    logger.debug("Initializing %d variables with their current weights", n)
    for variable in model.variables:
        value = variable.eval(sess)
        initial_value = tf.constant(value=value, dtype=value.dtype.name)
        variable._initializer_op = variable.assign(initial_value)
        variable._initial_value = initial_value


def apply_trainable_variables_on_keras_model(model, trainable_variables, mode):
    """Changes the trainable status of layers in a keras model.
    It can only turn off the trainable status of layer.

    Parameters
    ----------
    model : object
        A Keras model
    trainable_variables : list or None
        See bob.learn.tensorflow.estimators.Logits
    mode : str
        One of tf.estimator.ModeKeys
    """
    for layer in model.layers:
        trainable = is_trainable(layer.name, trainable_variables, mode=mode)
        if layer.trainable:
            layer.trainable = trainable


def restore_model_variables_from_checkpoint(model, checkpoint, session=None):
    if session is None:
        session = tf.keras.backend.get_session()

    # removes duplicates
    var_list = set(model.variables)
    assert len(var_list)
    saver = tf.train.Saver(var_list=var_list)
    ckpt_state = tf.train.get_checkpoint_state(checkpoint)
    logger.info("Loading checkpoint %s", ckpt_state.model_checkpoint_path)
    saver.restore(session, ckpt_state.model_checkpoint_path)


def initialize_model_from_checkpoint(model, checkpoint, normalizer=None):
    if normalizer is None:
        def normalizer(name):
            return name.split(":")[0]
    assignment_map = {normalizer(v.name): v for v in model.variables}
    assert len(assignment_map)
    tf.train.init_from_checkpoint(checkpoint, assignment_map=assignment_map)
