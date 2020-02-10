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


def _create_var_map(variables, normalizer=None):
    if normalizer is None:

        def normalizer(name):
            return name.split(":")[0]

    assignment_map = {normalizer(v.name): v for v in variables}
    assert len(assignment_map)
    return assignment_map


def restore_model_variables_from_checkpoint(
    model, checkpoint, session=None, normalizer=None
):
    if session is None:
        session = tf.keras.backend.get_session()

    var_list = _create_var_map(model.variables, normalizer=normalizer)
    saver = tf.train.Saver(var_list=var_list)
    ckpt_state = tf.train.get_checkpoint_state(checkpoint)
    logger.info("Loading checkpoint %s", ckpt_state.model_checkpoint_path)
    saver.restore(session, ckpt_state.model_checkpoint_path)


def initialize_model_from_checkpoint(model, checkpoint, normalizer=None):
    assignment_map = _create_var_map(model.variables, normalizer=normalizer)
    tf.train.init_from_checkpoint(checkpoint, assignment_map=assignment_map)


def model_summary(model, do_print=False):
    try:
        from tensorflow.python.keras.utils.layer_utils import count_params
    except ImportError:
        from tensorflow_core.python.keras.utils.layer_utils import count_params
    nest = tf.nest

    if model.__class__.__name__ == "Sequential":
        sequential_like = True
    elif not model._is_graph_network:
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(nest.flatten(v[0].inbound_layers)) > 1
            ):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        # header names for the different log elements
        to_display = ["Layer (type)", "Details", "Output Shape", "Number of Parameters"]
    else:
        # header names for the different log elements
        to_display = [
            "Layer (type)",
            "Details",
            "Output Shape",
            "Number of Parameters",
            "Connected to",
        ]
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    rows = [to_display]

    def print_row(fields):
        for i, v in enumerate(fields):
            if isinstance(v, int):
                fields[i] = f"{v:,}"
        rows.append(fields)

    def layer_details(layer):
        cls_name = layer.__class__.__name__
        details = []
        if "Conv" in cls_name and "ConvBlock" not in cls_name:
            details += [f"filters={layer.filters}"]
            details += [f"kernel_size={layer.kernel_size}"]

        if "Pool" in cls_name and "Global" not in cls_name:
            details += [f"pool_size={layer.pool_size}"]

        if (
            "Conv" in cls_name
            and "ConvBlock" not in cls_name
            or "Pool" in cls_name
            and "Global" not in cls_name
        ):
            details += [f"strides={layer.strides}"]

        if (
            "ZeroPad" in cls_name
            or cls_name in ("Conv1D", "Conv2D", "Conv3D")
            or "Pool" in cls_name
            and "Global" not in cls_name
        ):
            details += [f"padding={layer.padding}"]

        if "Cropping" in cls_name:
            details += [f"cropping={layer.cropping}"]

        if cls_name == "Dense":
            details += [f"units={layer.units}"]

        if cls_name in ("Conv1D", "Conv2D", "Conv3D") or cls_name == "Dense":
            act = layer.activation.__name__
            if act != "linear":
                details += [f"activation={act}"]

        if cls_name == "Dropout":
            details += [f"drop_rate={layer.rate}"]

        if cls_name == "Concatenate":
            details += [f"axis={layer.axis}"]

        if cls_name == "Activation":
            act = layer.get_config()["activation"]
            details += [f"activation={act}"]

        if "InceptionModule" in cls_name:
            details += [f"b1_c1={layer.filter_1x1}"]
            details += [f"b2_c1={layer.filter_3x3_reduce}"]
            details += [f"b2_c2={layer.filter_3x3}"]
            details += [f"b3_c1={layer.filter_5x5_reduce}"]
            details += [f"b3_c2={layer.filter_5x5}"]
            details += [f"b4_c1={layer.pool_proj}"]

        if cls_name == "LRN":
            details += [f"depth_radius={layer.depth_radius}"]
            details += [f"alpha={layer.alpha}"]
            details += [f"beta={layer.beta}"]

        if cls_name == "ConvBlock":
            details += [f"filters={layer.num_filters}"]
            details += [f"bottleneck={layer.bottleneck}"]
            details += [f"dropout_rate={layer.dropout_rate}"]

        if cls_name == "DenseBlock":
            details += [f"layers={layer.num_layers}"]
            details += [f"growth_rate={layer.growth_rate}"]
            details += [f"bottleneck={layer.bottleneck}"]
            details += [f"dropout_rate={layer.dropout_rate}"]

        if cls_name == "TransitionBlock":
            details += [f"filters={layer.num_filters}"]

        if cls_name == "InceptionA":
            details += [f"pool_filters={layer.pool_filters}"]

        if cls_name == "InceptionResnetBlock":
            details += [f"block_type={layer.block_type}"]
            details += [f"scale={layer.scale}"]
            details += [f"n={layer.n}"]

        if cls_name == "ReductionA":
            details += [f"k={layer.k}"]
            details += [f"kl={layer.kl}"]
            details += [f"km={layer.km}"]
            details += [f"n={layer.n}"]

        if cls_name == "ReductionB":
            details += [f"k={layer.k}"]
            details += [f"kl={layer.kl}"]
            details += [f"km={layer.km}"]
            details += [f"n={layer.n}"]
            details += [f"no={layer.no}"]
            details += [f"p={layer.p}"]
            details += [f"pq={layer.pq}"]

        if cls_name == "ScaledResidual":
            details += [f"scale={layer.scale}"]

        return ", ".join(details)

    def print_layer_summary(layer):
        """Prints a summary for a single layer.

        Arguments:
                layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "multiple"
        except RuntimeError:  # output_shape unknown in Eager mode.
            output_shape = "?"
        name = layer.name
        cls_name = layer.__class__.__name__
        fields = [
            name + " (" + cls_name + ")",
            layer_details(layer),
            output_shape,
            layer.count_params(),
        ]
        print_row(fields)

    def print_layer_summary_with_connections(layer):
        """Prints a summary for a single layer (including topological connections).

        Arguments:
                layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "multiple"
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue

            for inbound_layer, node_index, tensor_index, _ in node.iterate_inbound():
                connections.append(
                    "{}[{}][{}]".format(inbound_layer.name, node_index, tensor_index)
                )

        name = layer.name
        cls_name = layer.__class__.__name__
        if not connections:
            first_connection = ""
        else:
            first_connection = connections[0]
        fields = [
            name + " (" + cls_name + ")",
            layer_details(layer),
            output_shape,
            layer.count_params(),
            first_connection,
        ]
        print_row(fields)
        if len(connections) > 1:
            for i in range(1, len(connections)):
                fields = ["", "", "", "", connections[i]]
                print_row(fields)

    layers = model.layers
    for i in range(len(layers)):
        if sequential_like:
            print_layer_summary(layers[i])
        else:
            print_layer_summary_with_connections(layers[i])

    model._check_trainable_weights_consistency()
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)

    print_row([])
    print_row(
        [
            "Model",
            f"Parameters: total={trainable_count + non_trainable_count:,}, trainable={trainable_count:,}",
        ]
    )

    if do_print:
        from tabulate import tabulate

        print()
        print(tabulate(rows, headers="firstrow"))
        print()

    return rows
