import tensorflow as tf


def call_on_frozen_graph(
    graph_def_path,
    input,
    return_elements,
    input_name,
    name=None,
    **kwargs
):
    """Loads a frozen graph def file (.pb) and replaces its input with the given input
    and return the requested output tensors.

    Parameters
    ----------
    graph_def_path : str
        Path to the graph definition file
    input : object
        Input tensor
    return_elements : [str]
        A list of strings which corresponds to operations in the graph.
    input_name : str, optional
        The name of input in the graph that will be replaced by input.
    name : str, optional
        The scope of the imported operations. Defaults to "import".
    **kwargs
        Extra arguments to be passed to tf.import_graph_def

    Returns
    -------
    list
        List of requested operations. Normally you would use
        ``returned_operations[0].outputs[0]``
    """
    with tf.gfile.GFile(graph_def_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    input_map = {input_name: input}

    return tf.import_graph_def(
        graph_def,
        input_map=input_map,
        return_elements=return_elements,
        name=name,
        **kwargs
    )
