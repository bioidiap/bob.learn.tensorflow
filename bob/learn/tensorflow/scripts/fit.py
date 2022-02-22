#!/usr/bin/env python
"""Trains networks using Keras Models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os

import click
import tensorflow as tf

from bob.extension.scripts.click_helper import ConfigCommand
from bob.extension.scripts.click_helper import ResourceOption
from bob.extension.scripts.click_helper import log_parameters
from bob.extension.scripts.click_helper import verbosity_option

logger = logging.getLogger(__name__)


@click.command(entry_point_group="bob.learn.tensorflow.config", cls=ConfigCommand)
@click.option(
    "--model-fn",
    "-m",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.model",
    help="The keras model that will be trained.",
)
@click.option(
    "--train-input-fn",
    "-i",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.input_fn",
    help="A function that will return the training data as a tf.data.Dataset "
    "or tf.data.Iterator. This will be given as `x` to "
    "tf.keras.Model.fit.",
)
@click.option(
    "--epochs",
    "-e",
    default=10,
    type=click.types.INT,
    cls=ResourceOption,
    help="Number of epochs to train model. See " "tf.keras.Model.fit.",
)
@click.option(
    "--callbacks",
    cls=ResourceOption,
    multiple=True,
    entry_point_group="bob.learn.tensorflow.callback",
    help="List of tf.keras.callbacks. Used for callbacks " "inside the training loop.",
)
@click.option(
    "--eval-input-fn",
    "-i",
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.input_fn",
    help="A function that will return the validation data as a tf.data.Dataset"
    " or tf.data.Iterator. This will be given as `validation_data` to "
    "tf.keras.Model.fit.",
)
@click.option(
    "--class-weight", "-c", cls=ResourceOption, help="See tf.keras.Model.fit."
)
@click.option(
    "--steps-per-epoch",
    type=click.types.INT,
    cls=ResourceOption,
    help="See tf.keras.Model.fit.",
)
@click.option(
    "--validation-steps",
    type=click.types.INT,
    cls=ResourceOption,
    help="See tf.keras.Model.fit.",
)
@click.option(
    "--dask-client",
    "-l",
    entry_point_group="dask.client",
    default=None,
    help="Dask client for the execution of the pipeline.",
    cls=ResourceOption,
)
@click.option(
    "--strategy-fn",
    entry_point_group="bob.learn.tensorflow.strategy",
    default=None,
    help="The strategy to be used for distributed training.",
    cls=ResourceOption,
)
@click.option(
    "--mixed-precision-policy",
    default=None,
    help="The mixed precision policy to be used for training.",
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def fit(
    model_fn,
    train_input_fn,
    epochs,
    verbose,
    callbacks,
    eval_input_fn,
    class_weight,
    steps_per_epoch,
    validation_steps,
    dask_client,
    strategy_fn,
    mixed_precision_policy,
    **kwargs,
):
    """Trains networks using Keras models."""
    from tensorflow.keras import mixed_precision

    from bob.extension.log import set_verbosity_level
    from bob.extension.log import setup as setup_logger

    from ..utils import FloatValuesEncoder
    from ..utils import compute_tf_config_from_dask_client

    log_parameters(logger)

    # Train
    save_callback = [
        c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint)
    ]
    model_dir = None
    if save_callback:
        model_dir = save_callback[0].filepath
        logger.info("Training a model in %s", model_dir)
    callbacks = list(callbacks) if callbacks else None

    def train(tf_config=None):
        # setup verbosity again in case we're in a dask worker
        setup_logger("bob")
        set_verbosity_level("bob", verbose)

        if tf_config is not None:
            logger.info("Setting up TF_CONFIG with %s", tf_config)
            os.environ["TF_CONFIG"] = json.dumps(tf_config)

        if mixed_precision_policy is not None:
            logger.info("Using %s mixed precision policy", mixed_precision_policy)
            mixed_precision.set_global_policy(mixed_precision_policy)

        validation_data = None

        if strategy_fn is None:
            model: tf.keras.Model = model_fn()
            x = train_input_fn()
            if eval_input_fn is not None:
                validation_data = eval_input_fn()
        else:
            strategy = strategy_fn()
            with strategy.scope():
                model: tf.keras.Model = model_fn()
                x = strategy.distribute_datasets_from_function(train_input_fn)
                if eval_input_fn is not None:
                    validation_data = strategy.distribute_datasets_from_function(
                        eval_input_fn
                    )

        # swap 1 and 2 verbosity values for Keras as verbose=1 is more verbose model.fit
        fit_verbose = {0: 0, 1: 2, 2: 1}[min(verbose, 2)]

        click.echo(
            f"""Calling {model}.fit with:(
            x={x},
            epochs={epochs},
            verbose={fit_verbose},
            callbacks={callbacks},
            validation_data={validation_data},
            class_weight={class_weight},
            steps_per_epoch={steps_per_epoch},
            validation_steps={validation_steps},
        )
        and optimizer: {model.optimizer}
        """
        )
        history = model.fit(
            x=x,
            epochs=epochs,
            verbose=fit_verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            class_weight=class_weight,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
        if model_dir is not None:
            with open(os.path.join(model_dir, "keras_fit_history.json"), "w") as f:
                json.dump(history.history, f, cls=FloatValuesEncoder)

        return history.history

    if dask_client is None:
        history = train()
    else:
        tf_configs, workers_ips = compute_tf_config_from_dask_client(dask_client)
        future_histories = []
        for tf_spec, ip in zip(tf_configs, workers_ips):
            future = dask_client.submit(train, tf_spec, workers=ip)
            future_histories.append(future)

        try:
            history = dask_client.gather(future_histories)
        finally:
            try:
                logger.debug("Printing dask logs:")
                for key, value in dask_client.cluster.get_logs().items():
                    logger.debug(f"{key}:")
                    logger.debug(value)
                logger.debug(dask_client.cluster.job_script())
            except Exception:
                pass

    logger.debug("history:")
    logger.debug(history)
    return history
