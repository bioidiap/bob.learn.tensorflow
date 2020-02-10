#!/usr/bin/env python
"""Saves predictions or embeddings of tf.estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import click
from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    log_parameters,
)
from collections import defaultdict
import numpy as np
import tensorflow as tf
from bob.io.base import create_directories_safe
from bob.bio.base.utils import save
from bob.bio.base.tools.grid import indices
from bob.learn.tensorflow.dataset.bio import BioGenerator

try:
    import bob.bio.video
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)


def make_output_path(output_dir, key):
    """Returns an output path used for saving keys. You need to make sure the
    directories leading to this output path exist.

    Parameters
    ----------
    output_dir : str
        The root directory to save the results
    key : str
        The key of the sample. Usually biofile.make_path("", "")

    Returns
    -------
    str
        The path for the provided key.
    """
    return os.path.join(output_dir, key + ".hdf5")


def non_existing_files(paths, force=False):
    if force:
        for i in range(len(paths)):
            yield i
        return
    for i, path in enumerate(paths):
        if not os.path.isfile(path):
            yield i


def save_predictions(output_dir, key, pred_buffer, video_container, remove_nan=False):
    outpath = make_output_path(output_dir, key)
    create_directories_safe(os.path.dirname(outpath))
    logger.debug("Saving predictions for %s", key)
    if video_container:
        fc = bob.bio.video.FrameContainer()
        for i, v in enumerate(pred_buffer[key]):
            if remove_nan and np.isnan(v):
                continue
            fc.add(i, v)
        data = fc
    else:
        if remove_nan:
            pred_buffer[key] = np.array(pred_buffer[key])
            pred_buffer[key] = pred_buffer[key][~np.isnan(pred_buffer[key])]
        data = np.mean(pred_buffer[key], axis=0)
    save(data, outpath)


@click.command(entry_point_group="bob.learn.tensorflow.config", cls=ConfigCommand)
@click.option(
    "--estimator",
    "-e",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.estimator",
    help="The estimator that will be evaluated.",
)
@click.option(
    "--database",
    "-d",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.bio.database",
    help="A bio database. Its original_directory must point to the correct " "path.",
)
@click.option(
    "--biofiles",
    required=True,
    cls=ResourceOption,
    help="The list of the bio files. You can only provide this through config "
    "files.",
)
@click.option(
    "--bio-predict-input-fn",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.biogenerator_input",
    help="A callable with the signature of "
    "`input_fn = bio_predict_input_fn(generator, output_types, output_shapes)`"
    " The inputs are documented in :any:`tf.data.Dataset.from_generator`"
    " and the output should be a function with no arguments and is passed"
    " to :any:`tf.estimator.Estimator.predict`.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    cls=ResourceOption,
    help="The directory to save the predictions.",
)
@click.option(
    "--load-data",
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.load_data",
    help="A callable with the signature of "
    "``data = load_data(database, biofile)``. "
    ":any:`bob.bio.base.read_original_data` is used by default.",
)
@click.option(
    "--hooks",
    cls=ResourceOption,
    multiple=True,
    entry_point_group="bob.learn.tensorflow.hook",
    help="List of SessionRunHook subclass instances.",
)
@click.option(
    "--predict-keys",
    "-k",
    multiple=True,
    default=None,
    cls=ResourceOption,
    help="List of `str`, name of the keys to predict. It is used if the "
    "`EstimatorSpec.predictions` is a `dict`. If `predict_keys` is used "
    "then rest of the predictions will be filtered from the dictionary. "
    "If `None`, returns all.",
)
@click.option(
    "--checkpoint-path",
    "-c",
    cls=ResourceOption,
    help="Path of a specific checkpoint to predict. If `None`, the "
    "latest checkpoint in `model_dir` is used. This can also "
    'be a folder which contains a "checkpoint" file where the '
    "latest checkpoint from inside this file will be used as "
    "checkpoint_path.",
)
@click.option(
    "--multiple-samples",
    "-m",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, it assumes that the db interface returns "
    "several samples from a biofile. This option can be used "
    "when you are working with videos.",
)
@click.option(
    "--array",
    "-t",
    type=click.INT,
    default=1,
    cls=ResourceOption,
    help="Use this option alongside gridtk to submit this script as " "an array job.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    cls=ResourceOption,
    help="Whether to overwrite existing predictions.",
)
@click.option(
    "--video-container",
    "-vc",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, the predictions will be written in FrameContainers from"
    " bob.bio.video. You need to install bob.bio.video as well.",
)
@click.option(
    "--remove-nan",
    "-rn",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, will remove nans before computing the mean or remove nans "
    "from the frame container.",
)
@verbosity_option(cls=ResourceOption)
def predict_bio(
    estimator,
    database,
    biofiles,
    bio_predict_input_fn,
    output_dir,
    load_data,
    hooks,
    predict_keys,
    checkpoint_path,
    multiple_samples,
    array,
    force,
    video_container,
    remove_nan,
    **kwargs
):
    """Saves predictions or embeddings of tf.estimators.

    This script works with bob.bio.base databases. This script works with
    tensorflow 1.4 and above.

    An example configuration for a trained model and its evaluation could be::

        import tensorflow as tf

        # define the database:
        from bob.bio.base.test.dummy.database import database

        # load the estimator model
        estimator = tf.estimator.Estimator(model_fn, model_dir)

        groups = ['dev']
        biofiles = database.all_files(groups)


        # the ``dataset = tf.data.Dataset.from_generator(generator,
        # output_types, output_shapes)`` line is mandatory in the function
        # below. You have to create it in your configuration file since you
        # want it to be created in the same graph as your model.
        def bio_predict_input_fn(generator, output_types, output_shapes):
            def input_fn():
                dataset = tf.data.Dataset.from_generator(
                    generator, output_types, output_shapes)
                # apply all kinds of transformations here, process the data
                # even further if you want.
                dataset = dataset.prefetch(1)
                dataset = dataset.batch(10**3)
                images, labels, keys = dataset.make_one_shot_iterator().get_next()

                return {'data': images, 'key': keys}, labels
            return input_fn
    """
    log_parameters(logger, ignore=("biofiles",))
    logger.debug("len(biofiles): %d", len(biofiles))

    assert len(biofiles), "biofiles are empty!"

    if array > 1:
        start, end = indices(biofiles, array)
        biofiles = biofiles[start:end]

    # filter the existing files
    paths = [make_output_path(output_dir, f.make_path("", "")) for f in biofiles]
    indexes = non_existing_files(paths, force)
    biofiles = [biofiles[i] for i in indexes]

    if len(biofiles) == 0:
        logger.warning("The biofiles are empty after checking for existing files.")
        return

    generator = BioGenerator(
        database, biofiles, load_data=load_data, multiple_samples=multiple_samples
    )

    predict_input_fn = bio_predict_input_fn(
        generator, generator.output_types, generator.output_shapes
    )

    logger.info("Saving the predictions of %d files in %s", len(generator), output_dir)
    generic_predict(
        estimator=estimator,
        predict_input_fn=predict_input_fn,
        output_dir=output_dir,
        predict_keys=predict_keys,
        checkpoint_path=checkpoint_path,
        hooks=hooks,
        video_container=video_container,
        remove_nan=remove_nan,
    )


@click.command(entry_point_group="bob.learn.tensorflow.config", cls=ConfigCommand)
@click.option(
    "--estimator",
    "-e",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.estimator",
    help="The estimator that will be evaluated.",
)
@click.option(
    "--predict-input-fn",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.input_fn",
    help="A callable with no arguments which will be used in estimator.predict.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    cls=ResourceOption,
    help="The directory to save the predictions.",
)
@click.option(
    "--predict-keys",
    "-k",
    multiple=True,
    default=None,
    cls=ResourceOption,
    help="List of `str`, name of the keys to predict. It is used if the "
    "`EstimatorSpec.predictions` is a `dict`. If `predict_keys` is used "
    "then rest of the predictions will be filtered from the dictionary. "
    "If `None`, returns all.",
)
@click.option(
    "--checkpoint-path",
    "-c",
    cls=ResourceOption,
    help="Path of a specific checkpoint to predict. If `None`, the "
    "latest checkpoint in `model_dir` is used. This can also "
    'be a folder which contains a "checkpoint" file where the '
    "latest checkpoint from inside this file will be used as "
    "checkpoint_path.",
)
@click.option(
    "--hooks",
    cls=ResourceOption,
    multiple=True,
    entry_point_group="bob.learn.tensorflow.hook",
    help="List of SessionRunHook subclass instances.",
)
@click.option(
    "--video-container",
    "-vc",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, the predictions will be written in FrameContainers from"
    " bob.bio.video. You need to install bob.bio.video as well.",
)
@click.option(
    "--remove-nan",
    "-rn",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, will remove nans before computing the mean or remove nans "
    "from the frame container.",
)
@verbosity_option(cls=ResourceOption)
def predict(
    estimator,
    predict_input_fn,
    output_dir,
    predict_keys,
    checkpoint_path,
    hooks,
    video_container,
    remove_nan,
    **kwargs
):

    generic_predict(
        estimator=estimator,
        predict_input_fn=predict_input_fn,
        output_dir=output_dir,
        predict_keys=predict_keys,
        checkpoint_path=checkpoint_path,
        hooks=hooks,
        video_container=video_container,
        remove_nan=remove_nan,
    )


def generic_predict(
    estimator,
    predict_input_fn,
    output_dir,
    predict_keys=None,
    checkpoint_path=None,
    hooks=None,
    video_container=False,
    remove_nan=False,
):
    # if the checkpoint_path is a directory, pick the latest checkpoint from
    # that directory
    if checkpoint_path:
        if os.path.isdir(checkpoint_path):
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                checkpoint_path = ckpt.model_checkpoint_path

        logger.info("Restoring the model from %s", checkpoint_path)

    predictions = estimator.predict(
        predict_input_fn,
        predict_keys=predict_keys,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
    )

    if video_container:
        try:
            import bob.bio.video
        except ModuleNotFoundError:
            raise click.ClickException(
                "Could not import bob.bio.video. Have you installed it?"
            )

    pred_buffer = defaultdict(list)
    for i, pred in enumerate(predictions):
        key = pred["key"]
        # key is in bytes format in Python 3
        if sys.version_info >= (3,):
            key = key.decode(errors="replace")
        if predict_keys:
            prob = pred[predict_keys[0]]
        else:
            prob = pred.get(
                "probabilities", pred.get("embeddings", pred.get("predictions"))
            )
        assert prob is not None
        pred_buffer[key].append(prob)
        if i == 0:
            last_key = key
        if last_key == key:
            continue
        else:
            save_predictions(output_dir, last_key, pred_buffer, video_container, remove_nan)
            # delete saved data so we don't run out of RAM
            del pred_buffer[last_key]
            # start saving this new key
            last_key = key

    try:
        key
        # save the final returned key as well:
        save_predictions(output_dir, key, pred_buffer, video_container, remove_nan)
    except UnboundLocalError:
        # if the input_fn was empty and hence key is not defined
        click.echo("predict_input_fn returned no samples.")
        pass
