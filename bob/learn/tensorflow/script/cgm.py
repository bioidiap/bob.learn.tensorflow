#!/usr/bin/env python
"""Converts datasets to TFRecords file formats.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tempfile
import os
import gridtk
import sys
import logging
import click
import tensorflow as tf
import h5py
import scipy
from bob.io.base import create_directories_safe
from bob.extension.scripts.click_helper import (
    verbosity_option,
    log_parameters,
    ConfigCommand,
    ResourceOption,
)
from glob import glob
import numpy
import time

logger = logging.getLogger(__name__)


@click.command(
    epilog="""\b
$ bob tf distance_matrix -vvv --hdf5 <hdf5-path> --n-chunks 10 [normalizer.py]
""",
    entry_point_group="bob.learn.tensorflow.config",
    cls=ConfigCommand,
)
@click.option(
    "--hdf5",
    required=True,
    cls=ResourceOption,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the hdf5 dataset.",
)
@click.option(
    "--n-chunks",
    type=click.INT,
    required=True,
    cls=ResourceOption,
    help="Divides the dataset into n_chunks**2 for parallel computation.",
)
@click.option(
    "--normalizer",
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.normalizer",
    help="A function that takes each data and returns a normalized data.",
)
@click.option(
    "--job-name", default="dis-mat", help="The prefix to use for SGE job submissions."
)
@click.option(
    "--queue-array",
    default="all.q",
    help="The queue to use for SGE array job submissions.",
)
@click.option(
    "--finalize",
    is_flag=True,
    help="If for some reason your array jobs are completed but you do not have"
    " your final distance matrix, use this option.",
)
@verbosity_option()
@click.option("--n-samples", type=click.INT, help="Internal option. Do not use.")
def distance_matrix(
    hdf5, n_chunks, normalizer, n_samples, job_name, queue_array, finalize, **kwargs
):
    """Computes distance matrix of an HDF5 dataset

    The data must be normalized -> zero-mean and histogram equalization

    Divides the dataset into N**2 chunks for parallel computation.
    """
    log_parameters(logger)

    if finalize:
        return _finalize(hdf5)

    if n_samples is None:
        create_directories_safe(_output_folder(hdf5))
        return _pre_compute(hdf5, n_chunks, job_name, queue_array)

    return _compute(hdf5, n_chunks, normalizer, n_samples)


def _pre_compute(hdf5, n_chunks, job_name, queue_array):
    with h5py.File(hdf5, "r") as f:

        # find the total number of samples in hdf5 file
        samples = sorted(int(k) for k in f.keys() if k.isdigit())
        n_samples = len(samples)

        logger.debug("There are %d samples in the hdf5", n_samples)

        # figure out the number of jobs jobs
        n_jobs = (n_chunks * (n_chunks + 1)) // 2

        # submit the array job with gridtk
        job_manager = gridtk.sge.JobManagerSGE()
        command = sys.argv + ["--n-samples", str(n_samples)]
        job_id = job_manager.submit(
            command, queue=queue_array, name=job_name + "-ary", array=(1, n_jobs, 1)
        )

    # wait for the jobs to finish and finalize
    def get_job():
        try:
            job_manager.lock()
            job_manager.communicate([job_id])
            job = job_manager.get_jobs([job_id])[0]
            return job
        finally:
            job_manager.unlock()

    job = get_job()
    while job.status not in ("success", "failure"):
        time.sleep(10)
        job = get_job()

    _finalize(hdf5)


def _compute(hdf5, n_chunks, normalizer, n_samples):
    with h5py.File(hdf5, "r") as f:

        # We should be in an array job
        job_id = int(os.getenv("SGE_TASK_ID", 1)) - 1

        output = os.path.join(_output_folder(hdf5), f"{job_id}.npz")
        logger.info("Saving output in %s", output)

        # calculate chunk size
        chunk_size = int(numpy.ceil(n_samples / n_chunks))

        # generate upper triangle indices
        if n_chunks > 100:
            logger.warning("The computation below is going to take forever!")
        indices = []
        for k in range(n_chunks ** 2):
            i, j = numpy.unravel_index(k, (n_chunks, n_chunks))
            if i <= j:
                i, j = i * chunk_size, j * chunk_size
                i = range(i, min(i + chunk_size, n_samples))
                j = range(j, min(j + chunk_size, n_samples))
                if (i, j) not in indices:
                    indices.append((i, j))

        def get_array(fd, id_range):
            array = []
            for i in id_range:
                array.append(fd[f"/{i}"]["data"].value.flatten())
            return array

        i, j = indices[job_id]
        if i == j:
            d = scipy.spatial.distance.pdist(get_array(f, i))
            d = scipy.spatial.distance.squareform(d)
        else:
            d = scipy.spatial.distance.cdist(get_array(f, i), get_array(f, j))

        # create sparse matrix
        sparse_d = scipy.sparse.dok_matrix((n_samples, n_samples))

        def _slice(_range):
            return slice(_range.start, _range.stop, _range.step)

        # assign this job's distance matrix
        sparse_d[_slice(i), _slice(j)] = d

        # remove extra numbers
        sparse_d = scipy.sparse.triu(sparse_d, k=1)

        # save
        scipy.sparse.save_npz(output, sparse_d)


def _finalize(hdf5):
    # load and sum all sparse matrices
    # save the final results
    # delete intermediate matrices
    out_fldr = _output_folder(hdf5)
    files = glob(os.path.join(out_fldr, "*.npz"))
    files = list(
        filter(lambda x: os.path.splitext(os.path.split(x)[1])[0].isdigit(), files)
    )

    dist = sum(scipy.sparse.load_npz(f) for f in files)
    output = os.path.join(out_fldr, "dist.npz")
    scipy.sparse.save_npz(output, dist)

    for path in files:
        os.remove(path)
        logger.debug("Deleted: %s", path)


def _output_folder(hdf5):
    return hdf5.rsplit(".hdf5", 1)[0]
