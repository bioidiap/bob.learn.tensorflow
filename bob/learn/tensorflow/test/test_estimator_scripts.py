from __future__ import print_function
import os
import shutil
from glob import glob
from tempfile import mkdtemp
from click.testing import CliRunner
from bob.extension.scripts.click_helper import assert_click_runner_result
from bob.io.base.test_utils import datafile

from bob.learn.tensorflow.script.db_to_tfrecords import db_to_tfrecords
from bob.learn.tensorflow.script.train import train
from bob.learn.tensorflow.script.eval import eval as eval_script
from bob.learn.tensorflow.script.train_and_evaluate import train_and_evaluate
from bob.learn.tensorflow.script.predict_bio import predict_bio
from nose.plugins.attrib import attr


db_to_tfrecords_config = datafile('db_to_tfrecords_config.py', __name__)
input_predict_bio_config = datafile('input_predict_bio_config.py', __name__)
input_biogenerator_config = datafile('input_biogenerator_config.py', __name__)


def input_tfrecords_config(tfrecord_path):
    with open(datafile('input_tfrecords_config.py', __name__)) as f:
        doc = '\n' + f.read() + '\n'
    return doc % {'tfrecord_filenames': tfrecord_path}


def estimator_atnt_faces_config(model_dir):
    with open(datafile('estimator_atnt_faces_config.py', __name__)) as f:
        doc = '\n' + f.read() + '\n'
    return doc % {'model_dir': model_dir}


def _create_tfrecord(test_dir):
    output = os.path.join(test_dir, 'dev.tfrecords')
    runner = CliRunner()
    result = runner.invoke(
        db_to_tfrecords, args=[db_to_tfrecords_config, '--output', output])
    assert_click_runner_result(result)
    return output


def _create_checkpoint(tmpdir, model_dir, tfrecord_path):
    config = input_tfrecords_config(
        tfrecord_path) + estimator_atnt_faces_config(model_dir)
    config_path = os.path.join(tmpdir, 'train_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    result = runner.invoke(train, args=[config_path])
    assert_click_runner_result(result)


def _eval(tmpdir, model_dir, tfrecord_path, extra_args=['--run-once']):
    config = input_tfrecords_config(
        tfrecord_path) + estimator_atnt_faces_config(model_dir)
    config_path = os.path.join(tmpdir, 'eval_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    result = runner.invoke(eval_script, args=[config_path] + extra_args)
    assert_click_runner_result(result)


def _train_and_evaluate(tmpdir, model_dir, tfrecord_path):
    config = input_tfrecords_config(
        tfrecord_path) + estimator_atnt_faces_config(model_dir)
    config_path = os.path.join(tmpdir, 'train_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    runner.invoke(train_and_evaluate, args=[config_path])


def _predict_bio(tmpdir, model_dir, tfrecord_path, extra_options=tuple()):
    config = input_tfrecords_config(
        tfrecord_path) + estimator_atnt_faces_config(model_dir)
    config_path = os.path.join(tmpdir, 'train_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    return runner.invoke(
        predict_bio,
        args=[config_path, input_predict_bio_config] + list(extra_options))

@attr('slow')
def test_eval():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')
        eval_dir = os.path.join(model_dir, 'eval')

        print('\nCreating a dummy tfrecord')
        tfrecord_path = _create_tfrecord(tmpdir)

        print('Training a dummy network')
        _create_checkpoint(tmpdir, model_dir, tfrecord_path)

        print('Evaluating a dummy network')
        _eval(tmpdir, model_dir, tfrecord_path)

        evaluated_path = os.path.join(eval_dir, 'evaluated')
        assert os.path.exists(evaluated_path), evaluated_path
        with open(evaluated_path) as f:
            doc = f.read()

        assert '0' in doc, doc
        assert '200' in doc, doc

        print('Train and evaluate a dummy network')
        _train_and_evaluate(tmpdir, model_dir, tfrecord_path)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

@attr('slow')
def test_eval_keep_n_model():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')
        eval_dir = os.path.join(model_dir, 'eval')

        print('\nCreating a dummy tfrecord')
        tfrecord_path = _create_tfrecord(tmpdir)

        print('Training a dummy network')
        _create_checkpoint(tmpdir, model_dir, tfrecord_path)

        print('Evaluating a dummy network')
        _eval(tmpdir, model_dir, tfrecord_path, ['-K', '1', '--run-once'])

        evaluated_path = os.path.join(eval_dir, 'evaluated')
        assert os.path.exists(evaluated_path), evaluated_path
        with open(evaluated_path) as f:
            doc = f.read()
        assert '0 ' in doc, doc
        assert '200 ' in doc, doc
        assert len(glob('{}/model.ckpt-*'.format(eval_dir))) == 3, \
            os.listdir(eval_dir)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

@attr('slow')
def test_predict_bio():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')

        tfrecord_path = _create_tfrecord(tmpdir)
        _create_checkpoint(tmpdir, model_dir, tfrecord_path)

        # Run predict_bio
        result = _predict_bio(
            tmpdir, model_dir, tfrecord_path, ['-o', tmpdir, '-vvv'])
        assert_click_runner_result(result)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

@attr('slow')
def test_predict_bio_empty_eval():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')
        eval_dir = os.path.join(model_dir, 'eval')

        tfrecord_path = _create_tfrecord(tmpdir)
        _create_checkpoint(tmpdir, model_dir, tfrecord_path)

        # Make an empty eval folder
        os.makedirs(eval_dir)
        open(os.path.join(eval_dir, 'checkpoint'), 'w')

        # Run predict_bio
        result = _predict_bio(
            tmpdir, model_dir, tfrecord_path,
            ['-o', tmpdir, '-c', eval_dir, '-vvv'])
        # the command should fail when the checkpoint path is empty
        assert_click_runner_result(result, 1)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


# uncomment to run this test locally
# def test_eval_too_many_open_files_with_biogenerator():
#     tmpdir = mkdtemp(prefix='bob_')
#     try:
#         # create estimator config file
#         model_dir = os.path.join(tmpdir, 'model_dir')
#         estimator_config = os.path.join(tmpdir, 'estimator_config.py')
#         with open(estimator_config, 'w') as f:
#             f.write(estimator_atnt_faces_config(model_dir))

#         runner = CliRunner()

#         # train and eval with biogenerators
#         result = runner.invoke(
#             train, args=[estimator_config, input_biogenerator_config])
#         assert_click_runner_result(result)

#         print("This test will not stop running. You should kill the process!")
#         result = runner.invoke(
#             eval_script, args=[estimator_config,
#                                input_biogenerator_config,
#                                '--force-re-run'])
#         assert_click_runner_result(result)

#     finally:
#         try:
#             shutil.rmtree(tmpdir)
#         except Exception:
#             pass
