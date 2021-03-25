import numpy as np

from bob.learn.tensorflow.layers import ModifiedSoftMaxLayer
from bob.learn.tensorflow.layers import SphereFaceLayer
from bob.learn.tensorflow.models import ArcFaceLayer
from bob.learn.tensorflow.models import ArcFaceLayer3Penalties


def test_arcface_layer():

    layer = ArcFaceLayer()
    np.random.seed(10)
    X = np.random.rand(10, 50)
    y = [np.random.randint(10) for i in range(10)]

    assert layer(X, y).shape == (10, 10)


def test_arcface_layer_3p():

    layer = ArcFaceLayer3Penalties()
    np.random.seed(10)
    X = np.random.rand(10, 50)
    y = [np.random.randint(10) for i in range(10)]

    assert layer(X, y).shape == (10, 10)


def test_sphereface():

    layer = SphereFaceLayer()
    np.random.seed(10)
    X = np.random.rand(10, 10)

    assert layer(X).shape == (10, 10)


def test_modsoftmax():

    layer = ModifiedSoftMaxLayer()
    np.random.seed(10)
    X = np.random.rand(10, 10)

    assert layer(X).shape == (10, 10)
