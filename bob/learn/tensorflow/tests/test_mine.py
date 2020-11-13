import numpy as np
import tensorflow as tf

from bob.learn.tensorflow.models import MineModel


def run_mine(is_mine_f):
    np.random.seed(10)
    N = 20000
    d = 1
    EPOCHS = 10

    X = np.sign(np.random.normal(0.0, 1.0, [N, d]))
    Z = X + np.random.normal(0.0, np.sqrt(0.2), [N, d])

    from sklearn.feature_selection import mutual_info_regression

    mi_numerical = mutual_info_regression(X.reshape(-1, 1), Z.ravel())[0]

    model = MineModel(is_mine_f=is_mine_f)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    callback = model.fit(x=[X, Z], epochs=EPOCHS, verbose=1, batch_size=100)
    mine = -np.array(callback.history["loss"])[-1]

    np.allclose(mine, mi_numerical, atol=0.01)


def test_mine():
    run_mine(False)


def test_mine_f():
    run_mine(True)
