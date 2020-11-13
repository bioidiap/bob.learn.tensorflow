"""
Implements the MINE loss from the paper:

Mutual Information Neural Estimation (https://arxiv.org/pdf/1801.04062.pdf)

"""

import tensorflow as tf


class MineModel(tf.keras.Model):
    """

    Parameters
    **********

      is_mine_f: bool
         If true, will implement MINE-F (equation 6), otherwise will implement equation 5
    """

    def __init__(self, is_mine_f=False, name="MINE", units=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.is_mine_f = is_mine_f

        self.transformer_x = tf.keras.layers.Dense(self.units)
        self.transformer_z = tf.keras.layers.Dense(self.units)
        self.transformer_xz = tf.keras.layers.Dense(self.units)
        self.transformer_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        def compute(x, z):
            h1_x = self.transformer_x(x)
            h1_z = self.transformer_z(z)
            h1 = tf.keras.layers.ReLU()(h1_x + h1_z)
            h2 = self.transformer_output(
                tf.keras.layers.ReLU()(self.transformer_xz(h1))
            )

            return h2

        def compute_lower_bound(x, z):
            t_xz = compute(x, z)
            z_shuffle = tf.random.shuffle(z)
            t_x_z = compute(x, z_shuffle)

            if self.is_mine_f:
                lb = -(
                    tf.reduce_mean(t_xz, axis=0)
                    - tf.reduce_mean(tf.math.exp(t_x_z - 1))
                )
            else:
                lb = -(
                    tf.reduce_mean(t_xz, axis=0)
                    - tf.math.log(tf.reduce_mean(tf.math.exp(t_x_z)))
                )

            self.add_loss(lb)
            return -lb

        x = inputs[0]
        z = inputs[1]

        return compute_lower_bound(x, z)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
