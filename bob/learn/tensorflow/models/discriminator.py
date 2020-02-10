"""Discriminator used in training of autoencoder in:
IMPROVING CROSS-DATASET PERFORMANCE OF FACE PRESENTATION ATTACK DETECTION SYSTEMS USING FACE RECOGNITION DATASETS,
Mohammadi, Amir and Bhattacharjee, Sushil and Marcel, Sebastien, ICASSP 2020
"""
import tensorflow as tf


def DenseDiscriminator(n_classes=1, name="DenseDiscriminator", **kwargs):
    """A discriminator that takes vectors as input and tries its best.
    Be careful, this one returns logits."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1000),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(n_classes),
        ],
        name=name,
        **kwargs
    )
