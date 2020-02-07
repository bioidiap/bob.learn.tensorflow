import tensorflow as tf
from ..utils import pdist_safe, upper_triangle

def total_pairwise_confusion(prelogits, name=None):
    """Total Pairwise Confusion Loss

        [1]X. Tu et al., “Learning Generalizable and Identity-Discriminative
        Representations for Face Anti-Spoofing,” arXiv preprint arXiv:1901.05602, 2019.
    """
    # compute L2 norm between all prelogits and sum them.
    with tf.name_scope(name, default_name="total_pairwise_confusion"):
        prelogits = tf.reshape(prelogits, (tf.shape(prelogits)[0], -1))
        loss_tpc = tf.reduce_mean(upper_triangle(pdist_safe(prelogits)))

    tf.summary.scalar("loss_tpc", loss_tpc)
    return loss_tpc
