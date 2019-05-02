import tensorflow as tf
import bob.measure
import numpy
from tensorflow.python.ops.metrics_impl import metric_variable
from ..utils import norm, predict_using_tensors
from .ContrastiveLoss import contrastive_loss


def logits_loss(
    bio_logits, pad_logits, bio_labels, pad_labels, bio_loss, pad_loss, alpha=0.5
):

    with tf.name_scope("Bio_loss"):
        bio_loss_ = bio_loss(logits=bio_logits, labels=bio_labels)

    with tf.name_scope("PAD_loss"):
        pad_loss_ = pad_loss(
            logits=pad_logits, labels=tf.cast(pad_labels, dtype="int32")
        )

    with tf.name_scope("EPSC_loss"):
        total_loss = (1 - alpha) * bio_loss_ + alpha * pad_loss_

    tf.add_to_collection(tf.GraphKeys.LOSSES, bio_loss_)
    tf.add_to_collection(tf.GraphKeys.LOSSES, pad_loss_)
    tf.add_to_collection(tf.GraphKeys.LOSSES, total_loss)

    tf.summary.scalar("bio_loss", bio_loss_)
    tf.summary.scalar("pad_loss", pad_loss_)
    tf.summary.scalar("epsc_loss", total_loss)

    return total_loss


def embedding_norm_loss(prelogits_left, prelogits_right, b, c, margin=10.0):
    with tf.name_scope("embedding_norm_loss"):
        prelogits_left = norm(prelogits_left)
        prelogits_right = norm(prelogits_right)

        loss = tf.add_n(
            [
                tf.reduce_mean(b * (tf.maximum(prelogits_left - margin, 0))),
                tf.reduce_mean((1 - b) * (tf.maximum(2 * margin - prelogits_left, 0))),
                tf.reduce_mean(c * (tf.maximum(prelogits_right - margin, 0))),
                tf.reduce_mean((1 - c) * (tf.maximum(2 * margin - prelogits_right, 0))),
            ],
            name="embedding_norm_loss",
        )
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        tf.summary.scalar("embedding_norm_loss", loss)
        # log norm of embeddings for BF and PA separately to see how their norm
        # evolves over time
        bf_norm = tf.concat(
            [
                tf.gather(prelogits_left, tf.where(b > 0.5)),
                tf.gather(prelogits_right, tf.where(c > 0.5)),
            ],
            axis=0,
        )
        pa_norm = tf.concat(
            [
                tf.gather(prelogits_left, tf.where(b < 0.5)),
                tf.gather(prelogits_right, tf.where(c < 0.5)),
            ],
            axis=0,
        )
        tf.summary.histogram("BF_embeddings_norm", bf_norm)
        tf.summary.histogram("PA_embeddings_norm", pa_norm)
    return loss


def siamese_loss(bio_logits, pad_logits, bio_labels, pad_labels, alpha=0.1):
    # prepare a, b, c
    with tf.name_scope("epsc_labels"):
        a = tf.to_float(
            tf.math.equal(bio_labels["left"], bio_labels["right"]), name="a"
        )
        b = tf.to_float(tf.math.equal(pad_labels["left"], True), name="b")
        c = tf.to_float(tf.math.equal(pad_labels["right"], True), name="c")
        tf.summary.scalar("Mean_a", tf.reduce_mean(a))
        tf.summary.scalar("Mean_b", tf.reduce_mean(b))
        tf.summary.scalar("Mean_c", tf.reduce_mean(c))

    prelogits_left = bio_logits["left"]
    prelogits_right = bio_logits["right"]

    bio_loss = contrastive_loss(prelogits_left, prelogits_right, labels=1 - a)

    pad_loss = alpha * embedding_norm_loss(prelogits_left, prelogits_right, b, c)

    with tf.name_scope("epsc_loss"):
        epsc_loss = (1 - alpha) * bio_loss + alpha * pad_loss
        tf.add_to_collection(tf.GraphKeys.LOSSES, epsc_loss)

    tf.summary.scalar("epsc_loss", epsc_loss)

    return epsc_loss


def py_eer(negatives, positives):
    def _eer(neg, pos):
        if neg.size == 0 or pos.size == 0:
            return numpy.array(0.0, dtype="float64")
        return bob.measure.eer(neg, pos)

    negatives = tf.reshape(tf.cast(negatives, "float64"), [-1])
    positives = tf.reshape(tf.cast(positives, "float64"), [-1])

    eer = tf.py_func(_eer, [negatives, positives], tf.float64, name="py_eer")

    return tf.cast(eer, "float32")


def epsc_metric(
    bio_embeddings,
    pad_probabilities,
    bio_labels,
    pad_labels,
    batch_size,
    pad_threshold=numpy.exp(-15),
):
    # math.exp(-2.0) = 0.1353352832366127
    # math.exp(-15.0) = 3.059023205018258e-07
    with tf.name_scope("epsc_metrics"):
        bio_predictions_op = predict_using_tensors(
            bio_embeddings, bio_labels, num=batch_size
        )

        # find the lowest value of bf and highest value of pa
        # their mean is the threshold
        # bf_probabilities = tf.gather(pad_probabilities, tf.where(pad_labels))
        # pa_probabilities = tf.gather(pad_probabilities, tf.where(tf.logical_not(pad_labels)))

        # eer = py_eer(pa_probabilities, bf_probabilities)
        # acc = 1 - eer

        # pad_threshold = (tf.reduce_max(pa_probabilities) + tf.reduce_min(bf_probabilities)) / 2
        # true_positives = tf.reduce_sum(tf.to_int32(bf_probabilities >= pad_threshold))
        # true_negatives = tf.reduce_sum(tf.to_int32(pa_probabilities < pad_threshold))
        # # pad_accuracy = metric_variable([], tf.float32, name='pad_accuracy')
        # acc = (true_positives + true_negatives) / batch_size

        # pad_accuracy, pad_update_ops = tf.metrics.mean(acc)

        # print_ops = [
        #     tf.print(pad_probabilities),
        #     tf.print(bf_probabilities, pa_probabilities),
        #     tf.print(pad_threshold),
        #     tf.print(true_positives, true_negatives),
        #     tf.print(pad_probabilities.shape[0]),
        #     tf.print(acc),
        # ]
        # update_op = tf.assign_add(pad_accuracy, tf.cast(acc, tf.float32))
        # update_op = tf.group([update_op] + print_ops)

        tp = tf.metrics.true_positives_at_thresholds(
            pad_labels, pad_probabilities, [pad_threshold]
        )
        fp = tf.metrics.false_positives_at_thresholds(
            pad_labels, pad_probabilities, [pad_threshold]
        )
        tn = tf.metrics.true_negatives_at_thresholds(
            pad_labels, pad_probabilities, [pad_threshold]
        )
        fn = tf.metrics.false_negatives_at_thresholds(
            pad_labels, pad_probabilities, [pad_threshold]
        )
        pad_accuracy = (tp[0] + tn[0]) / (tp[0] + tn[0] + fp[0] + fn[0])
        pad_accuracy = tf.reduce_mean(pad_accuracy)
        pad_update_ops = tf.group([x[1] for x in (tp, tn, fp, fn)])

        eval_metric_ops = {
            "bio_accuracy": tf.metrics.accuracy(
                labels=bio_labels, predictions=bio_predictions_op
            ),
            "pad_accuracy": (pad_accuracy, pad_update_ops),
        }
    return eval_metric_ops
