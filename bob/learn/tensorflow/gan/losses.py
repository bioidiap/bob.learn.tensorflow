import tensorflow as tf


def relativistic_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.25,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False,
):
    """Relativistic (average) loss

  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data`, and must be broadcastable to `real_data` (i.e., all
      dimensions must be either `1`, or the same as the corresponding
      dimension).
    generated_weights: Same as `real_weights`, but for `generated_data`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.compat.v1.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
    with tf.name_scope(
        scope,
        "discriminator_relativistic_loss",
        (
            discriminator_real_outputs,
            discriminator_gen_outputs,
            real_weights,
            generated_weights,
            label_smoothing,
        ),
    ) as scope:

        real_logit = discriminator_real_outputs - tf.reduce_mean(
            discriminator_gen_outputs
        )
        fake_logit = discriminator_gen_outputs - tf.reduce_mean(
            discriminator_real_outputs
        )

        loss_on_real = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(real_logit),
            real_logit,
            real_weights,
            label_smoothing,
            scope,
            loss_collection=None,
            reduction=reduction,
        )
        loss_on_generated = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(fake_logit),
            fake_logit,
            generated_weights,
            scope=scope,
            loss_collection=None,
            reduction=reduction,
        )

        loss = loss_on_real + loss_on_generated
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar("discriminator_gen_relativistic_loss", loss_on_generated)
            tf.summary.scalar("discriminator_real_relativistic_loss", loss_on_real)
            tf.summary.scalar("discriminator_relativistic_loss", loss)

    return loss


def relativistic_generator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.0,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False,
    confusion_labels=False,
):
    """Relativistic (average) loss

  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data`, and must be broadcastable to `real_data` (i.e., all
      dimensions must be either `1`, or the same as the corresponding
      dimension).
    generated_weights: Same as `real_weights`, but for `generated_data`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.compat.v1.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
    with tf.name_scope(
        scope,
        "generator_relativistic_loss",
        (
            discriminator_real_outputs,
            discriminator_gen_outputs,
            real_weights,
            generated_weights,
            label_smoothing,
        ),
    ) as scope:

        real_logit = discriminator_real_outputs - tf.reduce_mean(
            discriminator_gen_outputs
        )
        fake_logit = discriminator_gen_outputs - tf.reduce_mean(
            discriminator_real_outputs
        )

        if confusion_labels:
            real_labels = tf.ones_like(real_logit) / 2
            fake_labels = tf.ones_like(fake_logit) / 2
        else:
            real_labels = tf.zeros_like(real_logit)
            fake_labels = tf.ones_like(fake_logit)

        loss_on_real = tf.losses.sigmoid_cross_entropy(
            real_labels,
            real_logit,
            real_weights,
            label_smoothing,
            scope,
            loss_collection=None,
            reduction=reduction,
        )
        loss_on_generated = tf.losses.sigmoid_cross_entropy(
            fake_labels,
            fake_logit,
            generated_weights,
            scope=scope,
            loss_collection=None,
            reduction=reduction,
        )

        loss = loss_on_real + loss_on_generated
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar("generator_gen_relativistic_loss", loss_on_generated)
            tf.summary.scalar("generator_real_relativistic_loss", loss_on_real)
            tf.summary.scalar("generator_relativistic_loss", loss)

    return loss
