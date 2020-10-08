#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from functools import partial
from multiprocessing import cpu_count

import pkg_resources
import tensorflow as tf
from bob.learn.tensorflow.losses import CenterLoss, CenterLossLayer
from bob.learn.tensorflow.models.inception_resnet_v2 import InceptionResNetV2
from bob.learn.tensorflow.utils import predict_using_tensors
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from bob.extension import rc

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)


TRAIN_TF_RECORD_PATHS = (
    f"{rc['htface']}/databases/tfrecords/msceleba/"
    "tfrecord_182x_hand_prunned_44/*.tfrecord"
)
VALIDATION_TF_RECORD_PATHS = (
    f"{rc['htface']}/databases/tfrecords/lfw/182x/RGB/*.tfrecord"
)
# there are 2812 samples in the validation set
VALIDATION_SAMPLES = 2812

CHECKPOINT = (
    f"{rc['temp']}/models/inception_v2_batchnorm_rgb_msceleba_mixed_precision"
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
TFRECORD_PARALLEL_READ = cpu_count()
N_CLASSES = 87662
DATA_SHAPE = (182, 182, 3)  # size of faces
DATA_TYPE = tf.uint8
OUTPUT_SHAPE = (160, 160)

SHUFFLE_BUFFER = int(2e4)

LEARNING_RATE = 0.1
BATCH_SIZE = 90 * 2  # should be a multiple of 8
# we want to run 35 epochs of tfrecords. There are 959083 samples in train tfrecords,
# depending on batch size, steps per epoch, and keras epoch multiplier should change
EPOCHS = 35
# number of training steps to do before validating a model. This also defines an epoch
# for keras which is not really true. We want to evaluate every 180000 (90 * 2000)
# samples
STEPS_PER_EPOCH = 180000 // BATCH_SIZE
# np.ceil(959083/180000=5.33)
KERAS_EPOCH_MULTIPLIER = 6

VALIDATION_BATCH_SIZE = 38  # should be a multiple of 8


FEATURES = {
    "data": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "key": tf.io.FixedLenFeature([], tf.string),
}
LOSS_WEIGHTS = {"cross_entropy": 1.0, "center_loss": 0.01}


def decode_tfrecords(x):
    features = tf.io.parse_single_example(x, FEATURES)
    image = tf.io.decode_raw(features["data"], DATA_TYPE)
    image = tf.reshape(image, DATA_SHAPE)
    features["data"] = image
    return features


def get_preprocessor():
    preprocessor = tf.keras.Sequential(
        [
            # rotate before cropping
            # 5 random degree rotation
            layers.experimental.preprocessing.RandomRotation(5 / 360),
            layers.experimental.preprocessing.RandomCrop(
                height=OUTPUT_SHAPE[0], width=OUTPUT_SHAPE[1]
            ),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            # FIXED_STANDARDIZATION from https://github.com/davidsandberg/facenet
            # [-0.99609375, 0.99609375]
            layers.experimental.preprocessing.Rescaling(
                scale=1 / 128, offset=-127.5 / 128
            ),
        ]
    )
    return preprocessor


def preprocess(preprocessor, features, augment=False):
    image = features["data"]
    label = features["label"]
    image = preprocessor(image, training=augment)
    return image, label


def prepare_dataset(tf_record_paths, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.list_files(tf_record_paths, shuffle=shuffle)
    ds = tf.data.TFRecordDataset(ds, num_parallel_reads=TFRECORD_PARALLEL_READ)
    if shuffle:
        # ignore order and read files as soon as they come in
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = ds.with_options(ignore_order)
    ds = ds.map(decode_tfrecords).prefetch(buffer_size=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER).repeat(EPOCHS)
    preprocessor = get_preprocessor()
    ds = ds.batch(batch_size).map(
        partial(preprocess, preprocessor, augment=augment), num_parallel_calls=AUTOTUNE,
    )

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)
    # return ds.apply(tf.data.experimental.prefetch_to_device(
    #         device, buffer_size=AUTOTUNE))


def accuracy_from_embeddings(labels, prelogits):
    labels = tf.reshape(labels, (-1,))
    embeddings = tf.nn.l2_normalize(prelogits, 1)
    predictions = predict_using_tensors(embeddings, labels)
    return tf.math.equal(labels, predictions)


class CustomModel(tf.keras.Model):
    def compile(
        self,
        cross_entropy,
        center_loss,
        loss_weights,
        train_loss,
        train_cross_entropy,
        train_center_loss,
        test_acc,
        global_batch_size,
        **kwargs,
    ):
        super().compile(**kwargs)
        self.cross_entropy = cross_entropy
        self.center_loss = center_loss
        self.loss_weights = loss_weights
        self.train_loss = train_loss
        self.train_cross_entropy = train_cross_entropy
        self.train_center_loss = train_center_loss
        self.test_acc = test_acc
        self.global_batch_size = global_batch_size

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            logits, prelogits = self(images, training=True)
            loss_cross = self.cross_entropy(labels, logits)
            loss_center = self.center_loss(labels, prelogits)
            loss = (
                loss_cross * self.loss_weights[self.cross_entropy.name]
                + loss_center * self.loss_weights[self.center_loss.name]
            )
            unscaled_loss = tf.nn.compute_average_loss(
                loss, global_batch_size=self.global_batch_size
            )
            loss = self.optimizer.get_scaled_loss(unscaled_loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_loss(unscaled_loss)
        self.train_cross_entropy(loss_cross)
        self.train_center_loss(loss_center)
        return {
            m.name: m.result()
            for m in [self.train_loss, self.train_cross_entropy, self.train_center_loss]
        }

    def test_step(self, data):
        images, labels = data
        logits, prelogits = self(images, training=False)
        self.test_acc(accuracy_from_embeddings(labels, prelogits))
        return {m.name: m.result() for m in [self.test_acc]}


def create_model():

    model = InceptionResNetV2(
        include_top=True,
        classes=N_CLASSES,
        bottleneck=True,
        input_shape=OUTPUT_SHAPE + (3,),
    )
    float32_layer = layers.Activation("linear", dtype="float32")

    prelogits = model.get_layer("Bottleneck/BatchNorm").output
    prelogits = CenterLossLayer(
        n_classes=N_CLASSES, n_features=prelogits.shape[-1], name="centers"
    )(prelogits)
    prelogits = float32_layer(prelogits)
    logits = float32_layer(model.get_layer("logits").output)
    model = CustomModel(
        inputs=model.input, outputs=[logits, prelogits], name=model.name
    )
    return model


def build_and_compile_model(global_batch_size):
    model = create_model()

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, name="cross_entropy", reduction=tf.keras.losses.Reduction.NONE
    )
    center_loss = CenterLoss(
        centers_layer=model.get_layer("centers"),
        alpha=0.9,
        name="center_loss",
        reduction=tf.keras.losses.Reduction.NONE,
    )

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=LEARNING_RATE, rho=0.9, momentum=0.9, epsilon=1.0
    )
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale="dynamic")

    train_loss = tf.keras.metrics.Mean(name="loss")
    train_cross_entropy = tf.keras.metrics.Mean(name="cross_entropy")
    train_center_loss = tf.keras.metrics.Mean(name="center_loss")

    test_acc = tf.keras.metrics.Mean(name="accuracy")

    model.compile(
        optimizer=optimizer,
        cross_entropy=cross_entropy,
        center_loss=center_loss,
        loss_weights=LOSS_WEIGHTS,
        train_loss=train_loss,
        train_cross_entropy=train_cross_entropy,
        train_center_loss=train_center_loss,
        test_acc=test_acc,
        global_batch_size=global_batch_size,
    )
    return model


class CustomBackupAndRestore(tf.keras.callbacks.experimental.BackupAndRestore):
    def __inti__(self, custom_objects, **kwargs):
        super().__inti__(**kwargs)
        self.custom_objects = custom_objects
        self.custom_objects_path = os.path.join(self.backup_dir, "custom_objects.pkl")

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)

        # pickle custom objects
        with open(self.custom_objects_path, "wb") as f:
            pickle.dump(self.custom_objects, f)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        if not os.path.exists(self.custom_objects_path):
            return

        # load custom objects
        with open(self.custom_objects_path, "rb") as f:
            self.custom_objects = pickle.load(f)

    def on_train_end(self, logs=None):
        # do not delete backups
        pass


def train_and_evaluate(tf_config):
    os.environ["TF_CONFIG"] = json.dumps(tf_config)

    per_worker_batch_size = BATCH_SIZE
    num_workers = len(tf_config["cluster"]["worker"])

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    val_global_batch_size = VALIDATION_BATCH_SIZE * num_workers

    train_ds = prepare_dataset(
        TRAIN_TF_RECORD_PATHS, batch_size=global_batch_size, shuffle=True, augment=True
    )

    val_ds = prepare_dataset(
        VALIDATION_TF_RECORD_PATHS,
        batch_size=val_global_batch_size,
        shuffle=False,
        augment=False,
    )

    with strategy.scope():
        model = build_and_compile_model(global_batch_size=global_batch_size)

    val_metric_name = "val_accuracy"

    def scheduler(epoch, lr):
        # 20 epochs at 0.1, 10 at 0.01 and 5 0.001
        # The epoch number here is Keras's which is different from actual epoch number
        epoch = epoch // KERAS_EPOCH_MULTIPLIER
        if epoch in range(20):
            return 0.1
        elif epoch in range(20, 30):
            return 0.01
        else:
            return 0.001

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{CHECKPOINT}/latest", verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            f"{CHECKPOINT}/best",
            monitor=val_metric_name,
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"{CHECKPOINT}/logs", update_freq=15, profile_batch="10,50"
        ),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor=val_metric_name, factor=0.2, patience=5, min_lr=0.001
        # ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    callbacks.append(CustomBackupAndRestore(backup_dir=f"{CHECKPOINT}/backup", custom_objects=callbacks))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS * KERAS_EPOCH_MULTIPLIER,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2 if os.environ.get("SGE_TASK_ID") else 1,
    )


if __name__ == "__main__":
    train_and_evaluate({})
