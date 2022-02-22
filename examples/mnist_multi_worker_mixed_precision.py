import sys

import dask
import numpy as np
import tensorflow as tf
from dask.distributed import Client
from dask_jobqueue import SGECluster

from bob.extension import rc
from bob.learn.tensorflow.callbacks import add_backup_callback

mixed_precision_policy = "mixed_float16"
strategy_fn = "multi-worker-mirrored-strategy"


N_WORKERS = 2
BATCH_SIZE = 64 * N_WORKERS
checkpoint_path = "mnist_distributed_mixed_precision"
steps_per_epoch = 60000 // BATCH_SIZE
epochs = 2


def train_input_fn(ctx=None):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    batch_size = BATCH_SIZE
    if ctx is not None:
        # shard the dataset BEFORE any shuffling
        train_dataset = train_dataset.shard(
            ctx.num_replicas_in_sync, ctx.input_pipeline_id
        )
        # calculate batch size per worker
        batch_size = ctx.get_per_replica_batch_size(BATCH_SIZE)

    # create inifinite databases, `.repeat()`, for distributed training
    train_dataset = train_dataset.shuffle(60000).repeat().batch(batch_size)
    return train_dataset


def model_fn():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
            # to support mixed precision training, output(s) must be float32
            tf.keras.layers.Activation("linear", dtype="float32"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


# dask.config.set({"distributed.comm.timeouts.connect": "30s"})
dask.config.set({"jobqueue.sge.walltime": None})
dask.config.set({"distributed.worker.memory.target": False})  # Avoid spilling to disk
dask.config.set({"distributed.worker.memory.spill": False})  # Avoid spilling to disk

cluster = SGECluster(
    queue="q_short_gpu",
    memory="28GB",
    cores=1,
    processes=1,
    log_directory="./logs",
    silence_logs="debug",
    resource_spec="q_short_gpu=TRUE,hostname=vgne*",
    project=rc.get("sge.project"),
    env_extra=[
        "export PYTHONUNBUFFERED=1",
        f"export PYTHONPATH={':'.join(sys.path)}",
        #
        # may need to unset proxies (probably set by SGE) to make sure tensorflow workers can communicate
        # see: https://stackoverflow.com/a/66059809/1286165
        # "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY",
        #
        # May need to tell dask workers not to use daemonic processes
        # see: https://github.com/dask/distributed/issues/2718
        # "export DASK_DISTRIBUTED__WORKER__DAEMON=False",
        #
        # f"export LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
    ],
)
cluster.scale(N_WORKERS)
dask_client = Client(cluster, timeout="2m")
print(f"Waiting (max 2 hours) for {N_WORKERS} dask workers to come online ...")
dask_client.wait_for_workers(n_workers=N_WORKERS, timeout="2h")
print(f"All requested {N_WORKERS} dask workers are ready!")


def scheduler(epoch, lr):
    if epoch in range(20):
        return 0.1
    elif epoch in range(20, 30):
        return 0.01
    else:
        return 0.001


callbacks = {
    "latest": tf.keras.callbacks.ModelCheckpoint(
        f"{checkpoint_path}/latest", verbose=1
    ),
    "best": tf.keras.callbacks.ModelCheckpoint(
        f"{checkpoint_path}/best",
        save_best_only=True,
        monitor="accuracy",
        mode="max",
        verbose=1,
    ),
    "tensorboard": tf.keras.callbacks.TensorBoard(
        log_dir=f"{checkpoint_path}/logs", update_freq=15, profile_batch=0
    ),
    "lr": tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
    "nan": tf.keras.callbacks.TerminateOnNaN(),
}
callbacks = add_backup_callback(callbacks, backup_dir=f"{checkpoint_path}/backup")
