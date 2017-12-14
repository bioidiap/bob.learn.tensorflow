import tensorflow as tf
data = tf.feature_column.numeric_column('data', shape=[784])
estimator = tf.estimator.LinearClassifier(feature_columns=[data], n_classes=10)
