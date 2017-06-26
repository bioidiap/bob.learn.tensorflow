#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
numpy.random.seed(10)


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    with tf.name_scope('euclidean_distance') as scope:
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))
        return d


def load_mnist(perc_train=0.9):

    import bob.db.mnist
    db = bob.db.mnist.Database()
    raw_data = db.data()

    # data  = raw_data[0].astype(numpy.float64)
    data = raw_data[0]
    labels = raw_data[1]

    # Shuffling
    total_samples = data.shape[0]
    indexes = numpy.array(range(total_samples))
    numpy.random.shuffle(indexes)

    # Spliting train and validation
    n_train = int(perc_train*indexes.shape[0])
    n_validation = total_samples - n_train

    train_data = data[0:n_train, :]
    train_labels = labels[0:n_train]

    validation_data = data[n_train:n_train+n_validation, :]
    validation_labels = labels[n_train:n_train+n_validation]

    return train_data, train_labels, validation_data, validation_labels


def plot_embedding_pca(features, labels):
    """

    Trains a PCA using bob, reducing the features to dimension 2 and plot it the possible clusters

    :param features:
    :param labels:
    :return:
    """

    import bob.learn.linear
    import matplotlib.pyplot as mpl

    colors = ['#FF0000', '#FFFF00', '#FF00FF', '#00FFFF', '#000000',
             '#AA0000', '#AAAA00', '#AA00AA', '#00AAAA', '#330000']

    # Training PCA
    trainer = bob.learn.linear.PCATrainer()
    machine, lamb = trainer.train(features.astype("float64"))

    # Getting the first two most relevant features
    projected_features = machine(features.astype("float64"))[:, 0:2]

    # Plotting the classes
    n_classes = max(labels)+1
    fig = mpl.figure()

    for i in range(n_classes):
        indexes = numpy.where(labels == i)[0]

        selected_features = projected_features[indexes,:]
        mpl.scatter(selected_features[:, 0], selected_features[:, 1],
                 marker='.', c=colors[i], linewidths=0, label=str(i))
    mpl.legend()
    return fig

def plot_embedding_lda(features, labels):
    """

    Trains a LDA using bob, reducing the features to dimension 2 and plot it the possible clusters

    :param features:
    :param labels:
    :return:
    """

    import bob.learn.linear
    import matplotlib.pyplot as mpl

    colors = ['#FF0000', '#FFFF00', '#FF00FF', '#00FFFF', '#000000',
             '#AA0000', '#AAAA00', '#AA00AA', '#00AAAA', '#330000']
    n_classes = max(labels)+1


    # Training PCA
    trainer = bob.learn.linear.FisherLDATrainer(use_pinv=True)
    lda_features = []
    for i in range(n_classes):
        indexes = numpy.where(labels == i)[0]
        lda_features.append(features[indexes, :].astype("float64"))

    machine, lamb = trainer.train(lda_features)

    #import ipdb; ipdb.set_trace();


    # Getting the first two most relevant features
    projected_features = machine(features.astype("float64"))[:, 0:2]

    # Plotting the classes
    fig = mpl.figure()

    for i in range(n_classes):
        indexes = numpy.where(labels == i)[0]

        selected_features = projected_features[indexes,:]
        mpl.scatter(selected_features[:, 0], selected_features[:, 1],
                 marker='.', c=colors[i], linewidths=0, label=str(i))
    mpl.legend()
    return fig


def compute_eer(data_train, labels_train, data_validation, labels_validation, n_classes):
    import bob.measure
    from scipy.spatial.distance import cosine

    # Creating client models
    models = []
    for i in range(n_classes):
        indexes = labels_train == i
        models.append(numpy.mean(data_train[indexes, :], axis=0))

    # Probing
    positive_scores = numpy.zeros(shape=0)
    negative_scores = numpy.zeros(shape=0)

    for i in range(n_classes):
        # Positive scoring
        indexes = labels_validation == i
        positive_data = data_validation[indexes, :]
        p = [cosine(models[i], positive_data[j]) for j in range(positive_data.shape[0])]
        positive_scores = numpy.hstack((positive_scores, p))

        # negative scoring
        indexes = labels_validation != i
        negative_data = data_validation[indexes, :]
        n = [cosine(models[i], negative_data[j]) for j in range(negative_data.shape[0])]
        negative_scores = numpy.hstack((negative_scores, n))

    # Computing performance based on EER
    negative_scores = (-1) * negative_scores
    positive_scores = (-1) * positive_scores

    threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
    far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
    eer = (far + frr) / 2.

    return eer


def compute_accuracy(data_train, labels_train, data_validation, labels_validation, n_classes):
    from scipy.spatial.distance import cosine

    # Creating client models
    models = []
    for i in range(n_classes):
        indexes = labels_train == i
        models.append(numpy.mean(data_train[indexes, :], axis=0))

    # Probing
    tp = 0
    for i in range(data_validation.shape[0]):

        d = data_validation[i,:]
        l = labels_validation[i]

        scores = [cosine(m, d) for m in models]
        predict = numpy.argmax(scores)

        if predict == l:
            tp += 1

    return (float(tp) / data_validation.shape[0]) * 100
    
    
def debug_embbeding(image, architecture, embbeding_dim=2, feature_layer="fc3"):
    """
    """
    import tensorflow as tf
    from bob.learn.tensorflow.network import SequenceNetwork
    from bob.learn.tensorflow.utils.session import Session
    
    session = Session.instance(new=False).session    
    inference_graph = architecture.compute_graph(architecture.inference_placeholder, feature_layer=feature_layer, training=False)

    embeddings = numpy.zeros(shape=(image.shape[0], embbeding_dim))
    for i in range(image.shape[0]):
        feed_dict = {architecture.inference_placeholder: image[i:i+1, :,:,:]}
        embedding = session.run([tf.nn.l2_normalize(inference_graph, 1, 1e-10)], feed_dict=feed_dict)[0]
        embedding = numpy.reshape(embedding, numpy.prod(embedding.shape[1:]))
        embeddings[i] = embedding

    return embeddings

    
