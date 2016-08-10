#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:33 CEST

"""
Neural net work error rates analizer
"""
import tensor
import numpy
import bob.measure

class Analizer:
    """
    Analizer.

    I don't know if this is the best way to do, but what this class do is the following.

    As an enrollment sample, averare all the TRAINING samples for one particular class.
    The probing is done with the validation set

    """

    def __init__(self, analizer_architecture_file, n_classes, extractor_file, snapshot, mu,
                 single_batch=False, end_cnn="feature"):
        """
        Use the CNN as feature extractor for a n-class classification

        ** Parameters **
         temp_model: Caffe model
         analizer_architecture_file: prototxt with the architecture
         n_classes: Number of classes
         single_batch: Do the forwar in a single batch? For huge architectures will consume a lot of memory
        """

        self.temp_model = None
        self.analizer_architecture_file = analizer_architecture_file
        self.n_classes = n_classes
        self.single_batch = single_batch
        # Statistics
        self.eer = []
        self.far10 = []
        self.far100 = []
        self.far1000 = []
        self.validation_loss = []
        self.end_cnn = end_cnn

        self.extractor_file = extractor_file
        self.snapshot = snapshot
        self.mu = mu

    def update_model(self, model):
        self.temp_model = model

    def save_stats(self, it):
        # Saving statistics
        hdf5 = bob.io.base.HDF5File(self.extractor_file, "w")
        hdf5.set("iterations", it)
        hdf5.set("snapshot", self.snapshot)
        #hdf5.set("validationloss", loss)
        hdf5.set("eer", self.eer)
        hdf5.set("far10", self.far10)
        hdf5.set("far100", self.far100)
        hdf5.set("far1000", self.far1000)
        hdf5.set("mu", self.mu)

        del hdf5

    def compute_stats(self, negative_scores, positive_scores):
        """
        Compute some stats with the scores, such as:
          - EER
          - FAR 10
          - FAR 100
          - FAR 1000
          - RANK 1
          - RANK 10

        **Parameters**
          negative_scores:
          positive_scores:
        """

        # Compute EER
        threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        eer = (far + frr) / 2.
        self.eer .append(eer)

        # Computing FAR 10
        threshold = bob.measure.far_threshold(negative_scores, positive_scores, far_value=0.1)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        self.far10.append(frr)

        # Computing FAR 100
        threshold = bob.measure.far_threshold(negative_scores, positive_scores, far_value=0.01)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        self.far100.append(frr)

        # Computing FAR 1000
        threshold = bob.measure.far_threshold(negative_scores, positive_scores, far_value=0.001)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        self.far1000.append(frr)

    def compute_features(self, net, data):

        if self.single_batch:
            features = net.forward(data=data)[self.end_cnn]
        else:
            features = numpy.zeros(shape=0, dtype='float32')
            for d in data:
                d = numpy.reshape(d,(1,d.shape[0],d.shape[1],d.shape[2]))
                if features.shape[0] == 0:
                    features = net.forward(data=d, end=self.end_cnn)[self.end_cnn][0]
                else:
                    features = numpy.vstack((features, net.forward(data=d, end=self.end_cnn)[self.end_cnn][0]))
        return features

    def __call__(self, data_train, labels_train, data_validation, labels_validation, it):
        from scipy.spatial.distance import cosine

        net = caffe.Net(self.analizer_architecture_file, self.temp_model, caffe.TEST)
        labels_train = numpy.reshape(labels_train[:], (labels_train.shape[0],))
        labels_validation = numpy.reshape(labels_validation[:], (labels_validation.shape[0],))

        # Projecting the data to train the models
        features = self.compute_features(net, data_train)

        # Creating client models
        models = []
        for i in range(self.n_classes):
            indexes = labels_train == i
            models.append(numpy.mean(features[indexes, :], axis=0))

        # Projecting the data for probing
        del features
        features = self.compute_features(net, data_validation)

        # Probing
        positive_scores = numpy.zeros(shape=0)
        negative_scores = numpy.zeros(shape=0)

        for i in range(self.n_classes):
            # Positive scoring
            indexes = labels_validation == i
            positive_data = features[indexes, :]
            p = [cosine(models[i], positive_data[j]) for j in range(positive_data.shape[0])]
            positive_scores = numpy.hstack((positive_scores, p))

            # negative scoring
            indexes = labels_validation != i
            negative_data = features[indexes, :]
            n = [cosine(models[i], negative_data[j]) for j in range(negative_data.shape[0])]
            negative_scores = numpy.hstack((negative_scores, n))

        #Computing performance based on EER
        negative_scores = (-1) * negative_scores
        positive_scores = (-1) * positive_scores

        self.compute_stats(negative_scores, positive_scores)
        self.save_stats(it)

        return self.eer[-1]
