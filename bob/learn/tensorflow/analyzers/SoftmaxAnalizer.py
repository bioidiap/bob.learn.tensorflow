#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:33 CEST

"""
Neural net work error rates analizer
"""
import numpy
import bob.measure
from tensorflow.core.framework import summary_pb2


class SoftmaxAnalizer:
    """
    Analizer.
    """

    def __init__(self, data_shuffler, machine, session):
        """
        Softmax analizer

        ** Parameters **

          data_shuffler:
          graph:
          session:
          convergence_threshold:
          convergence_reference: References to analize the convergence. Possible values are `eer`, `far10`, `far10`


        """

        self.data_shuffler = data_shuffler
        self.machine = machine
        self.session = session


        """

        placeholder_data, placeholder_labels = data_shuffler.get_placeholders(name="validation")
        graph = machine.compute_graph(placeholder_data)

        loss_validation = self.loss(validation_graph, validation_placeholder_labels)
        tf.scalar_summary('accuracy', loss_validation, name="validation")
        merged_validation = tf.merge_all_summaries()
        """

    def __call__(self, graph=None):

        validation_graph = self.compute_graph(self.data_shuffler, name="validation")

        predictions = numpy.argmax(self.session.run(network,
            feed_dict={data_node: data[:]}), 1)

        return 100. * numpy.sum(predictions == labels) / predictions.shape[0]


        data, labels = self.data_shuffler.get_batch()

        feed_dict = {validation_placeholder_data: validation_data,
                     validation_placeholder_labels: validation_labels}

        # l, predictions = session.run([loss_validation, validation_prediction, ], feed_dict=feed_dict)
        # l, summary = session.run([loss_validation, merged_validation], feed_dict=feed_dict)
        # import ipdb; ipdb.set_trace();
        l = session.run(loss_validation, feed_dict=feed_dict)
        summaries = []
        summaries.append(summary_pb2.Summary.Value(tag="loss", simple_value=float(l)))
        validation_writer.add_summary(summary_pb2.Summary(value=summaries), step)



