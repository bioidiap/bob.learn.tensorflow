#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 19 Oct 23:43:22 2016

import numpy
import bob.core
from .Base import Base

from scipy.io.wavfile import read as readWAV

logger = bob.core.log.setup("bob.learn.tensorflow")
logger.propagate = False


class DiskAudio(Base):
    def __init__(self, data, labels,
                 input_dtype="float64",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 context_size=20,
                 win_length_ms=10,
                 rate=16000,
                 out_file=""
                 ):
        """
         This datashuffler deals with speech databases that are stored in the disk.
         The data is loaded and preprocessed on the fly.

        """
        self.out_file = out_file
        self.context_size = context_size
        self.win_length_ms = win_length_ms
        self.m_win_length = self.win_length_ms * rate / 1000  # number of values in a given window
        self.m_frame_length = self.m_win_length * (2 * self.context_size + 1)

        input_shape = [self.m_frame_length, 1]

        if isinstance(data, list):
            data = numpy.array(data)

        if isinstance(labels, list):
            labels = numpy.array(labels)

        super(DiskAudio, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation
        )
        # Seting the seed
        numpy.random.seed(seed)

        # a flexible queue that stores audio frames extracted from files
        self.frames_storage = []
        # a similar queue for the corresponding labels
        self.labels_storage = []
#        if self.out_file != "":
#            bob.io.base.create_directories_safe(os.path.dirname(self.out_file))
#            f = open(self.out_file, "w")
#            for i in range(0, self.data.shape[0]):
#                f.write("%d %s\n" % (self.labels[i], str(self.data[i])))
#            f.close()


    def load_from_file(self, file_name):
        rate, audio = readWAV(file_name)
        # We consider there is only 1 channel in the audio file => data[0]
        data = numpy.cast['float32'](audio)

        return rate, data

    def get_batch(self, noise=False):
        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)
        f = None
        if self.out_file != "":
            f = open(self.out_file, "a")
        i = 0
        # if not enough in the storage, we pre-load frames from the audio files
        while len(self.frames_storage) < self.batch_size:
            if f is not None:
                f.write("%s\n" % self.data[indexes[i]])
            frames, labels = self.extract_frames_from_file(self.data[indexes[i]], self.labels[indexes[i]])
            self.frames_storage.extend(frames)
            self.labels_storage.extend(labels)
            i += 1

        # our temp frame queue should have enough data
        selected_data = numpy.asarray(self.frames_storage[:self.batch_size])
        selected_labels = numpy.asarray(self.labels_storage[:self.batch_size])
        # remove them from the list
        del self.frames_storage[:self.batch_size]
        del self.labels_storage[:self.batch_size]
        selected_data = numpy.reshape(selected_data, (self.batch_size, -1, 1))
        if f is not None:
            f.close()
        return [selected_data.astype("float32"), selected_labels.astype("int64")]

    def extract_frames_from_file(self, filename, label):
        rate, wav_signal = self.load_from_file(filename)
        return self.extract_frames_from_wav(wav_signal, label)

    def extract_frames_from_wav(self, wav_signal, label):

        m_total_length = len(wav_signal)
        m_num_win = int(m_total_length / self.m_win_length)  # discard the tail of the signal

        # normalize the signal first
        wav_signal -= numpy.mean(wav_signal)
        wav_signal /= numpy.std(wav_signal)

        # make sure the array is divided into equal chunks
        windows = numpy.split(wav_signal[:self.m_win_length * m_num_win], m_num_win)

        final_frames = []
        final_labels = [label] * m_num_win
        # loop through the windows
        for i, window in zip(range(0, len(windows)), windows):
            # window with surrounding context will form the frame we seek

            # if we don't have enough frame for the context
            # copy the first frame necessary number of times
            if i < self.context_size:
                left_context = numpy.tile(windows[0], self.context_size - i)
                final_frames.append(numpy.append(left_context, windows[:i + self.context_size + 1]))
            elif (i + self.context_size) > (m_num_win - 1):
                right_context = numpy.tile(windows[-1], i + self.context_size - m_num_win + 1)
                final_frames.append(numpy.append(windows[i - self.context_size:], right_context))
            else:
                final_frames.append(numpy.ravel(windows[i - self.context_size:i + self.context_size + 1]))

        return final_frames, final_labels
