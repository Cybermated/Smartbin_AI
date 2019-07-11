# -*- coding: utf-8 -*-

"""
    TFRecord utils file.
    ======================
    Collection of useful functions for TFRecords generation.
"""

from utils import *
import tensorflow as tf


class RotatingRecord(object):
    """
    Class to write TFRecord files that don't exceed a certain size to make trainings more efficient.
    See https://www.tensorflow.org/guide/performance/overview
    """

    def __init__(self, directory, last, purpose, max_file_size):
        """
        Initializes file rotation.
        :param directory: file path.
        :param last: last file ID.
        :param purpose: record type.
        :param max_file_size: maximum allowed file size (bytes).
        """
        self.purpose = purpose
        self.directory = os.path.join(directory, self.purpose)
        self.max_file_size = max_file_size
        self.ii = int(last) + 1
        self.writer = None
        self.check_path()
        self.open()

    def rotate(self):
        """
        Checks if current file reached the maximum size and opens a new one if necessary.
        :return: void.
        """
        # File might be not on disk yet.
        try:
            if os.stat(self.filename_template).st_size > self.max_file_size:
                self.close()
                self.ii += 1
                self.open()
        except:
            pass

    def open(self):
        """
        Creates new empty file.
        :return: void.
        """
        self.writer = tf.python_io.TFRecordWriter(self.filename_template)

    def write(self, record):
        """
        Appends content in current file.
        :param record: TFRecord entry.
        :return: void.
        """
        self.rotate()
        self.writer.write(record.SerializeToString())

    def close(self):
        """
        Freezes current file.
        :return: void.
        """
        self.writer.close()

    def check_path(self):
        """
        Checks output directory.
        :return: void.
        """
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    @property
    def filename_template(self):
        """
        Returns current file name.
        :return: filename as string.
        """
        return os.path.join(self.directory, TFRECORD_CONFIG[self.purpose].format(id=self.ii))
