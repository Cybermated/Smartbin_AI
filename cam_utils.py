#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Cam utils file.
    ======================
    Collection of useful functions for webcam detection.
"""

from utils import *
from config import *
from threading import Thread


class FPS:
    """
    Class to get information about framerate.
    """

    def __init__(self):
        """
        Stores the start time, end time, and total number of frames that were examined
        between the start and end intervals.
        """
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        """
        Starts the timer.
        :return: itself.
        """
        self._start = datetime.now()
        return self

    def stop(self):
        """
        Stops the timer.
        :return: void.
        """
        self._end = datetime.now()

    def update(self):
        """
        Increments the total number of frames examined during the start and end intervals
        :return: void.
        """
        self._numFrames += 1

    def elapsed(self):
        """
        Returns the total number of seconds between the start and  end interval.
        :return: elapsed time in seconds.
        """
        return (self._end - self._start).total_seconds()

    def fps(self):
        """
        Computes the (approximate) frames per second.
        :return: number of frame per second.
        """
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    """
    Class to retrieve the videostream of a capture device frame per frame.
    """

    def __init__(self, src, width, height):
        """
        # Initializes the video camera stream and read the first frame from the stream.
        :param src: capture device identifier.
        :param width: width.
        :param height: height.
        """
        self.stream = cv.VideoCapture(src)
        self.stream.set(get_prop_id("FRAME_WIDTH"), width)
        self.stream.set(get_prop_id("FRAME_HEIGHT"), height)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        """
        Starts the Thread to read frames from the video stream.
        :return: itself.
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Keeps looping infinitely until the Thread is stopped.
        :return: void.
        """
        while True:
            if self.stopped:
                return

            # Read the next frame from the stream.
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """
        Returns the frame the most recently read.
        :return: frame.
        """
        return self.frame

    def stop(self):
        """
        Indicates that the Thread should be stopped.
        :return: void.
        """
        self.stopped = True
