#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Videostream capture.
    ======================

    Displays in real-time the videostream of any recording plugged device.
"""

from utils import *
from config import *
from argparse import ArgumentParser

__description__ = "Displays in real-time the videostream of any recording plugged device."

# Parse args.
parser = ArgumentParser(description=__description__)

parser.add_argument("-v-source", "--video-source", dest="video_source",
                    type=int,
                    default=DEVICE_CONFIG["id"],
                    help="Capture device identifier, default is {default}.".format(default=DEVICE_CONFIG["id"]))

parser.add_argument("-q", "--quality", dest="quality",
                    type=str,
                    default=DEVICE_CONFIG["resolution"],
                    help="Input quality, default is {default}.".format(default=DEVICE_CONFIG["resolution"]))

args = parser.parse_args()


def main():
    """
    Main program.
    :return: void.
    """

    # Get the first video input.
    try:
        # Retrieve video input.
        cap = cv.VideoCapture(args.video_source)

        width, height = INPUT_RESOLUTION[args.quality]["width"], INPUT_RESOLUTION[args.quality]["height"]

        # Set input properties.
        cap.set(get_prop_id("FRAME_WIDTH"), width)
        cap.set(get_prop_id("FRAME_HEIGHT"), height)

        while True:
            # Capture frame-by-frame.
            ret, frame = cap.read()

            if ret:
                # Display the frame.
                cv.imshow("Webcam videostream ({width} x {height})".format(width=width, height=height), frame)

                # Exit program on the Q click.
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release capture and close windows.
        cap.release()
        cv.destroyAllWindows()

    except Exception as ee:
        print("Device {device} not found : {error}.".format(device=args.device, error=ee))


if __name__ == "__main__":
    main()
