#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Videostream capture.
    ======================

    Displays in real-time the videostream of any recording plugged device.
"""

from config import *
from argparse import ArgumentParser
from pkg_resources import parse_version

__description__ = "Displays in real-time the videostream of any recording plugged device."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("--device", type=int, default=DEVICE_CONFIG["id"],
                    help="Capture device identifier, default is {default}.".format(default=DEVICE_CONFIG["id"]))
args = parser.parse_args()


def capPropId(prop):
    """
    Gets property identifier of the video capture device by name.
    :param prop: string.
    :return: void.
    """
    OPCV3 = parse_version(cv.__version__) >= parse_version("3")
    return getattr(cv if OPCV3 else cv.cv,
                   ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)


def main():
    """
    Main program.
    :return: void.
    """

    # Get the first video input.
    try:
        # Retrieve video input.
        cap = cv.VideoCapture(args.device)

        # Set input properties.
        cap.set(capPropId("FRAME_WIDTH"), DEVICE_CONFIG["width"])
        cap.set(capPropId("FRAME_HEIGHT"), DEVICE_CONFIG["height"])

        while True:
            # Capture frame-by-frame.
            ret, frame = cap.read()

            if ret:
                # Operations on the frame.
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Display the frame.
                cv.imshow("Webcam videostream", gray)

                # Exit program on the Q click.
                if cv.waitKey(1) & 0xFF == ord("q"):
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
