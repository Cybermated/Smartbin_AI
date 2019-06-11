#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Frames extraction.
    ======================

    Extracts frames of video files to populate the raw images dataset.

    Usage:
        extract_frames.py [--compress True --quality 85 --delete False]

    Options:
        compress (bool): Compress extracted frames.
        quality (int): Compression quality, applied if compression is set to True.
        delete (bool): Delete video files.
"""

from utils import *
from config import *
from argparse import ArgumentParser

__description__ = "Extracts frames of video files to populate the raw images dataset."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("--compress", type=bool, default=True,
                    help="Apply image compression, default is {default}.".format(default=True))
parser.add_argument("--quality", type=bool, default=FRAME_CONFIG['quality'],
                    help="Compression quality, only applied if compression is enabled, default is {default}.".format(
                        default=FRAME_CONFIG['quality']))
parser.add_argument("--delete", type=bool, default=False,
                    help="Delete video source files, default is {default}.".format(default=False))
args = parser.parse_args()


def extract_frame(video, output, frame_config, video_config):
    """
    Extracts and saves video frames.
    :param video: video file.
    :param output: output directory.
    :param frame_config: frame properties.
    :param video_config: video properties.
    :return: void.
    """
    print("Reading {video}.".format(video=os.path.basename(video)))
    f = 0
    s = 0
    cap = cv.VideoCapture(video)
    fps = cap.get(cv.CAP_PROP_FPS)
    modulo = fps // video_config["interval"]

    print("Framerate for file {file} is {fps}.".format(file=os.path.basename(video), fps=round(fps)))
    print("Extracting frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if not f % modulo:
                name = random_name(chars=frame_config["chars"], size=frame_config["size"],
                                   use_date=frame_config["date"])
                try:
                    if args.compress:
                        cv.imwrite(os.path.join(output, name + frame_config["ext"]), frame,
                                   [cv.IMWRITE_JPEG_QUALITY, args.quality])
                    else:
                        cv.imwrite(os.path.join(output, name + frame_config["ext"]), frame)
                    s += 1
                except Exception as ee:
                    print("Error while saving frame : {error}".format(error=ee))
            f += 1
        else:
            break
    print("{frame} frames extracted from {file}.".format(frame=s, file=os.path.basename(video)))
    cap.release()
    cv.destroyAllWindows()

    # Add a suffix to the video file.
    if not args.delete:
        add_suffix(video, video_config["suffix"])
    else:
        # Delete video file.
        try:
            os.remove(video)
        except Exception as ee:
            print("Error while deleting file {file} : {error}".format(file=video, error=ee))


def main():
    """
    Main program.
    :return: void.
    """
    for _ in list_files(RAW_VIDEOS_DIR, VIDEO_EXT, VIDEO_CONFIG["suffix"]):
        extract_frame(_, RAW_IMAGES_DIR, FRAME_CONFIG, VIDEO_CONFIG)


if __name__ == "__main__":
    main()
