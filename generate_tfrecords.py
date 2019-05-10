#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TFRecords generation.
    ======================

    Generates TFRecords for future trainings.
"""

import pandas as pd
import tensorflow as tf

from utils import *
from config import *
from argparse import ArgumentParser
from collections import namedtuple

from object_detection.utils import dataset_util

__description__ = "Generates TFRecords for future trainings."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("--record", default=TFRECORD_CONFIG["default"], choices=("train", "test"),
                    help="Purpose of the tensorflow recording file, default is {default}.".format(
                        default=TFRECORD_CONFIG["default"]))
args = parser.parse_args()


def group_rois(df, group):
    """
    Groups ROIs by their filenames.
    :param df: dataframe.
    :param group: which column to group on.
    :return: grouped dataframe.
    """
    data = namedtuple("data", ["Filename", "object"])
    gb = df.groupby(group)
    return [data(path, gb.get_group(x)) for path, x in zip(gb.groups.keys(), gb.groups)]


def create_tfrecords(df, filename, rows):
    """
    Creates TFRecords out of each dataset image.
    :param df: current dataframe.
    :param filename: ROI location.
    :param rows: ROI properties.
    :return: updated dataframe and TFRecord.
    """
    counter = 0
    file_fullpath = get_roi_fullpath(filename).encode("utf8")
    file_extension = filename.split('.')[-1]
    image_format = file_extension.encode("utf8")
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes_id = []

    with tf.gfile.GFile(file_fullpath, "rb") as fid:
        encoded_file = fid.read()

    for index, row in rows.iterrows():
        # Exclude non-relevant ROIs (depictions excluded).
        if not row["Is_extracted"] and row["Purpose"] == args.record and not row["Is_depiction"]:
            # Create a new TFRecord.
            xmins.append(row["Xmin"])
            xmaxs.append(row["Xmax"])
            ymins.append(row["Ymin"])
            ymaxs.append(row["Ymax"])
            classes_text.append(row["Class"].encode("utf8"))
            classes_id.append(class_text_to_int(row["Class"], ROI_CLASSES))
            counter += 1

            # Update ROI extraction status.
            df.loc[index, "Extraction_date"] = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
            df.loc[index, "Is_extracted"] = True

    if xmins:
        return df, tf.train.Example(features=tf.train.Features(feature={
            "image/height": dataset_util.int64_feature(row["Height"]),
            "image/width": dataset_util.int64_feature(row["Width"]),
            "image/filename": dataset_util.bytes_feature(file_fullpath),
            "image/source_id": dataset_util.bytes_feature(file_fullpath),
            "image/encoded": dataset_util.bytes_feature(encoded_file),
            "image/format": dataset_util.bytes_feature(image_format),
            "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
            "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
            "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
            "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
            "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
            "image/object/class/label": dataset_util.int64_list_feature(classes_id)
        }))

    return df, None


def main(_):
    """
    Main program.
    :param _: unused parameter.
    :return: void.
    """
    # Load the dataset CSV file.
    try:
        df = pd.read_csv(DATASET_CSV_PATH)
        print("Reading CSV file {csv}...".format(csv=DATASET_CSV_PATH))
        rows, cols = df.shape
    except Exception as ee:
        print("Error while reading CSV file {csv} : {error}.".format(csv=DATASET_CSV_PATH, error=ee))
        exit(1)

    # Check if file is not empty.
    if rows and cols:

        # Group ROIs by filename.
        grouped_df = group_rois(df, "Filename")

        # Init an empty record file.
        writer = tf.python_io.TFRecordWriter(os.path.join(TFRECORDS_DIR, TFRECORD_CONFIG[args.record]))

        # Read each ROI group.
        for filename, rows in grouped_df:
            df, tfrecord = create_tfrecords(df, filename, rows)
            if tfrecord is not None:
                writer.write(tfrecord.SerializeToString())

        # Close record file.
        writer.close()

        # Update the dataset CSV file.
        df.to_csv(DATASET_CSV_PATH, mode='w', header=True, index=False, quoting=CSV_CONFIG["quoting"],
                  quotechar=CSV_CONFIG["quotechar"])
    else:
        print("Nothing to create !")


if __name__ == "__main__":
    tf.app.run()
