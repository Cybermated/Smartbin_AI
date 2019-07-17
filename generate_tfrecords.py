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
from tfrecord_utils import RotatingRecord
from object_detection.utils import dataset_util

__description__ = 'Generates TFRecords for future trainings.'

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument('--record', default=TFRECORD_CONFIG['default'], choices=('train', 'test'),
                    help='Purpose of the tensorflow recording file, default is {default}.'.format(
                        default=TFRECORD_CONFIG['default']))
args = parser.parse_args()


def create_tfrecords(df, filename, rows):
    """
    Creates TFRecords out of each dataset image.
    :param df: current dataframe.
    :param filename: ROI location.
    :param rows: ROI properties.
    :return: updated dataframe and TFRecord.
    """
    counter = 0
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes_id = []

    for index, row in rows.iterrows():
        # Change class name if necessary.
        row['Class'] = manage_classes(row['Class'])

        # Create a new TFRecord.
        xmins.append(row['Xmin'])
        xmaxs.append(row['Xmax'])
        ymins.append(row['Ymin'])
        ymaxs.append(row['Ymax'])
        classes_text.append(row['Class'].encode('utf8'))
        classes_id.append(class_text_to_int(row['Class'], ROI_CLASSES))
        counter += 1

        # Update ROI extraction status.
        df.loc[index, 'Extraction_date'] = get_current_datetime()
        df.loc[index, 'Is_extracted'] = True

    # Get file path and extension.
    file_fullpath = get_roi_fullpath(filename).encode('utf8')
    image_format = get_file_extension(filename).encode('utf8')

    # Read file content.
    with tf.gfile.GFile(file_fullpath, 'rb') as fid:
        encoded_file = fid.read()

    # Return TFRecord.
    width, height = int(row['Width']), int(row['Height'])
    return df, tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(width),
        'image/width': dataset_util.int64_feature(height),
        'image/filename': dataset_util.bytes_feature(file_fullpath),
        'image/source_id': dataset_util.bytes_feature(file_fullpath),
        'image/encoded': dataset_util.bytes_feature(encoded_file),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes_id)
    }))


def main(_):
    """
    Main program.
    :param _: unused parameter.
    :return: void.
    """
    # Load the dataset CSV file.
    try:
        df = pd.read_csv(DATASET_CSV_PATH)
        print('Reading CSV file {csv}...'.format(csv=DATASET_CSV_PATH))
    except Exception as ee:
        print('Error while reading CSV file {csv} : {error}.'.format(csv=DATASET_CSV_PATH, error=ee))
        raise ()

    # Drop extracted, ignored and bad rows.
    indexes = list(set(df[df['Is_extracted'] == True].index) | set(df[df['Is_ignored'] == True].index) | set(
        df[df['Purpose'] != args.record].index) | set(df[df['Class'] not in ROI_CLASSES].index))
    cropped_df = df.drop(indexes, inplace=False)

    # Get dataframe dimensions.
    rows, cols = cropped_df.shape

    # Check if file is not empty.
    if rows and cols:
        # Group ROIs by filename.
        grouped_df = group_dataframe(cropped_df, 'Filename')

        # Instantiate record rotation.
        tfrecord_rotation = RotatingRecord(
            directory=TFRECORDS_DIR,
            last=find_latest_tfrecord(purpose=args.record),
            purpose=args.record,
            max_file_size=TFRECORD_CONFIG['max_size'],
        )

        # Read each ROI group.
        for filename, rows in grouped_df:
            # Generate TFRecord file.
            df, tfrecord = create_tfrecords(df, filename, rows)
            tfrecord_rotation.write(tfrecord)

        # Close last file.
        tfrecord_rotation.close()

        # Save dataset.
        df.to_csv(DATASET_CSV_PATH, mode='w', header=True, index=False, quoting=CSV_CONFIG['quoting'],
                  quotechar=CSV_CONFIG['quotechar'])
    else:
        print('Nothing to create !')


if __name__ == '__main__':
    tf.app.run()
