#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    ROIs extraction (Regions of Interest).
    ======================

    Extracts ROIs from CSV files for future trainings.

    Usage:
        extract_rois.py [--greyscale True --ignore-size False]

    Options:
        greyscale (bool): Convert ROIs in greyscale (improve training speed but may affect its accuracy if enabled).
        ignore-size (bool): Keep small boxes (may affect training accuracy if enabled).
"""

import pandas as pd

from sys import platform
from utils import *
from config import *
from argparse import ArgumentParser

__description__ = 'Extracts ROIs from CSV files for future trainings.'

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument('--greyscale', type=bool, default=True,
                    help='Extract images as greyscale, default is {default}.'.format(default=True))
parser.add_argument('--ignore-size', type=bool, default=True,
                    help='Keep small ROIs, default is {default}.'.format(
                        default=True))
args = parser.parse_args()


def get_rois(csv, csv_config):
    """
    Fetches and filters ROIs in CSV files.
    :param csv: csv file location.
    :param csv_config: csv properties.
    :return: void.
    """
    print('Reading {csv}...'.format(csv=os.path.basename(csv)))

    # Exclude empty ROIs.
    df = pd.read_csv(csv, delimiter=csv_config['delimiter'], header=0).dropna(subset=['Class'])
    if df.empty is True:
        print('{csv} with none ROI.'.format(csv=csv))
        return

    # Get raw images location.
    folder_name = os.path.splitext(os.path.basename(csv))[0].split('_')[1]
    img_location = os.path.join(FINAL_FOLDERS_DIR, folder_name)

    # Group ROIs by filename.
    grouped_df = group_dataframe(df, 'Path')

    # Extract ROIs from each group.
    for group in grouped_df:
        extract_rois(os.path.join(img_location, group.Path), group.object, ROI_CONFIG, folder_name)

    # Rename CSV file.
    add_suffix(csv, CSV_CONFIG['suffix'])


def extract_rois(img_path, rows, roi_config, folder_name):
    """
    Extract ROIs from frames to save them to local disk.
    :param img_path: frame path..
    :param rows: ROI properties.
    :param roi_config: ROI frame properties.
    :return: void.
    """
    if platform == 'linux':
        # Fix OS separator that might be wrong on Linux.
        img_path = img_path.replace('\\', os.sep)

    if os.path.isfile(img_path):
        try:
            # Init valid ROIs array.
            rois = []

            # Read frame.
            source = cv.imread(img_path)

            # Loop over ROIs.
            for index, row in rows.iterrows():
                rois.append(row)

            # Save frame if ROIs matched.
            if rois:
                # Generate image name.
                name = get_roi_name()

                if save_roi(source, name):
                    # Pick a random purpose.
                    purpose = random.choice(TFRECORD_CONFIG['weights'])

                    # Get image dimensions.
                    height, width, channels = source.shape

                    # Fill CSV file with ROIs.
                    for roi in rois:
                        fill_csv(DATASET_CSV_PATH, CSV_CONFIG, name, width, height, purpose, roi, folder_name)
        except Exception:
            pass


def save_roi(roi, roi_name):
    """
    Saves ROIs to local disk.
    :param roi: source image file.
    :param roi_name: filename.
    :return: void.
    """
    try:
        roi_path = get_roi_fullpath(roi_name)
        if not os.path.isfile(roi_path):
            cv.imwrite(roi_path, image_to_greyscale(roi) if args.greyscale else roi)
            return True
    except Exception:
        pass
    return False


def generate_csv(csv_path, csv_config):
    """
    Generates the dataset CSV file with headers.
    :param csv_path: CSV file path.
    :param csv_config: CSV properties.
    :return: void.
    """
    with open(csv_path, 'w+', newline=csv_config['newline']) as csv_file:
        fw = csv.writer(csv_file, delimiter=csv_config['delimiter'], quotechar=csv_config['quotechar'],
                        quoting=csv_config['quoting'])
        # Add headers.
        fw.writerow(CSV_STRUCTURE['dataset'])


def fill_csv(csv_path, csv_config, name, width, height, purpose, row, folder_name):
    """
    Fills the dataset CSV file with ROIs.
    :param csv_path:
    :param csv_config:
    :param name:
    :param width:
    :param height:
    :param purpose:
    :param row:
    :param folder_name:
    :return:
    """
    with open(csv_path, 'a', newline=csv_config['newline']) as csv_file:
        fw = csv.writer(csv_file, delimiter=csv_config['delimiter'], quotechar=csv_config['quotechar'],
                        quoting=csv_config['quoting'])
        # Append a new line.
        fw.writerow(
            [name, folder_name, width, height, row['Class'], row['Confidence'], row['Xmin'], row['Ymin'], row['Xmax'],
             row['Ymax'], row['Is_occluded'], row['Is_truncated'], row['Is_depiction'], 'False', 'False', 'False',
             str(ignore_roi(row)), '', purpose, get_current_datetime(), '', ''])


def main():
    """
    Main program.
    :return: void.
    """
    # Generate empty CSV file if it doesn't already exists.
    if not os.path.isfile(DATASET_CSV_PATH):
        generate_csv(DATASET_CSV_PATH, CSV_CONFIG)

    # Generate output directory if it doesn't already exists.
    if not os.path.isdir(ROIS_PATH):
        os.makedirs(ROIS_PATH)

    for csv_file in list_files(CSV_DIR, CSV_CONFIG['ext'], CSV_CONFIG['suffix']):
        get_rois(csv_file, CSV_CONFIG)


if __name__ == '__main__':
    main()
