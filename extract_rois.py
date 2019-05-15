#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    ROIs extraction (Regions of Interest).
    ======================

    Extracts ROIs from CSV files for future trainings.
"""


class Window:

    def __init__(self):
        pass


import pandas as pd

from utils import *
from config import *
from argparse import ArgumentParser
from collections import namedtuple

__description__ = "Extracts ROIs from CSV files for future trainings."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("--delete", type=bool, default=False,
                    help="Delete CSV source files instead of adding a suffix, default is {default}.".format(
                        default=False))
parser.add_argument("--ignore-size", type=bool, default=False,
                    help="Keep small ROIs, default is {default}.".format(
                        default=False))
args = parser.parse_args()


def group_rois(df, group):
    """
    Groups ROIs by their filenames.
    :param df:
    :param group:
    :return:
    """
    data = namedtuple("data", ["Path", "object"])
    gb = df.groupby(group)
    return [data(path, gb.get_group(x)) for path, x in zip(gb.groups.keys(), gb.groups)]


def get_rois(csv, csv_config):
    """
    Fetches and filters ROIs in CSV files.
    :param csv: csv file location.
    :param csv_config: csv properties.
    :return: void.
    """
    print("Reading {csv}...".format(csv=os.path.basename(csv)))

    # Exclude empty ROIs.
    df = pd.read_csv(csv, delimiter=csv_config["delimiter"], header=0).dropna(subset=["Class"])
    if df.empty is True:
        print("{csv} with none ROI.".format(csv=csv))
        return

    # Get raw images location.
    folder_name = os.path.splitext(os.path.basename(csv))[0].split('_')[1]
    img_location = os.path.join(FINAL_FOLDERS_DIR, folder_name)

    # Group ROIs by filename.
    grouped_df = group_rois(df, "Path")

    # Extract ROIs from each group.
    for group in grouped_df:
        extract_rois(os.path.join(img_location, group.Path), group.object, ROI_CONFIG, folder_name)

    # Delete or rename CSV file.
    if not args.delete:
        add_suffix(csv, CSV_CONFIG["suffix"])
    else:
        try:
            os.remove(csv)
        except Exception:
            pass


def bandw_roi(frame, color=cv.COLOR_RGB2GRAY):
    """
    Converts frames in greyscale.
    :param frame: source image file.
    :param color: color mode.
    :return: colorized frame.
    """
    return cv.cvtColor(frame, color)


def extract_rois(img_path, rows, roi_config, folder_name):
    """
    Extract ROIs from frames to save them to local disk.
    :param img_path: frame path..
    :param rows: ROI properties.
    :param roi_config: ROI frame properties.
    :return: void.
    """
    if os.path.isfile(img_path):
        try:
            # Read frame as greyscale image.
            source = bandw_roi(cv.imread(img_path))

            # Generate image fullname.
            name = name_roi(roi_config)

            # Get image dimensions.
            height, width = source.shape

            # Pick a random purpose.
            purpose = random.choice(TFRECORD_CONFIG["weights"])

            # Init valid record counter.
            roi_counter = 0

            # Loop over ROIs.
            for index, row in rows.iterrows():
                if keep_roi(row, (width, height)) or args.ignore_size:
                    write_csv(DATASET_CSV_PATH, CSV_CONFIG, name, width, height, purpose, row, folder_name)
                    roi_counter += 1

            # Save frame if ROIs matched.
            if roi_counter > 0:
                save_roi(source, name)
        except Exception:
            pass


def name_roi(roi_config):
    """
    Generates ROI filenames.
    :param roi_infos: ROI properties.
    :param chars: allowed chars.
    :param size: ROI size name.
    :param ext: ROI extension.
    :return: ROI full path.
    """
    return random_name(roi_config["chars"], roi_config["size"]) + roi_config["ext"]


def save_roi(roi, roi_name):
    """
    Saves ROIs to local disk.
    :param roi: source image file.
    :param roi_name: filename.
    :return: void.
    """
    roi_path = get_roi_fullpath(roi_name)
    try:
        if not os.path.isfile(roi_path):
            cv.imwrite(roi_path, roi)
            return True
    except Exception as ee:
        print("Error while saving ROI {roi_path} : {error}.".format(roi_path=roi_path, error=ee))
    return False


def generate_csv(csv_path, csv_config):
    """
    Generates the dataset CSV file with headers.
    :param csv_path: CSV file path.
    :param csv_config: CSV properties.
    :return: void.
    """
    with open(csv_path, 'w+', newline=csv_config["newline"]) as csv_file:
        fw = csv.writer(csv_file, delimiter=csv_config["delimiter"], quotechar=csv_config["quotechar"],
                        quoting=csv_config["quoting"])
        # Add headers.
        fw.writerow(["Filename", "Folder", "Width", "Height", "Class", "Confidence", "Xmin", "Ymin", "Xmax", "Ymax",
                     "Is_occluded", "Is_truncated", "Is_depiction", "Is_extracted", "Is_augmented", "Is_augmentation",
                     "Augmentation", "Purpose", "Generation_date", "Extraction_date", "Augmentation_date"])


def write_csv(csv_path, csv_config, name, width, height, purpose, row, folder_name):
    """
    Fills the CSV file with ROIs.
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
    with open(csv_path, 'a', newline=csv_config["newline"]) as csv_file:
        fw = csv.writer(csv_file, delimiter=csv_config["delimiter"], quotechar=csv_config["quotechar"],
                        quoting=csv_config["quoting"])
        # Append a new line.
        fw.writerow(
            [name, folder_name, width, height, row["Class"], row["Confidence"], row["Xmin"], row["Ymin"], row["Xmax"],
             row["Ymax"], row["Is_occluded"], row["Is_truncated"], row["Is_depiction"], "False", "False", "False", "",
             purpose, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "", ""])


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

    for csv_file in list_files(CSV_DIR, CSV_CONFIG["ext"], CSV_CONFIG["suffix"]):
        get_rois(csv_file, CSV_CONFIG)


if __name__ == "__main__":
    main()
