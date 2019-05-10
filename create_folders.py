#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Annotation folders creation.
    ======================

    Creates annotation folders for labelling purposes.
"""

import zipfile as zf

from utils import *
from config import *
from shutil import copy as cp
from shutil import move as mv
from shutil import rmtree as rt
from argparse import ArgumentParser

__description__ = "Extracts ROIs from CSV files for future trainings."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("--delete", type=bool, default=False,
                    help="Delete image source files, default is {default}.".format(default=False))
args = parser.parse_args()


def create_dir(pardir, subdir, chars, size):
    """
    Creates new annotation folders.
    :param pardir: parent directory.
    :param chars: allowed chars.
    :param size: directory size name.
    :return: directory path.
    """
    dir_name = random_name(chars, size, False)
    dir = {
        "name": dir_name,
        "root_path": os.path.join(pardir, dir_name + os.sep),
        "img_path": os.path.join(pardir, dir_name, subdir),
        "content": []
    }
    try:
        os.makedirs(dir["img_path"], exist_ok=True)
    except Exception as ee:
        raise
    return dir


def copy_file(file, dir):
    """
    Copies image files to their folders.
    :param file: source file path.
    :param dir: target directory path.
    :return: void.
    """
    try:
        cp(file, dir["img_path"])
        dir["content"].append(os.path.basename(dir["img_path"]) + os.path.basename(file))
        if args.delete:
            try:
                rt(file)
            except OSError as ose:
                print("Error while trying to delete image file : {error}".format(error=ose))
    except IOError as ioe:
        print("Error while trying to save image file : {error}".format(error=ioe))


def zip_directory(dir):
    """
    Zips annotation directories.
    :param dir: directory path.
    :return: void.
    """
    print("Compressing folder {folder}...".format(folder=dir["name"]))
    with zf.ZipFile(os.path.join(FOLDERS_DIR, dir["name"]) + ".zip", "w") as zip_file:
        paths = []

        # Remove absolute path in the Zip.
        abs_path = len(dir["root_path"])

        # Read all files in the annotation folder.
        for r, d, f in os.walk(dir["root_path"]):
            for filename in f:
                paths.append(os.path.join(r, filename))

        for file in paths:
            zip_file.write(file, compress_type=zf.ZIP_DEFLATED, arcname=file[abs_path:])


def generate_csv(dir, csv_config):
    print("Generating CSV file for {folder}...".format(folder=dir["name"]))
    """
    Generates CSV files for annotation.
    :param dir: directory path.
    :param csv_config: CSV file properties.
    :return: void.
    """
    with open(os.path.join(dir["root_path"] + csv_config["name_pattern"].format(folder=dir["name"])), "w+",
              newline=csv_config["newline"]) as csv_file:
        # Add headers.
        fw = csv.writer(csv_file, delimiter=csv_config["delimiter"], quotechar=csv_config["quotechar"],
                        quoting=csv_config["quoting"])
        fw.writerow(
            ["Path", "Class", "Xmin", "Ymin", "Xmax", "Ymax", "Confidence", "Is_occluded", "Is_truncated",
             "Is_depiction"])

        # Write file line by line.
        for file in dir["content"]:
            fw.writerow(
                [os.path.join(
                    os.path.basename(os.path.dirname(dir["img_path"])), file), "", "", "", "", "", "", "", "", ""])


def main():
    """
    Main program.
    :return: void.
    """
    curr_subdir = None
    i = 0
    files = list_files(RAW_IMAGES_DIR, IMG_EXT)
    for _ in files:

        # Create a new empty directory.
        if i % FOLDER_CONFIG["items"] == 0:
            curr_subdir = create_dir(FOLDERS_DIR, FOLDER_CONFIG["img_dir"], FOLDER_CONFIG["chars"],
                                     FOLDER_CONFIG["size"])
            print("Generating folder {folder}...".format(folder=curr_subdir["name"]))
        copy_file(_, curr_subdir)

        i += 1

        # If the folder reaches its max size or if it is the last image.
        if i % FOLDER_CONFIG["items"] == 0 or files.index(_) == len(files) - 1:
            print("Folder {folder} is full !".format(folder=curr_subdir["name"]))
            # Generate CSV file and zip directory.
            generate_csv(curr_subdir, CSV_CONFIG)
            zip_directory(curr_subdir)

            # Move folder for ROI extraction.
            mv(curr_subdir["root_path"], FINAL_FOLDERS_DIR)


if __name__ == "__main__":
    main()
