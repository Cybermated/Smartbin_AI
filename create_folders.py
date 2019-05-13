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

__description__ = "Creates annotation folders for labelling purposes."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("-del", "--delete", dest="delete",
                    type=bool,
                    default=True,
                    help="Delete image source files, default is {default}.".format(default=True))
parser.add_argument("-z", "--zip", dest="zip",
                    type=bool,
                    default=True,
                    help="Zip directories, default is {default}.".format(default=True))
args = parser.parse_args()


def create_dir(parent_dir, img_dir, chars, size):
    """
    Creates new annotation folders.
    :param parent_dir: parent directory.
    :param img_dir: images directory.
    :param chars: allowed chars.
    :param size: directory size name.
    :return: directory path.
    """
    dir_name = random_name(chars, size, False)
    dir_dict = {
        "name": dir_name,
        "root_path": os.path.join(parent_dir, dir_name),
        "img_path": os.path.join(parent_dir, dir_name, img_dir),
        "content": []
    }
    try:
        os.makedirs(dir_dict["img_path"], exist_ok=True)
    except Exception as ee:
        raise
    return dir_dict


def copy_file(file_path, dir_dict):
    """
    Copies image files to their folders.
    :param file_path: source file path.
    :param dir_dict: target directory path.
    :return: void.
    """
    try:
        cp(file_path, dir_dict["img_path"])
        dir_dict["content"].append(os.path.join(os.path.basename(dir_dict["img_path"]), os.path.basename(file_path)))
        if args.delete:
            try:
                rt(file_path)
            except OSError as ose:
                print("Error while trying to delete image file : {error}".format(error=ose))
    except IOError as ioe:
        print("Error while trying to save image file : {error}".format(error=ioe))


def zip_directory(dir_dict):
    """
    Zips annotation directories.
    :param dir_dict: directory dictionary.
    :return: void.
    """
    print("Compressing folder {folder}...".format(folder=dir_dict["name"]))
    with zf.ZipFile(os.path.join(FOLDERS_DIR, dir_dict["name"]) + ".zip", "w") as zip_file:
        paths = []

        # Remove absolute path in the Zip.
        abs_path = len(dir_dict["root_path"])

        # Read all files in the annotation folder.
        for r, d, f in os.walk(dir_dict["root_path"]):
            for filename in f:
                paths.append(os.path.join(r, filename))

        for file in paths:
            zip_file.write(file, compress_type=zf.ZIP_BZIP2, arcname=file[abs_path:])


def generate_csv(dir_dict, csv_config):
    print("Generating CSV file for {folder}...".format(folder=dir_dict["name"]))
    """
    Generates CSV files for annotation.
    :param dir: directory path.
    :param csv_config: CSV file properties.
    :return: void.
    """
    with open(os.path.join(dir_dict["root_path"], csv_config["name_pattern"].format(folder=dir_dict["name"])), "w+",
              newline=csv_config["newline"]) as csv_file:
        # Add headers.
        fw = csv.writer(csv_file, delimiter=csv_config["delimiter"], quotechar=csv_config["quotechar"],
                        quoting=csv_config["quoting"])
        fw.writerow(
            ["Path", "Class", "Xmin", "Ymin", "Xmax", "Ymax", "Confidence", "Is_occluded", "Is_truncated",
             "Is_depiction"])

        # Write file line by line.
        for file in dir_dict["content"]:
            fw.writerow([file, "", "", "", "", "", "", "", "", ""])


def main():
    """
    Main program.
    :return: void.
    """
    curr_folder = None
    i = 0
    imgs = list_files(RAW_IMAGES_DIR, IMG_EXT)
    for img in imgs:

        # Create a new empty directory.
        if i % FOLDER_CONFIG["items"] == 0:
            curr_folder = create_dir(
                parent_dir=FOLDERS_DIR,
                img_dir=FOLDER_CONFIG["img_dir"],
                chars=FOLDER_CONFIG["chars"],
                size=FOLDER_CONFIG["size"]
            )
            print("Generating folder {folder}...".format(folder=curr_folder["name"]))
        copy_file(img, curr_folder)

        i += 1

        # If the folder reaches its max size or if it is the last image.
        if i % FOLDER_CONFIG["items"] == 0 or imgs.index(img) == len(imgs) - 1:
            print("Folder {folder} is full !".format(folder=curr_folder["name"]))

            # Generate CSV file.
            generate_csv(curr_folder, CSV_CONFIG)

            # Zip directory if enabled.
            if args.zip:
                zip_directory(curr_folder)

            # Move folder for future extraction.
            mv(curr_folder["root_path"], FINAL_FOLDERS_DIR)


if __name__ == "__main__":
    main()
