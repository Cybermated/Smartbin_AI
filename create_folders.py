#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Annotation folders creation.
    ======================

    Generates annotation folders.

    Usage:
        create_folders.py [--delete False --greyscale True]

    Options:
        delete (bool): Delete image files.
        greyscale (bool): Use greyscale images.
"""

import zipfile as zf

from utils import *
from config import *
from argparse import ArgumentParser
from shutil import copy as cp, copytree as cpt, move as mv, rmtree as rt

__description__ = 'Creates annotation folders for labelling purposes.'

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument('--delete',
                    type=bool,
                    default=False,
                    help='Delete image source files, default is {default}.'.format(default=False))
parser.add_argument('--greyscale', type=bool, default=True,
                    help='Use greyscale images, default is {default}.'.format(
                        default=True))
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
        'name': dir_name,
        'root_path': os.path.join(parent_dir, dir_name),
        'img_path': os.path.join(parent_dir, dir_name, img_dir),
        'content': []
    }
    try:
        os.makedirs(dir_dict['img_path'], exist_ok=True)
    except Exception:
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
        name = random_name(use_date=False, chars=FRAME_CONFIG['chars'], size=FRAME_CONFIG['size'])
        cp(file_path, os.path.join(dir_dict['img_path'], name))
        dir_dict['content'].append(os.path.basename(file_path))
        if args.delete:
            try:
                rt(file_path)
            except OSError as ose:
                print('Error while trying to delete image file : {error}'.format(error=ose))
    except IOError as ioe:
        print('Error while trying to save image file : {error}'.format(error=ioe))


def convert_folder_greyscale(dir_dict):
    """
    Converts an annotation folder in greyscale mode.
    :param dir_dict:
    :return: void.
    """
    for image_path in dir_dict['content']:
        cv.imwrite(filename=os.path.join(dir_dict['img_path'], image_path),
                   img=cv.cvtColor(src=cv.imread(filename=os.path.join(dir_dict['img_path'], image_path)),
                                   code=cv.COLOR_BGR2GRAY))


def zip_directory(dir_dict):
    """
    Zips annotation directories.
    :param dir_dict: directory dictionary.
    :return: void.
    """
    print('Compressing folder {folder}...'.format(folder=dir_dict['name']))
    with zf.ZipFile(os.path.join(FOLDERS_DIR, dir_dict['name']) + '.zip', 'w') as zip_file:
        paths = []

        # Remove absolute path in the Zip.
        abs_path = len(dir_dict['root_path'])

        # Read all files in the annotation folder.
        for r, d, f in os.walk(dir_dict['root_path']):
            for filename in f:
                paths.append(os.path.join(r, filename))

        for file in paths:
            zip_file.write(file, compress_type=zf.ZIP_DEFLATED, arcname=file[abs_path:])


def generate_csv(dir_dict):
    print('Generating CSV file for {folder}...'.format(folder=dir_dict['name']))
    """
    Generates CSV files for annotation.
    :param dir: directory path.
    :param csv_config: CSV file properties.
    :return: void.
    """
    with open(os.path.join(dir_dict['root_path'], CSV_CONFIG['name_pattern'].format(folder=dir_dict['name'])), 'w+',
              newline=CSV_CONFIG['newline']) as csv_file:
        # Add headers.
        fw = csv.writer(csv_file, delimiter=CSV_CONFIG['delimiter'], quotechar=CSV_CONFIG['quotechar'],
                        quoting=CSV_CONFIG['quoting'])
        fw.writerow(
            ['Path', 'Class', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'Confidence', 'Is_occluded', 'Is_truncated',
             'Is_depiction'])

        # Write file line by line.
        for file in dir_dict['content']:
            fw.writerow(
                [os.path.join(os.path.basename(dir_dict['img_path']), file), '', '', '', '', '', '', '', '', ''])


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
        if i % FOLDER_CONFIG['items'] == 0:
            curr_folder = create_dir(
                parent_dir=FOLDERS_DIR,
                img_dir=FOLDER_CONFIG['img_dir'],
                chars=FOLDER_CONFIG['chars'],
                size=FOLDER_CONFIG['size']
            )
            print('Generating folder {folder}...'.format(folder=curr_folder['name']))
        copy_file(img, curr_folder)

        i += 1

        # If the folder reaches its max size or if it is the last image.
        if i % FOLDER_CONFIG['items'] == 0 or imgs.index(img) == len(imgs) - 1:
            print('Folder {folder} is full !'.format(folder=curr_folder['name']))

            # Generate the CSV file.
            generate_csv(curr_folder)

            # If greyscale mode is enabled.
            if args.greyscale:
                # Copy folder for future extraction.
                cpt(src=curr_folder['root_path'], dst=os.path.join(FINAL_FOLDERS_DIR, curr_folder['name']))

                # Convert images.
                convert_folder_greyscale(curr_folder)

                # Zip directory for annotation.
                zip_directory(curr_folder)

                # Remove remaining folder.
                rt(curr_folder['root_path'])
            else:
                # Zip directory for annotation.
                zip_directory(curr_folder)

                # Move file for future extraction.
                mv(curr_folder["root_path"], FINAL_FOLDERS_DIR)


if __name__ == '__main__':
    main()
