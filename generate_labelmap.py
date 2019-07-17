#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Labelmap file generations.
    ======================

    Generates the labelmap file for future trainings.

"""

from utils import *
from config import *
from argparse import ArgumentParser

__description__ = 'Generates the labelmap file for future trainings.'

# Parse args.
parser = ArgumentParser(description=__description__)
args = parser.parse_args()


def add_entry(roi_class):
    """
    Adds entry into the labelmap file.
    :param roi_class: roi class name.
    :return: labelmap file item.
    """
    class_id = class_text_to_int(roi_class)
    print('Adding class \'{roi_class}\' with ID {id}...'.format(roi_class=roi_class, id=class_id))
    lines = [
        "item {\n",
        "  id: {id}\n".format(id=class_id),
        "  name: '{name}'\n".format(name=roi_class),
        '}\n'
    ]
    return lines


def main():
    """
    Main program.
    :return: void.
    """
    labelmap_path = os.path.join(LABELMAP_CONFIG['path'], LABELMAP_FILE)
    try:
        # Remove the existing labelmap file.
        if os.path.isfile(labelmap_path):
            print('Removing previous file located at {file}...'.format(file=labelmap_path))
            os.remove(labelmap_path)
    except Exception as ee:
        print('Error while deleting previous file : {error}.'.format(error=ee))
        exit()

    with open(labelmap_path, 'w+') as labelmap_file:
        for roi_class in ROI_CLASSES:
            labelmap_file.writelines(add_entry(roi_class))


if __name__ == '__main__':
    main()
