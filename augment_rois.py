#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    ROIs augmentation.
    ======================

    Augments ROIs for future trainings.

    Usage:
        augment_roi.py

    Options:

"""

import pandas as pd

from utils import *
from config import *
from skimage import exposure, io
from argparse import ArgumentParser

__description__ = 'Augments ROIs for future trainings.'

# Parse args.
parser = ArgumentParser(description=__description__)
args = parser.parse_args()


def apply_transformations(df, augmented_df, filename, rows):
    """
    Applies transformation on images.
    :param df: original dataset.
    :param augmented_df: augmented dataset.
    :param filename: ROI filename.
    :param rows: rows to process.
    :return:
    """
    # Load image file.
    source_image = sk.io.imread(get_roi_fullpath(filename))

    # Apply each transformation.
    for transformation in ROI_TRANSFORMATIONS:

        # Get augmented image.
        dict = augmentation_router(source_image, transformation)

        # Ignore low-contrasted images.
        if sk.exposure.is_low_contrast(dict['image']):
            break

        try:
            # Save new image to disk.
            name = random_name(chars=ROI_CONFIG['chars'], size=ROI_CONFIG['size'], use_date=True) + ROI_CONFIG['ext']
            io.imsave(get_roi_fullpath(name), dict['image'])
        except Exception:
            break

        # Pick a random purpose.
        purpose = random.choice(TFRECORD_CONFIG['weights'])

        # Get coords of each item.
        for index, row in rows.iterrows():
            min, max = calculate_coords({'x': float(row['Xmin']), 'y': float(row['Ymin'])},
                                        {'x': float(row['Xmax']), 'y': float(row['Ymax'])},
                                        float(row['Width']), float(row['Height']), dict)

            # Update augmentation dataset.
            augmented_df = augmented_df.append(
                {
                    'Filename': name,
                    'Folder': row['Folder'],
                    'Width': row['Width'],
                    'Height': row['Height'],
                    'Class': row['Class'],
                    'Confidence': row['Confidence'],
                    'Xmin': min['x'],
                    'Ymin': min['y'],
                    'Xmax': max['x'],
                    'Ymax': max['y'],
                    'Is_occluded': str(row['Is_occluded']),
                    'Is_truncated': str(row['Is_truncated']),
                    'Is_depiction': str(row['Is_depiction']),
                    'Is_extracted': 'False',
                    'Is_augmented': 'False',
                    'Is_augmentation': 'True',
                    'Augmentation': dict['augmentation'],
                    'Purpose': purpose,
                    'Generation_date': get_current_datetime(),
                    'Extraction_date': '',
                    'Augmentation_date': ''
                }
                , ignore_index=True)

            # Update ROI augmentation status.
            df.loc[index, 'Augmentation_date'] = get_current_datetime()
            df.loc[index, 'Is_augmented'] = 'True'

    return df, augmented_df


def main():
    """Main program."""
    # Load the dataset CSV file.
    try:
        df = pd.read_csv(DATASET_CSV_PATH)
    except Exception:
        return

    # Drop augmented, depictions and ignored rows.
    indexes = list(set(df[df['Is_augmented'] == True].index) | set(df[df['Augmentation'] == ''].index) | set(
        df[df['Is_depiction'] == True].index) | set(df[df['Is_ignored'] == True].index))
    cropped_df = df.drop(indexes, inplace=False)

    # Get dataframe dimensions.
    rows, cols = cropped_df.shape

    # Check if file is not empty.
    if rows and cols:

        # Group ROIs by filename.
        grouped_df = group_dataframe(cropped_df, 'Filename')

        print('{count} augmentations are enabled, expecting {images} new images.'.format(
            count=len(ROI_TRANSFORMATIONS), images=len(ROI_TRANSFORMATIONS) * len(grouped_df)))

        # Init valid record counter and dataset.
        augmented_df = pd.DataFrame()

        # Read each ROI line.
        for filename, rows in grouped_df:
            print('Augmenting image {filename}...'.format(filename=filename))

            # Apply augmentations.
            df, augmented_df = apply_transformations(df, augmented_df, filename, rows)

        # Concat both datasets.
        final_df = pd.concat([df.astype(str), augmented_df], ignore_index=False, sort=True)

        # Write re-indexed dataset.
        write_df_as_csv(df=final_df.reindex(
            columns=CSV_STRUCTURE['dataset']),
            path=DATASET_CSV_PATH)

    else:
        print('Nothing to augment !')


if __name__ == '__main__':
    main()
