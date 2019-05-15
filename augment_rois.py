#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    ROIs augmentation.
    ======================

    Augments ROIs for future trainings.
"""

import pandas as pd

from utils import *
from config import *
from skimage import io
from skimage import exposure
from argparse import ArgumentParser

__description__ = "Augments ROIs for future trainings."

# Parse args.
parser = ArgumentParser(description=__description__)
args = parser.parse_args()


def apply_transformations(df, augmented_df, filename, rows, roi_config, transformations):
    """
    Applies transformations on ROIs.
    :param df: augmentation dataset.
    :param row: ROI properties.
    :param roi_config: ROIs config.
    :return: updated dataset.
    """
    # Load image file.
    source_image = sk.io.imread(get_roi_fullpath(filename))

    # Apply each transformation.
    for transformation in transformations:

        # Get augmented image.
        dict = augmentation_router(source_image, transformation)

        # Ignore low-contrasted image.
        if sk.exposure.is_low_contrast(dict["image"]):
            break

        try:
            # Save transformed image to disk.
            name = random_name(roi_config["chars"], roi_config["size"], True) + roi_config["ext"]
            io.imsave(get_roi_fullpath(name), dict["image"])
        except Exception:
            break

        # Pick a random purpose.
        purpose = random.choice(TFRECORD_CONFIG["weights"])

        # Get coords of each item.
        for index, row in rows.iterrows():
            min, max = calculate_coords({"x": float(row["Xmin"]), "y": float(row["Ymin"])},
                                        {"x": float(row["Xmax"]), "y": float(row["Ymax"])},
                                        float(row["Width"]), float(row["Height"]), dict)

            # Update augmentation dataset.
            augmented_df = augmented_df.append(
                {
                    "Filename": name,
                    "Folder": row["Folder"],
                    "Width": row["Width"],
                    "Height": row["Height"],
                    "Class": row["Class"],
                    "Confidence": row["Confidence"],
                    "Xmin": min["x"],
                    "Ymin": min["y"],
                    "Xmax": max["x"],
                    "Ymax": max["y"],
                    "Is_occluded": row["Is_occluded"],
                    "Is_truncated": row["Is_truncated"],
                    "Is_depiction": row["Is_depiction"],
                    "Is_extracted": "False",
                    "Is_augmented": "False",
                    "Is_augmentation": "True",
                    "Augmentation": dict["augmentation"],
                    "Purpose": purpose,
                    "Generation_date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "Extraction_date": "",
                    "Augmentation_date": ""
                }
                , ignore_index=True)

            # Update ROI augmentation status.
            df.loc[index, "Augmentation_date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            df.loc[index, "Is_augmented"] = "True"

    return df, augmented_df


def main():
    """
    Main program.
    :return: void.
    """
    # Load the dataset CSV file.
    try:
        df = pd.read_csv(DATASET_CSV_PATH)
        print("Reading CSV file {csv}...".format(csv=DATASET_CSV_PATH))
    except Exception as ee:
        print("Error while reading CSV file {csv} : {error}.".format(csv=DATASET_CSV_PATH, error=ee))
        return

    # Ignore augmented lines.
    indexes = list(set(df[df['Is_augmented'] == True].index) | set(df[df['Augmentation'] == ""].index))
    cropped_df = df.drop(indexes, inplace=False)

    # Get dataset dimensions.
    rows, cols = cropped_df.shape

    # Check if file is not empty.
    if rows and cols:

        # Group ROIs by filename.
        grouped_df = group_rois(cropped_df, "Filename")

        # Init valid record counter and dataset.
        augmented_df = pd.DataFrame()

        # Read each ROI line.
        for filename, rows in grouped_df:
            print("Augmenting image {filename}...".format(filename=filename))

            # Apply augmentations.
            df, augmented_df = apply_transformations(df, augmented_df, filename, rows, ROI_CONFIG, ROI_TRANSFORMATIONS)

        # Concat both datasets.
        final_df = pd.concat([df.astype(str), augmented_df], ignore_index=False, sort=True)

        # Reindex dataset.
        final_df = final_df.reindex(
            columns=["Filename", "Folder", "Width", "Height", "Class", "Confidence", "Xmin", "Ymin", "Xmax", "Ymax",
                     "Is_occluded", "Is_truncated", "Is_depiction", "Is_extracted", "Is_augmented", "Is_augmentation",
                     "Augmentation", "Purpose", "Generation_date", "Extraction_date", "Augmentation_date"])

        # Write dataset.
        write_df_as_csv(df=final_df, path=DATASET_CSV_PATH)

        print("{count} augmentations have been performed.".format(count=augmented_df.shape[0]))

    else:
        print("Nothing to augment !")


if __name__ == "__main__":
    main()
