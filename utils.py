#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Utils file.
    ======================
    Collection of useful functions.
"""

import numpy as np
import skimage as sk

from config import *
from skimage import util
from skimage import filters
from skimage import exposure
from skimage import transform
from datetime import datetime
from collections import namedtuple

augmentation_config = TRANSFORMATION_CONFIG


def frange(start, stop=None, step=None):
    if stop is None:
        stop = start + 0.0
        start = 0.0
    if step is None:
        step = 1.0
    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start)
        start = start + step


def random_name(chars, size, date=True):
    """
    Generates a random filename string.
    :param chars: allowed chars.
    :param size: random name size.
    :param date: add current date to string.
    :return: random filename string.
    """
    if date:
        return datetime.now().strftime("%Y%m%d%H%M%S") + "-" + "".join(random.choice(chars) for _ in range(size))
    return "".join(random.choice(chars) for _ in range(size))


def add_suffix(file, suffix):
    """
    Renames a file by adding a suffix.
    :param file: original filename.
    :param suffix: suffix to be added to the filename.
    :return: void.
    """
    dirname, filename = os.path.split(file)
    split = filename.split(".")
    new_filename = "{path}{sep}{file}{suffix}.{ext}".format(path=dirname, sep=os.sep, file=split[0], suffix=suffix,
                                                            ext=split[1])
    os.rename(file, new_filename)


def list_directories(dir):
    return [os.path.join(dir, x) for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]


def list_files(dir, extensions, suffix=None):
    """
    Returns all files in a specific directory with specific extensions.
    :param dir: target directory path.
    :param extensions: file extensions.
    :param suffix: filename suffix.
    :return: array of files.
    """

    if suffix is not None:
        return [os.path.join(dir, filename) for filename in os.listdir(dir) if
                get_file_extension(filename) in extensions and suffix not in filename]
    return [os.path.join(dir, filename) for filename in os.listdir(dir) if
            get_file_extension(filename) in extensions]


def get_file_extension(filename):
    """
    Return file extension.
    :param filename:
    :return:
    """
    try:
        return filename.split(".")[-1]
    except Exception as ee:
        return None


def get_roi_fullpath(name):
    """
    Returns ROI fullpaths.
    :param name: ROI filename.
    :return: ROI fullpath.
    """
    return os.path.join(ROIS_PATH, name)


def group_rois(df, column):
    """
    Groups dataframe by the specified column.
    :param df: input dataframe.
    :param column: which column to group by.
    :return: grouped dataframe.
    """
    data = namedtuple("data", [column, "object"])
    gb = df.groupby(column)
    return [data(path, gb.get_group(x)) for path, x in zip(gb.groups.keys(), gb.groups)]


def manage_classes(roi_class):
    """
    Manages ROI class names.
    :param roi_class: original class name.
    :return: final class name.
    """
    if roi_class == "tin_can":
        roi_class = "can"
    if roi_class == "plastic_goblet":
        roi_class = "goblet"
    return roi_class


def class_text_to_int(class_name, classes=ROI_CLASSES):
    """
    Returns the internal labelmap ID of a class.
    :param class_name: class name.
    :param classes: all classes.
    :return: internal ID of the class.
    """
    return classes.index(class_name) + LABELMAP_CONFIG["start"]


def int_to_class_text(class_id, classes=ROI_CLASSES):
    """
    Returns the class name from the internal labelmap ID.
    :param class_id: class id.
    :param classes: all classes.
    :return: class_name.
    """
    return classes[class_id - LABELMAP_CONFIG["start"]]


def write_df_as_csv(df, path):
    """
    Saves dataframe as CSV in the specified path.
    :param df:
    :param path:
    :return:
    """
    df.to_csv(path, mode='w', header=True, index=False, quoting=CSV_CONFIG["quoting"],
              quotechar=CSV_CONFIG["quotechar"])


def get_detection_boxes(boxes, classes, scores, category_index, tresh_level,
                        max_boxes_to_draw=DETECTION_CONFIG["max_boxes_to_draw"]):
    """
    Returns all the items detected on a frame as dictionary.
    :param boxes:
    :param classes:
    :param scores:
    :param category_index:
    :param tresh_level:
    :param max_boxes_to_draw:
    :return:
    """
    output = []

    min_score_thresh = DETECTION_CONFIG["score_treshes"][tresh_level]

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            class_name = "N/A"
            ymin, xmin, ymax, xmax = boxes[i].tolist()
            confidence = round(scores[i], 2)

            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]["name"]
            class_name = str(class_name)

            output.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "box": {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }
                }
            )
    return output


def augmentation_router(image_array, augmentations):
    """
    Applies the specified augmentation(s).
    :param image_array:
    :param augmentation:
    :return:
    """
    augmentation_queue = augmentations.split('+')

    dict = {
        "augmentation": augmentations,
        "image": image_array,
        "angle": 0
    }

    for augmentation in augmentation_queue:

        if augmentation == "random_rotation":
            dict["image"], dict["angle"] = random_rotation(dict["image"])

        elif augmentation == "random_pepper":
            dict["image"] = random_pepper(dict["image"])

        elif augmentation == "random_salt":
            dict["image"] = random_salt(dict["image"])

        elif augmentation == "random_sp":
            dict["image"] = random_sp(dict["image"])

        elif augmentation == "random_poisson":
            dict["image"] = random_poisson(dict["image"])

        elif augmentation == "random_gaussian":
            dict["image"] = random_gaussian(dict["image"])

        elif augmentation == "horizontal_flip":
            dict["image"] = horizontal_flip(dict["image"])

        elif augmentation == "vertical_flip":
            dict["image"] = vertical_flip(dict["image"])

        elif augmentation == "double_flip":
            dict["image"] = double_flip(dict["image"])

        elif augmentation == "random_contrast":
            dict["image"] = random_contrast(dict["image"])

        elif augmentation == "random_blur":
            dict["image"] = random_blur(dict["image"])

        elif augmentation == "adjust_gamma":
            dict["image"] = adjust_gamma(dict["image"])

        elif augmentation == "adjust_sigmoid":
            dict["image"] = adjust_sigmoid(dict["image"])

        elif augmentation == "adjust_log":
            dict["image"] = adjust_log(dict["image"])

    return dict


def calculate_coords(min, max, width, height, dict):
    """
    Calculates new coordinates after augmentation.
    :param min:
    :param max:
    :param width:
    :param height:
    :param dict:
    :return:
    """
    mini = min
    maxi = max

    augmentation_queue = dict["augmentation"].split("+")

    for augmentation in augmentation_queue:
        if augmentation == "horizontal_flip":
            mini = {"x": np.abs(mini["x"] - 1), "y": mini["y"]}
            maxi = {"x": np.abs(maxi["x"] - 1), "y": maxi["y"]}

        elif augmentation == "vertical_flip":
            mini = {"x": mini["x"], "y": np.abs(mini["y"] - 1)}
            maxi = {"x": maxi["x"], "y": np.abs(maxi["y"] - 1)}

        elif augmentation == "double_flip":
            mini = {"x": np.abs(mini["x"] - 1), "y": np.abs(mini["y"] - 1)}
            maxi = {"x": np.abs(maxi["x"] - 1), "y": np.abs(maxi["y"] - 1)}

        elif augmentation == "random_rotation":
            xc, yc = .5 * width, .5 * height

            # Calculate original edges.
            a = {"x": mini["x"] * width, "y": mini["y"] * height}
            b = {"x": maxi["x"] * width, "y": mini["y"] * height}
            c = {"x": mini["x"] * width, "y": maxi["y"] * height}
            d = {"x": maxi["x"] * width, "y": maxi["y"] * height}

            # Calculate new edges.
            a = rotate_coords(a, (xc, yc), dict["angle"])
            b = rotate_coords(b, (xc, yc), dict["angle"])
            c = rotate_coords(c, (xc, yc), dict["angle"])
            d = rotate_coords(d, (xc, yc), dict["angle"])

            # Calculate boxe edges.
            mini = {"x": b["x"] / width, "y": a["y"] / height}
            maxi = {"x": c["x"] / width, "y": d["y"] / height}

    return check_coords(mini), check_coords(maxi)


def random_rotation(image_array):
    """
    Rotates image by a certain angle around its center.
    :param image_array: input image.
    :return: output image.
    """
    random_degree = random.choice(augmentation_config["rotation"])
    return sk.transform.rotate(image_array, random_degree), random_degree


def random_pepper(image_array):
    """
    Replaces random pixels with 0.
    :param image_array: input image.
    :return: output image.
    """
    return sk.util.random_noise(image_array, mode="pepper")


def random_salt(image_array):
    """
    Replaces random pixels with 1.
    :param image_array: input image.
    :return: output image.
    """
    return sk.util.random_noise(image_array, mode="salt")


def random_sp(image_array):
    """
    Replaces random pixels with 0 or 1.
    :param image_array: input image.
    :return: output image.
    """
    return sk.util.random_noise(image_array, mode="s&p")


def random_poisson(image_array):
    """
    Poisson-distributed noise generated from the data.
    :param image_array: input image.
    :return: output image.
    """
    return sk.util.random_noise(image_array, mode="poisson")


def random_gaussian(image_array):
    """
    Gaussian-distributed additive noise.
    :param image_array: input image.
    :return: output image.
    """
    return sk.util.random_noise(image_array, mode="gaussian")


def horizontal_flip(image_array):
    """
    Flips image horizontally.
    :param image_array: input image.
    :return: output image.
    """
    return image_array[:, ::-1]


def vertical_flip(image_array):
    """
    Flips image vertically.
    :param image_array: input image.
    :return: output image.
    """
    return image_array[::-1, :]


def double_flip(image_array):
    """
    Applies both vertical and horizontal flips.
    :param image_array: input image.
    :return: output image.
    """
    return vertical_flip(horizontal_flip(image_array))


def random_contrast(image_array):
    """
    Returns image after stretching or shrinking its intensity levels.
    :param image_array: input image.
    :return: output image..
    """
    return sk.exposure.rescale_intensity(image_array)


def adjust_gamma(image_array):
    """
    Performs Gamma correction on the input image.
    :param image_array: input image.
    :return: output image.
    """
    gain, gamma = random.choice(augmentation_config["gain"]), random.choice(augmentation_config["gamma"])
    return sk.exposure.adjust_gamma(image_array, gamma, gain), gain, gamma


def adjust_log(image_array):
    """
    Performs Logarithmic correction on the input image.
    :param image_array: input image.
    :return: output image.
    """
    return sk.exposure.adjust_log(image_array)


def adjust_sigmoid(image_array):
    """
    Performs Sigmoid correction on the input image.
    :param image_array: input image.
    :return: output image.
    """
    return sk.exposure.adjust_sigmoid(image_array)


def random_blur(image_array):
    """
    Multi-dimensional Gaussian filter.
    :param image_array: input image.
    :return: output image.
    """
    return sk.filters.gaussian(image_array, sigma=random.choice(augmentation_config["blur"]))


def resize_image(image, ratios):
    """
    Resizes images.
    :param image_array: input image.
    :param ratios: output image ratios.
    :return: output image.
    """
    width, height = ratios
    return cv.resize(image, None, fx=width, fy=height)


def image_to_greyscale(image):
    """
    Converts an image in greyscale.
    :param image_array: input image.
    :return: output image.
    """
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def get_date_string(date=datetime.now(), base=DATE_FORMAT):
    """
    Returns current datetime as string.
    :param date:
    :param base:
    :return:
    """
    return base.format(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute,
                       second=date.second)


def rotate_coords(coords, center, angle):
    """
    Returns the new coords of a point after a counter clockwise rotation.
    :param coords: original coords.
    :param center: center of the image.
    :param angle: angle in degrees.
    :return: updated coords.
    """
    xc, yc = center
    angle = np.deg2rad(angle)

    return {
        "x": (coords["x"] - xc) * np.cos(angle) - (coords["y"] - yc) * np.sin(angle) + xc,
        "y": (coords["x"] - xc) * np.sin(angle) + (coords["y"] - yc) * np.cos(angle) + yc
    }


def keep_roi(row, roi_config):
    """

    :param row:
    :param roi_config:
    :return:
    """
    # Retrieve and round ROI positions.
    roi_width = row["Xmax"] * row["Width"] - row["Xmin"] * row["Width"]
    roi_height = row["Ymax"] * row["Height"] - row["Ymin"] * row["Height"]

    # Read frame ratios.
    width_ratio, height_ratio = roi_config["ratio"]

    if roi_width >= width_ratio * ROI_WIDTH or row["Height"] >= roi_height * height_ratio:
        if not row["Is_depiction"]:
            return True
    return False


def check_coords(coords):
    """
    Checks and corrects coordinates to avoid off-image ROIs.
    :param coords: original coords.
    :return: corrected coords.
    """
    for key, value in coords.items():
        if value > 1:
            coords[key] = 1
        elif value < 0:
            coords[key] = 0
        else:
            coords[key] = value
    return coords
