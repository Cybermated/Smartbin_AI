#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Config file.
    ======================
    Project configuration file.
"""
import os
import csv
import string
import cv2 as cv

from datetime import datetime

__author__ = "Raphaël POSIER"
__copyright__ = "Copyright 2019, The Smartbin Project"
__credits__ = ["Florian LESAGE", "Raphaël OCTAU"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Raphaël POSIER"
__email__ = "pro@raphaelposier.fr"
__status__ = "Production"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Files extensions.
IMG_EXT = ("jpg", "png", "jpeg", "bmp")
VIDEO_EXT = ("mp4", "avi", "mkv")

ROI_CLASSES = ["battery", "can", "carton_packaging", "glass_bottle", "glass_container", "plastic_bottle",
               "plastic_container", "goblet"]

# Project directories.
PRETRAINING_DIR = os.path.join("pretraining")
TRAINING_DIR = os.path.join("training")

# Pre-training directories and paths.
RAW_IMAGES_DIR = os.path.join(PRETRAINING_DIR, "raw_images")
RAW_VIDEOS_DIR = os.path.join(PRETRAINING_DIR, "raw_videos")
CSV_DIR = os.path.join(PRETRAINING_DIR, "annotations", "csv")
FINAL_FOLDERS_DIR = os.path.join(PRETRAINING_DIR, "annotations", "folders")
FOLDERS_DIR = os.path.join(PRETRAINING_DIR, "folders")
ROIS_DIR = os.path.join(PRETRAINING_DIR, "dataset")
ROIS_PATH = os.path.join(ROIS_DIR, "images")
DATASET_CSV_PATH = os.path.join(ROIS_DIR, "dataset.csv")

# Training directories and paths.
TFRECORDS_DIR = os.path.join(TRAINING_DIR, "tfrecords")
TRAINING_CONFIG_DIR = os.path.join(TRAINING_DIR, "config")
TUNE_CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, "tune_checkpoints")
CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(TRAINING_DIR, "outputs")
FROZEN_MODEL_PATH = os.path.join(OUTPUTS_DIR, "frozen_inference_graph.pb")

DATE_FORMAT = "{day}/{month}/{year} {hour}:{minute}:{second}"

# Detection configuration.
DETECTION_CONFIG = {
    "num_workers": 4,
    "queue_size": 12,
    "default_thresh": "medium",
    "line_thickness": 3,
    "use_normalize_coordinates": True,
    "max_boxes_to_draw": 5,
    "num_classes": len(ROI_CLASSES),
    "model_dir": os.path.join(OUTPUTS_DIR, "frozen_inference_graph.pb"),
    "labelmap_path": os.path.join(TRAINING_CONFIG_DIR, "smartbin_labelmap.pbtxt")
}

SCORE_TRESH = {
    "very_low": .3,
    "low": .5,
    "medium": .7,
    "high": .9,
}

# ROIs extraction.
ROI_WIDTH = pow(2, 8)
ROI_HEIGHT = ROI_WIDTH
ROI_CONFIG = {
    "chars": string.digits + string.ascii_lowercase,
    "size": pow(2, 4),
    "ratio": (2 / 3, 2 / 3),
    "ext": ".jpg",
    "date": True
}
CSV_CONFIG = {
    "name_pattern": "roi_{folder}.csv",
    "suffix": "_extracted",
    "delimiter": ',',
    "quotechar": '|',
    "quoting": csv.QUOTE_MINIMAL,
    "newline": "",
    "ext": "csv"
}

# Frames extractions.
VIDEO_CONFIG = {
    "interval": pow(2, 3),
    "suffix": "_extracted",
}
FRAME_CONFIG = {
    "chars": string.digits + string.ascii_lowercase,
    "size": pow(2, 4),
    "date": False,
    "ext": ".jpg"
}
FRAME_TRANSFORMATIONS = {
    "color": cv.COLOR_RGB2GRAY,
    "ratios": (2 / 3, 2 / 3)
}

# Folders generations.
FOLDER_CONFIG = {
    "items": pow(2, 10),
    "chars": string.digits + string.ascii_lowercase,
    "size": pow(2, 3),
    "date": False,
    "img_dir": "img"
}

# TFRecords generations.
TFRECORD_CONFIG = {
    "default": "train",
    "train": "train-{date}.record".format(date=datetime.now().strftime("%Y%m%d-%H%M%S")),
    "test": "test-{date}.record".format(date=datetime.now().strftime("%Y%m%d-%H%M%S")),
    "weights": ["train"] * 85 + ["test"] * 15
}

# Labelmap file.
LABELMAP_CONFIG = {
    "start": 1,
    "path": TRAINING_CONFIG_DIR,
    "filename": "smartbin_labelmap",
    "suffix": "-{date}".format(date=datetime.now().strftime("%Y%m%d")),
    "ext": ".pbtxt",
    "delete": True
}

# Capture options.
DEVICE_CONFIG = {
    "id": 0,
    "resolution": "hd",
}

INPUT_RESOLUTION = {
    "sd": {
        "width": 640,
        "height": 480
    },
    "hd": {
        "width": 1280,
        "height": 720
    },
    "full-hd": {
        "width": 1920,
        "height": 1080
    }
}

# Model trainer.
TRAINER_CONFIG = {
    "checkpoints_dir": CHECKPOINTS_DIR,
    "pipeline_config_path": os.path.join(TRAINING_CONFIG_DIR, "smartbin_pipeline.config"),
    "eval_training_data": False,
    "resume_from_ckpt": True,
    "hparams_override": None,
    "run_once": False,
    "num_train_steps": 200000,
    "sample_1_of_n_eval_examples": 5,
    "sample_1_of_n_eval_on_train_example": 1
}

# ROIs augmentations.
ROI_TRANSFORMATIONS = [
    "adjust_gamma",
    "adjust_log",
    "adjust_sigmoid",
    "random_blur",
    "random_contrast",
    "random_gaussian",
    "random_pepper",
    "random_poisson",
    "random_salt",
    "random_sp",

    "double_flip",
    "horizontal_flip",
    "random_rotation",
    "vertical_flip",

    "adjust_gamma+horizontal_flip",
    "adjust_gamma+vertical_flip",
    "adjust_gamma+double_flip",

    "adjust_log+horizontal_flip",
    "adjust_log+vertical_flip",
    "adjust_log+double_flip",

    "adjust_sigmoid+horizontal_flip",
    "adjust_sigmoid+vertical_flip",
    "adjust_sigmoid+double_flip",

    "random_blur+horizontal_flip",
    "random_blur+vertical_flip",
    "random_blur+double_flip",

    "random_contrast+horizontal_flip",
    "random_contrast+vertical_flip",
    "random_contrast+double_flip",

    "random_gaussian+horizontal_flip",
    "random_gaussian+vertical_flip",
    "random_gaussian+double_flip",

    "random_pepper+horizontal_flip",
    "random_pepper+vertical_flip",
    "random_pepper+double_flip",

    "random_salt+horizontal_flip",
    "random_salt+vertical_flip",
    "random_salt+double_flip",

    "random_sp+horizontal_flip",
    "random_sp+vertical_flip",
    "random_sp+double_flip",

    "random_poisson+horizontal_flip",
    "random_poisson+vertical_flip",
    "random_poisson+double_flip"
]

TRANSFORMATION_CONFIG = {
    "blur": [x for x in range(1, 2)],
    "rotation": [x for x in range(1, 6)],
    "gain": [x / 100 for x in range(50, 151, 10) if x != 100],
    "gamma": [x / 100 for x in range(50, 151, 10) if x != 100]
}
