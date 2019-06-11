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

from PIL import Image

__author__ = 'Raphaël POSIER'
__copyright__ = 'Copyright 2019, The Smartbin Project'
__credits__ = ['Florian LESAGE', 'Raphaël OCTAU']
__license__ = 'GPL'
__version__ = '1.0'
__maintainer__ = 'Raphaël POSIER'
__email__ = 'pro@raphaelposier.fr'
__status__ = 'Production'

# Tensorflow logs level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Files extensions.
IMG_EXT = ('jpg', 'png', 'jpeg', 'bmp')
VIDEO_EXT = ('mp4', 'avi', 'mkv')

ROI_CLASSES = ['battery', 'can', 'carton_packaging', 'glass_bottle', 'glass_container', 'plastic_bottle',
               'plastic_container', 'goblet']

# Project directories.
PRETRAINING_DIR = os.path.join('pretraining')
TRAINING_DIR = os.path.join('training')

# Pre-training directories and paths.
RAW_IMAGES_DIR = os.path.join(PRETRAINING_DIR, 'raw_images')
RAW_VIDEOS_DIR = os.path.join(PRETRAINING_DIR, 'raw_videos')
CSV_DIR = os.path.join(PRETRAINING_DIR, 'annotations', 'csv')
FINAL_FOLDERS_DIR = os.path.join(PRETRAINING_DIR, 'annotations', 'folders')
FOLDERS_DIR = os.path.join(PRETRAINING_DIR, 'folders')
ROIS_DIR = os.path.join(PRETRAINING_DIR, 'dataset')
ROIS_PATH = os.path.join(ROIS_DIR, 'images')
DATASET_CSV_PATH = os.path.join(ROIS_DIR, 'dataset.csv')

# Training directories and paths.
TFRECORDS_DIR = os.path.join(TRAINING_DIR, 'tfrecords')
TRAINING_CONFIG_DIR = os.path.join(TRAINING_DIR, 'config')
TUNE_CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, 'tune_checkpoints')
CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, 'checkpoints')
OUTPUTS_DIR = os.path.join(TRAINING_DIR, 'outputs')
FROZEN_MODEL_PATH = os.path.join(OUTPUTS_DIR, 'frozen_inference_graph.pb')

# Time configuration.
TIMEZONE = 'Europe/Paris'
DATE_FORMAT = '%d-%m-%Y_%H-%M-%S'

# CSV files structures.
CSV_STRUCTURE = {
    'annotation': ['Path', 'Class', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'Confidence', 'Is_occluded', 'Is_truncated',
                   'Is_depiction'],
    'dataset': ['Filename', 'Folder', 'Width', 'Height', 'Class', 'Confidence', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                'Is_occluded', 'Is_truncated', 'Is_depiction', 'Is_extracted', 'Is_augmented', 'Is_ignored',
                'Is_augmentation', 'Augmentation', 'Purpose', 'Generation_date', 'Extraction_date', 'Augmentation_date']
}

# ROIs dimensions.
ROI_WIDTH = 256
ROI_HEIGHT = 256

# Training purposes ratios.
TRAIN_RATIO = 85
TEST_RATIO = 15

# Default capture settings.
DEVICE_CONFIG = {
    'id': 0,
    'resolution': 'hd',
}

# Available capture resolutions.
INPUT_RESOLUTION = {
    'sd': {
        'width': 640,
        'height': 480
    },
    'hd': {
        'width': 1280,
        'height': 720
    },
    'full-hd': {
        'width': 1920,
        'height': 1080
    }
}

# Detections levels settings.
SCORE_TRESH = {
    'very_low': .1,
    'low': .3,
    'medium': .5,
    'high': .7,
    'fair': .75,
    'very_high': .9
}

# Frame extractions settings.
VIDEO_CONFIG = {
    'interval': 8,
    'suffix': '_extracted',
}

# Frames settings.
FRAME_CONFIG = {
    'quality': 85,
    'chars': string.digits + string.ascii_letters,
    'size': 16,
    'date': False,
    'ext': '.jpg'
}

# Exif orientation tags.
ORIENTATIONS_TAG = {
    1: ('Normal', 0),
    2: ('Mirrored left-to-right', 0),
    3: ('Rotated 180 degrees', Image.ROTATE_180),
    4: ('Mirrored top-to-bottom', 0),
    5: ('Mirrored along top-left diagonal', 0),
    6: ('Rotated 90 degrees', Image.ROTATE_270),
    7: ('Mirrored along top-right diagonal', 0),
    8: ('Rotated 270 degrees', Image.ROTATE_90)
}

# Annotation folders settings
FOLDER_CONFIG = {
    'items': 2048,
    'chars': string.digits + string.ascii_lowercase,
    'size': 8,
    'date': False,
    'img_dir': 'img'
}

# Annotation CSV settings.
CSV_CONFIG = {
    'name_pattern': 'roi_{folder}.csv',
    'name_pattern_ai': 'roi_{folder}-AI_{level}',
    'suffix': '_extracted',
    'delimiter': ',',
    'quotechar': '|',
    'quoting': csv.QUOTE_MINIMAL,
    'newline': '',
    'ext': 'csv',
}

# ROIs settings.
ROI_CONFIG = {
    'chars': string.digits + string.ascii_letters,
    'size': 32,
    'ratio': (2 / 3, 2 / 3),
    'ext': '.jpg',
    'date': True
}

# Items detection settings.
DETECTION_CONFIG = {
    'num_workers': 4,
    'queue_size': 12,
    'default_thresh': 'fair',
    'line_thickness': 3,
    'use_normalize_coordinates': True,
    'max_boxes_to_draw': 8,
    'num_classes': len(ROI_CLASSES),
    'model_dir': os.path.join(OUTPUTS_DIR, 'frozen_inference_graph.pb'),
    'labelmap_path': os.path.join(TRAINING_CONFIG_DIR, 'smartbin_labelmap.pbtxt')
}

# TFRecords settings.
TFRECORD_CONFIG = {
    'default': 'train',
    'train': 'train-{id}.record',
    'test': 'test-{id}.record',
    'weights': ['train'] * TRAIN_RATIO + ['test'] * TEST_RATIO
}

# Labelmap file settings.
LABELMAP_CONFIG = {
    'start': 1,
    'path': TRAINING_CONFIG_DIR,
    'filename': 'smartbin_labelmap',
    'ext': '.pbtxt',
    'delete': True
}

# Model trainer settings.
TRAINER_CONFIG = {
    'checkpoints_dir': CHECKPOINTS_DIR,
    'pipeline_config_path': os.path.join(TRAINING_CONFIG_DIR, 'smartbin_pipeline.config'),
    'eval_training_data': False,
    'resume_from_ckpt': True,
    'hparams_override': None,
    'run_once': False,
    'num_train_steps': 200000,
    'sample_1_of_n_eval_examples': 5,
    'sample_1_of_n_eval_on_train_example': 1
}

# Available augmentations.
ROI_TRANSFORMATIONS = [
    'adjust_gamma',
    'adjust_log',
    'adjust_sigmoid',
    'random_blur',
    'random_contrast',
    'random_gaussian',
    'random_pepper',
    'random_poisson',
    'random_salt',
    'random_sp',

    'random_pepper',
    'random_poisson',
    'random_salt',
    'random_sp',

    'double_flip',
    'horizontal_flip',
    'random_rotation',
    'vertical_flip',

    'adjust_gamma+horizontal_flip',
    'adjust_gamma+vertical_flip',
    'adjust_gamma+double_flip',

    'adjust_log+horizontal_flip',
    'adjust_log+vertical_flip',
    'adjust_log+double_flip',

    'adjust_sigmoid+horizontal_flip',
    'adjust_sigmoid+vertical_flip',
    'adjust_sigmoid+double_flip',

    'random_blur+horizontal_flip',
    'random_blur+vertical_flip',
    'random_blur+double_flip',

    'random_blur+horizontal_flip',
    'random_blur+vertical_flip',
    'random_blur+double_flip',

    'random_contrast+horizontal_flip',
    'random_contrast+vertical_flip',
    'random_contrast+double_flip',

    'random_gaussian+horizontal_flip',
    'random_gaussian+vertical_flip',
    'random_gaussian+double_flip',

    'random_pepper+horizontal_flip',
    'random_pepper+vertical_flip',
    'random_pepper+double_flip',

    'random_salt+horizontal_flip',
    'random_salt+vertical_flip',
    'random_salt+double_flip',

    'random_sp+horizontal_flip',
    'random_sp+vertical_flip',
    'random_sp+double_flip',

    'random_poisson+horizontal_flip',
    'random_poisson+vertical_flip',
    'random_poisson+double_flip'
]

# Augmentations randomness settings.
TRANSFORMATION_CONFIG = {
    'blur': [x for x in range(1, 4)],
    'rotation': [x for x in range(1, 11)],
    'gain': [x / 100 for x in range(50, 151, 10) if x != 100],
    'gamma': [x / 100 for x in range(50, 151, 10) if x != 100]
}
