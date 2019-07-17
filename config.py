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

ROI_CLASSES = [
    # 'battery',
    'can',
    'carton_packaging',
    'glass_bottle',
    'glass_container',
    'plastic_bottle',
    'plastic_container',
    # 'goblet'
]

# Project directories.
PRETRAINING_DIR = os.path.join('pretraining')
TRAINING_DIR = os.path.join('training')

# Project filenames.
DATASET_FILE = 'dataset.csv'
INFERENCE_GRAPH_FILE = 'frozen_inference_graph.pb'
LABELMAP_FILE = 'smartbin_labelmap.pbtxt'
PIPELINE_FILE = 'smartbin_pipeline.config'

# Pre-training directories and paths.
RAW_IMAGES_DIR = os.path.join(PRETRAINING_DIR, 'raw_images')
RAW_VIDEOS_DIR = os.path.join(PRETRAINING_DIR, 'raw_videos')
CSV_DIR = os.path.join(PRETRAINING_DIR, 'annotations', 'csv')
FINAL_FOLDERS_DIR = os.path.join(PRETRAINING_DIR, 'annotations', 'folders')
FOLDERS_DIR = os.path.join(PRETRAINING_DIR, 'folders')
ROIS_DIR = os.path.join(PRETRAINING_DIR, 'dataset')
ROIS_PATH = os.path.join(ROIS_DIR, 'images')
DATASET_CSV_PATH = os.path.join(ROIS_DIR, DATASET_FILE)

# Training directories and paths.
TFRECORDS_DIR = os.path.join(TRAINING_DIR, 'tfrecords')
TRAINING_CONFIG_DIR = os.path.join(TRAINING_DIR, 'config')
TUNE_CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, 'tune_checkpoints')
CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, 'checkpoints')
OUTPUTS_DIR = os.path.join(TRAINING_DIR, 'outputs')
FROZEN_MODEL_PATH = os.path.join(OUTPUTS_DIR, INFERENCE_GRAPH_FILE)
PIPELINE_CONFIG_PATH = os.path.join(TRAINING_CONFIG_DIR, PIPELINE_FILE)
LABELMAP_PATH = os.path.join(TRAINING_CONFIG_DIR, LABELMAP_FILE)
INFERENCE_GRAPH_PATH = os.path.join(OUTPUTS_DIR, INFERENCE_GRAPH_FILE)

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
    'name_pattern_ai': 'roi_{folder}_AI-{level}',
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
    # Width/Height ratio of the ROI size that the ROI must reach to be kept.
    'ratio': (.6, .6),
    'ext': '.jpg',
    'date': True,
    'greyscale': False,
    'ignore_size': False
}

# Items detection settings.
DETECTION_CONFIG = {
    'num_workers': 4,
    'queue_size': 8,
    # Minimum confidence to display a box.
    'default_thresh': 'high',
    # Line thickness.
    'line_thickness': 3,
    'use_normalize_coordinates': True,
    # Maximum boxes to display at a time.
    'max_boxes_to_draw': 8,
    'num_classes': len(ROI_CLASSES),
    'model_dir': INFERENCE_GRAPH_PATH,
    'labelmap_path': LABELMAP_PATH
}

# TFRecords settings.
TFRECORD_CONFIG = {
    # According to TensorFlow it's better to use small TFRecords files (~100MB).
    'max_size': 102400000,
    'default': 'train',
    'train': 'train-{id}.record',
    'test': 'test-{id}.record',
    'weights': ['train'] * TRAIN_RATIO + ['test'] * TEST_RATIO
}

# Labelmap file settings.
LABELMAP_CONFIG = {
    'start': 1,
    'delete': True
}

# Model trainer settings.
TRAINER_CONFIG = {
    # Path to output model directory where event and checkpoint files will be written.
    'checkpoints_dir': CHECKPOINTS_DIR,
    # Path to pipeline config file.
    'pipeline_config_path': PIPELINE_CONFIG_PATH,
    # Test dataset against training data.
    'eval_training_data': False,
    # If previous checkpoints are found, start from the latest one.
    'resume_from_ckpt': True,
    # Override advanced parameters.
    'hparams_override': None,
    # If running in eval-only mode, whether to run just one round of eval vs running continuously.
    'run_once': False,
    # Number of train steps.
    'num_train_steps': 200000,
    # Sample one of every n eval input examples, where n is provided.
    'sample_1_of_n_eval_examples': 5,
    # Sample one of every n train input examples for evaluation, where n is provided.
    'sample_1_of_n_eval_on_train_example': 1
}

# Available augmentations.
ROI_TRANSFORMATIONS = [
    'random_contrast',
    'random_blur',
    'random_poisson',
    'random_gaussian'

    'horizontal_flip',
    'vertical_flip',
    'double_flip',

    'random_contrast+horizontal_flip',
    'random_contrast+vertical_flip',
    'random_contrast+double_flip',

    'random_blur+horizontal_flip',
    'random_blur+vertical_flip',
    'random_blur+double_flip',

    'random_rescale',

    'random_rescale+horizontal_flip',
    'random_rescale+vertical_flip',
    'random_rescale+double_flip',

    'random_poisson+horizontal_flip',
    'random_poisson+vertical_flip',
    'random_poisson+double_flip'
]

# Augmentations randomness settings.
TRANSFORMATION_CONFIG = {
    'blur': [x for x in range(1, 4)],
    'rotation': [x for x in range(1, 11)],
    'gain': [x / 100 for x in range(50, 151, 10) if x != 100],
    'gamma': [x / 100 for x in range(50, 151, 10) if x != 100],
    'rescale': [x / 100 for x in range(15, 51)]
}
