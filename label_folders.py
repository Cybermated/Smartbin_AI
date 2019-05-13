#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Folders annotation.
    ======================

    Pre-annotates folders using AI model.
"""
import pandas as pd
import tensorflow as tf

from utils import *
from config import *
from argparse import ArgumentParser
from object_detection.utils import label_map_util

__description__ = "Pre-annotates folders using AI model."

# Parse args.
parser = ArgumentParser(description=__description__)

parser.add_argument('-min-c', "--min-confidence", dest="min_confidence",
                    type=str,
                    choices=list(SCORE_TRESH.keys()),
                    default=DETECTION_CONFIG["default_thresh"],
                    help="Required confidence to display a box, default is {default}.".format(
                        default=DETECTION_CONFIG["default_thresh"]))

parser.add_argument("-max-b", "--max-boxes", dest='max_boxes',
                    type=int,
                    default=DETECTION_CONFIG["max_boxes_to_draw"],
                    help="Max number of boxes to draw at a time, default is {default}.".format(
                        default=DETECTION_CONFIG["max_boxes_to_draw"]))

args = parser.parse_args()

# Load labelmap file.
label_map = label_map_util.load_labelmap(DETECTION_CONFIG["labelmap_path"])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=DETECTION_CONFIG["num_classes"],
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loads a frozen Tensorflow model in memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_MODEL_PATH, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def detect_items(image_path, session):
    # Read input image.
    try:
        image = cv.imread(image_path)
    except Exception as ee:
        print(ee)
        return

    # Expand image.
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

    # Retrieve boxes, scores, classes and number of detections.
    boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
    scores = detection_graph.get_tensor_by_name("detection_scores:0")
    classes = detection_graph.get_tensor_by_name("detection_classes:0")
    num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    # Actual detection.
    (boxes, scores, classes, num_detections) = session.run([boxes, scores, classes, num_detections],
                                                           feed_dict={image_tensor: image_np_expanded})

    return get_detection_boxes(boxes=np.squeeze(boxes), classes=np.squeeze(classes).astype(np.int32),
                               scores=np.squeeze(scores), category_index=category_index,
                               tresh_level=args.min_confidence, max_boxes_to_draw=args.max_boxes)


def annotate_folder(folder_path):
    """

    :param folder_path:
    :return:
    """
    new_df = pd.DataFrame()
    csv_path = os.path.join(folder_path, "roi_{folder}.csv".format(folder=os.path.basename(folder_path)))

    # Skip empty folders.
    if not os.path.isfile(csv_path):
        return

    # Skip if CSV is missing.
    try:
        df = pd.read_csv(csv_path)
        print("Reading CSV file {csv}...".format(csv=DATASET_CSV_PATH))
    except Exception as ee:
        print("Error while reading CSV file {csv} : {error}.".format(csv=DATASET_CSV_PATH, error=ee))
        return

    # Detect items on each frame.
    with tf.Session(graph=detection_graph) as sess:
        for index, row in df.iterrows():
            detections = detect_items(image_path=os.path.join(folder_path, row["Path"]), session=sess)
            if not detections:
                new_df = new_df.append({
                    "Path": row["Path"],
                    "Class": "",
                    "Xmin": "",
                    "Ymin": "",
                    "Xmax": "",
                    "Ymax": "",
                    "Confidence": "",
                    "Is_occluded": "",
                    "Is_truncated": "",
                    "Is_depiction": ""
                }, ignore_index=True)
            else:
                for detection in detections:
                    new_df = new_df.append({
                        "Path": row["Path"],
                        "Class": detection["class"],
                        "Xmin": detection["box"]["xmin"],
                        "Ymin": detection["box"]["ymin"],
                        "Xmax": detection["box"]["xmax"],
                        "Ymax": detection["box"]["ymax"],
                        "Confidence": detection["confidence"],
                        "Is_occluded": False,
                        "Is_truncated": False,
                        "Is_depiction": False
                    }, ignore_index=True)
    return new_df


def main():
    """
    Main program.
    :return: void.
    """
    for folder_path in list_directories(FOLDERS_DIR):
        # Retrieve annotated folder as a dataframe.
        df = annotate_folder(folder_path)

        # Save dataframe as CSV if not empty.
        if df is not None:
            write_df_as_csv(df=df, path=os.path.join(folder_path, "roi-ai-{tresh}_{name}.csv".format(
                tresh=args.min_confidence, name=os.path.basename(folder_path))))
        else:
            print("No detections / Missing files.")


if __name__ == "__main__":
    main()
