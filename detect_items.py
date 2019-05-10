#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Items detection.
    ======================
    Retrieves videostream and shows detected items.
"""
import tensorflow as tf

from utils import *
from config import *
from argparse import ArgumentParser
from multiprocessing import Queue, Pool
from cam_utils import FPS, WebcamVideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

__description__ = "Retrieves videostream and shows detected items."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("--video-source", dest="video_source", type=int, default=DEVICE_CONFIG["id"],
                    help="Capture device identifier, default is {default}.".format(default=DEVICE_CONFIG["id"]))
parser.add_argument('-wd', '--width', dest='width', type=int, default=DEVICE_CONFIG["width"],
                    help='Width of the frames in the video stream, default is {default}.'.format(
                        default=DEVICE_CONFIG["height"]))
parser.add_argument('-ht', '--height', dest='height', type=int, default=DEVICE_CONFIG["height"],
                    help='Height of the frames in the video stream, default is {default}.'.format(
                        default=DEVICE_CONFIG["height"]))
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=DETECTION_CONFIG["num_workers"],
                    help='Number of workers, default is {default}.'.format(default=DETECTION_CONFIG["num_workers"]))
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=DETECTION_CONFIG["queue_size"],
                    help='Size of the queue, default is {default}.'.format(default=DETECTION_CONFIG["queue_size"]))
parser.add_argument("--min-confidence", type=str, choices=list(DETECTION_CONFIG["score_treshes"].keys()),
                    default="medium",
                    help="Min confidence to reach to display a box, default is {default}.".format(default="medium"))
parser.add_argument("--max-boxes", type=int, default=DETECTION_CONFIG["max_boxes_to_draw"],
                    help="Max number of boxes to draw, default is {default}.".format(
                        default=DETECTION_CONFIG["max_boxes_to_draw"]))
args = parser.parse_args()

# Load labelmap file.
label_map = label_map_util.load_labelmap(DETECTION_CONFIG["labelmap_path"])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=DETECTION_CONFIG["num_classes"],
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
GRAPH_PATH = OUTPUTS_DIR + "frozen_inference_graph.pb"


def detect_objects(image_np, sess, detection_graph):
    """
    Detects objects on a frame.
    :param image_np: input frame.
    :param sess: Tensorflow session.
    :param detection_graph: Tensorflow model.
    :return: detected items.
    """
    # Expand dimensions of the model.
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents the level of confidence for each object.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=["use_normalized_coordinates"],
        line_thickness=DETECTION_CONFIG["line_thickness"],
        max_boxes_to_draw=args.max_boxes,
        min_score_thresh=DETECTION_CONFIG["score_treshes"][args.min_confidence])
    return image_np


def worker(input_q, output_q):
    """
    Loads a frozen Tensorflow model in memory.
    :param input_q:
    :param output_q:
    :return:
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        output_q.put(detect_objects(frame, sess, detection_graph))

    fps.stop()
    sess.close()


def main():
    """
    Main program.
    :return: void.
    """
    # Create a Thread pool.
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    # Grab video input.
    video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()
    fps = FPS().start()

    # Init a full-screen window.
    #cv.namedWindow('Webcam input', cv.WND_PROP_FULLSCREEN)
    #cv.setWindowProperty('Webcam input', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Read video input.
    while True:
        # Grab frame.
        frame = video_capture.read()

        # Send frame to AI.
        input_q.put(frame)

        # Show video input.
        cv.imshow('Webcam input', output_q.get())

        # Loop breaker.
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()

    # End program properly.
    pool.terminate()
    video_capture.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
