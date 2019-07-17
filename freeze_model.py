#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Model exportation.
    ======================

    Exports/Freezes training checkpoints for future detections.
"""

import tensorflow as tf

from utils import *
from config import *
from argparse import ArgumentParser
from object_detection import exporter
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

__description__ = "Exports/Freezes training checkpoints for future detections."

# Parse args.
parser = ArgumentParser(description=__description__)

parser.add_argument("--input-type", type=str, choices=("image_tensor", "tf_example", "encoded_image_string_tensor"),
                    default="image_tensor",
                    help="Type of input node, default is {default}.".format(default="image_tensor"))
parser.add_argument("--config-path", type=str, default=PIPELINE_CONFIG_PATH,
                    help="Path of the model config file, default is {default}.".format(
                        default=PIPELINE_CONFIG_PATH))
parser.add_argument("--checkpoint-prefix", type=str, default="model.ckpt",
                    help="Path to trained checkpoint, typically of the form path/to/model.ckpt, default is {default}.".format(
                        default="model.ckpt"))
parser.add_argument("--output-directory", type=str, default=OUTPUTS_DIR,
                    help="Path to write outputs, default is {default}.".format(
                        default=OUTPUTS_DIR))
parser.add_argument("--config-override", type=str, default="",
                    help="Override pipeline config file content, default is {default}.".format(
                        default="''"))
parser.add_argument("--write-inference-graph", type=bool, default=False,
                    help="Write inference graph to disk, default is {default}.".format(
                        default=False))
args = parser.parse_args()


def main(_):
    """
    Main program.
    :param _: unused parameter.
    :return: void.
    """

    # Retrieve latest checkpoint prefix.
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    latest_checkpoint = find_latest_checkpoint(dir=CHECKPOINTS_DIR, prefix=args.checkpoint_prefix)

    if latest_checkpoint is None:
        return

    # Read model config file.
    with tf.gfile.GFile(args.config_file, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    # Override config file if any updates are provided.
    text_format.Merge(args.config_override, pipeline_config)

    # Freeze checkpoint file.
    exporter.export_inference_graph(
        args.input_type, pipeline_config, latest_checkpoint,
        args.output_directory, write_inference_graph=args.write_inference_graph)


if __name__ == "__main__":
    tf.app.run()
