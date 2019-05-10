#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Model training.
    ======================

    Runs both train and evaluation on object detection model.
    Copyright 2017 The TensorFlow Authors. All Rights Reserved.
"""

import sys
import tensorflow as tf

sys.path.append("models/research/slim")

from utils import *
from config import *
from argparse import ArgumentParser
from object_detection import model_lib
from object_detection import model_hparams

__description__ = "Runs both train and evaluation on object detection model."

# Parse args.
parser = ArgumentParser(description=__description__)
parser.add_argument("--checkpoints", type=bool, default=TRAINER_CONFIG["resume_from_ckpt"],
                    help="Reuse previous checkpoints if any, default is {default}.".format(
                        default=TRAINER_CONFIG["resume_from_ckpt"]))
parser.add_argument("--run-once", type=bool, default=TRAINER_CONFIG["run_once"],
                    help="If running in evaluation-only mode, whether to run just one round of evaluation vs running "
                         "continuously (default), default is {default}.".format(default=TRAINER_CONFIG["run_once"]))
args = parser.parse_args()


def main(_):
    """
    Main program.
    :param _: unused parameter.
    :return: void.
    """
    config = tf.estimator.RunConfig(model_dir=TRAINER_CONFIG["checkpoints_dir"])

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(TRAINER_CONFIG["hparams_override"]),
        pipeline_config_path=TRAINER_CONFIG["pipeline_config_path"],
        train_steps=TRAINER_CONFIG["num_train_steps"],
        sample_1_of_n_eval_examples=TRAINER_CONFIG["sample_1_of_n_eval_examples"],
        sample_1_of_n_eval_on_train_examples=(TRAINER_CONFIG["sample_1_of_n_eval_on_train_example"]))

    estimator = train_and_eval_dict["estimator"]
    train_input_fn = train_and_eval_dict["train_input_fn"]
    eval_input_fns = train_and_eval_dict["eval_input_fns"]
    eval_on_train_input_fn = train_and_eval_dict["eval_on_train_input_fn"]
    predict_input_fn = train_and_eval_dict["predict_input_fn"]
    train_steps = train_and_eval_dict["train_steps"]

    # Train model from checkpoints.
    if args.checkpoints and len(os.listdir(TRAINER_CONFIG["checkpoints_dir"])) > 1:
        if TRAINER_CONFIG["eval_training_data"]:
            name = "training_data"
            input_fn = eval_on_train_input_fn
        else:
            name = "validation_data"
            input_fn = eval_input_fns[0]

        if args.run_once:
            estimator.evaluate(input_fn, steps=None,
                               checkpoint_path=tf.train.latest_checkpoint(TRAINER_CONFIG["checkpoints_dir"]))

        else:
            model_lib.continuous_eval(estimator, TRAINER_CONFIG["checkpoints_dir"], input_fn,
                                      train_steps, name)
    # Train model from scratch.
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=TRAINER_CONFIG["eval_training_data"])

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == "__main__":
    tf.app.run()
