#!/usr/bin/env python3
"""Training entry point that records the full L2-regularized objective.

This module deliberately leaves train.py unchanged.  It reuses its training
loop and wraps only the MNIST logistic task's reporting methods.
"""

from functools import wraps
import os

import train as base_train


def l2_penalty(parameters, weight_decay):
    """Return (weight_decay / 2) * ||parameters||_2^2 as a Python float."""
    return 0.5 * weight_decay * sum(
        parameter.detach().pow(2).sum().item() for parameter in parameters
    )


def configure_logistic_l2_task():
    task = original_configure_task()
    config = base_train.config

    if config["task"] != "MNIST" or config["model_name"] not in (
        "Logistic-L2",
        "Logistic",
    ):
        raise ValueError(
            "train_logistic_l2.py only supports the MNIST Logistic-L2 model."
        )

    weight_decay = config["weight_decay"]
    original_loss_and_gradient = task.loss_and_gradient
    original_evaluate = task.evaluate

    @wraps(original_loss_and_gradient)
    def loss_and_gradient_with_l2(parameters, state, batch, random_seed=None):
        cross_entropy, gradients, new_state = original_loss_and_gradient(
            parameters, state, batch, random_seed=random_seed
        )
        objective = cross_entropy + l2_penalty(parameters, weight_decay)
        return objective, gradients, new_state

    @wraps(original_evaluate)
    def evaluate_with_l2(dataset, parameters, state):
        quality = original_evaluate(dataset, parameters, state)

        # Keep test_epoch as cross-entropy + accuracy.  Training evaluations
        # (loss_meanx and loss_fxi) report cross-entropy + the L2 penalty.
        if dataset is not task._test_data:
            quality = dict(quality)
            quality["loss"] += l2_penalty(parameters, weight_decay)
        return quality

    task.loss_and_gradient = loss_and_gradient_with_l2
    task.evaluate = evaluate_with_l2
    return task


original_configure_task = base_train.configure_task
base_train.configure_task = configure_logistic_l2_task


if __name__ == "__main__":
    # train.py builds output filenames with string concatenation (path + name),
    # so its path must end with a directory separator.
    base_train.path = os.path.join(base_train.path, "")
    base_train.main()
