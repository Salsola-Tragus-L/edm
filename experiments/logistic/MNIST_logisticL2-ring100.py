#!/usr/bin/env python3

import os
import signal
import subprocess
import sys

import numpy as np


gpus_per_node = 4
num_workers = 16

which_exp = "MNIST-logisticL2-ring100step-beta-lr"
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_script = os.path.join(project_root, "train.py")
log_root = os.environ.get("RELAYSGD_LOG_ROOT", "/root/autodl-tmp/edm")
diverge_hist_path = os.path.join(log_root, f"logs-{which_exp}", "diverge_hist.txt")

description = "Repetitions"

base_config = {
    "task": "MNIST",
    "model_name": "Logistic-L2",
    "overlap_communication": False,
    "base_optimizer": "SGD",
    "num_epochs": 20,
    "num_lr_warmup_epochs": 5,
    "lr_schedule_milestones": [(150, 0.1), (180, 0.1)],
    "batch_size": 32,
    # The task adds weight_decay * parameter to the data-loss gradient,
    # corresponding to the penalty (weight_decay / 2) * ||theta||_2^2.
    "weight_decay": 1e-4,
    "l1_norm": 1e-4,
    "data_split_method": "dirichlet",
    "non_iid_alpha": None,
    "distributed_world_size": num_workers,
    "gpus_per_node": gpus_per_node,
    "distributed_rank": 0,
    "log_verbosity": 10,
    "distributed_backend": "mpi",
    "test_interval": 2,
    "simulated_dropped_message_probability": 0.0,
    "K": 3,
}


def run_cmd(cmd, echo_print=True):
    if echo_print:
        print(f'Executing command="{" ".join(cmd)}"')
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        start_new_session=True,
    )
    has_diverged = False

    def stop_process():
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()

    try:
        for line in process.stdout:
            print(line, end="")
            if "RuntimeError: diverged" in line:
                has_diverged = True
                stop_process()
                break
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    except KeyboardInterrupt:
        stop_process()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if has_diverged:
            return process.returncode, has_diverged
        raise
    return process.returncode, has_diverged


def already_diverged(diverge_record):
    if not os.path.isfile(diverge_hist_path):
        return False
    with open(diverge_hist_path, "r") as file:
        return diverge_record in {line.strip() for line in file}


alg_list = ["edm", "dsmt", "dmsgt_hb", "quasi-global-momentum", "decent_lam"]

for seed in [1]:
    for alpha in [0.1]:
        for mom in [0.8, 0.9, 0.95, 0.99]:
            for topology in ["ring"]:
                for lr in [0.01, 0.025, 0.05, 0.1]:
                    for alg in alg_list:
                        config = {
                            **base_config,
                            "learning_rate": lr,
                            "momentum": mom,
                            "topology": topology,
                            "non_iid_alpha": alpha,
                            "seed": seed,
                            "algorithm": alg,
                        }
                        job_name = (
                            "{task}-{model_name}/alpha{non_iid_alpha}-{algorithm}-"
                            "{topology}-mom{momentum}-lr{learning_rate}-seed{seed}"
                        ).format(**config)
                        run_name = (
                            "alpha{non_iid_alpha}-{algorithm}-{topology}-mom{momentum}-"
                            "lr{learning_rate}-seed{seed}"
                        ).format(**config)
                        diverge_record = run_name + ": diverged"
                        logdir = os.path.join(log_root, f"logs-{which_exp}", job_name)
                        os.makedirs(logdir, exist_ok=True)
                        np.save(os.path.join(logdir, "config.npy"), config)

                        if len(os.listdir(logdir)) >= 96:
                            print(f"Skip completed: {run_name}")
                            continue
                        if already_diverged(diverge_record):
                            print(f"Skip diverged: {run_name}")
                            continue

                        returncode, has_diverged = run_cmd(
                            [
                                "mpirun",
                                "-np",
                                str(num_workers),
                                sys.executable,
                                train_script,
                                "--path",
                                logdir,
                            ]
                        )
                        if returncode != 0 and has_diverged:
                            os.makedirs(os.path.dirname(diverge_hist_path), exist_ok=True)
                            with open(diverge_hist_path, "a") as file:
                                file.write(diverge_record + "\n")
                            print(f"Record diverged: {run_name}")

os.system("shutdown -s -t 10")
