#!/usr/bin/env python3

import os
import json
import subprocess
import os,sys 
from datetime import datetime
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
from train import main
import numpy as np

gpus_per_node = 4
num_workers = 16

which_exp = "ring100step-beta-lr"
base_logdir = "/root/autodl-tmp/edm/logs-{}/".format(which_exp)
status_path = os.path.join(base_logdir, "run_status.json")

description = "Repetitions"

base_config = {
    # "seed": 1,
    "task": "Cifar", # "FashionMNIST",
    # "model_name": "VGG-11", #"ResNet_evo",  #"VGG-11",
    "model_name": "VGG-11",
    # "algorithm": "dmsgd_gt",   #'PDFP',#'gradient-tracking',
    "overlap_communication": False,
    "base_optimizer": "SGD",
    "num_epochs": 100,
    "num_lr_warmup_epochs": 5,
    "lr_schedule_milestones": [(150, 0.1), (180, 0.1)],
    "batch_size": 32,
    "weight_decay": 1e-4,
    "l1_norm": 1e-4,
    "data_split_method": "dirichlet",
    "non_iid_alpha": None,
    "distributed_world_size": num_workers, 
    "gpus_per_node": gpus_per_node,
    "distributed_rank":0,
    "log_verbosity":10,
    "distributed_backend":"mpi",
    "test_interval":2,
    "simulated_dropped_message_probability":0.0,
    "K":3,
}

def load_status():
    if not os.path.isfile(status_path):
        return {}
    with open(status_path, "r") as f:
        return json.load(f)


def save_status(status):
    if not os.path.isdir(base_logdir):
        os.makedirs(base_logdir)
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2, sort_keys=True)
    os.replace(tmp_path, status_path)


def job_key(config):
    key_fields = [
        "task",
        "model_name",
        "non_iid_alpha",
        "algorithm",
        "topology",
        "momentum",
        "learning_rate",
        "seed",
    ]
    return "|".join("{}={}".format(field, config[field]) for field in key_fields)


def log_contains_diverged(log_path):
    if not os.path.isfile(log_path):
        return False
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "RuntimeError: diverged" in line or "diverged" in line:
                return True
    return False


def run_cmd(cmd_str='', echo_print=1, log_path=None):
    if echo_print == 1:
        print('run cmd="{}"'.format(cmd_str))
    if log_path is None:
        return subprocess.run(cmd_str, shell=True).returncode

    with open(log_path, "a") as log_file:
        log_file.write("\n[{}] cmd={}\n".format(datetime.now().isoformat(timespec="seconds"), cmd_str))
        completed = subprocess.run(
            cmd_str,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.write("[{}] returncode={}\n".format(
            datetime.now().isoformat(timespec="seconds"),
            completed.returncode,
        ))
    return completed.returncode


# alg_list = ["edm","d2","dsmt","dmsgt_hb", "gradient-tracking","quasi-global-momentum","decent_lam"]
alg_list = ["edm","dsmt","dmsgt_hb","quasi-global-momentum","decent_lam"]
run_status = load_status()

for seed in [1]:                                      # [2,3]
    for alpha in [0.1]:
        for mom in [0.8, 0.9, 0.95, 0.99]:                               # [1, 0.1, 0.01]
            for topology in ["ring"]:                     # ['fully-connected'], ["ring"]
                for lr in [0.01, 0.025, 0.05, 0.1]:
                    for alg in alg_list:
                        config = {**base_config,
                                "learning_rate": lr,
                                "momentum": mom,
                                # "momentum": 0,
                                "topology": topology,
                                "non_iid_alpha": alpha,
                                "seed": seed,
                                "algorithm": alg}
                        job_name = "{task}-{model_name}/alpha{non_iid_alpha}-{algorithm}-{topology}-mom{momentum}-lr{learning_rate}-seed{seed}".format(**config)
                        logdir = os.path.join(base_logdir, job_name) + '/'
                        key = job_key(config)
                        previous_status = run_status.get(key, {}).get("status")
                        if previous_status in ["success", "diverged"]:
                            print("Skip {}: {}".format(previous_status, job_name))
                            continue
                        if not os.path.isdir(logdir):
                            os.makedirs(logdir)
                        np.save(logdir+'config.npy', config)
                        run_status[key] = {
                            "status": "running",
                            "job_name": job_name,
                            "logdir": logdir,
                            "config": config,
                            "started_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        save_status(run_status)

                        cmd = 'mpirun -np {} python /root/autodl-tmp/edm/train.py --path "{}"'.format(num_workers, logdir)
                        cmd_log_path = os.path.join(logdir, "run_cmd.log")
                        try:
                            returncode = run_cmd(cmd, log_path=cmd_log_path)
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            returncode = None
                            run_status[key]["error"] = repr(exc)

                        if returncode == 0:
                            status = "success"
                        elif log_contains_diverged(cmd_log_path):
                            status = "diverged"
                        else:
                            status = "failed"

                        run_status[key].update({
                            "status": status,
                            "returncode": returncode,
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                            "cmd_log_path": cmd_log_path,
                        })
                        save_status(run_status)
                        print("Recorded {}: {}".format(status, job_name))

os.system("shutdown -s -t 10")
