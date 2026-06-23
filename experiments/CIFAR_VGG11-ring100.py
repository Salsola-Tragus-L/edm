#!/usr/bin/env python3

import os
import json
import subprocess
import os,sys 
import signal
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
from train import main
import numpy as np

gpus_per_node = 4
num_workers = 16

which_exp = "ring100step-beta-lr"
diverge_hist_path = "/root/autodl-tmp/edm/logs-{}/diverge_hist.txt".format(which_exp)

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

def run_cmd(cmd_str='', echo_print=1):
    """
    执行cmd命令，不显示执行过程中弹出的黑框
    备注：subprocess.run()函数会将本来打印到cmd上的内容打印到python执行界面上，所以避免了出现cmd弹出框的问题
    :param cmd_str: 执行的cmd命令
    :return: 
    """
    if echo_print == 1:
        print('执行cmd指令="{}"'.format(cmd_str))
    process = subprocess.Popen(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
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
    with open(diverge_hist_path, "r") as f:
        return diverge_record in set(line.strip() for line in f)


# alg_list = ["edm","d2","dsmt","dmsgt_hb", "gradient-tracking","quasi-global-momentum","decent_lam"]
alg_list = ["edm","dsmt","dmsgt_hb","quasi-global-momentum","decent_lam"]

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
                        run_name = "alpha{non_iid_alpha}-{algorithm}-{topology}-mom{momentum}-lr{learning_rate}-seed{seed}".format(**config)
                        diverge_record = run_name + ": diverged"
                        logdir = "/root/autodl-tmp/edm/logs-{}/".format(which_exp)+job_name + '/'
                        if not os.path.isdir(logdir):
                            os.makedirs(logdir)
                        np.save(logdir+'config.npy', config)
                        is_completed = os.path.isdir(logdir) and len(os.listdir(logdir)) >= 96
                        if is_completed:
                            print("Skip completed: {}".format(run_name))
                            continue
                        if already_diverged(diverge_record):
                            print("Skip diverged: {}".format(run_name))
                            continue

                        returncode, has_diverged = run_cmd('mpirun -np {} python /root/autodl-tmp/edm/train.py --path "{}"'.format(num_workers, logdir))
                        if returncode != 0 and has_diverged:
                            with open(diverge_hist_path, "a") as f:
                                f.write(diverge_record + "\n")
                            print("Record diverged: {}".format(run_name))

os.system("shutdown -s -t 10")
