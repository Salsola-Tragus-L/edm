#!/usr/bin/env python3

import os
import json
import subprocess
import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
from train import main
import numpy as np

gpus_per_node = 4
num_workers = 16

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
    "lr_schedule_milestones": [(60, 0.1), (80, 0.1)],
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

def run_cmd( cmd_str='', echo_print=1):
    """
    执行cmd命令，不显示执行过程中弹出的黑框
    备注：subprocess.run()函数会将本来打印到cmd上的内容打印到python执行界面上，所以避免了出现cmd弹出框的问题
    :param cmd_str: 执行的cmd命令
    :return: 
    """
    from subprocess import run
    if echo_print == 1:
        print('执行cmd指令="{}"'.format(cmd_str))
    run(cmd_str, shell=True)
    
# best_lrs = {
#     (1, "double-binary-trees"): 0.1,
#     (.1, "double-binary-trees"): 0.025,
#     (.01, "double-binary-trees"): 0.025,
#     (1, "ring"): 0.1,
#     (.1, "ring"): 0.025,
#     (.01, "ring"): 0.05,
# }
# alg_list = ["d2"]
alg_list = ["edm","d2","dmsgt_hb","quasi-global-momentum","gradient-tracking","decent_lam"]
# alg_list = ["edm", "d2", "gradient-tracking", "quasi-global-momentum"]

for seed in [1,2,3]:                                      # [2,3]
    for alg in alg_list:
        for alpha in [1]:                               # [1, 0.1, 0.01]
            for topology in ["ring"]:                     # ['fully-connected'], ["ring"]
                lr = 0.1                                 # best_lrs[alpha, topology]
                config = {**base_config,
                          "learning_rate": lr,
                          "momentum": 0.9,
                          # "momentum": 0,
                          "topology": topology,
                          "non_iid_alpha": alpha,
                          "seed": seed,
                          "algorithm": alg}
                job_name = "{task}-{model_name}/alpha{non_iid_alpha}-{algorithm}-{topology}-mom{momentum}-lr{learning_rate}-seed{seed}".format(**config)
                logdir = "/root/autodl-tmp/relaysgd/logs/"+job_name + '/'
                if not os.path.isdir(logdir):
                    os.makedirs(logdir)
                np.save(logdir+'config.npy', config)
                run_cmd('mpirun -np {} python /root/autodl-tmp/relaysgd/train.py --path "{}"'.format(num_workers, logdir))

os.system("shutdown -s -t 10")