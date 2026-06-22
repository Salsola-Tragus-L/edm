#!/usr/bin/env python3

import datetime
import math
import os
import sys

import numpy as np
import torch
import argparse
from algorithms import train
from tasks.api import Task
from utils.accumulators import Mean as MeanAccumulator
# from utils.communication import get_rank
from utils.timer import Timer
from utils.communication import (
    MultiTopologyGossipMechanism,
    MultiTopologyRelayMechanism,
    get_rank,
    get_world_size,
    isend,
    num_bytes,
    pack,
    recv,
    unpack,
)

# output_dir = "./output.tmp"  # can be overwritten by the code running this script

parser = argparse.ArgumentParser()

# Add a positional argument for the path
parser.add_argument('--path', type=str, help='File path',default="/root/autodl-tmp/edm/")

# Parse command line arguments
args = parser.parse_args()

# Access the path argument
path = args.path

# Use the path as needed
print(f"Received path: {path}")

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def main():
    
    global config
    
    config = np.load(path+'/config.npy',allow_pickle=True).item()

    torch.manual_seed(config["seed"] + config["distributed_rank"])
    np.random.seed(config["seed"] + config["distributed_rank"])

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)

    init_distributed_pytorch()

    task = configure_task()

    epoch_metrics = MeanAccumulator()
    
    sys.stdout = Logger(path+"text.txt")
    
    loss_epoch,loss_meanx,test_epoch,loss_fxi, var_localx = {},{},{},{},{}#{'-1':0},{'-1':0}
    # np.save(path+'loss.npy', loss_epoch)
    # np.save(path+'test.npy', test_epoch)

        
    for train_stats, batch_stats, parameters, state in train(config, task, timer):
        epoch = train_stats.step
        training_time = {"epoch": epoch, "mb": train_stats.bytes_sent / 1024 / 1024}

        if batch_stats.loss is not None:
            # loss_epoch = np.load(path+'loss.npy',allow_pickle=True).item()
            if(epoch not in loss_epoch.keys()):
                loss_epoch[epoch] = []
            loss_epoch[epoch].append(batch_stats.loss)
            # np.save(path+'loss.npy', loss_epoch)
            
            epoch_metrics.add({"loss": batch_stats.loss})
            if math.isnan(batch_stats.loss):
                raise RuntimeError("diverged")
                
        ## loss_meanx f(\bar{x})
        if epoch % config["test_interval"] == 0:
            with timer("global_loss_calculation"):
                parameters_meanx = [p.clone() for p in parameters]
                buffer, shapes = pack(parameters_meanx)
                torch.distributed.all_reduce(buffer)
                buffer /= get_world_size()
                parameters_meanx = unpack(buffer, shapes)
                mean_stats = task.evaluate(task.data, parameters_meanx, state)
                
                if(epoch not in loss_meanx.keys()):
                    loss_meanx[epoch] = []
                loss_meanx[epoch].append(mean_stats)
                
#         ## training loss f(x_i)
#         if epoch % config["test_interval"] == 0:
#             with timer("training_loss_for_f"):
#                 train_fxi_stats = task.evaluate(task._train_data_total, parameters, state)
                
#                 if(epoch not in loss_fxi.keys()):
#                     loss_fxi[epoch] = []
#                 loss_fxi[epoch].append(train_fxi_stats)
                
        ## training ||x - x_i||^2
        if epoch % config["test_interval"] == 0:
            with timer("cal_var_localx"):
                squared_diffs = []
                for param, param_meanx in zip(parameters, parameters_meanx):
                    diff = param - param_meanx  # 计算差值
                    squared_diff = diff.pow(2)   # 计算平方
                    squared_diffs.append(squared_diff)
                    
                total_squared_diff = sum(s.sum().item() for s in squared_diffs)
                
                if(epoch not in var_localx.keys()):
                    var_localx[epoch] = []
                var_localx[epoch].append(total_squared_diff)
                
        ## test_loss f(x_i)
        if epoch % config["test_interval"] == 0:
            with timer("test"):
                test_stats = task.evaluate(task._test_data, parameters, state)
                
                # test_epoch = np.load(path+'test.npy',allow_pickle=True).item()
                if(epoch not in test_epoch.keys()):
                    test_epoch[epoch] = []
                test_epoch[epoch].append(test_stats)
                
                for key, value in test_stats.items():
                    log_metric(
                        key,
                        {"value": value, **training_time},
                        tags={"split": "test", "worker": get_rank()},
                    )

        # if (epoch <= 5 and (epoch % 1 == 0)) or epoch % config["test_interval"] == 0:
        if epoch % config["test_interval"] == 0 and get_rank()==0:
            # print('time!!!!!!!!!!!')
            for entry in timer.transcript():
                print( entry["event"], entry["mean"], entry["std"], entry["instances"])
                log_runtime(
                    entry["event"], entry["mean"], entry["std"], entry["instances"]
                )

        if epoch >= config["num_epochs"]:
            info({"state.progress": 1.0})
            timer.save_summary(path+"time{}.txt".format(get_rank()))
            np.save(path+'loss{}.npy'.format(get_rank()), loss_epoch)
            np.save(path+'loss_meanx{}.npy'.format(get_rank()), loss_meanx)
            np.save(path+'test{}.npy'.format(get_rank()), test_epoch)
            # np.save(path+'loss_fxi{}.npy'.format(get_rank()), loss_fxi)
            np.save(path+'var_localx{}.npy'.format(get_rank()), var_localx)
            break


def configure_task() -> Task:
    if config["task"] == "Cifar":
        from tasks.cifar import CifarTask, download

        if config["distributed_world_size"] > 1:
            if torch.distributed.get_rank() == 0:
                download()
            torch.distributed.barrier()

        return CifarTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "FashionMNIST":
        from tasks.FashionMNIST import FashionTask, download

        if config["distributed_world_size"] > 1:
            if torch.distributed.get_rank() == 0:
                download()
            torch.distributed.barrier()

        return FashionTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "MNIST":
        from tasks.mnist import MNISTTask, download

        if config["distributed_world_size"] > 1:
            if torch.distributed.get_rank() == 0:
                download()
            torch.distributed.barrier()

        return MNISTTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "ImageNet":
        from tasks.imagenet import ImageNetTask

        return ImageNetTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "DeIT":
        from tasks.deit import ImagenetTask

        return ImagenetTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif "BERT" in config["task"]:
        from tasks.bert import BERTTask

        return BERTTask(
            weight_decay=config["weight_decay"],
            data_name=config["task"].split("-")[-1],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "Quadratics":
        from tasks.quadratics import QuadraticsTask

        return QuadraticsTask(
            d=config["quadratics_d"],
            non_iidness=config["quadratics_non_iidness"],
            sgd_noise_variance=config["quadratics_sgd_noise_variance"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "Delivery":
        from tasks.delivery import DeliveryTask

        return DeliveryTask()
    else:
        raise ValueError("Unsupported task {}".format(config["task"]))


def init_distributed_pytorch():
    if config["distributed_world_size"] > 1:
        if config["distributed_backend"] == "mpi":
            print("Initializing with MPI")
            torch.distributed.init_process_group("mpi")
            print(
                "Rank",
                torch.distributed.get_rank(),
                "world size",
                torch.distributed.get_world_size(),
            )
            torch.cuda.set_device(
                torch.distributed.get_rank() % config["gpus_per_node"]
            )
        else:
            if config["distributed_init_file"] is None:
                config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
            print(
                "Distributed init: rank {}/{} - {}".format(
                    config["distributed_rank"],
                    config["distributed_world_size"],
                    config["distributed_init_file"],
                )
            )
            torch.distributed.init_process_group(
                backend=config["distributed_backend"],
                init_method="file://"
                + os.path.abspath(config["distributed_init_file"]),
                timeout=datetime.timedelta(seconds=120),
                world_size=config["distributed_world_size"],
                rank=config["distributed_rank"],
            )


def log_info(info_dict):
    """Add any information to MongoDB
    This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
    This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
 
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags)) 


def log_runtime(label, mean_time, std, instances):
    """This function will be overwritten when called through run.py"""
    pass


def info(*args, **kwargs):
    if config["distributed_rank"] == 0:
        log_info(*args, **kwargs)


def metric(*args, **kwargs):
    if config["distributed_rank"] == 0:
        log_metric(*args, **kwargs)


if __name__ == "__main__":
    main()
