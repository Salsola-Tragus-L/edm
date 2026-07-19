#!/usr/bin/env python3
"""Reproduce DSMT Section 5.1 on CIFAR-10 airplanes versus trucks.

With no arguments this script launches the experiment sweep.  MPI workers are
started with ``--worker`` and use this same file as a non-invasive adapter for
the hyphenated task module; no existing project source file needs modification.
"""

import importlib.util
import math
import os
import signal
import subprocess
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK_PATH = os.path.join(PROJECT_ROOT, "tasks", "cifar-air-truck.py")
THIS_SCRIPT = os.path.abspath(__file__)
SOLVER_SCRIPT = os.path.join(
    os.path.dirname(THIS_SCRIPT), "solve_CIFAR_airplanes_trucks_logisticL2.py"
)


def rho_tag(rho):
    return format(rho, ".12g").replace("-", "m").replace(".", "p")


def load_task_module():
    # A package-qualified synthetic name lets relative imports such as .api work.
    spec = importlib.util.spec_from_file_location("tasks.cifar_air_truck", TASK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_worker():
    # train.py parses argv at import time and only accepts --path.
    sys.argv.remove("--worker")
    sys.path.insert(0, PROJECT_ROOT)
    import train as base_train
    import torch

    task_module = load_task_module()
    original_train = base_train.train
    original_init_distributed = base_train.init_distributed_pytorch

    def init_distributed_for_selected_device():
        requested = os.environ.get("RELAYSGD_DEVICE", "auto").strip().lower()
        if requested != "cpu":
            return original_init_distributed()

        # Keep the MPI process group used by the decentralized algorithms, but
        # do not execute train.py's unconditional torch.cuda.set_device().
        if base_train.config["distributed_world_size"] > 1:
            if base_train.config["distributed_backend"] != "mpi":
                raise ValueError("The CPU adapter currently expects the MPI backend.")
            print("Initializing with MPI on CPU")
            torch.distributed.init_process_group("mpi")
            print(
                "Rank",
                torch.distributed.get_rank(),
                "world size",
                torch.distributed.get_world_size(),
                "device cpu",
            )

    base_train.init_distributed_pytorch = init_distributed_for_selected_device

    def train_with_distance_to_xstar(config, task, timer):
        artifact = torch.load(config["xstar_path"], map_location=task._device)
        artifact_rho = float(artifact["rho"])
        config_rho = float(config["weight_decay"])
        if not math.isclose(artifact_rho, config_rho, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError(
                f"x* has rho={artifact_rho}, but this run uses rho={config_rho}."
            )
        xstar = [value.to(task._device) for value in artifact["parameters"]]
        rank = base_train.get_rank()
        steps = []
        squared_distances = []
        output_path = os.path.join(
            base_train.path,
            f"distance_to_xstar_rho{rho_tag(config_rho)}_rank{rank}.pt",
        )

        def save_distances():
            torch.save(
                {
                    "rho": config_rho,
                    "f_xstar": float(artifact["f_xstar"]),
                    "xstar_path": os.path.abspath(config["xstar_path"]),
                    "metric": "squared_l2_distance_to_xstar",
                    "rank": rank,
                    "iterations": list(range(len(steps))),
                    "epoch_steps": steps,
                    "values": squared_distances,
                },
                output_path,
            )

        for item in original_train(config, task, timer):
            train_stats, _, parameters, _ = item
            distance = sum(
                (parameter.detach() - optimum).pow(2).sum().item()
                for parameter, optimum in zip(parameters, xstar)
            )
            steps.append(float(train_stats.step))
            squared_distances.append(distance)
            if len(steps) % 1000 == 0 or train_stats.step >= config["num_epochs"]:
                save_distances()
            yield item

    base_train.train = train_with_distance_to_xstar

    def configure_air_truck_task():
        config = base_train.config
        if config["task"] != "CIFAR-Air-Truck":
            raise ValueError("This adapter only supports task='CIFAR-Air-Truck'.")

        if config["distributed_world_size"] > 1:
            if base_train.torch.distributed.get_rank() == 0:
                task_module.download()
            base_train.torch.distributed.barrier()

        return task_module.CIFARAirTruckTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )

    base_train.configure_task = configure_air_truck_task
    base_train.path = os.path.join(base_train.path, "")
    base_train.main()


def run_cmd(cmd):
    print(f'Executing command="{" ".join(cmd)}"')
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        start_new_session=True,
    )

    def stop_process():
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()

    try:
        for line in process.stdout:
            print(line, end="")
        return process.wait()
    except KeyboardInterrupt:
        stop_process()
        process.wait()
        raise


def ensure_xstar(rho, log_root):
    optimum_root = os.environ.get(
        "RELAYSGD_XSTAR_ROOT", os.path.dirname(THIS_SCRIPT)
    )
    os.makedirs(optimum_root, exist_ok=True)
    output = os.path.abspath(
        os.path.join(
            optimum_root,
            f"CIFAR_air_truck_logisticL2_xstar_rho{rho_tag(rho)}.pt",
        )
    )
    if not os.path.isfile(output):
        command = [
            sys.executable,
            SOLVER_SCRIPT,
            "--rho",
            str(rho),
            "--output",
            output,
        ]
        data_root = os.environ.get("RELAYSGD_DATA_ROOT")
        if data_root:
            command.extend(["--data-root", data_root])
        if run_cmd(command) != 0:
            raise SystemExit("Failed to compute x*.")
    return output


def aggregate_distances(logdir, num_workers, rho):
    import torch

    records = [
        torch.load(
            os.path.join(
                logdir, f"distance_to_xstar_rho{rho_tag(rho)}_rank{rank}.pt"
            ),
            map_location="cpu",
        )
        for rank in range(num_workers)
    ]
    steps = records[0]["epoch_steps"]
    if any(record["epoch_steps"] != steps for record in records[1:]):
        raise RuntimeError("Worker distance histories have inconsistent steps.")
    mean_values = np.asarray([record["values"] for record in records]).mean(axis=0)
    output = os.path.join(
        logdir, f"mean_distance_to_xstar_rho{rho_tag(rho)}.pt"
    )
    torch.save(
        {
            "rho": rho,
            "f_xstar": records[0]["f_xstar"],
            "xstar_path": records[0]["xstar_path"],
            "metric": "mean_i_squared_l2_distance_to_xstar",
            "num_workers": num_workers,
            "iterations": records[0]["iterations"],
            "epoch_steps": steps,
            "values": mean_values.tolist(),
        },
        output,
    )
    print(f"Saved averaged distance history to {output}")


def launch_experiments():
    gpus_per_node = int(os.environ.get("RELAYSGD_GPUS_PER_NODE", "4"))
    log_root = os.environ.get("RELAYSGD_LOG_ROOT", "/root/autodl-tmp/edm")
    rho = 0.2
    xstar_path = ensure_xstar(rho, log_root)

    # Names map onto the implementations already present in algorithms.py.
    # d2 is EDAS; gossip is DSGD; gradient-tracking is DSGT.
    decentralized_algorithms = [
        "dsmt",
        "dmsgt_hb",       # DSMT without LCA
        "dsgt_hb",
        "gradient-tracking",
        "d2",
        "gossip",
    ]

    def paper_ring_beta(num_workers):
        # With neighbour weights 1/6 and self-weight 2/3, the second eigenvalue
        # is lambda = 1 - (1/3)(1-cos(2*pi/n)).  This gives the spectral gaps
        # 6.6e-4 and 2.6e-3 stated under Figures 2a and 2b.  Lemma 2.1 defines
        # eta_w=1/(1+sqrt(1-lambda^2)) and tilde-rho_w=sqrt(eta_w).
        lambda_w = 1.0 - (1.0 / 3.0) * (
            1.0 - math.cos(2.0 * math.pi / num_workers)
        )
        eta_w = 1.0 / (1.0 + math.sqrt(1.0 - lambda_w * lambda_w))
        return math.sqrt(eta_w)

    for num_workers in (100, 50):
        # Each worker has 100 (n=100) or 200 (n=50) examples.  Batch size 10
        # therefore makes 10 or 20 local updates per epoch.
        num_epochs = 800 if num_workers == 100 else 400
        base_config = {
            "task": "CIFAR-Air-Truck",
            "model_name": "Logistic-L2",
            "overlap_communication": False,
            "base_optimizer": "SGD",
            "num_epochs": num_epochs,
            "num_lr_warmup_epochs": 0,
            "lr_schedule_milestones": [],
            "batch_size": 10,
            "weight_decay": rho,
            "l1_norm": 0.0,
            "data_split_method": "label_sorted",
            "non_iid_alpha": None,
            "distributed_world_size": num_workers,
            "gpus_per_node": gpus_per_node,
            "distributed_rank": 0,
            "log_verbosity": 10,
            "distributed_backend": "mpi",
            "test_interval": 10,
            "simulated_dropped_message_probability": 0.0,
            "K": 1,
            "learning_rate": 0.01,
            "topology": "ring",
            "gossip_weight": 1.0 / 6.0,
            "seed": 1,
            "xstar_path": xstar_path,
        }

        for seed in range(1, 11):
            for algorithm in decentralized_algorithms:
                config = dict(base_config, algorithm=algorithm, momentum=0.0, seed=seed)
                # DSMT and momentum baselines use beta = tilde-rho_w.  DSMT
                # computes its matching LCA coefficient from the ring matrix.
                if algorithm in ("dsmt", "dmsgt_hb", "dsgt_hb"):
                    config["momentum"] = paper_ring_beta(num_workers)

                run_name = f"n{num_workers}-{algorithm}-ring-lr0.01-seed{seed}"
                logdir = os.path.join(
                    log_root,
                    "logs-CIFAR-airplanes-trucks-logisticL2",
                    run_name,
                )
                os.makedirs(logdir, exist_ok=True)
                np.save(os.path.join(logdir, "config.npy"), config)

                if os.path.isfile(os.path.join(logdir, "loss0.npy")):
                    print(f"Skip completed: {run_name}")
                    continue

                returncode = run_cmd(
                    [
                        "mpirun",
                        "-np",
                        str(num_workers),
                        sys.executable,
                        THIS_SCRIPT,
                        "--worker",
                        "--path",
                        logdir,
                    ]
                )
                if returncode != 0:
                    raise SystemExit(
                        f"Experiment {run_name} failed with code {returncode}."
                    )
                aggregate_distances(logdir, num_workers, rho)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker()
    else:
        launch_experiments()
