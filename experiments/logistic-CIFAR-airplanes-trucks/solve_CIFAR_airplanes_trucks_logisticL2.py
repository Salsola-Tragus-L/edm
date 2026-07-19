#!/usr/bin/env python3
"""Compute x* for the strongly convex CIFAR airplane/truck objective.

The saved artifact is self-describing: it contains rho, f(x*), the data-loss
and regularization components, the gradient norm, and the model parameters.
"""

import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK_PATH = os.path.join(PROJECT_ROOT, "tasks", "cifar-air-truck.py")


def load_task_module():
    sys.path.insert(0, PROJECT_ROOT)
    spec = importlib.util.spec_from_file_location("tasks.cifar_air_truck", TASK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rho_tag(rho):
    return format(rho, ".12g").replace("-", "m").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the CIFAR airplane/truck objective.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("RELAYSGD_DATA_ROOT", "/root/autodl-tmp/edm/data"),
    )
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument("--tolerance-grad", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to CIFAR_air_truck_logisticL2_xstar_rho<RHO>.pt.",
    )
    return parser.parse_args()


def objective_components(model, loader, rho, device, backward=False):
    num_examples = len(loader.dataset)
    data_loss_sum = 0.0
    correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_sum = F.softplus(-labels * model(inputs)).sum()
        if backward:
            (batch_sum / num_examples).backward()
        data_loss_sum += batch_sum.detach().item()
        correct += torch.where(model(inputs).detach() >= 0, 1.0, -1.0).eq(labels).sum().item()

    penalty_tensor = 0.5 * rho * sum(p.pow(2).sum() for p in model.parameters())
    if backward:
        penalty_tensor.backward()
    data_loss = data_loss_sum / num_examples
    penalty = penalty_tensor.detach().item()
    return data_loss + penalty, data_loss, penalty, correct / num_examples


def main():
    args = parse_args()
    if args.rho <= 0:
        raise ValueError("--rho must be positive for strong convexity.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    os.environ["RELAYSGD_DATA_ROOT"] = os.path.abspath(args.data_root)
    task_module = load_task_module()
    dataset = task_module.CIFARAirTruckDataset("train", device="cpu")._set
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(args.device == "cuda"),
        num_workers=0,
    )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = task_module.BinaryLogisticRegression().to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=args.max_iter,
        history_size=args.history_size,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        objective, _, _, _ = objective_components(
            model, loader, args.rho, device, backward=True
        )
        print(f"closure={closure_calls:4d} objective={objective:.12f}")
        return torch.tensor(objective, dtype=torch.float64, device=device)

    optimizer.step(closure)

    optimizer.zero_grad(set_to_none=True)
    objective, data_loss, penalty, accuracy = objective_components(
        model, loader, args.rho, device, backward=True
    )
    gradient_norm = torch.sqrt(
        sum(p.grad.detach().pow(2).sum() for p in model.parameters())
    ).item()
    parameters = [p.detach().cpu().clone() for p in model.parameters()]

    result = {
        "rho": args.rho,
        "objective_min": objective,
        "f_xstar": objective,
        "data_loss_at_xstar": data_loss,
        "l2_penalty_at_xstar": penalty,
        "gradient_norm": gradient_norm,
        "train_accuracy": accuracy,
        "num_examples": len(dataset),
        "closure_calls": closure_calls,
        "model": "bias-free binary logistic regression",
        "classes": {"airplane": -1, "truck": 1},
    }
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"CIFAR_air_truck_logisticL2_xstar_rho{rho_tag(args.rho)}.pt",
    )
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(
        {
            "metadata": result,
            "rho": args.rho,
            "f_xstar": objective,
            "parameters": parameters,
            "model_state_dict": model.state_dict(),
        },
        output,
    )
    print(json.dumps(result, indent=2))
    print(f"Saved x* to {output}")


if __name__ == "__main__":
    main()
