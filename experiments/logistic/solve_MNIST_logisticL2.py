#!/usr/bin/env python3
"""Compute the minimum of L2-regularized multinomial logistic regression on MNIST.

The optimized objective is

    (1 / n) * sum_i CrossEntropy(W x_i + b, y_i)
        + (weight_decay / 2) * (||W||_F^2 + ||b||_2^2).

This matches MNISTTask.loss_and_gradient: both ``linear.weight`` and
``linear.bias`` are included in the L2 penalty.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tasks.mnist import LogisticRegression  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve the strongly convex MNIST Logistic-L2 objective."
    )
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/edm/data",
        help="Directory containing the MNIST dataset.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--tolerance-grad", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--output",
        default="MNIST_logisticL2_optimum.pt",
        help="File used to save the optimum and model state.",
    )
    return parser.parse_args()


def make_loader(data_root, batch_size, device):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device == "cuda"),
        num_workers=2,
    )


def l2_penalty(model, weight_decay):
    squared_norm = sum(parameter.pow(2).sum() for parameter in model.parameters())
    return 0.5 * weight_decay * squared_norm


def full_objective_and_accuracy(model, loader, weight_decay, device):
    model.eval()
    cross_entropy_sum = 0.0
    correct = 0
    num_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            cross_entropy_sum += F.cross_entropy(
                logits, targets, reduction="sum"
            ).item()
            correct += logits.argmax(dim=1).eq(targets).sum().item()
            num_examples += targets.numel()

        cross_entropy = cross_entropy_sum / num_examples
        penalty = l2_penalty(model, weight_decay).item()
    return cross_entropy + penalty, cross_entropy, penalty, correct / num_examples


def main():
    args = parse_args()
    if args.weight_decay <= 0:
        raise ValueError("--weight-decay must be positive for strong convexity.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    loader = make_loader(args.data_root, args.batch_size, args.device)
    num_examples = len(loader.dataset)

    model = LogisticRegression().to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=args.max_iter,
        history_size=args.history_size,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)

        # Accumulate the exact gradient of the mean loss without retaining the
        # computation graph for all 60,000 examples at once.
        cross_entropy_value = 0.0
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_loss_sum = F.cross_entropy(
                model(inputs), targets, reduction="sum"
            )
            (batch_loss_sum / num_examples).backward()
            cross_entropy_value += batch_loss_sum.detach().item()

        penalty = l2_penalty(model, args.weight_decay)
        penalty.backward()
        objective = cross_entropy_value / num_examples + penalty.detach().item()
        print(f"closure={closure_calls:4d} objective={objective:.12f}")
        return torch.tensor(objective, device=device)

    optimizer.step(closure)

    objective, cross_entropy, penalty, accuracy = full_objective_and_accuracy(
        model, loader, args.weight_decay, device
    )
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    ).item()

    result = {
        "objective_min": objective,
        "cross_entropy": cross_entropy,
        "l2_penalty": penalty,
        "train_accuracy": accuracy,
        "gradient_norm": gradient_norm,
        "weight_decay": args.weight_decay,
        "num_examples": num_examples,
        "closure_calls": closure_calls,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"result": result, "model_state_dict": model.state_dict()}, args.output
    )

    print(json.dumps(result, indent=2))
    print(f"Saved optimum to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
