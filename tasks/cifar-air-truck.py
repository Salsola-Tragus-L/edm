"""Binary CIFAR-10 logistic-regression task from DSMT, Section 5.1.

Only airplane (CIFAR-10 class 0) and truck (class 9) examples are retained.
Labels are mapped to {-1, +1}, and the optimized local objective is

    mean_j softplus(-y_j <x, u_j>) + (weight_decay / 2) ||x||^2.

The linear classifier intentionally has no intercept, matching equation (43),
where x is an element of R^p and u_j is the image vector.
"""

import contextlib
import itertools
import os
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.random import fork_rng
from torch.utils.data import DataLoader, Subset, random_split

from .api import Batch, Dataset, Gradient, Loss, Parameters, Quality, State, Task


AIRPLANE_CLASS = 0
TRUCK_CLASS = 9
DEFAULT_DATA_ROOT = os.environ.get("RELAYSGD_DATA_ROOT", "/root/autodl-tmp/edm/data")


def configured_device():
    requested = os.environ.get("RELAYSGD_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("RELAYSGD_DEVICE requests CUDA, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError(
            "RELAYSGD_DEVICE must be 'auto', 'cpu', or 'cuda', got "
            f"{requested!r}."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BinaryLogisticRegression(nn.Module):
    """A bias-free linear classifier on flattened 32 x 32 RGB images."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 32 * 32, 1, bias=False)

    def forward(self, inputs):
        return self.linear(torch.flatten(inputs, start_dim=1)).squeeze(1)


class AirplaneTruckSubset(torch.utils.data.Dataset):
    """Filter CIFAR-10 and map airplane/truck labels to -1/+1."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [
            index
            for index, target in enumerate(dataset.targets)
            if target in (AIRPLANE_CLASS, TRUCK_CLASS)
        ]
        self.targets = np.asarray(
            [-1 if dataset.targets[index] == AIRPLANE_CLASS else 1 for index in self.indices],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        return image, torch.tensor(float(self.targets[index]), dtype=torch.float32)


class PyTorchDataset:
    def __init__(self, dataset, device):
        self._set = dataset
        self._device = device

    def __len__(self):
        return len(self._set)

    def random_split(self, fractions: List[float], seed: int = 0) -> List[Dataset]:
        lengths = [int(fraction * len(self)) for fraction in fractions]
        lengths[0] += len(self) - sum(lengths)
        return [
            PyTorchDataset(split, self._device)
            for split in random_split(
                self._set, lengths, generator=torch.Generator().manual_seed(seed)
            )
        ]

    def label_sorted_split(self, num_workers: int) -> List[Dataset]:
        """Sort by label and make equal contiguous shards, as in the paper."""
        targets = np.asarray(self._set.targets)
        sorted_indices = np.argsort(targets, kind="stable")
        shards = np.array_split(sorted_indices, num_workers)
        return [
            PyTorchDataset(Subset(self._set, indices.tolist()), self._device)
            for indices in shards
        ]

    def prepare_batch(self, batch):
        return Batch(*batch).to(self._device)

    def iterator(
        self, batch_size: int, shuffle=True, repeat=True, ref_num_data=None
    ) -> Iterable[Tuple[float, Batch]]:
        reference_size = len(self) if ref_num_data is None else ref_num_data
        batches_per_epoch = max(1, int(reference_size / batch_size))
        loader = DataLoader(
            self._set,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(self._device.type == "cuda"),
            drop_last=True if repeat else False,
            # This module's requested filename contains a hyphen and is loaded
            # dynamically by the experiment adapter.  Worker subprocesses
            # cannot import that name reliably (notably with Windows spawn).
            num_workers=0,
        )
        step = 0
        for _ in itertools.count() if repeat else [0]:
            for batch in loader:
                yield float(step) / batches_per_epoch, self.prepare_batch(batch)
                step += 1


class CIFARAirTruckDataset(PyTorchDataset):
    max_batch_size = 128

    def __init__(self, split, data_root=DEFAULT_DATA_ROOT, device="cuda", download=False):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split {split!r}.")
        # The paper does not state data augmentation.  ToTensor preserves the
        # fixed finite-sum objective and scales pixels to [0, 1].
        transform = torchvision.transforms.ToTensor()
        cifar = torchvision.datasets.CIFAR10(
            root=data_root,
            train=(split == "train"),
            download=download,
            transform=transform,
        )
        super().__init__(AirplaneTruckSubset(cifar), device=device)


class CIFARAirTruckTask(Task):
    def __init__(
        self, weight_decay, model_name, data_split_method, non_iid_alpha=None, seed=0
    ):
        del non_iid_alpha, seed
        if model_name not in ("Logistic-L2", "Logistic"):
            raise ValueError(f"Unknown airplane/truck model {model_name!r}.")
        if weight_decay <= 0:
            raise ValueError("weight_decay must be positive for strong convexity.")

        self._device = configured_device()
        full_train = CIFARAirTruckDataset("train", device=self._device)
        self.max_batch_size = full_train.max_batch_size

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num_workers = torch.distributed.get_world_size()
            if data_split_method == "label_sorted":
                splits = full_train.label_sorted_split(num_workers)
            elif data_split_method == "random":
                splits = full_train.random_split([1 / num_workers] * num_workers)
            else:
                raise ValueError(
                    "data_split_method must be 'label_sorted' or 'random', got "
                    f"{data_split_method!r}."
                )
            self.mean_num_data_per_worker = sum(map(len, splits)) / num_workers
            self.data = splits[torch.distributed.get_rank()]
            if torch.distributed.get_rank() == 0:
                print("Airplane/truck samples per worker:", [len(split) for split in splits])
        else:
            self.data = full_train
            self.mean_num_data_per_worker = len(full_train)

        self._train_data_total = full_train
        self._test_data = CIFARAirTruckDataset("test", device=self._device)
        self._weight_decay = float(weight_decay)
        self._model = self._create_model()

    def _create_model(self):
        model = BinaryLogisticRegression().to(self._device)
        model.train()
        return model

    def initialize(self, seed=42) -> Tuple[Parameters, State]:
        with fork_rng_with_seed(seed):
            self._model = self._create_model()
        return [p.data for p in self._model.parameters()], []

    def _forward(self, inputs, parameters, is_training=False):
        self._model.train(is_training)
        for parameter, value in zip(self._model.parameters(), parameters):
            parameter.data = value
        return self._model(inputs)

    def _penalty(self, parameters):
        return 0.5 * self._weight_decay * sum(p.pow(2).sum() for p in parameters)

    def loss(self, parameters, state, batch, random_seed=None) -> Tuple[Loss, State]:
        del random_seed
        with torch.no_grad():
            logits = self._forward(batch._x, parameters, is_training=True)
            objective = F.softplus(-batch._y * logits).mean() + self._penalty(parameters)
        return objective.item(), state

    def loss_and_gradient(
        self, parameters, state, batch, random_seed=None
    ) -> Tuple[Loss, Gradient, State]:
        del random_seed
        logits = self._forward(batch._x, parameters, is_training=True)
        objective = F.softplus(-batch._y * logits).mean() + self._penalty(parameters)
        gradients = torch.autograd.grad(objective, list(self._model.parameters()))
        return objective.item(), gradients, state

    def quality(self, parameters, state, batch) -> Quality:
        del state
        with torch.no_grad():
            logits = self._forward(batch._x, parameters, is_training=False)
            data_loss = F.softplus(-batch._y * logits).mean()
            objective = data_loss + self._penalty(parameters)
            predictions = torch.where(logits >= 0, 1.0, -1.0)
            accuracy = predictions.eq(batch._y).float().mean()
        return {"loss": objective.item(), "accuracy": accuracy.item()}

    def evaluate(self, dataset, parameters, state) -> Quality:
        totals = {"loss": 0.0, "accuracy": 0.0}
        count = 0
        for _, batch in dataset.iterator(batch_size=250, shuffle=False, repeat=False):
            quality = self.quality(parameters, state, batch)
            count += len(batch)
            for key in totals:
                totals[key] += len(batch) * quality[key]
        if count == 0:
            raise RuntimeError("Cannot evaluate an empty dataset.")
        return {key: value / count for key, value in totals.items()}


@contextlib.contextmanager
def fork_rng_with_seed(seed):
    if seed is None:
        yield
    else:
        with fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


def download():
    CIFARAirTruckDataset("train", device="cpu", download=True)
    CIFARAirTruckDataset("test", device="cpu", download=True)
