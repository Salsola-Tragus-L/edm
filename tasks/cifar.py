import os
import contextlib
import itertools
import re
from typing import Iterable, List, Tuple

import torch
from torch.random import fork_rng
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split
from .utils.non_iid_dirichlet import distribute_data_dirichlet

from .api import Batch, Dataset, Gradient, Loss, Parameters, Quality, State, Task

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
    
class CifarTask(Task):
    def __init__(
        self, weight_decay, model_name, data_split_method, non_iid_alpha=None, seed=0
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = CifarDataset("train", device=self._device)
        self.max_batch_size = self.data.max_batch_size

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Splitting data by worker
            num_workers = torch.distributed.get_world_size()
            if data_split_method == "dirichlet":
                splits, indices_per_worker = self.data.dirichlet_split(
                    num_workers, non_iid_alpha, seed=seed
                )
                self._indices_per_worker = indices_per_worker
            elif data_split_method == "random":
                splits = self.data.random_split(
                    fractions=[1 / num_workers for _ in range(num_workers)], seed=seed
                )
            else:
                raise ValueError(
                    f"Unknown value {data_split_method} for data_split_method"
                )
            self.mean_num_data_per_worker = (
                sum(len(split) for split in splits) / num_workers
            )
            print(
                f"Splitting data using {data_split_method} according to",
                [len(split) for split in splits],
            )
            self.data = splits[torch.distributed.get_rank()]
        else:
            self.mean_num_data_per_worker = len(self.data)
        
        self._train_data_total = CifarDataset("train", device=self._device)
        self._test_data = CifarDataset("test", device=self._device)

        self._model_name = model_name
        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self._weight_decay_per_param = [
            0 if parameter_type(p) == "batch_norm" else weight_decay
            for p, _ in self._model.named_parameters()
        ]

    def initialize(self, seed=42) -> Tuple[Parameters, State]:
        with fork_rng_with_seed(seed):
            self._model = self._create_model()
        parameters = [p.data for p in self._model.parameters()]
        state = [b.data for b in self._model.buffers()]
        return parameters, state

    def loss(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, State]:
        with torch.no_grad():
            with fork_rng_with_seed(random_seed):
                output, state = self.forward(
                    batch._y, parameters, state, is_training=True
                )
        loss = self._criterion(output, batch._y).item()
        return loss, state

    def loss_and_gradient(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:
        with fork_rng_with_seed(random_seed):
            output, state = self._forward(batch._x, parameters, state, is_training=True)
        loss = self._criterion(output, batch._y)
        gradients = torch.autograd.grad(loss, list(self._model.parameters()))

        for g, wd, p in zip(gradients, self._weight_decay_per_param, parameters):
            g.add_(p, alpha=wd)
            
        return loss.item(), gradients, state

    def quality(
        self, parameters: List[torch.Tensor], state: List[torch.Tensor], batch: Batch
    ) -> Quality:
        """Average quality on the batch"""
        with torch.no_grad():
            output, _ = self._forward(batch._x, parameters, state, is_training=False)
        accuracy = torch.argmax(output, 1).eq(batch._y).sum().float() / len(batch)
        loss = self._criterion(output, batch._y)
        return {"loss": loss.item(), "accuracy": accuracy.item()}

    def evaluate(
        self,
        dataset: Dataset,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
    ) -> Quality:
        """Average quality on a dataset"""
        mean_quality = None
        count = 0
        for _, batch in dataset.iterator(batch_size=250, shuffle=False, repeat=False):
            quality = self.quality(parameters, state, batch)
            if mean_quality is None:
                count = len(batch)
                mean_quality = quality
            else:
                count += len(batch)
                weight = float(len(batch)) / count
                for key, value in mean_quality.items():
                    mean_quality[key] += weight * (quality[key] - mean_quality[key])
        return mean_quality

    def _forward(
        self,
        input,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        is_training=False,
    ) -> Tuple[torch.Tensor, State]:
        if is_training:
            self._model.train()
        else:
            self._model.eval()

        for param, value in zip(self._model.parameters(), parameters):
            param.data = value

        for buffer, value in zip(self._model.buffers(), state):
            buffer.data = value

        output = self._model(input)
        state = [b.data for b in self._model.buffers()]

        return output, state

    def _create_model(self):
        if self._model_name == "ResNet20":
            from .models.resnet20 import ResNet20

            model = ResNet20()
            model.to(self._device)
            model.train()
        elif self._model_name == "VGG-11":
            from .models.vgg import vgg11

            model = vgg11()
            model.to(self._device)
            model.train()
        elif self._model_name == "ResNet_evo20":
            from .models.resnet_evonorm import ResNet_cifar

            model = ResNet_cifar(20)
            model.to(self._device)
            model.train()
        elif self._model_name == "ResNet18":
            from .models.resnext import ResNet18
            
            model = ResNet18()
            model.to(self._device)
            model.train()
            return model
        return model


def parameter_type(parameter_name):
    if "conv" in parameter_name and "weight" in parameter_name:
        return "convolution"
    elif re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name):
        return "batch_norm"
    else:
        return "other"


@contextlib.contextmanager
def fork_rng_with_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


class PyTorchDataset(object):
    def __init__(self, dataset, device, prepare_batch=None):
        self._set = dataset
        self._device = device
        if prepare_batch is not None:
            self.prepare_batch = prepare_batch

    def __len__(self):
        return len(self._set)

    def random_split(self, fractions: List[float], seed: int = 0) -> List[Dataset]:
        lengths = [int(f * len(self._set)) for f in fractions]
        lengths[0] += len(self._set) - sum(lengths)
        return [
            PyTorchDataset(split, self._device)
            for split in random_split(
                self._set, lengths, torch.Generator().manual_seed(seed)
            )
        ]

    def dirichlet_split(
        self,
        num_workers: int,
        alpha: float = 1,
        seed: int = 0,
        distribute_evenly: bool = True,
    ) -> List[Dataset]:
        indices_per_worker = distribute_data_dirichlet(
            self._set.targets, alpha, num_workers, num_auxiliary_workers=10, seed=seed
        )

        if distribute_evenly:
            indices_per_worker = np.array_split(
                np.concatenate(indices_per_worker), num_workers
            )

        return ([
            PyTorchDataset(Subset(self._set, indices), self._device)
            for indices in indices_per_worker
        ],indices_per_worker)

    def prepare_batch(self, batch):
        return Batch(*batch).to(self._device)

    def iterator(
        self, batch_size: int, shuffle=True, repeat=True, ref_num_data=None
    ) -> Iterable[Tuple[float, Batch]]:
        if ref_num_data is not None:
            nn = int(ref_num_data / batch_size)
        else:
            nn = int(len(self) / batch_size)

        loader = DataLoader(
            self._set,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
        )
        # print(len(self._set))

        step = 0
        for _ in itertools.count() if repeat else [0]:
            for i , batch in enumerate(loader):#in range(nn):#, batch in enumerate(loader):
                # batch = torch.utils.data.RandomSampler(self._set, replacement=True, num_samples=batch_size)
                epoch_fractional = float(step) / nn
                # print(batch)
                # (x,y) = batch
                # print("********")
                # print(x)
                yield epoch_fractional, self.prepare_batch(batch)
                step += 1


class CifarDataset(PyTorchDataset):
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    max_batch_size = 128

    def __init__(
        self, split, data_root='/root/autodl-tmp/relaysgd/data', device="cuda"
    ):
        if split == "train":
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            )
            # transform = torchvision.transforms.Compose([
            #                        torchvision.transforms.ToTensor(),
            #                        torchvision.transforms.Normalize(
            #                            (0.1307,), (0.3081,))
            #                    ])

        elif split == "test":
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            )
            # transform = torchvision.transforms.Compose([
            #                        torchvision.transforms.ToTensor(),
            #                        torchvision.transforms.Normalize(
            #                            (0.1307,), (0.3081,))
            #                    ])
        else:
            raise ValueError(f"Unknown split '{split}'.")

        # dataset = torchvision.datasets.CIFAR10(
        #     root=data_root, train=(split == "train"), download=False, transform=transform
        # )
        # super().__init__(dataset, device=device)
        
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=(split == "train"), download=True, transform=transform
        )
        super().__init__(dataset, device=device)



def download():
    CifarDataset("train")