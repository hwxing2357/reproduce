# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18


from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from torchvision.datasets import CIFAR10


class TCIFAR10(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10(ContinualDataset):
    """Sequential CIFAR10 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data."""

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    # N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    def get_examples_number(self) -> int:
        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                   download=True)
        return len(train_dataset.data)

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                   download=True, transform=transform)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TCIFAR10(base_path() + 'CIFAR10',train=False,
                                     download=True, transform=test_transform)

        train, test, context = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test, context
        #
        # test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False,
        #                          download=True, transform=test_transform)
        #
        # train, test = store_masked_loaders(train_dataset, test_dataset, self)
        #
        # return train, test

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS)
    #
    # @staticmethod
    # def get_transform():
    #     transform = transforms.Compose(
    #         [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
    #     return transform

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    # @set_default_from_args("backbone")
    # def get_backbone():
    #     return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR10.MEAN, SequentialCIFAR10.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR10.MEAN, SequentialCIFAR10.STD)
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 10

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR10.get_batch_size()

    @staticmethod
    def get_np_head(np_name, test_time_oracle=False):
        hierarchy=True
        if np_name == "anp":
            from backbone.neural_processes.ANP import ANP_HEAD as NP
            # from backbone.neural_processes.ANP_SetTransformer import ANP_HEAD as NP
        elif 'npcl' in np_name:
            # from backbone.neural_processes.NPCL import NPCL as NP
            from backbone.neural_processes.NPCL_robust import NPCL as NP
            if 'no-hierarchy' in np_name:
                hierarchy = False
        else:
            raise NotImplementedError("Enter valid NP type")
        return NP(512, 128,
                       SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS,
                       n_tasks=SequentialCIFAR10.N_TASKS, cls_per_task=SequentialCIFAR10.N_CLASSES_PER_TASK,
                       xavier_init=True, test_oracle=test_time_oracle, hierarchy=hierarchy)
