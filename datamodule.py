# -*- coding: utf-8 -*-

import torchvision.transforms as T

import cl_lite.core as cl

class DataModule(cl.SplitedDataModule):
    finetuning: bool = False

    @property
    def train_transforms(self):
        return [
            T.RandomCrop(self.dims[-2:], padding=4),
            T.RandomHorizontalFlip(),
        ]
