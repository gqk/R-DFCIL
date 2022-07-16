# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pytorch_lightning as pl


from cl_lite.head.dynamic_simple import DynamicSimpleHead
from cl_lite.optim import ConstantLR, SequentialLR
from cl_lite.nn import freeze, unfreeze

from datamodule import DataModule


class FinetuningMixin:
    hparams: object
    trainer: pl.Trainer
    datamodule: DataModule
    backbone: nn.Module
    head: DynamicSimpleHead
    current_epoch: int
    _finetuning_state: dict = {}

    @property
    def finetuning(self):
        assert hasattr(self.datamodule, "finetuning")
        return self.datamodule.finetuning

    @finetuning.setter
    def finetuning(self, finetuning: bool):
        if finetuning:
            self._finetuning_state["backbone"] = freeze(self.backbone)
            self._finetuning_state["head"] = freeze(self.head)
            if self.datamodule.current_task > 0:
                unfreeze(self.head.classifiers)
            self.head.train()
        else:
            unfreeze(self.backbone, self._finetuning_state.pop("backbone", {}))
            unfreeze(self.head, self._finetuning_state.pop("head", {}))

        print(f"\n==> finetuning mode: {finetuning}")

        self.datamodule.finetuning = finetuning
        self.trainer.reset_train_dataloader(self.trainer.lightning_module)

    def _create_finetuning_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        assert hasattr(self.hparams, "finetuning_epochs")
        assert hasattr(self.hparams, "finetuning_lr")
        assert hasattr(self.hparams, "base_lr")

        num_ft_epochs = self.hparams.finetuning_epochs
        if not (0 < num_ft_epochs < self.trainer.max_epochs):
            return None

        scheduler_ft = ConstantLR(
            optimizer,
            factor=self.hparams.finetuning_lr / self.hparams.base_lr,
            total_iters=num_ft_epochs,
            last_epoch=-2,  # prevent the step during initialization
        )

        return scheduler_ft

    def add_finetuning_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        scheduler_ft = self._create_finetuning_lr_scheduler(optimizer)
        if scheduler_ft is None:
            return scheduler

        num_ft_epochs = self.hparams.finetuning_epochs
        milestones = [self.trainer.max_epochs - num_ft_epochs]
        scheduler = SequentialLR(
            optimizer, [scheduler, scheduler_ft], milestones
        )

        return scheduler

    def should_start_finetuning(self, at_epoch_end: bool = True):
        if self.finetuning or self.hparams.finetuning_epochs < 1:
            return False

        num_ft_epochs = self.hparams.finetuning_epochs
        current_epoch = self.current_epoch
        if at_epoch_end:
            current_epoch = current_epoch + 1
        return current_epoch >= (self.trainer.max_epochs - num_ft_epochs)

    def should_stop_finetuning(self, at_epoch_end: bool = True):
        if not self.finetuning or self.hparams.finetuning_epochs < 1:
            return False

        current_epoch = self.current_epoch
        if at_epoch_end:
            current_epoch = current_epoch + 1
        return current_epoch >= self.trainer.max_epochs

    def check_finetuning(self, at_epoch_end: bool = True):
        if self.should_start_finetuning(at_epoch_end):
            self.finetuning = True
            return

        if self.should_stop_finetuning():
            self.finetuning = False
