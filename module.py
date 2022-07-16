# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
from copy import deepcopy
from math import log, sqrt
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import cl_lite.backbone as B
import cl_lite.core as cl
from cl_lite.deep_inversion import GenerativeInversion
from cl_lite.head import DynamicSimpleHead
from cl_lite.mixin import FeatureHookMixin
from cl_lite.nn import freeze, RKDAngleLoss

from datamodule import DataModule
from mixin import FinetuningMixin


class Module(FeatureHookMixin, FinetuningMixin, cl.Module):
    datamodule: DataModule
    evaluator_cls = cl.ILEvaluator

    def __init__(
        self,
        base_lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        lr_factor: float = 0.1,
        milestones: List[int] = [80, 120],
        finetuning_epochs: int = 40,
        finetuning_lr: float = 0.005,
        lambda_ce: float = 0.5,
        lambda_hkd: float = 0.15,
        lambda_rkd: float = 0.5,
        num_inv_iters: int = 5000,
        inv_lr: float = 0.001,
        inv_tau: float = 1000.0,
        inv_alpha_pr: float = 0.001,
        inv_alpha_rf: float = 50.0,
        inv_resume_from: str = None,
    ):
        """Module of joint project

        Args:
            base_lr: Base learning rate
            momentum: Momentum value for SGD optimizer
            weight_decay: Weight decay value
            lr_factor: Learning rate decay factor
            milestones: Milestones for reducing learning rate
            finetuning_epochs: the number of finetuning epochs,
            finetuning_lr: the learning rate of finetuning,
            lambda_ce: the scale factor of cross entropy loss,
            lambda_hkd: the scale factor of stablility knowledge distillation,
            lambda_rkd: the scale factor of relation knowledge distillation,
            num_inv_iters: number of inversion iterations
            inv_lr: inversion learning rate
            inv_tau: temperature of inversion cross entropy loss
            inv_alpha_pr: factor of inversion image prior regularization
            inv_alpha_rf: factor of inversion feature statistics regularization
            inv_resume_from: resume inversion from a checkpoint
        """

        super().__init__()
        self.save_hyperparameters()

    def register_losses(self):
        self.register_loss(
            "ce",
            nn.CrossEntropyLoss(),
            ["prediction", "target"],
        )

        if self.model_old is None:
            return

        alpha = log(self.datamodule.num_classes / 2 + 1, 2)
        beta2 = self.model_old.head.num_classes / self.datamodule.num_classes
        beta = sqrt(beta2)

        self.set_loss_factor(
            "ce", self.hparams.lambda_ce * (1 + 1 / alpha) / beta
        )

        self.register_loss(
            "hkd",
            nn.L1Loss(),
            ["input_hkd", "target_hkd"],
            self.hparams.lambda_hkd * alpha * beta,
        )

        self.register_loss(
            "rkd",
            self.rkd,
            ["input_rkd", "target_rkd"],
            self.hparams.lambda_rkd * alpha * beta,
        )

    def update_old_model(self):
        model_old = [("backbone", self.backbone), ("head", self.head)]
        self.model_old = deepcopy(nn.Sequential(OrderedDict(model_old))).eval()
        freeze(self.model_old)

        self.inversion = GenerativeInversion(
            model=deepcopy(self.model_old),
            dataset=self.datamodule.dataset,
            batch_size=self.datamodule.batch_size,
            max_iters=self.hparams.num_inv_iters,
            lr=self.hparams.inv_lr,
            tau=self.hparams.inv_tau,
            alpha_pr=self.hparams.inv_alpha_pr,
            alpha_rf=self.hparams.inv_alpha_rf,
        )

        self.rkd = RKDAngleLoss(
            self.backbone.num_features,
            proj_dim=2 * self.backbone.num_features,
        )

        self.register_feature_hook("pen", "head.neck")

    def init_setup(self, stage=None):
        if self.datamodule.dataset.startswith("imagenet"):
            self.backbone = B.resnet.resnet18()
        else:
            self.backbone = B.resnet_cifar.resnet32()
        kwargs = dict(num_features=self.backbone.num_features, bias=False)
        self.head = DynamicSimpleHead(**kwargs)
        self.model_old, self.inversion, self.rkd = None, None, nn.Identity()

        for task_id in range(0, self.datamodule.current_task + 1):
            if task_id > 0 and task_id == self.datamodule.current_task:
                self.update_old_model()  # load from checkpoint
            self.head.append(self.datamodule[task_id].num_classes)

    def setup(self, stage=None):
        current_task = self.datamodule.current_task
        resume_from_checkpoint = self.trainer.resume_from_checkpoint

        if current_task == 0 or resume_from_checkpoint is not None:
            self.init_setup(stage)
        else:
            self.update_old_model()
            self.head.append(self.datamodule.num_classes)

        self.cls_count = torch.zeros(self.head.num_classes)
        self.cls_weight = torch.ones(self.head.num_classes)
        self.register_losses()

        self.print(f"=> Network Overview \n {self}")

    def forward(self, input):
        output = self.backbone(input)
        output = self.head(output)
        return output

    def check_finetuning(self, at_epoch_end: bool = True):
        if not self.should_start_finetuning(at_epoch_end):
            return super().check_finetuning(at_epoch_end)

        result = super().check_finetuning(at_epoch_end)
        if self.model_old is not None:
            _ = [self.unregister_loss(name) for name in ["rkd"]]
            self.register_loss(
                "ce",
                nn.functional.cross_entropy,
                ["prediction", "target", "ft_weight"],
                self.hparams.lambda_ce,
            )
        return result

    def on_train_start(self):
        super().on_train_start()
        if self.model_old is not None:
            ckpt_path = self.hparams.inv_resume_from
            if ckpt_path is None:
                self.inversion()
                log_dir = self.trainer.logger.log_dir
                ckpt_path = os.path.join(log_dir, "inversion.ckpt")
                print("\n==> Saving inversion states to", ckpt_path)
                torch.save(self.inversion.state_dict(), ckpt_path)
            else:
                print("\n==> Restoring inversion states from", ckpt_path)
                state = torch.load(ckpt_path, map_location=self.device)
                self.inversion.load_state_dict(state)
                self.hparams.inv_resume_from = None

    def training_step(self, batch, batch_idx):
        input, target = batch
        if self.finetuning and self.datamodule.current_task == 0:
            zeros = torch.zeros_like(input, requires_grad=True)
            return zeros.sum()

        target_t = self.datamodule.transform_target(target)
        kwargs = dict(
            input=input,
            target=target_t,
            prediction=self(input),
        )

        target_all = target_t
        if self.model_old is not None:
            _ = self.model_old.eval() if self.model_old.training else None
            _ = self.inversion.eval() if self.inversion.training else None

            input_rh, target_rh = self.inversion.sample(input.shape[0])
            target_all = torch.cat([target_t, target_rh])

            feature_new = self.current_feature("pen")  # save
            prediction_old = self(input_rh)
            feature_old = self.current_feature("pen")  # save

            n_old = self.model_old.head.num_classes
            kwargs["input_hkd"] = prediction_old[:, :n_old]
            kwargs["target_hkd"] = self.model_old(input_rh)
            if not self.finetuning:
                self.model_old.head.feature_mode = True
                kwargs["input_rkd"] = feature_new
                kwargs["target_rkd"] = self.model_old(input)
                self.model_old.head.feature_mode = False

                kwargs["prediction"] = kwargs["prediction"][:, n_old:]
                kwargs["target"] = kwargs["target"] - n_old
            else:
                pred = torch.cat([kwargs["prediction"], prediction_old])
                kwargs["prediction"], kwargs["target"] = pred, target_all
                kwargs["ft_weight"] = self.cls_weight.to(self.device)

        loss, loss_dict = self.compute_loss(**kwargs)
        self.log_dict({f"loss/{key}": val for key, val in loss_dict.items()})

        indices, counts = target_all.cpu().unique(return_counts=True)
        self.cls_count[indices] += counts

        return loss

    def training_epoch_end(self, *args, **kwargs):
        if self.model_old is not None:
            cls_weight = self.cls_count.sum() / self.cls_count.clamp(min=1)
            self.cls_weight = cls_weight.div(cls_weight.min())
        self.check_finetuning()
        return super().training_epoch_end(*args, **kwargs)

    def configure_optimizers(self):
        module = nn.ModuleList([self.backbone, self.head, self.rkd])
        optimizer = optim.SGD(
            module.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.milestones,
            gamma=self.hparams.lr_factor,
        )

        scheduler = self.add_finetuning_lr_scheduler(optimizer, scheduler)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
