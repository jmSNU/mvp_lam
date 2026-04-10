from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from bitsandbytes.optim import Adam8bit
from accelerate import PartialState
from collections import defaultdict
from latent_action_model.genie.metric_util import *
from pathlib import Path

OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules import DINOLatentActionModel
from latent_action_model.genie.modules.action_vq import ActionVQVAE

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class DINO_LAM(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            optimizer: OptimizerCallable = Adam8bit,
            make_data_pair: bool = False,
    ) -> None:
        super(DINO_LAM, self).__init__()

        lam = DINOLatentActionModel

        self.lam = lam(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                )
        
        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair
        
        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def refresh_buffer(self):
        self.idx_buffer = defaultdict(list)
        self.latent_buffer = defaultdict(list)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys ['videos', 'task_instruction', 'action', 'dataset_names', 'state', 'coeff']
        _ = batch.pop("state_by_dataset")
        dataset_names = batch.pop("dataset_names")  # list/array of dataset-name identifiers

        ds_str_list = [
            ds.decode() if isinstance(ds, bytes) else str(ds)
            for ds in dataset_names
        ]
        def _group_name(name: str) -> str:
            if name.startswith("egoexo4d_split_"):
                return "egoexo4d"
            return name

        def reduce_per_sample(x: torch.Tensor) -> torch.Tensor:
            """Reduce any [B, ...] tensor to [B] by averaging all non-batch dimensions."""
            return x.view(x.shape[0], -1).mean(dim=1)

        def _loss(gt, pred, coeff):
            if coeff.ndim == 1:
                coeff = coeff.view(-1, *([1] * (gt.ndim - 1)))

            mse = ((gt - pred['recon'])**2) * coeff
            q   = ((pred['emb'].detach() - pred['z'])**2) * coeff
            commit = ((pred['emb'] - pred['z'].detach())**2) * coeff

            mse_per_sample     = reduce_per_sample(mse)
            q_per_sample       = reduce_per_sample(q)
            commit_per_sample  = reduce_per_sample(commit)

            # Total loss per sample and batch mean
            loss_per_sample = mse_per_sample + q_per_sample + self.vq_beta * commit_per_sample
            loss = loss_per_sample.mean()

            # Code usage statistics
            unique, counts = torch.unique(pred["indices"], return_counts=True)
            index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long, device=gt.device)
            index_counts[unique] = counts
            code_usage = (index_counts != 0).float().mean()

            loss_logs = {
                "mse_loss": mse_per_sample.mean(),
                "q_loss": q_per_sample.mean(),
                "commit_loss": commit_per_sample.mean(),
                "code_usage": code_usage,
            }

            per_sample_logs = {
                "mse_loss": mse_per_sample,        # [B]
                "q_loss": q_per_sample,            # [B]
                "commit_loss": commit_per_sample,  # [B]
            }

            indices = pred["indices"].detach()
            return loss, loss_per_sample, loss_logs, per_sample_logs, indices

        coeff = batch.pop('coeff').to(self.device, dtype=self.dtype)

        with torch.autocast("cuda"):
            outputs = self.lam(batch)

        loss = 0.0
        loss_logs = []
        per_sample_loss = None
        loss_logs_per_sample_per_view_all: Dict[str, torch.Tensor] = {}
        indices_per_view: Dict[str, torch.Tensor] = {}

        for obs_view in outputs.keys():
            for act_view in outputs[obs_view].keys():
                gt_future_frames = outputs[obs_view][act_view]["target"]
                pred = outputs[obs_view][act_view]
                (
                    loss_per_view,
                    loss_per_sample_per_view,
                    loss_logs_per_view,
                    loss_logs_per_sample_per_view,
                    indices_view,
                ) = _loss(gt_future_frames, pred, coeff)

                if obs_view != act_view:
                    loss_per_view = loss_logs_per_view["mse_loss"]
                    loss_per_sample_per_view = loss_logs_per_sample_per_view["mse_loss"]

                loss = loss + loss_per_view * 0.5
                if per_sample_loss is None:
                    per_sample_loss = 0.5 * loss_per_sample_per_view
                else:
                    per_sample_loss = per_sample_loss + loss_per_sample_per_view * 0.5

                view_prefix = f"{obs_view}-{act_view}"
                for k, v in loss_logs_per_view.items():
                    loss_logs.append((f"{view_prefix}/{k}", v))

                for k, v in loss_logs_per_sample_per_view.items():
                    key = f"{view_prefix}/{k}"
                    loss_logs_per_sample_per_view_all[key] = v

                indices_per_view[view_prefix] = indices_view

        group_names = set(_group_name(n) for n in ds_str_list)
        group_to_mask: Dict[str, torch.Tensor] = {}
        for g in group_names:
            mask = torch.tensor(
                [_group_name(n) == g for n in ds_str_list],
                device=self.device,
                dtype=torch.bool,
            )
            group_to_mask[g] = mask

        for group_name_str, mask in group_to_mask.items():
            ds_loss = per_sample_loss[mask].mean()
            loss_logs.append((f"dataset/{group_name_str}/loss", ds_loss))

            for k, v in loss_logs_per_sample_per_view_all.items():
                ds_loss_logs = v[mask].mean()
                loss_logs.append((f"dataset/{group_name_str}/{k}", ds_loss_logs))

            for k, indices in indices_per_view.items():
                idx_ds = indices[mask]
                unique_idx_ds = torch.unique(idx_ds)
                code_usage_ds = unique_idx_ds.numel() / float(self.lam_num_latents)
                loss_logs.append(
                    (f"dataset/{group_name_str}/{k}/code_usage", code_usage_ds)
                )
        return outputs, loss, loss_logs


    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"valid_loss": loss}, **{f"valid/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"valid_loss": loss}, **{f"valid/{k}": v for k, v in aux_losses}})

        return loss


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False
        )

        return loss

    def on_train_epoch_end(self):
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        params = [
            p for n, p in self.named_parameters()
        ]
        optim = self.optimizer(params)
        return optim

    # def configure_optimizers(self):
    #     return DeepSpeedCPUAdam(self.parameters(), lr=1e-4, weight_decay=1e-2)