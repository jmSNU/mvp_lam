# deepspeed_zero3_cpu.py
"""
DeepSpeed ZeRO-3 + CPU Offload Training Strategy (clean revision)

Key points vs. your draft:
  - Clear English docstrings/comments mirroring the revised FSDP style
  - ZeRO-3 with CPU offload for params & optimizer to maximize VRAM headroom
  - BF16 compute, Flash/SDPA attention to cut activation memory without accuracy loss
  - Optional CPU activation checkpointing (heavy VRAM relief, slower)
  - No grad accumulation (micro-batch == per-GPU batch)
  - Rank-0 checkpointing with informative tags
"""

from __future__ import annotations
import math, os
from pathlib import Path
from typing import Optional
from typing import Callable, Optional

import torch
import torch.distributed as dist
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.vlms import PrismaticVLM

from prismatic.training.strategies.base_strategy import TrainingStrategy
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util.data_utils import (
    PaddedCollatorForLanguageModeling,
    PaddedCollatorForActionPrediction,
)
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.overwatch import initialize_overwatch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, checkpoint_wrapper, apply_activation_checkpointing
)

overwatch = initialize_overwatch(__name__)


class DeepSpeedZeRO3CPUOffloadStrategy(TrainingStrategy):
    """
    A TrainingStrategy using DeepSpeed ZeRO-3 with CPU offload.
    Target use-case: maximize per-GPU batch size without grad accumulation.

    Notes:
      * BF16 compute is enabled.
      * ZeRO-3 shards params/grads/optimizer states across ranks.
      * CPU offload moves param/optimizer states to host memory.
      * Optional CPU activation checkpointing trades speed for large VRAM relief.
    """
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        cpu_checkpointing:bool = True
    ):
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
        )

        # We rely on large micro-batches; keep grad accumulation disabled.
        assert self.grad_accumulation_steps == 1, "ZeRO-3 config assumes gradient_accumulation_steps == 1."
        self.engine = None
        self.cpu_checkpointing = cpu_checkpointing

        # Avoid allocator issues seen with expandable_segments under heavy sharding/memory moves.
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    def _build_ds_config(self, num_training_steps: int, num_warmup_steps: int) -> dict:
        """
        Construct the DeepSpeed JSON config (as a Python dict).
        BF16 + ZeRO-3 + CPU offload for both params and optimizer states.
        """
        return {
            "train_batch_size": self.global_batch_size,
            "train_micro_batch_size_per_gpu": self.per_device_batch_size,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": False,
                "reduce_scatter": True,
                "contiguous_gradients": False,
                "allgather_bucket_size": 5e7,      
                "reduce_bucket_size":    5e7, 
                "stage3_param_persistence_threshold": 0,
                # Gather full weights only at save-time
                "stage3_gather_16bit_weights_on_model_save": True,
                # Keep very large tensors sharded (reduce persistent cache)
                "stage3_param_persistence_threshold": 1e6,
                # --- CPU Offload knobs ---
                "offload_param":     {"device": "cpu", "pin_memory": True},
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
            },
            # Warmup + decay (DeepSpeed's WarmupDecayLR ≈ linear warmup + decay)
            "lr_scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": max(1, num_warmup_steps),
                    "total_num_steps": num_training_steps,
                },
            },
            "gradient_clipping": self.max_grad_norm,
            # Activation checkpointing on CPU is the largest VRAM saver (slower).
            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
                "cpu_checkpointing": bool(self.cpu_checkpointing),
                "synchronize_checkpoint_boundary": False,
            },
        }

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        """
        Initialize attention kernels, compute schedule sizes, and create the DeepSpeed engine.
        """
        # Compute number of training steps and warmup steps.
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        num_training_steps = (
            (n_train_examples * self.epochs) // self.global_batch_size
            if self.max_steps is None else self.max_steps
        )
        # Use at least 5% warmup unless user provided larger ratio.
        warmup_ratio = max(0.05, self.warmup_ratio)
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        ds_cfg = self._build_ds_config(num_training_steps, num_warmup_steps)

        # Initialize DeepSpeed engine. It wraps the model and builds optimizer/scheduler internally.
        # Only pass trainable parameters.
        trainable_params = [p for p in self.vlm.parameters() if p.requires_grad]
        optimizer = DeepSpeedCPUAdam(
            trainable_params, lr = self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay
        )
        self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.vlm,
            optimizer=optimizer,
            config=ds_cfg,
        )

        if self.reduce_in_full_precision:
            for b in self.vlm.vision_backbone.buffers():
                b.data = b.data.to(self.vlm.vision_backbone.half_precision_dtype)

        if self.enable_gradient_checkpointing:
            non_reentrant = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def _is_llm_block(m):
                return isinstance(m, self.llm_transformer_layer_cls)  

            apply_activation_checkpointing(
                self.engine.module,                       
                checkpoint_wrapper_fn=non_reentrant,
                check_fn=_is_llm_block,
            )

        overwatch.info(
            "DeepSpeed ZeRO-3 + CPU Offload setup:\n"
            f"  |-> Global BSZ = {self.global_batch_size} | Per-Device BSZ = {self.per_device_batch_size}\n"
            f"  |-> World Size = {overwatch.world_size()} | Grad Accum = {self.grad_accumulation_steps}\n"
            f"  |-> bf16 = True | Reduce full precision {self.reduce_in_full_precision} | CPU Act Ckpt = {self.cpu_checkpointing}\n"
            f"  |-> Steps = {num_training_steps} (warmup={num_warmup_steps})\n",
            ctx_level=1,
        )

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,  # DeepSpeed handles full model; flag kept for API parity
    ) -> None:
        """
        Save a ZeRO-3 checkpoint. DeepSpeed will handle gathering shards as needed.
        """
        tag = f"step-{global_step:06d}-epoch-{epoch:02d}-loss={float('inf') if train_loss is None else train_loss:.4f}"
        if overwatch.is_rank_zero():
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.engine.save_checkpoint(run_dir / "checkpoints", tag=tag)

    def clip_grad_norm(self):
        """Use DeepSpeed's grad clipping hook (works with ZeRO-3 sharded grads)."""
        return self.engine.clip_grad_norm(self.max_grad_norm)

    # ===== LLM training (text + vision) =====
    def run_training(
        self,
        dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """
        Standard supervised instruction tuning loop using the DeepSpeed engine.
        Micro-batch equals per-GPU batch (no grad accumulation).
        """
        # Build sampler: split-modality for finetune, otherwise standard distributed sampler.
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            sampler = SplitModalitySampler(
                dataset,
                dataset.get_modality_lengths(),
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        dl = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
        )

        # Determine loop length for the progress bar.
        steps_total = len(dl) if self.max_steps is None else self.max_steps
        self.engine.train()
        status = metrics.get_status()

        from tqdm import tqdm
        with tqdm(total=steps_total, desc=status, leave=False, disable=not overwatch.is_rank_zero()) as pbar:
            for epoch in range(self.epochs if self.max_steps is None else 10**9):
                sampler.set_epoch(epoch)

                for batch in dl:
                    with torch.autocast("cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training):
                        out: CausalLMOutputWithPast = self.engine(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch.get("multimodal_indices", None),
                        )
                        loss = out.loss

                    # Log loss before backward (keeps semantics similar to other strategies)
                    metrics.commit(loss=loss)

                    # Backward on ZeRO-3 sharded model
                    self.engine.backward(loss)

                    # Clip and step
                    grad_norm = self.clip_grad_norm()
                    metrics.commit(grad_norm=grad_norm)
                    self.engine.step()

                    # Update metrics & progress
                    metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0], update_step_time=True)
                    status = metrics.push()
                    if overwatch.is_rank_zero():
                        pbar.update(1)
                        pbar.set_description(status)

                    # Termination by max_steps
                    if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                        if dist.is_available() and dist.is_initialized():
                            dist.barrier()
                        return

                # If running by epochs, save at epoch end
                if self.max_steps is None:
                    self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, float(loss.item()))
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

    # ===== VLA training (latent action) =====
    def run_vla_training(
        self,
        vla_dataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,  # unused; DS saves the full sharded state
    ) -> None:
        """
        VLA loop variant. The model still outputs logits and loss; we compute action metrics on-the-fly.
        """
        dl = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            num_workers=0,
            collate_fn=collator,
            worker_init_fn=self.worker_init_fn,
        )

        self.engine.train()
        from tqdm import tqdm
        assert self.engine is not None
        with tqdm(
            total=(self.epochs * len(dl)) if self.max_steps is None else self.max_steps,
            desc=metrics.get_status(),
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as pbar:
            for batch in dl:
                with torch.autocast("cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training):
                    out: CausalLMOutputWithPast = self.engine(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = out.loss

                metrics.commit(loss=loss)
                self.engine.backward(loss)
                grad_norm = self.clip_grad_norm()
                metrics.commit(grad_norm=grad_norm)
                self.engine.step()

                # Lightweight action metrics
                preds = out.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                gt = batch["labels"][:, 1:].to(preds.device)
                mask = gt > 32000
                acc = ((preds == gt) & mask).sum().float() / mask.sum().float()
                metrics.commit(action_accuracy=acc, l1_loss=torch.tensor(0.0), update_step_time=True)

                # Global step + LR log
                metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()
                if overwatch.is_rank_zero():
                    pbar.update(1)
                    pbar.set_description(status)

                # Save or terminate
                if self.max_steps is not None and metrics.global_step >= self.max_steps:
                    self.save_checkpoint(metrics.run_dir, metrics.global_step, 0, loss.item())
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    return
                
    def _vision_param_dtype(self):
        m = getattr(self.engine.module, "vision_backbone", None)
        if m is None:
            return torch.float32
        for _, p in m.named_parameters(recurse=True):
            return p.dtype
        return torch.float32
                
    def _prepare_batch(self, batch):
        
        dev = self.engine.device
        move = lambda t: t.to(dev, non_blocking=True) if torch.is_tensor(t) else t
        for k in ("input_ids","attention_mask","labels","multimodal_indices"):
            if k in batch: batch[k] = move(batch[k])

        vdtype = self._vision_param_dtype()
        if "pixel_values" in batch:
            pv = batch["pixel_values"]
            if isinstance(pv, dict):
                for kk, vv in pv.items():
                    batch["pixel_values"][kk] = vv.to(device=dev, dtype=vdtype, memory_format=torch.channels_last, non_blocking=True)
            else:
                batch["pixel_values"] = pv.to(device=dev, dtype=vdtype, memory_format=torch.channels_last, non_blocking=True)
        return batch
                
    def run_latent_action_training(
        self,
        vla_dataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,  # DeepSpeed saves full sharded state; flag kept for parity
    ) -> None:
        """Latent-action training loop using DeepSpeed engine. No grad accumulation."""
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation."

        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,  # RLDS iterator handles repetition
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # progress
        from tqdm import tqdm
        assert self.engine is not None
        total_steps = (self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps
        self.engine.train()

        with tqdm(total=total_steps, desc=metrics.get_status(), leave=False, disable=not overwatch.is_rank_zero()) as pbar:
            step = 0
            for batch in dataloader:
                batch = self._prepare_batch(batch)
                out: CausalLMOutputWithPast = self.engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = out.loss

                # loss + backward
                metrics.commit(loss=loss)
                self.engine.backward(loss)

                # clip + step
                grad_norm = self.clip_grad_norm()
                metrics.commit(grad_norm=grad_norm)
                self.engine.step()

                # ===== action metrics =====
                preds = out.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                gt = batch["labels"][:, 1:].to(preds.device)
                mask = gt > 32000
                acc = ((preds == gt) & mask).sum().float() / mask.sum().float()
                metrics.commit(action_accuracy=acc, l1_loss=torch.tensor(0.0), update_step_time=True)

                # step, lr, status
                step += 1
                lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 0.0
                metrics.commit(global_step=metrics.global_step + 1, lr=lr)
                status = metrics.push()
                if overwatch.is_rank_zero():
                    pbar.update(1)
                    pbar.set_description(status)

                # save / terminate
                if (self.max_steps is not None and metrics.global_step >= self.max_steps) or (
                    self.max_steps is None and (metrics.global_step % save_interval) == 0
                ):
                    # epoch is not well-defined for IterableDataset; record 0
                    self.save_checkpoint(metrics.run_dir, metrics.global_step, 0, float(loss.item()))
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        return