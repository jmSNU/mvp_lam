from pathlib import Path
import torch
import os, re, json
from typing import *
import numpy as np

class ActionVQVae:
    def __init__(self, vq_vae_path, device: str = "cpu"):
        self.device = device

        from vqvae.vqvae import VqVae
        self.vq_path = Path(vq_vae_path)
        assert self.vq_path.exists(), f"Missing VQ VAE path: {self.vq_path}"
        vq_model_path = self.vq_path / "checkpoints" / "model.pt"
        vq_config_path = self.vq_path / "config.json"
        assert vq_model_path.exists(), f"Missing VQ checkpoint path: {vq_model_path}"
        assert vq_config_path.exists(), f"Missing VQ config path: {vq_config_path}"
        with open(vq_config_path, "r") as f:
            vq_config = dict(json.load(f))
        # set the load checkpoint
        vq_config["load_dir"] = vq_model_path
        vq_config["eval"] = True
        vq_config["device"] = self.device
        # instantiate the vqvae and load
        self.vq_vae = VqVae(**vq_config)
        self.horizon: int = int(self.vq_vae.input_dim_h)    # H (e.g. 8)
        self.action_dim: int = int(self.vq_vae.input_dim_w) # A (e.g. 7)
        self.groups: int = int(self.vq_vae.vqvae_groups)

    def _to_action_tensor(self, action: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(action)}")

        if action.ndim == 2:
            # (H, A)
            assert action.shape == (self.horizon, self.action_dim), \
                f"Expected action shape ({self.horizon}, {self.action_dim}), got {action.shape}"
            action = action[None, ...]  # (1, H, A)
            is_single = True
        elif action.ndim == 3:
            # (B, H, A)
            assert action.shape[1] == self.horizon and action.shape[2] == self.action_dim, \
                f"Expected action shape (B, {self.horizon}, {self.action_dim}), got {action.shape}"
            is_single = False
        else:
            raise ValueError(f"Expected action ndim 2 or 3, got {action.ndim}")

        action_tensor = action.to(self.device)
        return action_tensor, is_single

    @torch.no_grad()
    def encode(self, action: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        action_tensor, is_single = self._to_action_tensor(action)

        latent, vq_code = self.vq_vae.get_code(action_tensor)  # latent: (B, D), vq_code: (B, G)
        latents = latent.detach().cpu().numpy()
        codes = vq_code.detach().cpu().numpy()

        if is_single:
            latents = latents[0]  # (D,)
            codes = codes[0]      # (G,)
        return latents, codes

    @torch.no_grad()
    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim == 1:
            # (G,)
            assert codes.shape[0] == self.groups, \
                f"Expected code length {self.groups}, got {codes.shape[0]}"
            codes = codes[None, :]  # (1, G)
            is_single = True
        elif codes.ndim == 2:
            # (B, G)
            assert codes.shape[1] == self.groups, \
                f"Expected code shape (B, {self.groups}), got {codes.shape}"
            is_single = False
        else:
            raise ValueError(f"Expected codes ndim 1 or 2, got {codes.ndim}")

        codes_tensor = torch.from_numpy(codes).to(self.device)

        latent = self.vq_vae.draw_code_forward(codes_tensor)
        actions = self.vq_vae.get_action_from_latent(latent)
        actions = actions.detach().cpu().numpy()

        if is_single:
            actions = actions[0]  # (H, A)
        return actions