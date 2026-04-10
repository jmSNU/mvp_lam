from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from latent_action_model.genie.modules.blocks import VectorQuantizer
from typing import *

class MLP(nn.Module):
    """Simple MLP encoder/decoder used in action VQ-VAE."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers = []
        dim = in_dim

        # hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim

        # output layer
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ActionVQVAE:
    """
    Vanilla VQ-VAE for actions.

    Input:
        batch[view_name]["actions"]: (B, T, A)

    Output:
        outputs[view_name] = {
            "recon":   (B, T, A),        # reconstructed actions
            "target":  (B, T, A),        # original actions
            "emb":     (B, num_codes, latent_dim),
            "z":       (B, num_codes, latent_dim),
            "z_q":     (B, num_codes, latent_dim),
            "indices": (B, num_codes),   # VQ code indices
        }
    """

    def __init__(
        self,
        action_dim: int,            # A
        horizon: int,               # T
        latent_dim: int = 128,      # E
        num_latents: int = 16,      # codebook size
        num_codes: int = 4,         # number of latent slots per sample
        hidden_dim: int = 128,
        num_layers: int = 2,
        device = "cuda"
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.num_codes = num_codes

        input_dim = action_dim * horizon          # T * A
        latent_flat_dim = latent_dim * num_codes  # num_codes * E

        # Encoder: (B, T, A) -> (B, num_codes * latent_dim)
        self.encoder = MLP(
            in_dim=input_dim,
            out_dim=latent_flat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        # VQ layer: same interface as your VectorQuantizer
        self.vq_layer = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        ).to(device)

        # Decoder: (B, num_codes * latent_dim) -> (B, T * A)
        self.decoder = MLP(
            in_dim=latent_flat_dim,
            out_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        self.beta = 0.25  
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.vq_layer.parameters())
        )
        self.vqvae_optimizer = torch.optim.Adam(
            params, lr=1e-4, weight_decay=0.0001
        )
        
        self.device = device

    def vq_encode(self, actions: Tensor) -> Dict[str, Tensor]:
        """
        Encode actions to VQ latents.

        actions: (B, T, A)
        returns:
            {
                "z_cont": (B, num_codes, latent_dim),   # pre-quantization continuous latent
                "z_q":    (B, num_codes, latent_dim),   # quantized latent (straight-through)
                "z":      (B, num_codes, latent_dim),   # codebook embeddings
                "emb":    (B, num_codes, latent_dim),   # original encoder output before VQ (from VectorQuantizer)
                "indices":(B, num_codes),               # VQ code indices
            }
        """
        B, T, A = actions.shape
        assert T == self.horizon, f"expected horizon={self.horizon}, got T={T}"
        assert A == self.action_dim, f"expected action_dim={self.action_dim}, got A={A}"

        # Flatten time and action dimensions
        x = rearrange(actions, "b t a -> b (t a)")               # (B, T*A)

        # Continuous latent
        z_cont_flat = self.encoder(x)                            # (B, num_codes * latent_dim)
        z_cont = z_cont_flat.view(B, self.num_codes, self.latent_dim)  # (B, num_codes, latent_dim)

        # Vector quantization
        # Your VectorQuantizer returns: z_q, z, x_in, indices
        z_q, z, emb, indices = self.vq_layer(z_cont)                   # all: (B, num_codes, latent_dim), indices: (B, num_codes)

        return {
            "z_cont": z_cont,
            "z_q":    z_q,
            "z":      z,
            "emb":    emb,
            "indices": indices,
        }

    def vqvae_update(self, act: torch.Tensor):
        """
        Single VQ-VAE update step.

        act: (B, T, A)

        Returns:
            encoder_loss:      L1 reconstruction loss (detached)
            vq_loss_state:     VQ loss (codebook + commitment, detached)
            vq_code:           code indices (B, num_codes)
            vqvae_recon_loss:  MSE reconstruction loss as float
        """
        # move to device / dtype
        act = act.to(self.device)              # (B, T, A)
        B, T, A = act.shape
        assert T == self.horizon, f"expected horizon={self.horizon}, got {T}"
        assert A == self.action_dim, f"expected action_dim={self.action_dim}, got {A}"

        # ---- encode & VQ ----
        vq_out = self.vq_encode(act)
        z_cont = vq_out["z_cont"]       # (B, num_codes, latent_dim) continuous encoder output
        z_q    = vq_out["z_q"]          # (B, num_codes, latent_dim) quantized (straight-through)
        z      = vq_out["z"]            # (B, num_codes, latent_dim) codebook embeddings
        vq_code = vq_out["indices"]     # (B, num_codes)

        # ---- decode ----
        z_q_flat = z_q.view(B, -1)      # (B, num_codes*latent_dim)
        dec_out_flat = self.decoder(z_q_flat)              # (B, T*A)
        dec_out = dec_out_flat.view(B, self.horizon, self.action_dim)  # (B, T, A)


        encoder_loss = (act - dec_out).abs().mean()
        vqvae_recon_loss = F.mse_loss(act, dec_out)

        e_latent_loss = F.mse_loss(z.detach(), z_cont)
        q_latent_loss = F.mse_loss(z, z_cont.detach())

        vq_loss_state = e_latent_loss + self.beta * q_latent_loss

        encoder_mult = 1.0
        rep_loss = encoder_loss * encoder_mult + (vq_loss_state * 5.0)

        # optimize
        self.vqvae_optimizer.zero_grad()
        rep_loss.backward()
        self.vqvae_optimizer.step()

        return (
            encoder_loss.detach(),
            vq_loss_state.detach(),
            vq_code.detach(),
            vqvae_recon_loss.item(),
        )
    
    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.vqvae_optimizer.state_dict(),
            "vq_embedding": self.vq_layer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.vqvae_optimizer.load_state_dict(state_dict["optimizer"])
        self.vq_layer.load_state_dict(state_dict["vq_embedding"])
        self.vq_layer.eval()

    def _to_action_tensor(self, act: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Normalize action shape to (B, T, A)."""

        if act.ndim == 2:
            # (T, A)
            assert act.shape == (self.horizon, self.action_dim), \
                f"Expected action shape ({self.horizon}, {self.action_dim}), got {act.shape}"
            act = act[None, ...]   # (1, T, A)
            is_single = True
        elif act.ndim == 3:
            # (B, T, A)
            assert act.shape[1] == self.horizon and act.shape[2] == self.action_dim, \
                f"Expected action shape (B, {self.horizon}, {self.action_dim}), got {act.shape}"
            is_single = False
        else:
            raise ValueError(f"Expected action ndim 2 or 3, got {act.ndim}")

        act = act.to(self.device)
        return act, is_single
    
    @torch.inference_mode()
    def encode(self, act: torch.Tensor):
        """
        Encode actions to latent and codes.

        act: (T, A) or (B, T, A)

        Returns:
            latents: (T*A_latent,) or (B, num_codes * latent_dim)
            codes:   (num_codes,) or (B, num_codes)
        """
        act_tensor, is_single = self._to_action_tensor(act)   # (B, T, A)

        with torch.autocast("cuda"):
            vq_out = self.vq_encode(act_tensor)
        z_q = vq_out["z_q"]          # (B, num_codes, latent_dim)
        codes = vq_out["indices"]    # (B, num_codes)

        # flatten quantized latent per sample
        latents = z_q.view(z_q.shape[0], -1)   # (B, num_codes*latent_dim)

        if is_single:
            latents = latents[0]   # (num_codes*latent_dim,)
            codes = codes[0]       # (num_codes,)

        return latents, codes

    
    @torch.inference_mode()
    def decode(self, codes: torch.Tensor):
        if codes.ndim == 1:
            # (num_codes,)
            assert codes.shape[0] == self.num_codes, \
                f"Expected code length {self.num_codes}, got {codes.shape[0]}"
            codes = codes[None, :]   # (1, num_codes)
            is_single = True
        elif codes.ndim == 2:
            # (B, num_codes)
            assert codes.shape[1] == self.num_codes, \
                f"Expected code shape (B, {self.num_codes}), got {codes.shape}"
            is_single = False
        else:
            raise ValueError(f"Expected codes ndim 1 or 2, got {codes.ndim}")

        codes = codes.to(self.device).long()      # (B, num_codes)

        with torch.autocast("cuda"):
            z = self.vq_layer.codebook(codes)         # (B, num_codes, latent_dim)

            # flatten and decode
            B = z.shape[0]
            z_flat = z.view(B, -1)                    # (B, num_codes*latent_dim)
            out_flat = self.decoder(z_flat)           # (B, T*A)
            actions = out_flat.view(B, self.horizon, self.action_dim)  # (B, T, A)

        if is_single:
            actions = actions[0]   # (T, A)
        return actions
    
    def eval(self):
        for module in [self.encoder, self.decoder, self.vq_layer]:
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
        self.vqvae_optimizer = None