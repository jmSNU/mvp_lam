from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from latent_action_model.genie.modules.blocks import SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer

from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(DINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )


    def vq_encode(self, videos: Tensor, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        with torch.no_grad():
            dino_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dino_features = rearrange(dino_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dino_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, attention_mask) 
      
        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dino_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }
    
    def forward(self, batch: Dict) -> Dict:
        view_names = list(batch.keys())
        V = len(view_names)
        
        videos_per_view = [batch[name]["videos"] for name in view_names]  # each: (B, T, C, H, W)
        B, T = videos_per_view[0].shape[:2]
        videos_cat = torch.cat(videos_per_view, dim=0)                    # (V * B, T, C, H, W)

        encode_out = self.vq_encode(videos_cat)  # "patches": (V*B, T, Lv, Dd)
        video_patches_all = self.patch_up(encode_out["patches"][:, :-1])   # (V*B, T-1, Lv, Dm)
        action_patches_all = self.action_up(encode_out["z_q"])             # (V*B, T-1, num_codes, Dm)
        targets_all = encode_out["patches"][:, [-1]]                        # (V*B, 1, Lv, Dd)

        def as_view_batch(x: torch.Tensor) -> torch.Tensor:
            return x.view(V, B, *x.shape[1:])

        video_patches_view_batch = as_view_batch(video_patches_all)   # (V, B, T-1, Lv, Dm)
        action_patches_view_batch = as_view_batch(action_patches_all) # (V, B, T-1, n_code, Dm)
        targets_view_batch = as_view_batch(targets_all)               # (V, B, 1, Lv, Dd)

        # VQ auxiliary tensors (reshape for later attachment per action-view)
        n_code = self.num_codes
        latent_dim = self.latent_dim
        emb_view_batch = encode_out["emb"].view(V, B, T - 1, n_code, latent_dim)
        z_view_batch = encode_out["z"].view(V, B, T - 1, n_code, latent_dim)
        z_q_view_batch = encode_out["z_q"].view(V, B, T - 1, n_code, latent_dim)
        indices_view_batch = encode_out["indices"].view(V, B, -1)  

        obs_idx = torch.arange(V, device=self.device).repeat_interleave(V) # [view1, view1, view2, view2]
        act_idx = torch.arange(V, device=self.device).repeat(V)            # [view1, view2, view1, view2]

        obs_token = video_patches_view_batch[obs_idx] # (V*V, B, T-1, Lv, Dm)
        act_token = action_patches_view_batch[act_idx] # (V*V, B, T-1, n, Dm)

        video_action_tokens = torch.cat([act_token, obs_token], dim=3)  # (V*V, B, T-1, Lv+n_code, Dm)

        shape_vid = video_action_tokens.shape[2:]
        decoder_input = video_action_tokens.reshape(B*V**2, *shape_vid)

        recon_all: torch.Tensor = self.decoder(decoder_input)  # (V*V*B, T-1, L_out, Dd)

        num_video_tokens = video_patches_view_batch.shape[3]
        recon_all = recon_all[:, :, -num_video_tokens:]  # (V*V*B, T-1, Lv, Dd)

        recon_grid = recon_all.view(V, V, B, *recon_all.shape[1:])

        targets_grid = targets_view_batch[:, None].expand(V, V, B, 1, num_video_tokens, targets_view_batch.shape[-1])

        outputs: Dict[str, Dict[str, Dict]] = {obs_name: {} for obs_name in view_names}
        for obs_view_index, obs_view_name in enumerate(view_names):
            for act_view_index, act_view_name in enumerate(view_names):
                outputs[obs_view_name][act_view_name] = {
                    "recon":   recon_grid[obs_view_index, act_view_index],  # (B, T-1, Lv, Dd)
                    "target":  targets_grid[obs_view_index, act_view_index],# (B, 1,   Lv, Dd)
                    "emb":     emb_view_batch[act_view_index],              # (B, T-1, num_codes, latent_dim)
                    "z":       z_view_batch[act_view_index],                # (B, T-1, num_codes, latent_dim)
                    "indices": indices_view_batch[act_view_index],          # as produced by VQ
                    "z_q":     z_q_view_batch[act_view_index],              # (B, T-1, num_codes, latent_dim)
                }

        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
