import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import draccus
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.models.vision_transformer import LayerScale
from transformers import AutoTokenizer

from prismatic.conf import ModelConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# ===== LayerScale patch (avoid `gamma` key issues with HF) =====
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    if hasattr(ls_module, "gamma") and not hasattr(ls_module, "scale_factor"):
        ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
        ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
        del ls_module.gamma


def patch_all_layerscale(root: nn.Module):
    for m in root.modules():
        if isinstance(m, LayerScale):
            ls_apply_patch(m)


# ===== Projector mapping =====
PROJECTOR_KEY_MAPPING = {
    "projector.0.weight": "projector.fc1.weight",
    "projector.0.bias": "projector.fc1.bias",
    "projector.2.weight": "projector.fc2.weight",
    "projector.2.bias": "projector.fc2.bias",
    "projector.4.weight": "projector.fc3.weight",
    "projector.4.bias": "projector.fc3.bias",
}


def map_projector_key(k: str) -> str:
    # k may be "projector.0.weight" (original) or "0.weight" (subdict) or "fc1.weight"
    if k in PROJECTOR_KEY_MAPPING:
        return PROJECTOR_KEY_MAPPING[k]
    if k.startswith("projector.") and k in PROJECTOR_KEY_MAPPING:
        return PROJECTOR_KEY_MAPPING[k]

    kk = k
    if kk.startswith("projector."):
        kk = kk[len("projector.") :]

    # If already in HF fc format
    if kk.startswith("fc"):
        return f"projector.{kk}"

    # If original sequential indices but missing "projector." prefix
    k2 = f"projector.{kk}"
    if k2 in PROJECTOR_KEY_MAPPING:
        return PROJECTOR_KEY_MAPPING[k2]

    # Fallback: try to keep under projector.*
    return f"projector.{kk}"


def map_llm_key(k: str) -> str:
    # Expected target: "language_model.<...>".
    # Common sources:
    # - "llm.model...." -> replace llm. with language_model.
    # - "model...." -> prefix language_model.
    # - already "language_model...." -> keep
    if k.startswith("language_model."):
        return k
    if k.startswith("llm."):
        return k.replace("llm.", "language_model.", 1)
    # If the subdict is raw llama keys like "model.layers..."
    if k.startswith("model.") or k.startswith("lm_head.") or k.startswith("embed_tokens."):
        return "language_model." + k
    # Otherwise, just prefix (best-effort)
    return "language_model." + k


def remap_state_dicts_for_hf(
    projector_sd: Dict[str, torch.Tensor],
    llm_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    hf_sd: Dict[str, torch.Tensor] = {}

    for k, v in projector_sd.items():
        hf_sd[map_projector_key(k)] = v

    for k, v in llm_sd.items():
        hf_sd[map_llm_key(k)] = v

    return hf_sd


def strip_prefixes(d: Dict[str, torch.Tensor], prefixes: Tuple[str, ...] = ("module.", "model.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in d.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p) :]
        out[kk] = v
    return out


def load_norm_stats(dataset_statistics_path: Optional[Union[str, Path]]) -> Tuple[Dict, Optional[Path]]:
    if dataset_statistics_path is None:
        return {}, None
    p = Path(dataset_statistics_path)
    if not p.exists():
        return {}, None
    with open(p, "r") as f:
        return json.load(f), p


@dataclass
class HFConvertConfig:
    # Prismatic run directory OR HF hub subpath
    openvla_model_path_or_id: Union[str, Path] = "/path/to/model"

    # Checkpoint file name (commonly "latest-checkpoint.pt")
    ckpt_name: str = "latest-checkpoint.pt"

    # Output HF directory
    output_hf_model_local_path: Path = Path("/path/to/hf_model")

    # Base model spec (bypass config.json parsing if needed)
    base_vlm: str = "prism-dinosiglip-224px+7b"

    # Optional dataset_statistics.json (you may pass SimplER stats here)
    dataset_statistics_path: Optional[Union[str, Path]] = None

    # HF token (env var name or token file path); default reads HF_TOKEN env
    hf_token_env: str = "HF_TOKEN"
    hf_token_file: Optional[Union[str, Path]] = None

    # Local TIMM weight files (optional). If None, timm may download weights.
    dino_weight_file: Optional[Union[str, Path]] = "/path/to/dino"
    siglip_weight_file: Optional[Union[str, Path]] = "/path/to/siglip"

    # Latent action tokens
    codebook_size: int = 16


def resolve_hf_token(cfg: HFConvertConfig) -> Optional[str]:
    if cfg.hf_token_file is not None:
        p = Path(cfg.hf_token_file)
        return p.read_text().strip()
    tok = os.environ.get(cfg.hf_token_env, None)
    return tok


def resolve_local_paths(run_dir: Path, ckpt_name: str) -> Tuple[Path, Path, Optional[Path]]:
    config_json = run_dir / "config.json"

    # Common checkpoint locations
    candidates = [
        run_dir / "checkpoints" / ckpt_name,
        run_dir / ckpt_name,
        run_dir / "checkpoints" / "latest-checkpoint.pt",
    ]
    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found. Tried: {candidates}")

    # dataset_statistics may be missing for your case
    ds = run_dir / "dataset_statistics.json"
    return config_json, ckpt_path, (ds if ds.exists() else None)


@draccus.wrap()
def convert_openvla_weights_to_hf(cfg: HFConvertConfig) -> None:
    print(f"[*] Converting `{cfg.openvla_model_path_or_id}` -> HF `{cfg.output_hf_model_local_path}`")
    torch.set_default_dtype(torch.bfloat16)

    hf_token = resolve_hf_token(cfg)

    # ===== Locate inputs =====
    if os.path.isdir(str(cfg.openvla_model_path_or_id)):
        run_dir = Path(cfg.openvla_model_path_or_id)
        print(f"[*] Loading from local: {run_dir}")
        config_json, checkpoint_pt, local_ds = resolve_local_paths(run_dir, cfg.ckpt_name)

        # dataset_statistics: prefer user-provided, else local if exists
        ds_path = Path(cfg.dataset_statistics_path) if cfg.dataset_statistics_path is not None else local_ds
        norm_stats, norm_stats_path = load_norm_stats(ds_path)
    else:
        # HF Hub
        # NOTE: adjust repo_id/filenames to your actual hub layout
        print(f"[*] Downloading from HF Hub path: {cfg.openvla_model_path_or_id}")
        repo_id = "openvla/openvla-dev"
        sub = str(cfg.openvla_model_path_or_id).strip("/")
        config_json = hf_hub_download(repo_id, f"{sub}/config.json", token=hf_token)
        checkpoint_pt = hf_hub_download(repo_id, f"{sub}/checkpoints/{cfg.ckpt_name}", token=hf_token)

        # dataset_statistics: optional
        try:
            ds = hf_hub_download(repo_id, f"{sub}/dataset_statistics.json", token=hf_token)
            norm_stats, norm_stats_path = load_norm_stats(ds)
        except Exception:
            norm_stats, norm_stats_path = {}, None

    # ===== Derive prismatic_config =====
    # Your local config.json may not contain {"vla": ...}. Use cfg.base_vlm as source of truth.
    prismatic_config = ModelConfig.get_choice_class(cfg.base_vlm)().__dict__

    # ===== Build HF config =====
    hf_config = OpenVLAConfig(
        vision_backbone_id=prismatic_config["vision_backbone_id"],
        llm_backbone_id=prismatic_config["llm_backbone_id"],
        arch_specifier=prismatic_config["arch_specifier"],
        image_resize_strategy=prismatic_config["image_resize_strategy"],
        llm_max_length=prismatic_config["llm_max_length"],
        torch_dtype=torch.bfloat16,
        norm_stats=norm_stats,
    )

    # ===== Tokenizer =====
    print("[*] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        model_max_length=hf_config.llm_max_length,
        token=hf_token,
        padding_side="right",
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.add_special_tokens({"additional_special_tokens": [f"<ACT_{i}>" for i in range(cfg.codebook_size)]})
    tokenizer.init_kwargs.pop("add_prefix_space", None)

    # Ensure vocab is padded to multiple-of (robust)
    pad_mult = int(getattr(hf_config, "pad_to_multiple_of", 64))
    target_vocab = len(tokenizer)
    if target_vocab % pad_mult != 0:
        target_vocab = ((target_vocab // pad_mult) + 1) * pad_mult

    hf_config.text_config.vocab_size = max(int(hf_config.text_config.vocab_size), target_vocab)
    hf_config.text_config.pad_token_id = hf_config.pad_token_id
    hf_config.text_config.torch_dtype = torch.bfloat16

    # ===== TIMM vision backbones + image stats =====
    print("[*] Loading TIMM vision backbone(s) + building image processor")
    timm_models = []
    input_sizes, interpolations, means, stds = [], [], [], []

    for idx, timm_model_id in enumerate(hf_config.timm_model_ids):
        pretrained_cfg = None
        if "dino" in timm_model_id and cfg.dino_weight_file is not None:
            pretrained_cfg = {"file": str(cfg.dino_weight_file)}
        if ("siglip" in timm_model_id or "SigLIP" in timm_model_id) and cfg.siglip_weight_file is not None:
            pretrained_cfg = {"file": str(cfg.siglip_weight_file)}

        m = timm.create_model(
            timm_model_id,
            pretrained=True,
            num_classes=0,
            img_size=hf_config.image_sizes[idx],
            act_layer=hf_config.timm_override_act_layers[idx],
            pretrained_cfg=pretrained_cfg,
        )

        patch_all_layerscale(m)

        data_cfg = timm.data.resolve_model_data_config(m)
        input_sizes.append((3, hf_config.image_sizes[idx], hf_config.image_sizes[idx]))
        interpolations.append(data_cfg["interpolation"])
        means.append(data_cfg["mean"])
        stds.append(data_cfg["std"])

        timm_models.append((timm_model_id, m))

    hf_image_processor = PrismaticImageProcessor(
        use_fused_vision_backbone=hf_config.use_fused_vision_backbone,
        image_resize_strategy=hf_config.image_resize_strategy,
        input_sizes=input_sizes,
        interpolations=interpolations,
        means=means,
        stds=stds,
    )
    hf_processor = PrismaticProcessor(image_processor=hf_image_processor, tokenizer=tokenizer)

    # ===== Load checkpoint =====
    print("[*] Loading checkpoint")
    ckpt = torch.load(checkpoint_pt, map_location="cpu")

    model_blob = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if not isinstance(model_blob, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(model_blob)}")

    model_blob = strip_prefixes(model_blob)

    # Accept formats:
    # 1) {"projector": {...}, "llm_backbone": {...}, ["vision_backbone": {...}]}
    # 2) flattened keys, but your log suggests (1) without vision_backbone
    if "projector" in model_blob and "llm_backbone" in model_blob:
        projector_sd = model_blob["projector"]
        llm_sd = model_blob["llm_backbone"]
    else:
        # flattened fallback
        projector_sd = {k[len("projector.") :]: v for k, v in model_blob.items() if k.startswith("projector.")}
        llm_sd = {k[len("llm_backbone.") :]: v for k, v in model_blob.items() if k.startswith("llm_backbone.")}
        if len(llm_sd) == 0:
            llm_sd = {k: v for k, v in model_blob.items() if k.startswith("llm.") or k.startswith("model.")}

        if len(projector_sd) == 0 or len(llm_sd) == 0:
            print("Top-level keys sample:", list(model_blob.keys())[:50])
            raise KeyError("Cannot find projector/llm keys in checkpoint.")

    # projector_sd may be a subdict with keys already including "projector."
    # normalize to the expected key patterns for mapping
    proj_norm = {}
    for k, v in projector_sd.items():
        kk = k if k.startswith("projector.") else f"projector.{k}"
        proj_norm[kk] = v
    projector_sd = proj_norm

    # ===== Convert projector + LLM keys to HF =====
    print("[*] Remapping projector + LLM keys")
    converted_state_dict = remap_state_dicts_for_hf(projector_sd, llm_sd)

    # ===== Build HF model and load weights =====
    print("[*] Building HF model")
    hf_model = OpenVLAForActionPrediction(hf_config)
    patch_all_layerscale(hf_model)

    print("[*] Loading projector + LLM weights (strict=False)")
    missing, unexpected = hf_model.load_state_dict(converted_state_dict, strict=False, assign=True)
    print(f"    missing={len(missing)} unexpected={len(unexpected)}")

    # ===== Load vision weights from TIMM pretrained (since checkpoint lacks vision_backbone) =====
    print("[*] Loading vision weights from TIMM pretrained")
    if getattr(hf_config, "use_fused_vision_backbone", False):
        dino = None
        sig = None
        for mid, m in timm_models:
            if dino is None and "dino" in mid:
                dino = m
            if sig is None and ("siglip" in mid or "SigLIP" in mid):
                sig = m
        if dino is None or sig is None:
            raise RuntimeError(f"Fused vision expected DINO+SigLIP, got ids={hf_config.timm_model_ids}")

        patch_all_layerscale(hf_model.vision_backbone)
        hf_model.vision_backbone.featurizer.load_state_dict(dino.state_dict(), strict=False)
        hf_model.vision_backbone.fused_featurizer.load_state_dict(sig.state_dict(), strict=False)
    else:
        # single backbone
        m0 = timm_models[0][1]
        patch_all_layerscale(hf_model.vision_backbone)
        hf_model.vision_backbone.featurizer.load_state_dict(m0.state_dict(), strict=False)

    # Cast to BF16 before save
    hf_model.to(torch.bfloat16)

    # ===== Save =====
    out_dir = Path(cfg.output_hf_model_local_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Saving model + processor")
    hf_model.save_pretrained(out_dir, max_shard_size="7GB")
    hf_image_processor.save_pretrained(out_dir)
    hf_processor.save_pretrained(out_dir)

    # Save dataset_statistics if provided
    if norm_stats_path is not None:
        shutil.copyfile(norm_stats_path, out_dir / "dataset_statistics.json")
    elif cfg.dataset_statistics_path is not None and Path(cfg.dataset_statistics_path).exists():
        shutil.copyfile(Path(cfg.dataset_statistics_path), out_dir / "dataset_statistics.json")
    else:
        # Still write something to make downstream code happy
        with open(out_dir / "dataset_statistics.json", "w") as f:
            json.dump(norm_stats, f)

    print(f"[*] Done. Saved to: {out_dir}")


if __name__ == "__main__":
    convert_openvla_weights_to_hf()
