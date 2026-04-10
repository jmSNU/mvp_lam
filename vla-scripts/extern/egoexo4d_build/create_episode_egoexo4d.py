import os
import re
import json
import argparse
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm


# =======================
# Utils
# =======================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create fake episodes from EgoExo4D denseclips with files like 00001_ego.npy / 00001_exo.npy (no cam subfolders)."
    )
    parser.add_argument(
        "--source_dir", type=str, required=True,
        help="Root dir: {take_id}/{action_name}/{00001_ego.npy, 00001_exo.npy, ...}"
    )
    parser.add_argument(
        "--target_dir", type=str, required=True,
        help="Output dir to save episodes (.npy)"
    )
    parser.add_argument(
        "--annotation_file", type=str, required=True,
        help="Path to annotations.json or info_clips.json"
    )
    parser.add_argument(
        "--processes", type=int, default=16,
        help="Number of worker processes (default: 16)"
    )
    parser.add_argument(
        "--target_size", type=int, nargs=2, default=[224, 224],
        help='Resize to "height width" (default: 224 224)'
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After saving, load each episode to verify"
    )
    return parser.parse_args()

def build_episode_id_map(source_dir):
    pairs = []
    for video_name in tqdm(sorted(os.listdir(source_dir)), desc="Building id_map"):
        vdir = os.path.join(source_dir, video_name)
        if not os.path.isdir(vdir):
            continue
        for action_name in sorted(os.listdir(vdir)):
            adir = os.path.join(vdir, action_name)
            if os.path.isdir(adir):
                pairs.append((video_name, action_name))

    id_map = {}
    for i, (v, a) in enumerate(pairs, start=1):
        id_map[(v, a)] = i
    return id_map

def center_crop_and_resize(image, target_hw=(224, 224)):
    """image: HxWxC (RGB). Return resized RGB uint8."""
    h, w, _ = image.shape
    if h <= 0 or w <= 0:
        return np.zeros((target_hw[0], target_hw[1], 3), dtype=np.uint8)
    if h < w:
        crop = h
        x0, y0 = (w - crop) // 2, 0
    else:
        crop = w
        x0, y0 = 0, (h - crop) // 2
    cropped = image[y0:y0+crop, x0:x0+crop, :]
    pil = Image.fromarray(cropped)
    resized = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def normalize_annotation(anno):
    """
    Accept either:
      - dict: { take_id: [ {action_name, narration_text|language, ...}, ... ] }
      - list: [ {take_id, action_name, narration_text|language, ...}, ... ]
    Return: dict { take_id: [ {action_name, narration_text}, ... ] } (deduped by (take_id, action_name))
    """
    if isinstance(anno, dict):
        # ensure only needed fields
        norm = {}
        for tid, actions in anno.items():
            seen = set()
            norm[tid] = []
            for a in actions:
                an = a.get("action_name")
                if not an or (tid, an) in seen:
                    continue
                seen.add((tid, an))
                narr = a.get("narration_text", "") or a.get("language", "")
                norm[tid].append({"action_name": an, "narration_text": narr})
        return norm

    # list
    norm = {}
    seen = set()
    for row in anno:
        tid = row.get("take_id")
        an  = row.get("action_name")
        if not tid or not an or (tid, an) in seen:
            continue
        seen.add((tid, an))
        narr = row.get("narration_text", "") or row.get("language", "")
        norm.setdefault(tid, []).append({"action_name": an, "narration_text": narr})
    return norm


FRAME_RE = re.compile(r"^(?P<idx>\d{5})_(?P<role>ego|exo)\.npy$", re.IGNORECASE)


def list_frame_pairs(action_dir):
    """
    action_dir contains files like:
      00001_ego.npy, 00001_exo.npy, 00002_ego.npy, 00002_exo.npy, ...
    Return:
      sorted_indices: sorted list of available integer indices
      paths: dict idx -> {'ego': path or None, 'exo': path or None}
    """
    files = os.listdir(action_dir)
    paths = {}
    for f in files:
        m = FRAME_RE.match(f)
        if not m:
            continue
        idx = int(m.group("idx"))
        role = m.group("role").lower()
        paths.setdefault(idx, {"ego": None, "exo": None})
        paths[idx][role] = os.path.join(action_dir, f)

    sorted_indices = sorted(paths.keys())
    return sorted_indices, paths


def load_and_preprocess(path, target_hw):
    """
    Load npy (BGR saved by OpenCV), convert to RGB, center-crop & resize.
    If path is None -> return zeros.
    """
    if path is None:
        return None
    try:
        arr = np.load(path)
        # If grayscale or shape unexpected, guard:
        if arr.ndim == 2:  # HxW
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 3:
            arr = arr[:, :, ::-1]  # BGR->RGB
        else:
            # Fallback to 3-channel
            arr = np.repeat(arr[..., :1], 3, axis=-1)
        return center_crop_and_resize(arr, target_hw)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


# =======================
# Core
# =======================

def create_fake_episode(clip_dir, save_dir, annotation, target_hw, verify, id_map):
    """
    clip_dir: {source_dir}/{take_id}/{action_name}
    inside:   00001_ego.npy, 00001_exo.npy, ...
    Save:     {save_dir}/{take_id}__{action_name}.npy
    """
    take_id     = os.path.basename(os.path.dirname(clip_dir))
    action_name = os.path.basename(clip_dir)

    # find caption from annotation (safe if missing)
    actions = annotation.get(take_id, [])
    caption = ""
    for a in actions:
        if a.get("action_name") == action_name:
            caption = a.get("narration_text", "") or ""
            break

    # collect frame pairs
    indices, paths = list_frame_pairs(clip_dir)
    if not indices:
        print(f"[WARN] No frame files in {clip_dir}")
        return

    episode = []
    for idx in indices:
        p_ego = paths[idx]["ego"]
        # print(p_ego)
        p_exo = paths[idx]["exo"]
        # main/exo and wrist/ego (or zeros if missing)
        img_main  = load_and_preprocess(p_exo, target_hw)  # treat exo as main
        img_wrist = load_and_preprocess(p_ego, target_hw)  # treat ego as wrist

        episode.append({
            "image":          img_main.astype(np.uint8),
            "wrist_image":    img_wrist.astype(np.uint8),
            "state":          np.zeros(7, dtype=np.float32),
            "action":         np.zeros(7, dtype=np.float32),
            "language_instruction": caption,
            "frame_index":    int(idx),
        })

    if not episode:
        print(f"[WARN] Empty episode after parsing {clip_dir}")
        return

    os.makedirs(save_dir, exist_ok=True)
    # action_name = action_name.replace(".","")
    episode_id = id_map[(take_id, action_name)]
    save_name = f"episode_{episode_id}.npy"
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, episode)

    if verify:
        try:
            _ = np.load(save_path, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Verify failed for {save_path}: {e}")


def process_take_dir(take_dir, target_dir, annotation, target_hw, verify=False, id_map = None):
    """
    take_dir: {source_dir}/{take_id}
    contains action folders (e.g., action_000_...).
    """
    for action_name in sorted(os.listdir(take_dir)):
        action_dir = os.path.join(take_dir, action_name)
        if not os.path.isdir(action_dir):
            continue
        create_fake_episode(
            clip_dir=action_dir,
            save_dir=target_dir,
            annotation=annotation,
            target_hw=target_hw,
            verify=verify,
            id_map=id_map
        )


def main():
    args = parse_arguments()
    os.makedirs(args.target_dir, exist_ok=True)

    # load annotations and normalize structure
    with open(args.annotation_file, "r") as f:
        raw_anno = json.load(f)
    annotation = normalize_annotation(raw_anno)

    id_map = build_episode_id_map(args.source_dir)

    # list take directories
    take_dirs = [
        os.path.join(args.source_dir, d)
        for d in sorted(os.listdir(args.source_dir))
        if os.path.isdir(os.path.join(args.source_dir, d))
    ]

    print(f"Processing {len(take_dirs)} takes using {args.processes} workers...")
    worker = partial(
        process_take_dir,
        target_dir=args.target_dir,
        annotation=annotation,
        target_hw=tuple(args.target_size),
        verify=args.verify,
        id_map = id_map,
    )

    # multiprocessing over take directories
    with Pool(processes=args.processes) as pool:
        list(tqdm(pool.imap_unordered(worker, take_dirs),
                  total=len(take_dirs),
                  desc="Processing takes"))

    print("EgoExo4D episode creation completed!")


if __name__ == "__main__":
    main()
