import os
import re
import json
import cv2
import argparse
from collections import defaultdict
from math import isfinite
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============== Utilities ===============

def sanitize(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^a-zA-Z0-9_\-\.\s]", "", text)
    return (text.strip().replace(" ", "_") or "untitled")[:-1]

def norm_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}

def sec_to_frame(sec, fps):
    return int(round(float(sec) * float(fps)))

def read_frame_by_num(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"read fail @ frame {frame_num}")
    return frame

def frames_last_index(cap):
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return max(0, n_total - 1)

# =============== takes.json index ===============
def load_takes_index(takes_json_path):
    with open(takes_json_path, "rb") as f:
        takes = json.load(f)

    index = {}
    for t in takes:
        take_uid  = t.get("take_uid")
        root_dir = t.get("root_dir")
        fav = t.get("frame_aligned_videos", {}) or {}
        if not take_uid:
            print(f"No take_uid")
            continue

        buckets = {"ego": [], "exo": []}
        for cam_id, streams in fav.items():
            if cam_id in ("collage", "best_exo"):
                continue
            if not isinstance(streams, dict) or len(streams) == 0:
                continue

            cam_id_l = str(cam_id).lower()
            if cam_id_l.startswith("aria"):
                for key in ["rgb", "slam-left", "slam-right", "et"]:
                    if key in streams and isinstance(streams[key], dict) and streams[key].get("relative_path"):
                        chosen = streams[key]
                        break
                else:
                    print("No ego")
                    continue
            else:
                if "0" in streams and streams["0"].get("relative_path"):
                    chosen = streams["0"]
                else:
                    chosen = next(iter(streams.values()))
                    if not chosen.get("relative_path"):
                        print("No exo")
                        continue

            rel_name = chosen["relative_path"].split("/")[-1]
            role = "ego" if cam_id_l.startswith("aria") else "exo"
            buckets[role].append({
                "cam_id": str(cam_id),
                "file_path": os.path.join(
                    root_dir, "frame_aligned_videos", "downscaled", "448", rel_name
                )
            })
        if buckets["ego"] or buckets["exo"]:
            index[take_uid] = buckets
        else:
            print(f"{take_uid} is skipped: {buckets['ego']} and {buckets['exo']}")
    assert len(takes) == len(index)
    return index


# =============== Annotations loader ===============

def load_and_merge_annotations(ann_dir):
    """
    Expect: annotations/train.json and annotations/val.json
    Schema:
      {
        "annotations": {
            <take_uid>: [ { "descriptions": [ {text, timestamp, ego_visible, best_exo{cam_id} ...}, ... ] }, ... ],
            ...
        },
        ... (ignored fields)
      }
    Returns: dict[take_uid] -> list[description dict]
    """
    merged = defaultdict(list)
    rejected_clip_num = 0
    for name in ("atomic_descriptions_train.json", "atomic_descriptions_val.json"):
        p = os.path.join(ann_dir, name)
        if not os.path.exists(p):
            print(f"Not exist path {p}")
            continue
        with open(p, "rb") as f:
            blob = json.load(f)
        ann = blob.get("annotations", {}) or {}
        for take_uid, ann_list in ann.items():
            for a in (ann_list or []):                
                if a.get("rejected") in (True, "1", "true", "True"):
                    rejected_clip_num += 1
                    continue
                for d in (a.get("descriptions") or []):
                    ts = d.get("timestamp")
                    try:
                        ts = float(ts)
                    except Exception as e:
                        print(f"[INFO] {take_uid} is skipped: {e}")
                        continue
                    if not isfinite(ts):
                        print(f"[INFO] {take_uid} is skipped: infinite timestep")
                        continue
                    merged[take_uid].append(d)

    # sort by timestamp per take
    for k in list(merged.keys()):
        merged[k].sort(key=lambda d: float(d.get("timestamp", 0.0)))
    print(f"[INFO] {rejected_clip_num} clips are skipped due to rejection")
    return dict(merged)

# =============== Media selection ===============

def pick_ego_media(take_medias):
    for m in take_medias.get("ego", []):
        return m

def pick_exo_media(take_medias, desc):
    target = None
    best = desc.get("best_exo") or {}
    if isinstance(best, dict) and best.get("cam_id"):
        target = str(best["cam_id"]).lower()

    candidates = []
    if target:
        candidates.append(target)
    else:
        candidates.extend(["cam01", "gp01"])

    exos = take_medias.get("exo", [])
    for cand in candidates:
        for m in exos:
            if str(m["cam_id"]).lower() == cand:
                return m
            
# =============== Core processing ===============

def export_for_take(root_path, out_dir, take_uid, descriptions, take_medias, info_clips):
    takes_base = root_path
    ego_m = pick_ego_media(take_medias)
    if ego_m is None:
        print("Fail to pick ego")
        return
    ego_path = os.path.join(takes_base, ego_m["file_path"])
    ego_cap = cv2.VideoCapture(ego_path)

    if not ego_cap.isOpened():
        print("Fail to open ego")
        return
    ego_fps = ego_cap.get(cv2.CAP_PROP_FPS) or 30.0
    ego_last = frames_last_index(ego_cap)

    entries = []

    valid_idx = [i for i, d in enumerate(descriptions) if norm_bool(d.get("ego_visible"))]
    if not valid_idx:
        ego_cap.release()
        return

    for pos, i in enumerate(valid_idx):
        d = descriptions[i]
        if i + 1 < len(descriptions):
            post_t = float(descriptions[i + 1].get("timestamp", d["timestamp"]))
        else:
            post_t = None

        exo_m = pick_exo_media(take_medias, d)
        if exo_m is None:
            print(f"Fail to pick exo")
            continue
        exo_path = os.path.join(takes_base, exo_m["file_path"])
        exo_cap = cv2.VideoCapture(exo_path)
        if not exo_cap.isOpened():
            print(f"Fail to open exo")
            continue
        exo_fps = exo_cap.get(cv2.CAP_PROP_FPS) or 30.0
        exo_last = frames_last_index(exo_cap)

        t = float(d["timestamp"])
        ego_pre  = max(0, min(sec_to_frame(t, ego_fps), ego_last))
        exo_pre  = max(0, min(sec_to_frame(t, exo_fps), exo_last))
        if post_t is None:
            ego_post = ego_last
            exo_post = exo_last
        else:
            ego_post = max(0, min(sec_to_frame(post_t, ego_fps), ego_last))
            exo_post = max(0, min(sec_to_frame(post_t, exo_fps), exo_last))

        try:
            ego_pre_img  = read_frame_by_num(ego_cap, ego_pre)
            ego_post_img = read_frame_by_num(ego_cap, ego_post)
            exo_pre_img  = read_frame_by_num(exo_cap, exo_pre)
            exo_post_img = read_frame_by_num(exo_cap, exo_post)
        except Exception:
            exo_cap.release()
            print(f"Fail to read frame by number")
            continue

        text = sanitize(d.get("text") or "")
        desc_dir = os.path.join(out_dir, take_uid, f"{str(i).zfill(6)}_{text}")
        # os.makedirs(desc_dir, exist_ok=True)

        ego_cam = sanitize(ego_m["cam_id"])
        exo_cam = sanitize(exo_m["cam_id"])

        rel_ego_pre  = os.path.join(take_uid, f"{str(i).zfill(6)}_{text}", f"pre_frame_{ego_cam}.jpg")
        rel_ego_post = os.path.join(take_uid, f"{str(i).zfill(6)}_{text}", f"post_frame_{ego_cam}.jpg")
        rel_exo_pre  = os.path.join(take_uid, f"{str(i).zfill(6)}_{text}", f"pre_frame_{exo_cam}.jpg")
        rel_exo_post = os.path.join(take_uid, f"{str(i).zfill(6)}_{text}", f"post_frame_{exo_cam}.jpg")

        # cv2.imwrite(os.path.join(out_dir, rel_ego_pre),  ego_pre_img)
        # cv2.imwrite(os.path.join(out_dir, rel_ego_post), ego_post_img)
        # cv2.imwrite(os.path.join(out_dir, rel_exo_pre),  exo_pre_img)
        # cv2.imwrite(os.path.join(out_dir, rel_exo_post), exo_post_img)

        entries.append({
            "action_name": f"{str(i).zfill(6)}_{text}",
            "narration_text": d.get("text") or "",
            "timestamp": float(d["timestamp"]),
            "ego": {
                "camera_id": ego_m["cam_id"],
                "pre_frame":  {"frame_num": int(ego_pre),  "path": rel_ego_pre},
                "post_frame": {"frame_num": int(ego_post), "path": rel_ego_post},
            },
            "exo": {
                "camera_id": exo_m["cam_id"],
                "pre_frame":  {"frame_num": int(exo_pre),  "path": rel_exo_pre},
                "post_frame": {"frame_num": int(exo_post), "path": rel_exo_post},
            }
        })

        exo_cap.release()

    ego_cap.release()

    if entries:
        info_clips.setdefault(take_uid, []).extend(entries)

def worker_run_one_take(args):
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    root, out_dir, take_uid, descriptions, take_medias = args
    local_info = {}
    try:
        export_for_take(root, out_dir, take_uid, descriptions, take_medias, local_info)
    except Exception as e:
        return (take_uid, [], f"[WORKER-ERR] {take_uid}: {e}")
    entries = local_info.get(take_uid, [])
    return (take_uid, entries, None)

# =============== Main ===============

def main():
    parser = argparse.ArgumentParser(description="Export pre/post frames (description-based) for EgoExo4D.")
    parser.add_argument("--root_path", type=str, required=True,
                        help="Dataset root (contains takes.json, takes/, annotations/).")
    args = parser.parse_args()

    root = args.root_path
    takes_json_path = os.path.join(root, "takes.json")
    ann_dir = os.path.join(root, "annotations")
    out_dir = os.path.join(root, "clips_jpgs", "processed")
    os.makedirs(out_dir, exist_ok=True)

    takes_index = load_takes_index(takes_json_path)

    desc_by_take = load_and_merge_annotations(ann_dir)
    info_clips = {}
    cnt = 0
    tasks = []
    for take_uid, descriptions in tqdm.tqdm(desc_by_take.items()):
        take_medias = takes_index.get(take_uid)
        if not take_medias:
            # print(f"[INFO] No take media for {take_uid}")
            cnt+=1
            continue
        tasks.append((root, out_dir, take_uid, descriptions, take_medias))

    with ProcessPoolExecutor(max_workers=32) as ex:
        futures = [ex.submit(worker_run_one_take, t) for t in tasks]    
        for fut in tqdm.tqdm(as_completed(futures), total = len(futures), desc= "Takes"):
            take_uid, entries, err = fut.result()
            if err:
                print(err)
            if entries:
                info_clips.setdefault(take_uid, []).extend(entries)

    with open(os.path.join(out_dir, "info_clips.json"), "w") as f:
        json.dump(info_clips, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
