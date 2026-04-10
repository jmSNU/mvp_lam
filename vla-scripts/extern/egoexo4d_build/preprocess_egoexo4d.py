import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import signal, sys
import multiprocessing as mp

# --- OpenCV/FFmpeg 안정 옵션 ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")  # Linux에선 무시
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "protocol_whitelist;file,crypto,data,rtp,udp,tcp,https,tls")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------- helpers ----------
PREFERRED_EGO_BASENAMES = [
    "aria01_214-1.mp4",    # rgb (example)
    "aria01_1201-1.mp4",   # slam-left (example)
    "aria01_1201-2.mp4",   # slam-right (example)
    "aria01_211-1.mp4",    # et (example)
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Ego/Exo dense clips (per cam) into frame sequences.')
    parser.add_argument('--denseclips_dir', type=str, required=True,
                        help='Root directory for denseclips output')
    parser.add_argument('--info_clips_json', type=str, required=True,
                        help='Path to info_clips.json created earlier')
    parser.add_argument('--source_videos_dir', type=str, required=True,
                        help='Directory containing source frame-aligned videos (root of takes/...)')
    parser.add_argument('--frame_interval', type=int, default=15,
                        help='Interval between saved frames (default: 15)')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of parallel processes (default: 1)')
    return parser.parse_args()

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

def read_segment_frames(video_path, start_idx, end_idx, interval):
    """Read frames [start_idx, end_idx] with interval. Returns list of ndarray."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return []

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_idx = max(0, min(start_idx, max(0, total - 1)))
    end_idx   = max(0, min(end_idx,   max(0, total - 1)))
    if end_idx < start_idx:
        cap.release()
        return []

    # 한 번만 정확히 시킹, 그 뒤 연속 read 하며 interval로 샘플링
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    cur = start_idx
    step = max(1, interval)
    pick = 0
    while cur <= end_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if pick % step == 0:
            frames.append(frame)
        pick += 1
        cur  += 1

    cap.release()
    return frames

def _init_worker(takes_json_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    global TAKES_INDEX
    TAKES_INDEX = load_takes_index(takes_json_path)

def cam_id_to_role(id_str):
    return "exo" if str(id_str).lower().startswith("cam") else "ego"

def process_take(take_id, actions, args):
    """
    For a single take_id:
      - iterate actions
      - for each of {ego, exo} (if present) in the action
      - slice frames [pre_frame.frame_num, post_frame.frame_num] and save:
          denseclips_dir/take_id/action_name/{00001_<role>.npy, ...}
    """
    global TAKES_INDEX
    take_medias = TAKES_INDEX.get(take_id)
    if not take_medias or not isinstance(take_medias, dict):
        print(f"[WARN][{take_id}] No medias in TAKES_INDEX or wrong type: {type(take_medias)}")
        return []

    cam2path = {}
    for role_key, lst in take_medias.items():
        if not isinstance(lst, list):
            continue
        for m in lst:
            if not isinstance(m, dict):
                continue
            cid = str(m.get('cam_id'))
            fpath = m.get('file_path')
            if cid and fpath:
                cam2path[cid] = fpath

    out_entries = []

    for aidx, action in enumerate(actions):
        action_name = action.get('action_name', f'action_{aidx:06d}')
        lang = action.get('narration_text', '')

        for role in ('ego', 'exo'):
            seg = action.get(role)
            if not isinstance(seg, dict):
                print(f"[SKIP][{take_id}][{action_name}] role={role}: no segment dict")
                continue

            cam_id = str(seg.get('camera_id') or '')
            pre = seg.get('pre_frame') or {}
            post = seg.get('post_frame') or {}
            if ('frame_num' not in pre) or ('frame_num' not in post) or not cam_id:
                print(f"[SKIP][{take_id}][{action_name}] role={role}: missing pre/post or cam_id "
                      f"(pre={pre}, post={post}, cam_id={cam_id})")
                continue

            start = int(pre['frame_num'])
            end   = int(post['frame_num'])
            if end < start:
                print(f"[SKIP][{take_id}][{action_name}] role={role}: end<start [{start}:{end}]")
                continue

            rel_path = cam2path.get(cam_id)
            if not rel_path:
                print(f"[SKIP][{take_id}][{action_name}] role={role}: cam_id={cam_id} not in TAKES_INDEX")
                continue

            if rel_path.startswith("takes/"):
                video_path = os.path.join(args.source_videos_dir, rel_path[len("takes/"):])
            else:
                video_path = os.path.join(args.source_videos_dir, rel_path)

            if not os.path.exists(video_path):
                print(f"[SKIP][{take_id}][{action_name}] role={role}: video not found -> {video_path}")
                continue
            frames = read_segment_frames(video_path, start, end, args.frame_interval)
            if not frames:
                print(f"[FAIL][{take_id}][{action_name}] role={role}: no frames read (interval={args.frame_interval})")
                continue

            save_dir = os.path.join(args.denseclips_dir, take_id, action_name)
            os.makedirs(save_dir, exist_ok=True)

            cnt_saved = 0
            for i, frame in enumerate(frames, start=1):
                npy_name = os.path.join(save_dir, f"{i:05d}_{role}.npy")
                if not os.path.exists(npy_name):
                    np.save(npy_name, frame)
                    cnt_saved += 1

            out_entries.append({
                'take_id': take_id,
                'action_name': action_name,
                'cam_id': cam_id,
                'role': role,
                'source_video': video_path,
                'start_frame': start,
                'end_frame': end,
                'language': lang,
            })

    return out_entries



def _process_item(kv):
    take_id, actions, args = kv
    return process_take(take_id, actions, args)

def main():
    args = parse_arguments()

    with open(args.info_clips_json, 'r') as f:
        info_clips = json.load(f)  # { take_id: [ {action_name, narration_text, pre_frame_camXX, post_frame_camXX, ...}, ...] }

    takes_json_path = os.path.join(args.source_videos_dir, "..", "takes.json")
    items = list(info_clips.items())

    if args.processes > 1:
        ctx = mp.get_context("spawn")
        info = []
        with ctx.Pool(processes=args.processes,
                      initializer=_init_worker,
                      initargs=(takes_json_path,),
                      maxtasksperchild=1) as pool:
            try:
                iterator = pool.imap_unordered(_process_item, [(t, a, args) for t, a in items], chunksize=1)
                for part in tqdm(iterator, total=len(items), desc="Processing takes (mp)"):
                    if part:
                        info.extend(part)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("\n[WARN] KeyboardInterrupt: terminating pool...", file=sys.stderr)
                pool.terminate()
                pool.join()
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
                pool.terminate()
                pool.join()

    else:
        _init_worker(takes_json_path)
        info = []
        for take_id, actions in tqdm(items, desc="Processing takes"):
            part = process_take(take_id, actions, args)

            if part:
                info.extend(part)

    os.makedirs(args.denseclips_dir, exist_ok=True)
    with open(os.path.join(args.denseclips_dir, 'annotations.json'), 'w') as f:
        json.dump(info, f, indent=4)

if __name__ == '__main__':
    main()
