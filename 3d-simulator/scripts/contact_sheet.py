#!/usr/bin/env python3
"""批跑结果拼版：每个 run 取 sbs.mp4 中段一帧，按网格拼成一张 contact sheet（附场景 id/桶/结果）。
    python3 scripts/contact_sheet.py out/batch/batch20 -o out/batch/batch20/contact_sheet.jpg
"""
import argparse, glob, json, os, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(); ap.add_argument("batch_dir"); ap.add_argument("-o", "--out"); ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--frac", type=float, default=0.55, help="取视频时长的该比例处的帧")
a = ap.parse_args()
runs = sorted(d for d in glob.glob(os.path.join(a.batch_dir, "*")) if os.path.isfile(os.path.join(d, "summary.json")))
tiles = []
W, H = 640, 240
for d in runs:
    s = json.load(open(os.path.join(d, "summary.json"))); js = s.get("judge_summary", {})
    scene = json.load(open(s["scene_path"])) if s.get("scene_path") and os.path.exists(s["scene_path"]) else {}
    bucket = (scene.get("source_2d") or {}).get("bucket", "?")
    vid = os.path.join(d, "sbs.mp4")
    img = Image.new("RGB", (W, H), (40, 40, 40))
    if os.path.exists(vid):
        dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", vid],
                                   capture_output=True, text=True).stdout.strip() or 0)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            subprocess.run(["ffmpeg", "-y", "-ss", f"{dur * a.frac:.2f}", "-i", vid, "-vframes", "1", "-vf", f"scale={W}:{H}", tf.name],
                           capture_output=True)
            try: img = Image.open(tf.name).convert("RGB")
            except Exception: pass
        os.unlink(tf.name)
    dr = ImageDraw.Draw(img)
    ok = js.get("success"); viol = len(js.get("violations", []))
    label = f"{os.path.basename(d)}  {bucket}  {'SUCCESS' if ok else 'FAIL'}  viol={viol}  ticks={js.get('total_ticks')}  {js.get('geo', {}).get('path_len_m', 0):.0f}m"
    dr.rectangle([0, 0, W, 18], fill=(0, 120, 0) if ok and viol == 0 else (150, 30, 30))
    dr.text((4, 3), label, fill=(255, 255, 255))
    tiles.append(img)
if not tiles: sys.exit("no runs")
cols = a.cols; rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * W, (i // cols) * H))
out = a.out or os.path.join(a.batch_dir, "contact_sheet.jpg")
sheet.save(out, quality=85); print(f"{len(tiles)} tiles -> {out}")
