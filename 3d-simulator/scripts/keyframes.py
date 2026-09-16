#!/usr/bin/env python3
"""把一次 run 的每个逻辑 tick 抽一帧，拼成带动作标注的关键帧序列。

    python3 scripts/keyframes.py out/batch/<tag>/<run> -o out/keyframes.png [--view god|fpv|both]

tick 边界取自 trace.jsonl 的 result.geo.sim_time_s（累计仿真秒），
再用 frames.jsonl 的逐帧 t 反查帧号；视频时间 = 帧号 / fps。
"""
import argparse, json, os, subprocess, tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--view", choices=["god", "fpv", "both"], default="god")
ap.add_argument("--fps", type=float, default=25.0)
ap.add_argument("--at", type=float, default=0.75, help="取每个 tick 时段的该比例处")
ap.add_argument("--width", type=int, default=1500, help="每格宽度")
a = ap.parse_args()

run = Path(a.run_dir)
summary = json.load(open(run / "summary.json"))
scene = json.load(open(summary["scene_path"]))
trace = [json.loads(l) for l in open(run / "trace.jsonl")]
frames = [json.loads(l) for l in open(run / "frames.jsonl")]

# tick -> 视频秒：tick k 的执行区间 [S_{k-1}, S_k]，取 at 比例处那一刻的帧号 / fps
bounds, prev = [], 0.0
for t in trace:
    s = float((t["result"].get("geo") or {}).get("sim_time_s") or prev)
    bounds.append((prev, s)); prev = s


def frame_at(sim_t):
    i = min(range(len(frames)), key=lambda k: abs(frames[k]["t"] - sim_t))
    return frames[i]["i"] / a.fps


def grab(vsec):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{vsec:.2f}", "-i", str(run / "sbs.mp4"),
                        "-frames:v", "1", "-y", tf.name], check=True)
        return Image.open(tf.name).convert("RGB")


def split(img):
    w, h = img.size
    god, fpv = img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))
    return {"god": god, "fpv": fpv, "both": img}[a.view]


def autocrop(img, pad=12):
    """裁到内容外框：俯视图里墙是深色线，背景是浅灰。"""
    g = img.convert("L")
    px = g.load(); w, h = g.size
    xs, ys = [], []
    step = 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            if px[x, y] < 110:            # 墙线 / 禁区地贴 / 色点
                xs.append(x); ys.append(y)
    if not xs:
        return img
    box = (max(0, min(xs) - pad), max(0, min(ys) - pad),
           min(w, max(xs) + pad), min(h, max(ys) + pad))
    return img.crop(box)


tiles = []
for k, (t0, t1) in enumerate(bounds, start=1):
    rec = trace[k - 1]
    img = split(grab(frame_at(t0 + (t1 - t0) * a.at)))
    if a.view == "god":
        img = autocrop(img)
    res = rec["result"]
    tiles.append((img, f"t{k}  {rec['action']}  →  {res.get('outcome')}  @{res.get('current_zone')}"))

BAR = 46
cw = a.width
scaled = []
for img, cap in tiles:
    ch = max(1, round(img.height * cw / img.width))
    scaled.append((img.resize((cw, ch), Image.LANCZOS), cap, ch))

total_h = sum(ch + BAR for _, _, ch in scaled)
sheet = Image.new("RGB", (cw, total_h), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 30)
except OSError:
    font = ImageFont.load_default()

y = 0
for img, cap, ch in scaled:
    sheet.paste(img, (0, y)); y += ch
    draw.rectangle([0, y, cw, y + BAR], fill=(248, 248, 246))
    draw.text((14, y + 8), cap, fill=(20, 20, 20), font=font)
    draw.line([0, y, cw, y], fill=(200, 200, 200), width=1)
    y += BAR

os.makedirs(Path(a.out).parent, exist_ok=True)
sheet.save(a.out, quality=94)
print(f"{len(tiles)} ticks -> {a.out}  ({sheet.width}x{sheet.height})")
