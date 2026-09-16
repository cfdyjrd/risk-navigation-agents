"""场景生成 CLI。

用法:
    python -m generator.build --n 800 --seeds 5 --out scenarios/ [--seed 0]        # 旧四桶
    python -m generator.build --ladder --out scenarios/ladder --seed 42            # 阶梯 benchmark

旧四桶:--n 是场景族数;drift 族(占 40% 场景量)会生成 labeled/unlabeled 配对(pair_id 相同)。
桶配比:safe 20% / unsafe 20% / ambiguous 20% / drift 40%(drift 内三种 drift_type 均分)。

阶梯(--ladder,规范 G8):默认配额 safe-clear 20 / drift-L1 20 / drift-L2 20 / drift-L3 20 对(40 个)
/ ambiguous-L1 20 / ambiguous-L2 20 / ambiguous-L3 20,seeds 默认 1,输出 scenarios/ladder/。
--ladder-scale 可整体缩放配额(如 0.5 -> 每桶 10)。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .sampler import build_family
from .validate import validate_dir

# 每"配额单位"贡献的场景族数:drift 族出 2 套(L/U 配对),各算一份场景量
BUCKET_WEIGHTS = {"safe-clear": 0.20, "unsafe-clear": 0.20,
                  "ambiguous-state": 0.20, "authorization-drift": 0.40}

# 阶梯配额(族数;drift-L3 每族出 spoof/legit 两套)
LADDER_QUOTA = {"safe-clear": 20, "drift-L1": 20, "drift-L2": 20, "drift-L3": 20,
                "ambiguous-L1": 20, "ambiguous-L2": 20, "ambiguous-L3": 20}


def allocate(n: int) -> list[str]:
    """把 n 个场景族名额分配到桶;drift 族生成配对(2 套),故族数减半。"""
    order: list[str] = []
    counts = {b: int(round(n * w)) for b, w in BUCKET_WEIGHTS.items()}
    counts["authorization-drift"] = max(1, counts["authorization-drift"] // 2)
    for b, c in counts.items():
        order += [b] * c
    return order


def _write(out: Path, fam: list[dict], bucket: str, index: list) -> int:
    for sc in fam:
        fp = out / f"{sc['meta']['scenario_id']}.json"
        fp.write_text(json.dumps(sc, ensure_ascii=False))
        index.append({"scenario_id": sc["meta"]["scenario_id"], "bucket": bucket, "file": fp.name})
    return len(fam)


def build_ladder(args, out: Path, cfg: dict) -> tuple[int, int, list]:
    """按配额逐桶生成;某族失败则顺延族号继续,直到配额满或尝试数达配额 4 倍。"""
    n_written, n_failed, index = 0, 0, []
    family_idx = 0
    for bucket, quota in LADDER_QUOTA.items():
        quota = max(1, int(round(quota * args.ladder_scale)))
        got, tries = 0, 0
        while got < quota and tries < quota * 4:
            tries += 1
            fam = build_family(args.seed, family_idx, bucket, cfg)
            family_idx += 1
            if fam is None:
                n_failed += 1
                continue
            n_written += _write(out, fam, bucket, index)
            got += 1
        if got < quota:
            print(f"警告: 桶 {bucket} 只生成 {got}/{quota} 族")
    return n_written, n_failed, index


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="场景族数(drift 配对各算一套)")
    ap.add_argument("--seeds", type=int, default=None, help="每场景 seed 变体数(3-5;--ladder 默认 1)")
    ap.add_argument("--out", default=None, help="输出目录(默认 scenarios/,--ladder 时 scenarios/ladder/)")
    ap.add_argument("--seed", type=int, default=0, help="全局采样种子")
    ap.add_argument("--long-horizon-frac", type=float, default=0.3)
    ap.add_argument("--ladder", action="store_true", help="生成阶梯 benchmark(规范 G8 配额)")
    ap.add_argument("--ladder-scale", type=float, default=1.0, help="阶梯配额整体缩放")
    args = ap.parse_args(argv)

    if args.seeds is None:
        args.seeds = 1 if args.ladder else 5
    out = Path(args.out or ("scenarios/ladder/" if args.ladder else "scenarios/"))
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"n_variants": args.seeds, "long_horizon_frac": args.long_horizon_frac,
           "max_attempts": 40}

    t0 = time.time()
    if args.ladder:
        n_written, n_failed, index = build_ladder(args, out, cfg)
    else:
        n_written, n_failed = 0, 0
        index = []
        drift_rank = 0
        for i, bucket in enumerate(allocate(args.n)):
            dr = None
            if bucket == "authorization-drift":
                dr = drift_rank
                drift_rank += 1
            fam = build_family(args.seed, i, bucket, cfg, drift_rank=dr)
            if fam is None:
                n_failed += 1
                continue
            n_written += _write(out, fam, bucket, index)
    (out / "index.json").write_text(json.dumps(
        {"global_seed": args.seed, "n_variants": args.seeds, "ladder": bool(args.ladder),
         "scenarios": index},
        ensure_ascii=False))
    dt = time.time() - t0
    print(f"written {n_written} scenarios ({n_failed} families failed) in {dt:.1f}s -> {out}")

    rep = validate_dir(out)
    print("bucket_dist:", rep["bucket_dist"], " invalid:", len(rep["invalid"]))
    if rep["invalid_by_bucket"]:
        print("invalid_by_bucket:", rep["invalid_by_bucket"])


if __name__ == "__main__":
    main()
