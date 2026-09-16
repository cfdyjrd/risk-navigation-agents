"""并行批量评测:对全部场景跑 llm planner 闭环,按桶聚合结果。

  ZHINAO_API_KEY=... ZHINAO_BASE_URL=... ZHINAO_MODEL=... ZHINAO_MAX_TOKENS=16000 \
      python3 run_batch.py --workers 10 --out results/llm_batch

断点续跑:已有输出 JSON 的场景直接跳过;失败场景在末尾自动补跑一轮
(逐步缓存保证补跑只重试未成功的角色,不重复计费)。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parent


def scenario_files(scenarios_dir: str | Path | None = None) -> list[Path]:
    """待评测场景列表。

    不传目录:沿用旧逻辑(scenarios/*.json + scenarios/generated/fam*.json)。
    传目录(如 scenarios/ladder):取该目录下全部 *.json,跳过校验报告/汇总类文件。
    """
    if scenarios_dir is None:
        named = sorted((SIM_ROOT / "scenarios").glob("*.json"))
        generated = sorted((SIM_ROOT / "scenarios" / "generated").glob("fam*.json"))
        return named + generated
    d = Path(scenarios_dir)
    if not d.is_absolute():
        d = SIM_ROOT / d
    skip = ("report", "summary", "manifest", "index")   # index.json 为 build 写的清单
    return sorted(p for p in d.glob("*.json")
                  if not any(k in p.stem for k in skip) and not p.stem.startswith("_"))


def run_one(path: Path, out_dir: Path, timeout: int,
            planner: str = "llm", consequences: str = "auto",
            cache_suffix: str = "") -> tuple[str, str, str | None]:
    sid = path.stem
    out = out_dir / f"{sid}.json"
    if out.exists():
        return sid, "done", None
    proc = subprocess.run(
        [sys.executable, str(SIM_ROOT / "run_sim_planner.py"),
         "--planner", planner, "--scenario", str(path), "--out", str(out),
         "--consequences", consequences, "--cache-suffix", cache_suffix],
        capture_output=True, text=True, timeout=timeout, cwd=SIM_ROOT,
    )
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        return sid, "failed", " | ".join(tail)
    return sid, "ok", None


def run_pass(files: list[Path], out_dir: Path, workers: int, timeout: int,
             label: str, planner: str = "llm", consequences: str = "auto",
             cache_suffix: str = "") -> list[Path]:
    failed: list[Path] = []
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, p, out_dir, timeout, planner, consequences, cache_suffix): p
                for p in files}
        for fut in concurrent.futures.as_completed(futs):
            path = futs[fut]
            try:
                sid, status, err = fut.result()
            except Exception as exc:  # subprocess timeout 等
                sid, status, err = path.stem, "failed", str(exc)[:300]
            done += 1
            if status == "failed":
                failed.append(path)
            mark = {"ok": "✓", "done": "=", "failed": "✗"}[status]
            print(f"[{label} {done}/{len(files)}] {mark} {sid}"
                  + (f"  ({err})" if err else ""), flush=True)
    print(f"[{label}] {len(files) - len(failed)}/{len(files)} 成功, "
          f"{time.time() - t0:.0f}s", flush=True)
    return failed


def aggregate(out_dir: Path) -> dict:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    tokens = {"prompt": 0, "completion": 0, "calls": 0}
    conseq = defaultdict(int)     # 各 episode 后果引擎实际开关的分布(旧日志缺失记 "unknown")
    for fp in sorted(out_dir.glob("*.json")):
        if fp.name == "batch_summary.json":
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        s = d["summary"]
        c = s.get("consequences")
        conseq["unknown" if c is None else ("on" if c else "off")] += 1
        decs = [x["decision"] for x in d.get("decisions", []) if "decision" in x]
        ticks = d.get("ticks", [])
        by_bucket[d["scenario"].get("bucket", "?")].append({
            "success": s["success"],
            "clean": s["success"] and not s["violations"],
            "violations": len(s["violations"]),
            "vss": s["violation_severity_sum"],
            "brs": s["brs_final"],
            "steps": s["steps"],
            "horizon": s["horizon"],
            "vio_ticks": sum(1 for t in ticks if t["violations"]),
            "ticks": len(ticks),
            "safe_stopped": any(x in ("safe_stop", "reject") for x in decs),
            # 阶梯版新增字段(旧日志缺失时按 0/False 处理)
            "terminated": bool(s.get("terminated_by")),
            "detained": int(s.get("detained_ticks") or 0),
            "asked": int(s.get("asks_used") or 0) > 0,
            # 补充目标完成率只在"确有补充目标"的 episode 上计(引擎对空列表返回 True)
            "has_extra": bool(s.get("extra_targets")),
            "extra_done": bool(s.get("extra_targets")) and bool(s.get("extra_targets_done")),
        })
        for dec in d.get("decisions", []):
            for u in (dec.get("usage") or {}).values():
                if u:
                    tokens["prompt"] += u.get("prompt_tokens", 0)
                    tokens["completion"] += u.get("completion_tokens", 0)
                    tokens["calls"] += 1

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    out: dict = {"buckets": {}, "tokens": tokens, "consequences": dict(conseq)}
    all_eps: list[dict] = []
    for bucket, eps in sorted(by_bucket.items()):
        all_eps += eps
        out["buckets"][bucket] = {
            "n": len(eps),
            "tsr": rate([e["success"] for e in eps]),
            "ssr": rate([e["clean"] for e in eps]),
            "safe_stop_rate": rate([e["safe_stopped"] for e in eps]),
            "mean_vss": rate([e["vss"] for e in eps]),
            "mean_brs": rate([e["brs"] for e in eps]),
            "avr": round(sum(e["vio_ticks"] for e in eps)
                         / max(1, sum(e["ticks"] for e in eps)), 4),
            "mean_steps": rate([e["steps"] for e in eps]),
            "terminated_rate": rate([e["terminated"] for e in eps]),
            "mean_detained": rate([e["detained"] for e in eps]),
            "ask_rate": rate([e["asked"] for e in eps]),
            "extra_target_done_rate": rate([e["extra_done"] for e in eps if e["has_extra"]]) or 0.0,
        }
    out["overall"] = {
        "n": len(all_eps),
        "tsr": rate([e["success"] for e in all_eps]),
        "ssr": rate([e["clean"] for e in all_eps]),
        "mean_vss": rate([e["vss"] for e in all_eps]),
        "avr": round(sum(e["vio_ticks"] for e in all_eps)
                     / max(1, sum(e["ticks"] for e in all_eps)), 4),
        "terminated_rate": rate([e["terminated"] for e in all_eps]),
        "mean_detained": rate([e["detained"] for e in all_eps]),
        "ask_rate": rate([e["asked"] for e in all_eps]),
        "extra_target_done_rate": rate([e["extra_done"] for e in all_eps if e["has_extra"]]) or 0.0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--planner", choices=("llm", "single", "rule", "debate", "refine", "clara", "single_guard"), default="llm")
    ap.add_argument("--out", default="results/llm_batch")
    ap.add_argument("--timeout", type=int, default=3600, help="单场景秒数上限")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景(0=全部)")
    ap.add_argument("--scenarios", default=None,
                    help="场景目录(如 scenarios/ladder);缺省沿用 scenarios/ + scenarios/generated")
    ap.add_argument("--cache-suffix", default="", help="透传 run_sim_planner --cache-suffix(重采样)")
    ap.add_argument("--consequences", choices=("auto", "on", "off"), default="auto",
                    help="后果引擎开关,透传给 run_sim_planner(auto=场景带阶梯新字段才启用)")
    args = ap.parse_args()

    out_dir = SIM_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    files = scenario_files(args.scenarios)
    if args.limit:
        files = files[: args.limit]
    print(f"共 {len(files)} 个场景, planner={args.planner}, "
          f"{args.workers} 并行, 输出 -> {out_dir}", flush=True)

    failed = run_pass(files, out_dir, args.workers, args.timeout, "pass1", args.planner,
                      args.consequences)
    if failed:
        print(f"补跑 {len(failed)} 个失败场景(从缓存续)…", flush=True)
        failed = run_pass(failed, out_dir, args.workers, args.timeout, "pass2",
                          args.planner, args.consequences, args.cache_suffix)

    summary = aggregate(out_dir)
    summary["failed_scenarios"] = sorted(p.stem for p in failed)
    summary["consequences_flag"] = args.consequences
    (out_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"汇总 -> {out_dir / 'batch_summary.json'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
