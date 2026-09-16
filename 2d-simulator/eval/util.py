"""eval 公用工具:jsonl.gz 读写、场景加载、配置加载(YAML 兼容 JSON)。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path


def write_jsonl_gz(path: str | Path, records) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlGzWriter:
    def __init__(self, path: str | Path):
        self.f = gzip.open(path, "wt", encoding="utf-8")

    def write(self, r: dict):
        self.f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def close(self):
        self.f.close()


def read_jsonl_gz(path: str | Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_scenarios(path: str | Path, max_scenarios: int | None = None,
                   buckets: list[str] | None = None, family_seed_only: bool = False):
    """按文件名排序加载场景目录。"""
    path = Path(path)
    out = []
    for fp in sorted(path.glob("*.json")):
        if fp.name in ("index.json", "validation_report.json"):
            continue
        sc = json.loads(fp.read_text())
        if buckets and sc["meta"]["bucket"] not in buckets:
            continue
        if family_seed_only and sc["meta"]["seed"] != 0:
            continue
        out.append(sc)
        if max_scenarios and len(out) >= max_scenarios:
            break
    return out


def load_config(path: str | Path) -> dict:
    """配置文件:YAML 是 JSON 的超集,本仓库的 .yaml 配置一律写成 JSON 兼容子集
    (零第三方依赖;若装了 PyYAML 也能读任意 YAML)。支持 // 与 # 整行注释。"""
    text = Path(path).read_text()
    try:
        lines = [l for l in text.splitlines()
                 if not l.strip().startswith(("#", "//"))]
        return json.loads("\n".join(lines))
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except ImportError as e:
            raise ValueError(f"{path} 不是 JSON 兼容 YAML,且未安装 PyYAML") from e
