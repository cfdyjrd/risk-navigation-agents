#!/usr/bin/env python3
"""镜像 Isaac Sim 5.1 官方资产到本地（设计文档 §2.2，零第三方依赖）。

来源：omniverse-content-production S3 公开桶，匿名 GET。
目标：~/isaacsim_assets/Assets/Isaac/5.1/（get_assets_root_path 校验
<root>/Isaac 与 <root>/NVIDIA 两个目录都存在，NVIDIA/ 仅占位）。

用法：python3 scripts/mirror_assets.py [--dest DIR]
已存在且字节数一致的文件跳过，可断点续传。
"""

import argparse
import concurrent.futures
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ROOT_PREFIX = "Assets/Isaac/5.1/"
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# 相对 assets root 的目录前缀（设计文档 §2.2 离线化清单）
PREFIXES = [
    "Isaac/Samples/Policies/Spot_Policies/",
    "Isaac/Samples/Policies/H1_Policies/",
    "Isaac/Samples/Policies/Anymal_Policies/",
    "Isaac/Robots/BostonDynamics/spot/",
    "Isaac/Robots/Unitree/H1/",
    "Isaac/Robots/NVIDIA/Jetbot/",
    "Isaac/Robots/ANYbotics/anymal_c/",
]


def list_keys(prefix):
    """list-objects-v2 全量枚举一个前缀，返回 [(key, size)]。"""
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": ROOT_PREFIX + prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            tree = ET.fromstring(r.read())
        for c in tree.iter(f"{S3NS}Contents"):
            key = c.find(f"{S3NS}Key").text
            size = int(c.find(f"{S3NS}Size").text)
            if not key.endswith("/"):
                keys.append((key, size))
        token_el = tree.find(f"{S3NS}NextContinuationToken")
        if token_el is None:
            return keys
        token = token_el.text


def fetch(key, size, dest_root):
    rel = key[len(ROOT_PREFIX):]
    dst = os.path.join(dest_root, rel)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return ("skip", rel, size)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if os.path.getsize(tmp) != size:
        os.remove(tmp)
        raise RuntimeError(f"size mismatch: {rel}")
    os.replace(tmp, dst)
    return ("get", rel, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.dest, "NVIDIA"), exist_ok=True)  # 占位，见 docstring

    todo = []
    for p in PREFIXES:
        ks = list_keys(p)
        if not ks:
            print(f"!! 前缀无对象（路径或版本有变？）: {p}", file=sys.stderr)
        todo += ks
    total = sum(s for _, s in todo)
    print(f"{len(todo)} 个对象，共 {total / 1e6:.1f} MB")

    got = skipped = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch, k, s, args.dest) for k, s in todo]
        for fu in concurrent.futures.as_completed(futs):
            try:
                st, rel, size = fu.result()
            except Exception as e:
                failed += 1
                print(f"FAIL {e}", file=sys.stderr)
                continue
            if st == "get":
                got += 1
                print(f"  {rel} ({size / 1e6:.1f} MB)")
            else:
                skipped += 1
    print(f"完成：下载 {got}，跳过 {skipped}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
