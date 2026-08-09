"""Minimal connectivity test for the 360 SmartBrain API."""

import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.getenv("ZHINAO_BASE_URL", "https://api.360.cn/v1").rstrip("/")
MODEL = os.getenv("ZHINAO_MODEL", "z-ai/glm-5.1")


def main() -> int:
    api_key = os.getenv("ZHINAO_API_KEY")
    if not api_key:
        print("错误：未设置环境变量 ZHINAO_API_KEY。", file=sys.stderr)
        return 2

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "只回复：API连接成功",
            }
        ],
        "temperature": 0,
        # GLM-5.1 may spend part of this budget on hidden reasoning tokens.
        "max_tokens": 256,
    }
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {details}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"网络调用失败：{exc}", file=sys.stderr)
        return 1

    try:
        message = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("接口已响应，但返回结构不符合预期：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    print(f"模型：{MODEL}")
    if not message or not message.strip():
        print("接口连接成功，但模型没有返回可见文本。")
        print("这通常表示输出额度被推理 token 耗尽。")
    else:
        print(f"回复：{message}")
    usage = result.get("usage")
    if usage:
        print(f"Token 用量：{json.dumps(usage, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
