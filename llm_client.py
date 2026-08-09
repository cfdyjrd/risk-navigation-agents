"""Small 360 SmartBrain Chat Completions client."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    """Raised when the model cannot provide a usable response."""


class ZhinaoClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ZHINAO_API_KEY")
        if not self.api_key:
            raise LLMError("未设置环境变量 ZHINAO_API_KEY")
        self.base_url = os.getenv(
            "ZHINAO_BASE_URL", "https://api.360.cn/v1"
        ).rstrip("/")
        self.model = os.getenv("ZHINAO_MODEL", "z-ai/glm-5.1")

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_data: dict[str, Any],
        max_tokens: int = 1600,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_data, ensure_ascii=False, indent=2),
                },
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw_response = json.load(response)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"360 API 返回 HTTP {exc.code}: {details}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"360 API 网络调用失败: {exc}") from exc

        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("360 API 响应缺少 choices[0].message.content") from exc

        return self._parse_json(content), raw_response.get("usage", {})

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        text = (content or "").strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            hint = ""
            if text.startswith("{") and not text.rstrip().endswith("}"):
                hint = "（输出疑似因 token 额度不足而被截断）"
            raise LLMError(
                f"模型没有返回合法 JSON{hint}。原始输出：{content!r}"
            ) from exc
        if not isinstance(value, dict):
            raise LLMError("模型 JSON 顶层必须是对象")
        return value
