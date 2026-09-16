"""Small 360 SmartBrain Chat Completions client."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import http.client
from typing import Any


class LLMError(RuntimeError):
    """Raised when the model cannot provide a usable response."""


class BalanceError(LLMError):
    """余额/额度类错误(HTTP 401 code 1004 / 402):可等待额度恢复后重试。"""


class NetworkError(LLMError):
    """DNS/连接/超时类错误:可等待网络恢复后重试。"""


def _is_balance_error(code: int, details: str) -> bool:
    text = details or ""
    return code == 402 or ("余额不足" in text) or ("insufficient_balance" in text.lower()) \
        or ("Insufficient Balance" in text) or ('"code":"1004"' in text.replace(" ", ""))


class ZhinaoClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ZHINAO_API_KEY")
        if not self.api_key:
            raise LLMError("未设置环境变量 ZHINAO_API_KEY")
        self.base_url = os.getenv(
            "ZHINAO_BASE_URL", "https://api.360.cn/v1"
        ).rstrip("/")
        self.model = os.getenv("ZHINAO_MODEL", "z-ai/glm-5.1")
        # 推理型模型的思维链 token 计入输出额度,可用 ZHINAO_MAX_TOKENS 整体调高
        self.max_tokens_override = int(os.getenv("ZHINAO_MAX_TOKENS", "0")) or None

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_data: dict[str, Any],
        max_tokens: int = 1600,
        retries: int = 2,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # 推理型模型偶发返回空内容或截断 JSON,自动重试可显著减少整链失败
        last_error: LLMError | None = None
        balance_wait = int(os.getenv("ZHINAO_BALANCE_WAIT", "0"))       # 秒;0 = 不等待
        balance_retries = int(os.getenv("ZHINAO_BALANCE_RETRIES", "0"))
        balance_tries = 0
        net_wait = int(os.getenv("ZHINAO_NET_WAIT", "0"))               # 秒;0 = 不等待
        net_retries = int(os.getenv("ZHINAO_NET_RETRIES", "0"))
        net_tries = 0
        attempt = 0
        while attempt <= retries:
            try:
                return self._chat_json_once(
                    system_prompt=system_prompt,
                    user_data=user_data,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except BalanceError as exc:
                last_error = exc
                if balance_wait > 0 and balance_tries < balance_retries:
                    balance_tries += 1
                    time.sleep(balance_wait)      # 等额度恢复,不计入普通重试次数
                    continue
                raise
            except NetworkError as exc:
                last_error = exc
                if net_wait > 0 and net_tries < net_retries:
                    net_tries += 1
                    time.sleep(net_wait)          # 等网络恢复,不计入普通重试次数
                    continue
                attempt += 1
            except LLMError as exc:
                last_error = exc
                attempt += 1
        raise last_error

    def _chat_json_once(
        self,
        *,
        system_prompt: str,
        user_data: dict[str, Any],
        max_tokens: int,
        temperature: float | None = None,
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
            "temperature": 0.2 if temperature is None else temperature,
            "max_tokens": self.max_tokens_override or max_tokens,
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
            if _is_balance_error(exc.code, details):
                raise BalanceError(f"360 API 返回 HTTP {exc.code}: {details}") from exc
            raise LLMError(f"360 API 返回 HTTP {exc.code}: {details}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException) as exc:
            raise NetworkError(f"360 API 网络调用失败: {exc}") from exc

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
