"""host 侧 Isaac 子进程管理与 JSON-lines IPC（设计文档 §8.1）。

host 运行在系统 Python 3.12，绝不 import isaac 模块；唯一例外是纯 stdlib 的
isaac/protocol.py，用 importlib 按文件路径加载，避免触碰 isaac 包内其他模块。

用法：
    with IsaacProc(log_path=out / "isaac.log") as proc:
        proc.request({"cmd": "init", ...})
        proc.request({"cmd": "observe"})
"""

from __future__ import annotations

import importlib.util
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

FENGWU_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = FENGWU_ROOT / "isaac" / "protocol.py"
SERVER_PATH = FENGWU_ROOT / "isaac" / "isaac_server.py"
DEFAULT_CONFIG = FENGWU_ROOT / "configs" / "run.yaml"

LOG_TAIL_LINES = 30
EXECUTE_TIMEOUT_S = 3600.0   # 楼梯跳 39 宏+渲染墙钟 >600s（实测），放宽
HEARTBEAT_TIMEOUT_S = 30.0  # 下限；run.yaml 里更大则取更大


def _load_protocol():
    spec = importlib.util.spec_from_file_location("3d-simulator_protocol", PROTOCOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


protocol = _load_protocol()


class IsaacCrashed(RuntimeError):
    """isaac 子进程死亡 / 超时 / 管道断裂。message 里带最后几十行日志。"""


def _read_run_config(path: Path) -> dict:
    """读 run.yaml 中 host 侧需要的三个键。优先 PyYAML，缺则用最小行解析。"""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        return {
            "isaac_python": data.get("isaac_python", ""),
            "assets_root": data.get("assets_root", ""),
            "heartbeat_timeout_s": float(
                (data.get("ipc") or {}).get("heartbeat_timeout_s", HEARTBEAT_TIMEOUT_S)
            ),
        }
    except ImportError:
        pass

    def scalar(key: str) -> str:
        m = re.search(rf"^{key}:\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE)
        if not m:
            return ""
        return m.group(1).strip().strip("\"'")

    hb = HEARTBEAT_TIMEOUT_S
    m = re.search(r"heartbeat_timeout_s:\s*([0-9.]+)", text)
    if m:
        hb = float(m.group(1))
    return {
        "isaac_python": scalar("isaac_python"),
        "assets_root": scalar("assets_root"),
        "heartbeat_timeout_s": hb,
    }


class IsaacProc:
    """isaac_server.py 子进程的生命周期 + 请求/应答。

    stdout 由后台线程逐行搬进队列（readline 在线程里阻塞无害，规避了
    selectors + TextIOWrapper 缓冲交互的坑）；request() 带超时地从队列取行，
    协议行（@FW 前缀）作为应答返回，其余行落日志文件。
    """

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG,
                 log_path: Path | str | None = None):
        self.config_path = Path(config_path)
        self.log_path = Path(log_path) if log_path else Path("isaac_server.log")
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._log_fh = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> "IsaacProc":
        cfg = _read_run_config(self.config_path)
        self._heartbeat = max(HEARTBEAT_TIMEOUT_S, cfg["heartbeat_timeout_s"])

        isaac_python = Path(os.path.expanduser(cfg["isaac_python"])).resolve()
        if not isaac_python.exists():
            raise FileNotFoundError(
                f"run.yaml isaac_python 不存在: {isaac_python}（检查 {self.config_path}）")
        if not SERVER_PATH.exists():
            raise FileNotFoundError(f"isaac server 脚本不存在: {SERVER_PATH}")

        env = os.environ.copy()
        env["FENGWU_ASSETS"] = os.path.expanduser(cfg["assets_root"])

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log_fh.write(f"===== IsaacProc start {time.strftime('%F %T')} =====\n")
        self._log_fh.flush()

        self._proc = subprocess.Popen(
            [str(isaac_python), str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_fh,
            env=env,
            cwd=str(FENGWU_ROOT),
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()
        return self

    def _pump_stdout(self) -> None:
        proc = self._proc
        try:
            for line in proc.stdout:  # 阻塞 readline，EOF 自然退出
                self._queue.put(line)
        except (ValueError, OSError):
            pass  # 流被关闭
        finally:
            self._queue.put(None)  # EOF 哨兵

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- request/response ----------------------------------------------------

    def request(self, msg: dict, timeout_s: float | None = None) -> dict:
        """发一条命令，阻塞等应答。超时/进程死亡 raise IsaacCrashed（带日志尾巴）。"""
        if self._proc is None:
            raise IsaacCrashed("IsaacProc 未 start()")
        if timeout_s is None:
            timeout_s = (EXECUTE_TIMEOUT_S if msg.get("cmd") == "execute_action"
                         else self._heartbeat)
        if self._proc.poll() is not None:
            self._raise_crashed(f"发送 {msg.get('cmd')} 前发现进程已退出"
                                f"（exit={self._proc.returncode}）")
        try:
            self._proc.stdin.write(protocol.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            self._raise_crashed(f"写 stdin 失败（{e!r}）")

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._raise_crashed(
                    f"等待 {msg.get('cmd')} 应答超时（{timeout_s:.0f}s）")
            try:
                line = self._queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self._proc.poll() is not None:
                    self._raise_crashed(
                        f"等待 {msg.get('cmd')} 应答期间进程退出"
                        f"（exit={self._proc.returncode}）")
                continue
            if line is None:  # EOF
                code = self._proc.poll()
                self._raise_crashed(f"stdout 关闭（exit={code}）")
            try:
                reply = protocol.parse_line(line)
            except Exception as e:  # 前缀对但 JSON 坏：记日志继续等
                self._log_line(f"[host] 坏协议行 {e!r}: {line.rstrip()}")
                continue
            if reply is None:
                self._log_line(line.rstrip("\n"))
                continue
            return reply

    # -- shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.stdin.write(protocol.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
        if self._reader is not None:
            self._reader.join(timeout=2)
        self._drain_queue_to_log()
        if self._log_fh is not None:
            try:
                self._log_fh.write(f"===== IsaacProc exit={proc.returncode} =====\n")
                self._log_fh.close()
            except (OSError, ValueError):
                pass
            self._log_fh = None
        self._proc = None

    def __enter__(self) -> "IsaacProc":
        if self._proc is None:
            self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # -- helpers -------------------------------------------------------------

    def _log_line(self, text: str) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.write(text + "\n")
                self._log_fh.flush()
            except (OSError, ValueError):
                pass

    def _drain_queue_to_log(self) -> None:
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                return
            if line is not None:
                self._log_line(line.rstrip("\n"))

    def _log_tail(self) -> str:
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
            except (OSError, ValueError):
                pass
        try:
            lines = self.log_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "(无日志)"
        return "\n".join(lines[-LOG_TAIL_LINES:]) or "(日志为空)"

    def _raise_crashed(self, reason: str):
        raise IsaacCrashed(
            f"isaac 子进程异常: {reason}\n"
            f"--- {self.log_path} 最后 {LOG_TAIL_LINES} 行 ---\n{self._log_tail()}")
