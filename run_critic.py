"""Run only the Risk Critic against the example navigation scenario."""

import json
from pathlib import Path

from agents.critic import analyze
from llm_client import LLMError, ZhinaoClient


ROOT = Path(__file__).resolve().parent


def main() -> int:
    scenario_path = ROOT / "scenarios" / "corridor_obstacle.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    try:
        report, usage = analyze(scenario, ZhinaoClient())
    except LLMError as exc:
        print(f"Risk Critic 运行失败：{exc}")
        return 1

    print("Risk Critic 报告：")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nToken 用量：")
    print(json.dumps(usage, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

