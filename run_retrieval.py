"""Retrieve risk experiences relevant to the example scenario."""

import json
from pathlib import Path

from experience_store import ExperienceError, ExperienceStore


ROOT = Path(__file__).resolve().parent


def main() -> int:
    scenario = json.loads(
        (ROOT / "scenarios" / "corridor_obstacle.json").read_text(encoding="utf-8")
    )
    store = ExperienceStore(ROOT / "experiences" / "risk_experiences.json")
    try:
        results = store.retrieve(scenario, top_k=3)
    except (ExperienceError, ValueError, json.JSONDecodeError) as exc:
        print(f"风险经验检索失败：{exc}")
        return 1

    print("与当前场景最相关的风险经验：")
    print(
        json.dumps(
            [item.as_dict() for item in results],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

