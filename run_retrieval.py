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
        bundles = {
            role: store.retrieve_memory_cards(
                scenario,
                role=role,
                top_k=3,
                token_budget=1800,
            )
            for role in ("advocate", "critic", "decision")
        }
    except (ExperienceError, ValueError, json.JSONDecodeError) as exc:
        print(f"风险经验检索失败：{exc}")
        return 1

    print("面向三个角色的风险记忆证据包：")
    print(json.dumps(bundles, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
