"""Offline demo: retrieve L2 risk rules for a selected scenario."""

import argparse
import json
from pathlib import Path

from risk_rule_store import RiskRuleStore, resolve_rule_conflicts


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线检索匹配的 L2 风险规则")
    parser.add_argument(
        "scenario",
        nargs="?",
        default=str(ROOT / "scenarios" / "corridor_obstacle.json"),
        help="场景 JSON 路径（默认使用窄走廊障碍场景）",
    )
    parser.add_argument(
        "--rules",
        default=str(ROOT / "experiences" / "risk_rules.json"),
        help="L2规则库JSON路径（默认使用正式规则库）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_path = Path(args.scenario).expanduser().resolve()
    rules_path = Path(args.rules).expanduser().resolve()
    scenario = json.loads(
        scenario_path.read_text(encoding="utf-8")
    )
    store = RiskRuleStore(rules_path)
    matches = store.retrieve_matching(scenario)
    resolution = resolve_rule_conflicts(
        matches, scenario.get("available_actions", [])
    )

    print("=== Retrieved L2 Risk Rules ===")
    print(
        json.dumps(
            {
                "scenario": str(scenario_path),
                "rules": str(rules_path),
                "matched_rule_count": len(matches),
                "items": matches,
                "resolution": resolution,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
