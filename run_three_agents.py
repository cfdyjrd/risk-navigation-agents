"""Run the first end-to-end three-agent deliberation."""

import json
import hashlib
from pathlib import Path

from agents.advocate import analyze as run_advocate
from agents.critic import analyze as run_critic
from agents.decision import decide
from experience_store import ExperienceStore, build_memory_card
from llm_client import LLMError, ZhinaoClient
from safety_guard import apply_safety_guard


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
PIPELINE_VERSION = "risk-memory-card-v2"


def print_section(title: str, value: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def pipeline_cache_dir(scenario: dict, experiences: list[dict]) -> Path:
    cache_input = {
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario,
        "retrieved_experiences": experiences,
    }
    encoded = json.dumps(
        cache_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return CACHE_ROOT / digest


def load_cached(cache_dir: Path, role: str):
    path = cache_dir / f"{role}.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    print(f"使用缓存：{role}")
    return saved["report"], saved.get("usage", {})


def save_cached(cache_dir: Path, role: str, report: dict, usage: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{role}.json"
    path.write_text(
        json.dumps(
            {"report": report, "usage": usage},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    scenario_path = ROOT / "scenarios" / "corridor_obstacle.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    store = ExperienceStore(ROOT / "experiences" / "risk_experiences.json")
    retrieved_experiences = [
        build_memory_card(item) for item in store.retrieve(scenario, top_k=3)
    ]
    cache_dir = pipeline_cache_dir(scenario, retrieved_experiences)

    try:
        client = ZhinaoClient()
        advocate_cached = load_cached(cache_dir, "advocate")
        if advocate_cached:
            advocate_report, advocate_usage = advocate_cached
        else:
            advocate_report, advocate_usage = run_advocate(
                scenario, client, retrieved_experiences
            )
            save_cached(
                cache_dir, "advocate", advocate_report, advocate_usage
            )

        critic_cached = load_cached(cache_dir, "critic")
        if critic_cached:
            critic_report, critic_usage = critic_cached
        else:
            critic_report, critic_usage = run_critic(
                scenario, client, retrieved_experiences
            )
            save_cached(cache_dir, "critic", critic_report, critic_usage)

        decision_cached = load_cached(cache_dir, "decision")
        if decision_cached:
            final_decision, decision_usage = decision_cached
        else:
            final_decision, decision_usage = decide(
                scenario,
                advocate_report,
                critic_report,
                client,
                retrieved_experiences,
            )
            save_cached(
                cache_dir, "decision", final_decision, decision_usage
            )
    except LLMError as exc:
        print(f"三 Agent 流程失败，机器人不得执行动作：{exc}")
        return 1

    print_section("Retrieved Risk Memory Cards", {"items": retrieved_experiences})
    print_section("Task Advocate", advocate_report)
    print_section("Risk Critic", critic_report)
    print_section("Safety Decision", final_decision)
    guarded_action = apply_safety_guard(scenario, final_decision)
    print_section("Safety Guard", guarded_action)

    usages = {
        "advocate": advocate_usage,
        "critic": critic_usage,
        "decision": decision_usage,
    }
    total_tokens = sum(int(item.get("total_tokens", 0)) for item in usages.values())
    print_section("Token Usage", {"by_agent": usages, "total_tokens": total_tokens})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
