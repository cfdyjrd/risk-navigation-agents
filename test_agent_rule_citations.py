"""Offline validation for Agent citations of matched L2 rules."""

import json
import unittest
from pathlib import Path

from agents.advocate import validate_report as validate_advocate
from agents.critic import validate_report as validate_critic
from agents.decision import validate_decision
from llm_client import LLMError
from risk_rule_store import RiskRuleStore


ROOT = Path(__file__).resolve().parent


class AgentRuleCitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = json.loads(
            (ROOT / "scenarios" / "corridor_obstacle.json").read_text(
                encoding="utf-8"
            )
        )
        self.rules = RiskRuleStore(
            ROOT / "experiences" / "risk_rules.json"
        ).retrieve_matching(self.scenario)

    def test_advocate_accepts_matched_rule_citation(self) -> None:
        report = {
            "recommendation": "observe_again",
            "proposed_action": {"name": "observe_again", "parameters": {}},
            "expected_benefit": "补充观测",
            "assumptions": [],
            "identified_risks": [],
            "mitigations": [],
            "cited_experience_ids": [],
            "cited_rule_ids": ["rule_narrow_corridor_001"],
            "confidence": 0.8,
        }
        validate_advocate(report, self.scenario, [], self.rules)

    def test_critic_accepts_matched_rule_citation(self) -> None:
        report = {
            "overall_risk": "high",
            "identified_risks": [],
            "missing_information": [],
            "stop_conditions": [],
            "recommendation": "observe_again",
            "cited_experience_ids": [],
            "cited_rule_ids": ["rule_narrow_corridor_001"],
            "confidence": 0.8,
        }
        validate_critic(report, self.scenario, [], self.rules)

    def test_decision_accepts_rule_as_supporting_evidence(self) -> None:
        report = {
            "decision": "observe_again",
            "selected_action": {"name": "observe_again", "parameters": {}},
            "reason": "命中窄通道规则",
            "supporting_evidence": [
                {"source": "rule", "evidence": "低置信度时先重新观测"}
            ],
            "accepted_risks": [],
            "unresolved_risks": [],
            "required_information": [],
            "cited_experience_ids": [],
            "cited_rule_ids": ["rule_narrow_corridor_001"],
            "confidence": 0.9,
        }
        validate_decision(report, self.scenario, [], self.rules)

    def test_rejects_invented_rule_id(self) -> None:
        report = {
            "recommendation": "observe_again",
            "proposed_action": {"name": "observe_again", "parameters": {}},
            "expected_benefit": "补充观测",
            "assumptions": [],
            "identified_risks": [],
            "mitigations": [],
            "cited_experience_ids": [],
            "cited_rule_ids": ["invented_rule"],
            "confidence": 0.8,
        }
        with self.assertRaises(LLMError):
            validate_advocate(report, self.scenario, [], self.rules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
