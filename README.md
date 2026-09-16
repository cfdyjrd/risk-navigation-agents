# Risk Navigation Agents

A risk-aware three-agent decision prototype for indoor mobile robots. The system
runs a Task Advocate, a Risk Critic and a Safety Decision Agent through the 360
Zhinao API. The model reports feed a deterministic constrained optimizer, and the
final action is reviewed once more by a Safety Guard.

For the G1 S1 narrow-corridor hardware experiment — entry point, calibration, the
B0/M/B1 conditions, the low-speed pilot and the full on-site procedure — see
[`docs/S1_narrow_corridor_operation.md`](docs/S1_narrow_corridor_operation.md).

The project currently focuses on two mechanisms:

1. Hierarchical risk memory: L1 Risk Memory Cards generated from past
   experiences, then generalized into reusable L2 Risk Rules.
2. Risk-aware retrieval: a different evidence pack is assembled for each of the
   three roles, based on scenario, risk severity, experience reliability,
   recency and agent role.

## Pipeline

```text
Past experiences (L0)
    ↓ compress, keeping provenance
Risk Memory Card (L1)
    ↓ generalize
Risk Rule (L2)
    ↓ scenario matching and conflict check
Task Advocate + Risk Critic
    ↓
Safety Decision Agent
    ↓
Deterministic constrained optimizer
    ↓
Safety Guard
    ↓
Robot execution layer (platform adapter and audit bridge)
```

L2 rules currently enter all three agents as traceable evidence. If several L2
rules give mutually incompatible advice, a deterministic gate in front of the
pipeline skips the model calls entirely, emits `safe_stop` and requires human
review.

## What each agent does

- **Task Advocate** — finds a workable way to complete the task, preferring
  successful experiences and effective mitigations.
- **Risk Critic** — actively hunts for risks such as collision, getting stuck and
  task failure, preferring failure cases and high-severity experiences.
- **Safety Decision Agent** — makes the final adjudication from the scenario
  facts, the two opposing reports, the L1 cards and the L2 rules.

The Safety Decision Agent's result is a model recommendation and does not reach
the robot execution layer directly. The deterministic constrained optimizer
re-selects a feasible action from task utility, relative risk and relative
uncertainty, and keeps `model_recommendation`, `optimized_action` and
`agrees_with_model` side by side for audit.

## Hardware execution interface

The G1 humanoid now has its own `G1LocoDriver` and `G1Config`, reusing the
execution bridge and the task loop; a restricted straight-line motion path has
been verified. The S1 corridor point-cloud calibration, dynamic-envelope
measurement and A/B low-speed pilot must still be done separately for every
on-site setup; for the on-site SDK checklist, read-only state subscription and
configuration, see [the G1 integration notes](docs/g1_connection.md). G1 does not
use the Go1/Go2 driver, and it refuses to move when the required FSM, pose or
synchronized observations are missing.

`robot_interface.py` defines a minimal platform-independent interface: read a
synchronized observation, execute one approved semantic action, and stop
independently. Before each dispatch, `execution_bridge.py` reads the latest
state, converts it into the existing scenario structure, runs Safety Guard again,
and executes only its `approved_action`. Malformed input, an action that did not
come from the optimizer, invalid Guard output or a platform exception all trigger
a fail-closed stop.

The G1's on-site DDS publishing clock has a stable offset of about 24.85 s from
the development machine. The state collector does not adjust the robot's or the
publisher's clock, and it does not simply add a fixed offset to timestamps;
instead it verifies, per telemetry stream, that source time increases
continuously, that the monotonic clock rate holds, and that the receive interval
and clock offset are stable. Only after those checks pass does it use the local
receive time to judge freshness, while keeping the raw DDS time for audit.
Passing this check still does not by itself authorize motion.

To integrate ROS 2 later, implement `RobotAdapter`: assemble the subscribed
odometry, LiDAR, battery and task state into a `RobotObservation`, and map the
seven semantic actions to navigation goals, velocity control, re-observation,
human requests or emergency stop. That adapter layer should contain no agent or
risk-reasoning logic.

After each execution the bridge produces an L0 record containing the
pre-execution scenario, the approved action, the outcome, risk feedback and
timestamps, which can be written to a persistent experience store through
`experience_sink`.

`unitree_adapter.py` already provides the Go1 high-level UDP driver and the Go2
SDK2 velocity driver, plus device-id verification, observation-freshness checks,
action-parameter clamping, in-motion Guard checks and a latchable software
emergency stop. Go2-1 is a device id; the model is `go2`. Go1 does not use the
reference library's G1 humanoid interface. Sensor subscription and fusion,
network parameters and chassis-level emergency stop still need to be worked out
against the physical robot. The acknowledgement distinguishes "command sent
successfully" from "command actually arrived", and has not yet been verified on
hardware. For configuration, wiring examples and offline tests, see
[the Go1 / Go2-1 integration notes](docs/unitree_connection.md).

Before connecting, run `robot_preflight.py` with the Go1 / Go2-1 configuration
templates under `configs/` to check that the SDK loads, that the local network is
configured, and that the sensor JSON snapshot is valid and updating. The tool
sends no robot control commands and emits a JSON report; a passing preflight does
not mean the robot is connected or cleared to move.

All three agents must explicitly declare the experiences and rules they actually
used, via `cited_experience_ids` and `cited_rule_ids`. Fabricated ids are
rejected in code.

## Environment setup

The core decision path and the offline tests use only the Python standard
library. The hardware adapter process additionally needs the corresponding
Unitree SDK — see the integration notes above. Python 3.10 or newer is
recommended.

Copy the environment template and edit your local `.env`:

```bash
cp .env.example .env
nano .env
```

The local file should contain:

```text
ZHINAO_API_KEY=your_real_api_key
ZHINAO_BASE_URL=https://api.360.cn/v1
ZHINAO_MODEL=z-ai/glm-5.1
```

Load it in every new shell:

```bash
set -a
source .env
set +a
```

Check that the key is loaded (this does not print the key):

```bash
python3 -c 'import os; print("API key loaded" if os.getenv("ZHINAO_API_KEY") else "API key not loaded")'
```

`.env` is already in `.gitignore`. Never put a real API key in code, in this
README, in `.env.example`, or in git.

## API connectivity test

```bash
python3 test_api.py
```

The expected output reports a successful API connection. This command calls the
360 API and consumes tokens.

## Running the full three-agent pipeline

```bash
python3 run_three_agents.py 2>&1 | tee results/latest_three_agents.txt
```

The program does the following:

1. Retrieves an L1 evidence pack for each of the three roles, each constrained to
   an estimated 1800-token budget.
2. Matches the L2 rules that apply to the current scenario and checks for
   conflicts.
3. Runs the three agents if there is no rule conflict.
4. Validates every JSON output, action set, confidence value and cited id.
5. Computes task utility, relative risk and relative uncertainty per action with
   the deterministic constrained optimizer.
6. Excludes actions that violate a hard rule, an L2 prohibition or a numeric
   constraint, and maximizes task utility over what is left.
7. Runs a pre-execution Safety Guard review of the optimized action.

Successful agent reports are cached in `.cache/`. The cache key covers the
scenario, the evidence packs, the rules and the pipeline version, so re-running
with the same inputs avoids duplicate calls and duplicate billing.

## Deterministic constrained action selection

`decision_optimizer.py` implements action selection explicitly as:

```text
maximize task_utility(action)
subject to:
  action passes the hard-safety and L2-prohibition filters
  relative_risk(action) <= 0.35
  relative_uncertainty(action) <= 0.45
```

Task utility is composed of task progress, time cost and energy cost. Relative
risk combines current scenario factors with past memories of the same action;
relative uncertainty combines observation confidence, memory evidence coverage
and the confidence-weighted disagreement among the three agents. If no action
satisfies the constraints, the system falls back conservatively to `safe_stop`.

The risk and uncertainty weights and thresholds are all currently marked
`hand_configured_uncalibrated_baseline`. They are an interpretable engineering
baseline, not learned or probability-calibrated parameters. The outputs therefore
use the explicit name `relative_*_score_not_probability` and must not be read as
collision probabilities or statistical confidence intervals.

## Per-role risk-aware retrieval

Past experiences live in `experiences/risk_experiences.json`. The retrieval score
covers:

- scenario similarity
- risk severity
- experience reliability
- experience recency
- agent-role fit
- a duplicate-evidence penalty

The full score decomposition is kept in `score_breakdown`. Advocate, Critic and
Decision each receive a differently ranked evidence pack.

Inspect retrieval offline:

```bash
python3 run_retrieval.py
```

## L1 Risk Memory Card

`experience_store.py` compresses a full past experience into an L1 card whose
main fields are:

- `memory_id` and the source trajectory
- scenario and trigger conditions
- hazard type and severity
- action, outcome and failure cause
- mitigations and stop conditions
- statistics, reliability and retrieval score
- links to L2 rules

The full raw experience stays in the experience store, so long text is not resent
with every agent request.

## L2 Risk Rule

The formal rule base is `experiences/risk_rules.json`. The current
narrow-corridor rule was generalized from one failure and one success: when total
passage clearance is at most 0.2 m and observation confidence is at most 0.8,
moving straight ahead is prohibited and the robot must re-observe first.

See what the formal rules match:

```bash
python3 run_rule_retrieval.py scenarios/corridor_obstacle.json
```

Verify that the rule does not falsely match a wide corridor:

```bash
python3 run_rule_retrieval.py scenarios/wide_corridor_clear.json
```

`matched_rule_count` should be 0.

## L2 rule-conflict experiment

`experiences/risk_rules_conflict_fixture.json` is a dedicated test fixture and is
not part of the formal rule base. It deliberately includes an overly broad rule
to create a conflict between `observe_again` and `slow_down`:

```bash
python3 run_rule_retrieval.py scenarios/corridor_obstacle.json \
  --rules experiences/risk_rules_conflict_fixture.json
```

Expected result:

```json
{
  "status": "conflict_detected",
  "selected_action": "safe_stop",
  "requires_human_review": true
}
```

On a conflict the system does not arbitrarily pick the more confident learned
rule; it falls back conservatively.

## A/B comparison

Two runs of the same scenario are already recorded:

- `results/baseline_l1_memory.txt` — L1 only.
- `results/l2_rule_integration.txt` — L1 plus L2.

Generate the offline comparison report:

```bash
python3 compare_memory_runs.py
```

The report is written to `results/l1_vs_l2_comparison.md`. In this single
experiment both arms chose `observe_again`; the L2 arm added explicit rule
evidence and decision traceability at a cost of 931 extra tokens (about 6.36%).
A single run cannot show that L2 improves accuracy — that needs repeated runs
across many scenarios.

## Automated tests

Run all offline tests:

```bash
python3 -m unittest discover -v
```

Current coverage:

- L1 experience loading, compression, retrieval, token budget and role
  differentiation
- L2 rule loading, field validation, hits and misses
- Agent rule citation and rejection of fabricated ids
- L2 rule-conflict detection and the deterministic `safe_stop` gate
- Hardware observation contract, post-Guard dispatch, blocking of raw model
  output and L0 record generation
- Safety Guard on emergency obstacles, low confidence, low battery,
  too-narrow passages and illegal actions
- Candidate-action hard-constraint and L2-prohibition filtering
- Task utility, scenario risk, memory risk and the three uncertainty terms
- Constrained optimization, the no-feasible-action fallback and result
  reproducibility
- Offline integration from the three agent reports through the optimizer to
  Safety Guard

These tests make no API calls and incur no model cost.

## Key files

```text
agents/                            Prompts, calls and output validation for the three agents
experiences/risk_experiences.json  L0 past experiences
experiences/risk_rules.json        Formal L2 rule base
experience_store.py                L1 cards and risk-aware retrieval
risk_rule_store.py                 L2 rule loading, matching and conflict handling
run_three_agents.py                Full main pipeline and the deterministic conflict gate
decision_optimizer.py              Safety filtering, scoring and deterministic constrained optimization
safety_guard.py                    Hard safety rules outside the model
run_retrieval.py                   L1 retrieval demo
run_rule_retrieval.py              L2 rule matching demo
compare_memory_runs.py             Offline L1 vs L1+L2 comparison
scenarios/                         Experiment scenarios
results/                           Experiment output and comparison reports
test_*.py                          Offline automated tests
2d-simulator/                      2D topological simulator and benchmark harness (see its README)
3d-simulator/                      3D embodied evaluation system on Isaac Sim (see its README)
docs/method_draft_zh.md            Chinese Method draft matching the current code
```

## Current limits

- Not yet connected to ROS, Nav2 or a real robot SDK.
- L2 rules are currently generalized and checked by hand; automatic clustering
  and updating from large numbers of L1 cards is not implemented.
- Learned rules do not replace Safety Guard's hard safety boundary.
- The experiments are small; they need more scenarios, more repetitions and
  quantitative metrics.
- The weights and thresholds for task utility, relative risk and relative
  uncertainty have not been calibrated on a simulated development set, so the
  current scores cannot be read as real event probabilities.

## Continuing to decide after re-observation

`robot_task_loop.py` provides `RobotTaskLoop`, which connects live observations,
memory retrieval, the three agents, the optimizer and `SafeExecutionBridge`. A
stop requested by `observe_again` is no longer treated as the end of the task:
the loop waits for two fresh observations after the action returns, confirms the
robot has actually come to rest, then updates the scenario and decides again.
Every execution still has the bridge re-read state and pass through the Guard;
the previous forward action is never resumed.

```python
from robot_task_loop import RobotTaskLoop, make_memory_decider

# bridge uses a configured RobotAdapter; store, rule_store and client are existing instances.
loop = RobotTaskLoop(
    bridge,
    make_memory_decider(store, rule_store, client, audit_sink=save_decision_audit),
    is_stationary=measured_stationary,
    goal_reached=verified_goal_reached,
    max_reobservations=3,
    max_steps=30,
    observation_timeout_s=3.0,
    task_timeout_s=120.0,
)
result = loop.run({"goal": "reach the target point past the corridor exit"})
```

`measured_stationary(observation)` must return a boolean based on real motion
feedback such as odometry. `verified_goal_reached(observation, task)` must judge
actual arrival independently and must not use "the command was accepted" as
evidence. `save_decision_audit(record)` stores each run's role evidence, reports
and optimization result. These callbacks come from the on-site sensors and
logging system; with no trustworthy feedback the loop never assumes the robot has
come to rest. Park the robot before entering the loop. Observation timestamps use
the real, timezone-aware acquisition time, so sensors and host must be
synchronized.

By default at most 3 consecutive re-observation requests are issued. Exceeding
that limit, stale frames that will not refresh, a timeout waiting for
rest confirmation, a decision exception, an aborted motion or a logging failure
all trigger an emergency stop and return `fail_closed`. `ask_human` returns
`needs_human` and never invents an answer; a Guard or decision stop returns
`stopped`; only the independent arrival criterion returns `completed`. A fresh
frame only means the data updated — it does not mean the occlusion is gone. The
loop generates no extra exploratory motion and provides neither full route
planning nor a human-dialogue module.

Synchronous perception, model and SDK callbacks must each set their own I/O
timeout. The loop checks its time budget while polling and after a decision
returns, but it cannot force-interrupt an external call that blocks forever;
hardware emergency stop and watchdogs remain the platform's responsibility.

Offline verification:
`python3 -m unittest test_robot_task_loop test_execution_bridge test_unitree_adapter`.
