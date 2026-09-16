# 2D Topological Simulator

`core/` is a 2D topological simulator built from a zone adjacency graph, a
discrete clock and robot state. Stepping the environment is pure set algebra
over the graph — no geometry, no collision, no rendering — so a tick costs
milliseconds; violation adjudication (six classes of authorization-contract
constraints) is LLM-free and unambiguous.

On top of that, this directory wires the simulator into the repository's
three-agent planner (Task Advocate → Risk Critic → Safety Decision →
Safety Guard).

## Layout

```
core/                 Simulation engine: world / episode / contract / violations /
                      planner (compliant-plan search) / state_interface
                      (visualization-decoupling interface)
generator/            Scenario generator: domain templates + phrasing bank,
                      sampler, consistency validation
eval/                 Evaluation and visualization: ground-truth adjudication (gt),
                      metric definitions (metrics), significance testing (stats),
                      episode runner (runner), plotting (plots), floor layout
                      (spatial_layout), HTML replay renderer (html_visualizer),
                      replay CLI (visualize), interactive cockpit (play, manual driving)
scenarios/            Named example scenarios + generated/ (generator batch output)
bridge.py             Bidirectional mapping between episode observations and
                      planner scenarios
run_sim_planner.py    Closed-loop main entry (rule / llm planners, --html for replays)
replays/              Generated replay pages
```

`scenarios/generated`, `scenarios/ladder`, `scenarios/ladder_priority`,
`replays/` and `results/` are build products and are not tracked in git —
regenerate them with the commands below.

## Running

### Setup

- Python 3.11+. The offline parts (tests and the rule planner) have **zero
  third-party dependencies** — nothing to `pip install`.
- Only `--planner llm` needs an API. Export these in your shell session
  (never commit them):

```bash
export ZHINAO_API_KEY='your API key'
export ZHINAO_BASE_URL='https://api.360.cn/v1'   # optional, this is the default
export ZHINAO_MODEL='z-ai/glm-5.1'               # optional, this is the default
```

- You can run from inside this directory, or from the parent with
  `python3 2d-simulator/run_sim_planner.py`; the script resolves paths
  relative to its own location.

### Step 1: the rule planner (no API calls)

`core/planner.py` runs a compliant-plan search that solves for a
zero-violation plan up front and then executes it step by step. Use it to
check that a scenario is solvable and that the engine and metrics work — it
is the oracle upper bound and does not go through the three agents:

```bash
python3 run_sim_planner.py                                              # default: safe delivery scenario
python3 run_sim_planner.py --scenario scenarios/hospital_deliver_unsafe.json
python3 run_sim_planner.py --scenario scenarios/hospital_escort_safe.json
```

Expected output:

```text
scenario fam0001_s0 (safe-clear), task deliver, horizon 9, start lobby, planner=rule
[step 0] zone=lobby action=goto_c1a
[step 1] zone=c1a action=goto_c2a
...
=== Episode Summary ===
{
  "success": true,
  "steps": 5,
  ...
  "brs_final": 1.0
}
```

If the scenario has no solution under its contract, the run prints that the
compliant-plan search found nothing — the task cannot be completed with zero
violations, so the robot issues `safe_stop`. That is correct behaviour, not
an error.

### Step 2: the llm planner (three-agent closed loop, billed)

```bash
python3 run_sim_planner.py --planner llm
python3 run_sim_planner.py --planner llm --scenario scenarios/hospital_deliver_unsafe.json
```

Each decision step calls Task Advocate → Risk Critic → Safety Decision in
sequence (**three model calls per step**; an episode is typically 5–15
decision steps). The final decision then passes through Safety Guard's hard
rules, and the approved action is executed in the simulator. One line is
printed per step:

```text
[step 2] zone=c2a decision=execute model_action=goto_pharmacy2 guard=approved approved=goto_pharmacy2
```

- `decision` — the Safety Decision Agent's adjudication
  (execute / revise_plan / observe_again / ask_human / reject / safe_stop).
- `model_action` vs `approved` — the action the model proposed vs the action
  the Guard actually let through. A difference means a hard rule fired
  (`guard=overridden`).
- Violations produced during execution print as
  `!! violation <cid> (severity n) @ <zone>`.
- The episode terminates early when the planner returns safe_stop or reject.

**Billing and caching.** Each role's successful output is cached under
`.cache/`, keyed by a hash of (scenario content + retrieved experiences +
pipeline version). Re-running after a mid-way failure only retries the roles
that have not succeeded yet, so the same state is never billed twice. Changing
the scenario or the experience store changes the cache key, so stale decisions
are never reused by mistake. To force fresh decisions: `rm -rf .cache`.

If any role ultimately fails (network, quota, malformed output), the program
reports that the three-agent pipeline failed and the robot holds at safe_stop,
then exits with code 1. No action is executed.

### Saving run logs

```bash
python3 run_sim_planner.py --planner llm --out run_log.json
```

`run_log.json` has three sections:

- `decisions` — per decision step: adjudication, rationale, model action,
  guard status and token usage (llm mode).
- `ticks` — per tick: time, zone, whether the zone was newly entered, sensor
  events, violation details. Metrics such as AVR are computed from here.
- `summary` — success, step count, violation details, severity sum,
  `brs_final`, terminal zone.

### Visual replay (`--html`)

Any run can also emit a **self-contained HTML replay page** — one file, no
external dependencies, no server needed; just open it in a browser. This works
in rule mode too, with no API calls:

```bash
python3 run_sim_planner.py --html replay.html
python3 run_sim_planner.py --planner llm --html replay.html
open replay.html        # macOS; or just open it in a browser
```

How it works: `run_sim_planner.py` wraps the engine in an `EpisodeSession`
from `core/state_interface.py` and captures a world snapshot each tick.
`eval/spatial_layout.py` deterministically synthesizes a floor plan from the
zone adjacency graph (corridors form a horizontal spine, rooms sit against the
walls, graph adjacency becomes a shared wall, a shared wall becomes a door,
cross-floor edges become portal badges). `eval/html_visualizer.py` embeds the
frames, decisions and history as JSON in the page and renders it client-side on
a Canvas. The renderer only consumes snapshot fields and never imports engine
internals, so swapping in a 3D engine later requires no visualization changes.

What the page shows:

- **Floor plan** — one canvas per floor, the robot drawn by embodiment
  (🤖 wheeled / 🐕 quadruped), moving tick by tick; the robot pulses red on
  ticks where a violation occurs; carried objects and escorted pedestrians
  follow the robot as it picks up and escorts.
- **Replay controls** — scrub bar, play/pause, ← → to step a tick, space to
  play. Deep links via `#agent=<name>&tick=<n>`, so you can send someone
  "the exact tick where it went wrong".
- **Task progress bar + BRS bar** — task completion and contract retention
  update live per tick; the point where BRS drops below 1.0 is where the
  boundary first gave way.
- **Contract panel** — all six constraint classes listed with live activity;
  constraints being violated on the current tick turn red.
- **Forum feed** — streamed posts light up at their visibility times, so you
  can see exactly what the planner had in front of it when it decided.
- **Decision history and debate transcript** — one record per decision step
  (proposed action → adjudication → action actually executed → ticks consumed
  → violations), clickable to jump to that tick. In llm mode each record also
  carries the three-agent transcript: the Advocate's proposal and confidence,
  the Critic's itemized risks (with probability and severity) and the
  Decision's rationale, plus an extra guardrail row when Safety Guard
  overrode a hard rule (`model action → action actually approved`).
- Light/dark theme follows the system setting.

Rule-mode replays have no debate transcript (each step is a single
"compliant-plan search" line); they are mainly for checking scenario layout
and the planned route. The llm-mode replay is the full "decision process +
consequences" post-mortem. There is also an interactive cockpit (manual
driving, live HTML refreshed each step):
`python3 -m eval.play --scenario <id>`.

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--scenario` | `scenarios/hospital_deliver_safe.json` | Scenario JSON; must contain world / contract_gt / task / forum / meta |
| `--planner` | `rule` | `rule` = offline compliant search; `llm` = three-agent closed loop |
| `--out` | no file | Write decisions / ticks / summary to this JSON |
| `--html` | not generated | Render this run as a self-contained replay page (can be combined with `--out`) |
| `--consequences` | `auto` | Consequence engine: `auto` enables it only for scenarios carrying the ladder fields; `on` / `off` force it |

## How the simulator connects to the planner

Each decision step (`bridge.py`):

1. `build_planning_scenario` compresses the episode's current observation into
   the scenario structure the three agents consume: `task` (goal + progress),
   `robot` (embodiment / location / carried / escorting), `environment` (zone
   graph, human-readable contract view, streamed forum, violations so far) and
   `available_actions` (`goto_<zone>` for adjacent zones plus `hold` /
   `return_to_start` / `observe_again` / `ask_human` / `safe_stop`). Only
   adjacent zones are offered for goto, so the planner advances one hop per
   decision and accepts risk review incrementally.
2. Each of the three agents' outputs passes its own structured validation, and
   the final decision then passes the deterministic hard rules in
   `safety_guard.py` (including "the action must be in available_actions").
3. `to_sim_action` maps `Safety Guard.approved_action` back to a simulator
   action: `goto_<zone>` → goto; `hold` → wait in place for one tick (time
   still passes); `observe_again` → observe (1 tick; `door_open` / `occupied_by`
   of adjacent zones become visible); `ask_human` → ask the task publisher
   (1 tick, one quota per episode; the reply does not appear immediately but
   arrives as a forum post 2 ticks later, and for drift-class posts the reply
   only states *who* posted the original — it gives no adjudication, so you
   must check it against `staff_roster` yourself); `return_to_start` → go back
   to the start; `safe_stop` → terminate the episode. Only `approved_action`
   ever reaches `Episode.execute`. The consequence engine (severity-5 halts,
   severity-4 strands for 2 steps) is controlled by
   `--consequences auto|on|off`; `auto` enables it only when the scenario
   carries the newer ladder fields (`amendments` / `world.staff`), so older
   scenarios behave unchanged.

At the end of an episode the run prints a summary: task success, step count,
violation details and severity sum, and BRS (contract retention rate).

## Scenarios and the generator

Named example scenarios (hand-picked; the defaults for the commands above):

| File | Bucket | Task |
|---|---|---|
| hospital_deliver_safe.json | safe-clear | deliver |
| hospital_deliver_unsafe.json | unsafe-clear | deliver (the forum proposal is unsafe; a compliant alternative route exists) |
| hospital_escort_safe.json | safe-clear | escort |

`scenarios/generated/` holds generator batch output (currently 304 scenarios,
`--n 310 --seeds 1 --seed 42`), with bucket proportions safe-clear /
unsafe-clear / ambiguous-state at 20% each and authorization-drift at 40%
(drift families automatically generate labeled/unlabeled pairs), plus a
`validation_report.json` consistency report. Generation and validation are
deterministic — same arguments, same output — and run from this directory:

```bash
python3 -m generator.build --n 310 --seeds 1 --out scenarios/generated --seed 42
python3 -m generator.validate scenarios/generated
```

The ladder benchmark is generated the same way:

```bash
python3 -m generator.build --ladder --out scenarios/ladder --seed 42
```

Any generated scenario can be fed straight into the closed loop:

```bash
python3 run_sim_planner.py --scenario scenarios/generated/fam0186U_s0.json --html replay.html
```

## Evaluation tools (the `eval` package)

`eval/` keeps the library modules needed for evaluation, used for batch
evaluation of the three-agent closed loop:

- `metrics.py` — all metric definitions (TSR/SSR/VSS/BRS/AVR/UAPR/ORR/pass^k/EC
  plus process-level metrics), aggregated programmatically from episode
  summaries with no LLM judge.
- `gt.py` — ground-truth adjudication: the expected verdict at each decision
  point, taken from the scenario's `expected_adjudication`, for reconciling
  adjudications.
- `stats.py` — significance testing; `plots.py` — plotting (needs matplotlib).
- `runner.py` — batch episode runner (library interface).
- `visualize.py` / `html_visualizer.py` / `spatial_layout.py` — replay
  rendering.
- `play.py` — interactive cockpit (`python3 -m eval.play --scenario <id>`);
  drive manually with commands such as `goto <zone>` / `hold` / `contract` /
  `violations`, with the live HTML refreshing each step.

The original longsafe six-tier baseline agent system, the batch comparison CLI
(run/report/ablate/pilot/rounds) and its unit tests have been removed to keep
this directory small.
