"""交互式驾驶舱(参考 simulator_bddl 的 bddl-sim 交互循环):

    python -m eval.play --scenario fam0408U_s0
    python -m eval.play --scenario fam0408U_s0 --agent full_system   # 可随时 auto 让 agent 接管

进入 REPL 后每执行一步都会覆写同一份 live HTML(浏览器打开后自动轮询刷新;
若浏览器禁用 file:// 轮询则手动刷新)。支持手动驾驶与 agent 托管混合:
你可以先手动把机器人开进歧义区,再 `auto` 看辩论系统如何接管收场。

命令(与 bddl-sim 同风格,动作支持 `goto w101` 与 `goto(w101)` 两种写法):
  goto <zone> / hold / return   手动动作
  auto [n]                      让 --agent 指定的 agent 决策 n 步(默认 1),打印辩论
  zones / posts / contract / obs / violations / summary   查看状态
  json                          打印当前帧 JSON
  browser                       (重新)在浏览器打开 HTML
  reset                         重置 episode
  help / quit
"""
from __future__ import annotations

import argparse
import json
import random
import re
import webbrowser
from pathlib import Path

from core.state_interface import EpisodeSession, decision_view

from .gt import GTAdjudicator
from .html_visualizer import HTMLVisualizer
from .util import load_scenarios

_ACT = re.compile(r"^(goto|g)\s*[( ]\s*([\w]+)\s*\)?$")


class PlaySession:
    """episode + 可视化 + 可选 agent 的组合,REPL 与测试共用。"""

    def __init__(self, scenario: dict, out_dir: str = "html_outputs",
                 agent_name: str | None = None, llm_kind: str = "scripted",
                 seed: int = 0, stream_forum: bool = True, consequences: bool | None = None):
        self.scenario = scenario
        # consequences=None:场景带阶梯新字段才启用后果引擎(旧场景关闭)
        self.sess = EpisodeSession(scenario, stream_forum=stream_forum, consequences=consequences)
        self.gt = GTAdjudicator(scenario)
        self.viz = HTMLVisualizer(scenario, live=True)
        sid = scenario["meta"]["scenario_id"]
        self.html_path = Path(out_dir) / f"{sid}_play.html"
        self.agent = None
        self._seed = seed
        self._agent_name = agent_name
        self._llm_kind = llm_kind
        if agent_name:
            self._make_agent()
        self.render()

    def _make_agent(self):
        # baseline agent 体系已移除:驾驶舱仅支持手动驾驶,不再提供 agent 托管
        try:
            from baseline_agents.registry import make_agent
        except ImportError as exc:
            raise SystemExit("agent 托管不可用(baseline_agents 已移除),"
                             "请去掉 --agent 参数用手动驾驶") from exc
        self.agent, _ = make_agent(self._agent_name, llm_kind=self._llm_kind)
        rng = random.Random(f"{self.scenario['meta']['scenario_id']}|play|{self._seed}")
        self.agent.begin_episode(self.sess.ep.static_observation(), rng,
                                 contract=self.scenario["contract_gt"],
                                 scenario=self.scenario)

    # ------------------------------------------------------------- 动作
    def manual(self, action: dict) -> list[dict]:
        label = {"goto": f"goto {action.get('zone')}", "hold": "hold",
                 "return": "return"}[action["type"]]
        exp = self.gt.expected_verdict(action.get("zone"), self.sess.ep.state.step + 1)
        entries = self.sess.act(action)
        self.sess.history.append({
            "step": entries[0]["step"] - 1 if entries else self.sess.ep.state.step,
            "label": f"[手动] {label}", "verdict": "accept", "expected": exp,
            "ok": None, "hops": len(entries),
            "viols": sum(len(e["violations"]) for e in entries),
            "blocked": bool(entries and not entries[-1]["entered"]
                            and action["type"] == "goto"
                            and action.get("zone") != self.sess.ep.state.zone),
        })
        self.render()
        return entries

    def auto(self) -> tuple[dict, list[dict]]:
        assert self.agent, "未指定 --agent"
        obs = self.sess.ep.observation()
        dec = self.agent.decide(obs)
        exp = self.gt.expected_verdict(dec.proposal_target, obs["step"] + 1)
        dv = decision_view({"obs": {"step": obs["step"]}, "decision": dec.to_dict(),
                            "expected": exp})
        self.sess.add_decision(dv)
        entries = self.sess.act(dec.action)
        act = dec.action
        label = {"goto": f"goto {act.get('zone')}", "hold": "hold",
                 "return": "return"}.get(act.get("type"), str(act))
        self.sess.history.append({
            "step": dv["at"], "label": f"[{self.agent.name}] {label}",
            "verdict": dv["verdict"], "expected": exp, "ok": dv["ok"],
            "hops": len(entries), "viols": sum(len(e["violations"]) for e in entries),
            "blocked": False,
        })
        self.render()
        return dv, entries

    def reset(self):
        self.sess.reset()
        if self._agent_name:
            self._make_agent()
        self.render()

    def render(self):
        self.viz.render_to_file(self.html_path, {"live": {
            "frames": self.sess.frames, "decisions": self.sess.decisions,
            "history": self.sess.history,
            "summary": {"success": self.sess.ep.success, "steps": self.sess.ep.state.step,
                        "violations": len(self.sess.ep.violations),
                        "vss": sum(v.severity for v in self.sess.ep.violations),
                        "mismatches": 0},
        }})


def _print_entries(entries):
    for e in entries:
        mark = "❌" if e["violations"] else "✅"
        vio = " ".join(f"{v['cid']}(sev{v['severity']})" for v in e["violations"])
        print(f"  {mark} step{e['step']:>3} {e['time']} @ {e['zone']}"
              + (f"  [{vio}]" if vio else "")
              + ("  📸传感事件" if e.get("sensor_event") else ""))


def interactive_loop(ps: PlaySession, auto_open: bool = True):
    ep = ps.sess.ep
    zones = {z.id: z.name for z in ep.world.zones.values()}
    print(f"\nlive HTML: {ps.html_path}")
    if auto_open:
        webbrowser.open(f"file://{ps.html_path.resolve()}")
        print("已在浏览器打开(每步自动覆写;若未自动刷新请手动刷新)")
    print(f"任务:{ps.viz._payload_static['task']['text']}"
          f" | horizon {ep.horizon} | 输入 help 看命令\n")
    while True:
        state = "✅完成" if ep.success else ("⏰超时" if ep.done else "")
        try:
            raw = input(f"[step {ep.state.step}/{ep.horizon} @ {ep.state.zone}{state}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        cmd = raw.lower()
        m = _ACT.match(cmd)
        if m:
            z = m.group(2)
            if z not in zones:
                print(f"  未知 zone:{z}(zones 命令查看全部)")
                continue
            _print_entries(ps.manual({"type": "goto", "zone": z}))
        elif cmd in ("hold", "h"):
            _print_entries(ps.manual({"type": "hold"}))
        elif cmd in ("return", "ret"):
            _print_entries(ps.manual({"type": "return"}))
        elif cmd.startswith("auto"):
            if not ps.agent:
                print("  未指定 --agent,无法托管")
                continue
            n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 1
            for _ in range(n):
                if ep.done:
                    break
                dv, entries = ps.auto()
                print(f"  ⚖️ 提议 {dv['proposal']} -> {dv['verdict']}"
                      f"(期望 {dv['expected']}{'' if dv['ok'] else ' ✗'})"
                      f" {dv['rounds']}轮/{dv['llm_calls']}调用")
                for t in dv["transcript"]:
                    print(f"     {t['role']:<11} {t['text'][:88]}")
                _print_entries(entries)
        elif cmd == "zones":
            for zid, name in sorted(zones.items()):
                z = ep.world.zones[zid]
                print(f"  {zid:<12} {name:<10} {z.kind:<10} F{z.attrs.get('floor', 1)}"
                      f"  {'restricted' if z.attrs.get('restricted') else ''}")
        elif cmd == "posts":
            for p in ep.visible_posts():
                print(f"  step{p['step']:>3} {p['author']}: {p['text']}")
        elif cmd == "contract":
            for c in ps.viz._payload_static["contract"]:
                mark = "❌" if c["cid"] in ep.violated_cids else "  "
                print(f"  {mark}{c['cid']:<13} sev{c['sev']}  {c['text']}")
        elif cmd == "obs":
            print(json.dumps(ep.observation(), ensure_ascii=False, indent=2))
        elif cmd == "violations":
            for v in ep.violations:
                print(f"  step{v.step:>3} {v.cid} sev{v.severity} @ {v.zone}  {v.detail}")
            if not ep.violations:
                print("  无违规 ✨")
        elif cmd == "summary":
            print(f"  success={ep.success} steps={ep.state.step} "
                  f"violations={len(ep.violations)} vss={sum(v.severity for v in ep.violations)}"
                  f" BRS={ep.brs_curve[-1]:.3f}")
        elif cmd == "json":
            print(json.dumps(ps.sess.get_frame(), ensure_ascii=False, indent=2))
        elif cmd == "browser":
            webbrowser.open(f"file://{ps.html_path.resolve()}")
        elif cmd == "reset":
            ps.reset()
            ep = ps.sess.ep
            print("  已重置")
        elif cmd in ("help", "?"):
            print(__doc__.split("命令")[1])
        elif cmd in ("quit", "q", "exit"):
            break
        else:
            print("  未知命令(help 查看)")
    print(f"\n最终:success={ep.success} steps={ep.state.step} "
          f"violations={len(ep.violations)} -> {ps.html_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="scenario_id,如 fam0408U_s0")
    ap.add_argument("--scenarios-dir", default="scenarios/")
    ap.add_argument("--agent", default=None, help="auto 托管用的 agent 名(如 full_system)")
    ap.add_argument("--llm", default="scripted", choices=["scripted", "api"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="html_outputs")
    ap.add_argument("--stream-forum", default="on", choices=["on", "off"])
    ap.add_argument("--consequences", default="auto", choices=["auto", "on", "off"],
                    help="后果引擎(拦停/滞留):auto=场景带阶梯新字段才启用")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    fp = Path(args.scenarios_dir) / f"{args.scenario}.json"
    if fp.exists():
        sc = json.loads(fp.read_text())
    else:  # 容错:按 id 搜
        matches = [s for s in load_scenarios(args.scenarios_dir)
                   if s["meta"]["scenario_id"] == args.scenario]
        if not matches:
            raise SystemExit(f"scenario {args.scenario} not found in {args.scenarios_dir}")
        sc = matches[0]
    ps = PlaySession(sc, out_dir=args.out, agent_name=args.agent, llm_kind=args.llm,
                     seed=args.seed, stream_forum=args.stream_forum == "on",
                     consequences={"auto": None, "on": True, "off": False}[args.consequences])
    interactive_loop(ps, auto_open=not args.no_browser)


if __name__ == "__main__":
    main()
