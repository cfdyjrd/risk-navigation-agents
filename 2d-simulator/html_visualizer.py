"""楼层平面图 HTML 可视化(参考 simulator_bddl/utils/html_visualizer.py 的实现方式)。

与该参考实现相同的骨架:
  - 可视化与仿真逻辑完全解耦,只消费 core.state_interface 产出的帧快照;
  - 状态 JSON 内嵌 <script type="application/json" id="state-data">,客户端 Canvas 渲染;
  - 每层楼一个 <canvas> 平面图(zone=房间色块+emoji,门/连线/跨层传送门);
  - 状态栏(任务进度条 + BRS 条 + 携带/护送/相机)、契约面板(逐条约束活性着色,
    对应参考实现的 predicates panel)、Forum 流、辩论转录、违规日志、决策历史;
  - live 模式用 renderId 轮询做免刷新更新(file:// 下浏览器若禁 fetch 则手动刷新)。

replay 模式把全部 tick 帧与多 agent 数据一次性内嵌,支持拖动回放与 agent 对比;
live 模式(eval/play.py)每步覆写同一文件。完全自包含,零外部依赖。
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

from core.state_interface import contract_views

from spatial_layout import layout_world

KIND_EMOJI = {"corridor": "", "ward": "🛏️", "icu": "🏥", "pharmacy": "💊",
              "stair": "🪜", "checkpoint": "🛂", "retail": "🛍️", "gate": "🛫",
              "vip": "🛋️", "lobby": "🏛️"}
ROBOT_EMOJI = {"wheeled": "🤖", "quadruped": "🐕"}


def _task_text(task: dict, names: dict) -> str:
    nm = lambda z: names.get(z, z)  # noqa: E731
    if task["type"] == "deliver":
        return f"把「{task.get('object_name', task['object'])}」从 {nm(task['pickup_zone'])} 送到 {nm(task['dropoff_zone'])}"
    if task["type"] == "visit":
        return "巡视 " + "、".join(nm(z) for z in task["targets"])
    if task["type"] == "escort":
        return f"把「{task.get('human_name', task['human'])}」从 {nm(task['from_zone'])} 护送到 {nm(task['to_zone'])}"
    return str(task)


class HTMLVisualizer:
    """scenario + agent 帧数据 -> 自包含 HTML。"""

    def __init__(self, scenario: dict, live: bool = False):
        self.scenario = scenario
        self.live = live
        self._payload_static = self._build_static()

    # ------------------------------------------------------------- payload
    def _build_static(self) -> dict:
        sc = self.scenario
        w, contract, ann = sc["world"], sc["contract_gt"], sc.get("annotations", {})
        names = {z["id"]: z.get("name", z["id"]) for z in w["zones"]}
        task = dict(sc["task"])
        for o in w.get("objects", []):
            if o["id"] == task.get("object"):
                task["object_name"] = o.get("name", o["id"])
        for h in w.get("humans", []):
            if h["id"] == task.get("human"):
                task["human_name"] = h.get("name", h["id"])

        allow, deny = set(), set()
        for e in contract.get("spatial", []):
            (allow if e["rule"] == "allow" else deny).update(e["zones"])
        temporal_zones, windows = set(), []
        for e in contract.get("temporal", []):
            if e["rule"] == "deny":
                temporal_zones.update(e.get("zones", []))
                windows.append({"a": e["window"][0], "b": e["window"][1]})

        meta_keys = ("scenario_id", "bucket", "domain", "horizon", "drift_type",
                     "drift_target_labeled", "drift_inject_step", "n_constraints",
                     "embodiment", "long_horizon")
        return {
            "mode": "live" if self.live else "replay",
            "meta": {**{k: sc["meta"].get(k) for k in meta_keys},
                     "clock": w["clock"], "start": w["start_zone"]},
            "task": {**task, "text": _task_text(task, names)},
            "zones": [{"id": z["id"], "name": z.get("name", z["id"]), "kind": z["kind"],
                       "floor": z["attrs"].get("floor", 1),
                       "restricted": bool(z["attrs"].get("restricted")),
                       "door_open": bool(z["attrs"].get("door_open")),
                       "occupied": z["attrs"].get("occupied_by"),
                       "gives": z["attrs"].get("gives_object")}
                      for z in w["zones"]],
            "zstatus": {"allow": sorted(allow), "deny": sorted(deny),
                        "drift": sorted(ann.get("drift_zones", [])),
                        "hazard": sorted(ann.get("hazard_zones", [])),
                        "temporal": sorted(temporal_zones), "windows": windows,
                        "visit": task.get("targets", []) if task["type"] == "visit" else []},
            "layout": layout_world(w),
            "posts": [{"step": p["step"], "author": p["author"], "text": p["text"],
                       "role": p.get("role", "noise")} for p in sc["forum"]["posts"]],
            "contract": contract_views(contract, w["zones"]),
        }

    def render(self, agents: dict[str, dict]) -> str:
        """agents: {name: {frames, decisions, history, summary}}(state_interface 产物)。"""
        if not agents:
            raise ValueError("no agent data to render")
        payload = dict(self._payload_static)
        payload["agents"] = agents
        payload["renderId"] = sum(len(a["frames"]) for a in agents.values())
        data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
        sid = _html.escape(payload["meta"]["scenario_id"] or "episode")
        return (TEMPLATE
                .replace("__SID__", sid)
                .replace("__MODE__", payload["mode"])
                .replace("__DATA__", data))

    def render_to_file(self, path: str | Path, agents: dict[str, dict]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(agents), encoding="utf-8")
        return p


# ==================================================================== 页面模板
TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LongSafe 平面复盘台 · __SID__</title>
<style>
:root{
  --bg1:#f2f5f4; --bg2:#e8edf1; --surface:#ffffff; --ink:#1d2724; --ink2:#5f6f69;
  --line:#dee5e1; --chip:#eef1ef; --accent:#0f766e; --accent-soft:#e0efec;
  --ok:#2e7d32; --ok-soft:#e6f2e7; --bad:#c62828; --bad-soft:#fbe9e8;
  --warn:#9a6700; --warn-soft:#fdf3dc; --drift:#6d4fc4; --drift-soft:#efeafb;
  --door:#b08a5a; --robot:#0f766e; --grid:#00000010;
  --zk-corridor:#eceff1; --zk-ward:#e3edf8; --zk-gate:#e3edf8; --zk-icu:#fbe6ee;
  --zk-pharmacy:#e6f2e7; --zk-stair:#efe9e4; --zk-checkpoint:#fdeedd;
  --zk-retail:#f3e8f7; --zk-vip:#fdf6dd; --zk-lobby:#e0f4f6;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg1:#10161a; --bg2:#151d23; --surface:#1a2228; --ink:#e4ebe7; --ink2:#93a39c;
    --line:#2b353b; --chip:#222c31; --accent:#4cc4b5; --accent-soft:#173c37;
    --ok:#7cc47f; --ok-soft:#1c3320; --bad:#ef6a5f; --bad-soft:#3a201d;
    --warn:#d9a521; --warn-soft:#37301a; --drift:#a68df0; --drift-soft:#2a2342;
    --door:#a3805a; --robot:#4cc4b5; --grid:#ffffff12;
    --zk-corridor:#242d33; --zk-ward:#20303f; --zk-gate:#20303f; --zk-icu:#3a2530;
    --zk-pharmacy:#213326; --zk-stair:#2f2a25; --zk-checkpoint:#37301f;
    --zk-retail:#2e2537; --zk-vip:#36301c; --zk-lobby:#1d3336;
  }
}
:root[data-theme="dark"]{
  --bg1:#10161a; --bg2:#151d23; --surface:#1a2228; --ink:#e4ebe7; --ink2:#93a39c;
  --line:#2b353b; --chip:#222c31; --accent:#4cc4b5; --accent-soft:#173c37;
  --ok:#7cc47f; --ok-soft:#1c3320; --bad:#ef6a5f; --bad-soft:#3a201d;
  --warn:#d9a521; --warn-soft:#37301a; --drift:#a68df0; --drift-soft:#2a2342;
  --door:#a3805a; --robot:#4cc4b5; --grid:#ffffff12;
  --zk-corridor:#242d33; --zk-ward:#20303f; --zk-gate:#20303f; --zk-icu:#3a2530;
  --zk-pharmacy:#213326; --zk-stair:#2f2a25; --zk-checkpoint:#37301f;
  --zk-retail:#2e2537; --zk-vip:#36301c; --zk-lobby:#1d3336;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:linear-gradient(150deg,var(--bg1),var(--bg2));color:var(--ink);
  font:14px/1.55 "Avenir Next","PingFang SC","Noto Sans CJK SC",system-ui,sans-serif;
  min-height:100vh;padding:18px}
.wrap{max-width:1280px;margin:0 auto}
header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:8px}
h1{font-size:20px;letter-spacing:.01em}
h1 small{color:var(--ink2);font-weight:500;font-size:13px;margin-left:6px}
.livebadge{background:var(--bad);color:#fff;border-radius:999px;font-size:11px;
  font-weight:700;padding:2px 10px;letter-spacing:.08em;animation:blink 1.6s infinite}
@keyframes blink{50%{opacity:.55}}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 12px}
.chip{background:var(--chip);border:1px solid var(--line);border-radius:999px;
  padding:2px 10px;font-size:12px;color:var(--ink2)}
.chip b{color:var(--ink);font-weight:600}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:14px;box-shadow:0 2px 8px #0000000f}
.statusbar{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;
  padding:10px 14px;margin-bottom:14px}
.sitem{display:flex;align-items:center;gap:7px;font-size:13px;white-space:nowrap}
.sitem b{font-variant-numeric:tabular-nums}
.sitem .lab{color:var(--ink2);font-size:12px}
.bar{width:120px;height:12px;background:var(--chip);border-radius:6px;overflow:hidden;
  border:1px solid var(--line)}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#6fd0c4);
  transition:width .25s}
.bar.brs i{background:linear-gradient(90deg,#6fd0c4,var(--ok))}
.bar.low i{background:linear-gradient(90deg,var(--warn),var(--bad))}
.time-alert{color:var(--bad);font-weight:700}
.win{background:var(--accent-soft);color:var(--accent);border-radius:999px;
  padding:1px 10px;font-weight:700;font-size:12px}
.cols{display:grid;grid-template-columns:minmax(430px,1fr) minmax(320px,430px);gap:14px}
@media(max-width:960px){.cols{grid-template-columns:1fr}}
.mapcard{position:sticky;top:12px;align-self:start}
.tabs{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}
.tab{border:1px solid var(--line);background:var(--surface);color:var(--ink2);
  border-radius:9px;padding:5px 12px;font-size:13px;cursor:pointer;display:flex;
  align-items:center;gap:6px}
.tab.on{border-color:var(--accent);color:var(--accent);background:var(--accent-soft);
  font-weight:600}
.tab .mini{font-size:11px;border-radius:6px;padding:0 5px}
.tab .mini.good{background:var(--ok-soft);color:var(--ok)}
.tab .mini.badm{background:var(--bad-soft);color:var(--bad)}
.floors{overflow-x:auto;display:flex;flex-direction:column;gap:10px}
.floor h4{font-size:12px;color:var(--ink2);letter-spacing:.08em;margin:0 0 4px}
canvas{display:block;border-radius:10px}
.controls{display:flex;align-items:center;gap:10px;margin-top:10px}
.controls button{border:1px solid var(--line);background:var(--surface);color:var(--ink);
  border-radius:8px;width:36px;height:30px;cursor:pointer;font-size:13px}
.controls button:focus-visible,.tab:focus-visible{outline:2px solid var(--accent)}
.controls input[type=range]{flex:1;accent-color:var(--accent)}
.ticklab{font-variant-numeric:tabular-nums;color:var(--ink2);font-size:12px;
  min-width:150px;text-align:right}
.legend{display:flex;flex-wrap:wrap;gap:10px 14px;font-size:11.5px;color:var(--ink2);
  margin-top:10px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block;border:1px solid var(--line)}
.panel{margin-bottom:13px}
.panel h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);
  margin:0 0 8px;font-weight:600;display:flex;gap:6px;align-items:center}
.tasktxt{font-size:13.5px;margin-bottom:6px}
.sub{display:flex;gap:6px;flex-wrap:wrap}
.sub span{font-size:12px;background:var(--chip);border-radius:8px;padding:2px 8px;
  color:var(--ink2)}
.sub span.done{background:var(--ok-soft);color:var(--ok);font-weight:600}
.ct-group{margin:7px 0}
.ct-cat{font-size:11px;color:var(--ink2);letter-spacing:.06em;margin-bottom:3px}
.ct{display:flex;align-items:flex-start;gap:7px;font-size:12.5px;padding:4px 8px;
  border-radius:8px;border:1px solid var(--line);margin:3px 0;background:var(--surface)}
.ct.violated{border-color:var(--bad);background:var(--bad-soft)}
.ct .rule{font-size:11px;font-weight:700;border-radius:6px;padding:0 6px;flex:0 0 auto;
  margin-top:1px}
.ct .rule.allow{background:var(--ok-soft);color:var(--ok)}
.ct .rule.deny{background:var(--bad-soft);color:var(--bad)}
.ct .sev{margin-left:auto;color:var(--warn);font-size:10px;letter-spacing:1px;flex:0 0 auto}
.ct.violated .cid{color:var(--bad);font-weight:700}
.post{border-left:3px solid var(--line);padding:4px 10px;margin:6px 0;font-size:13px;
  color:var(--ink2);border-radius:0 8px 8px 0}
.post.vis{color:var(--ink);background:var(--chip)}
.post .who{font-size:11px;color:var(--ink2)}
.post.origin{border-color:var(--accent)} .post.drift{border-color:var(--drift)}
.post.hazard{border-color:var(--bad)} .post.reinforce{border-color:var(--warn)}
.post .tag{font-size:10px;border-radius:4px;padding:0 5px;margin-left:6px;background:var(--chip)}
.post.drift .tag{background:var(--drift-soft);color:var(--drift)}
.post.hazard .tag{background:var(--bad-soft);color:var(--bad)}
.msg{display:flex;gap:8px;font-size:12.5px;margin:5px 0}
.msg .role{flex:0 0 74px;font-size:11px;color:var(--ink2);text-align:right;padding-top:2px}
.msg.planner .role{color:var(--accent)} .msg.critic .role{color:var(--bad)}
.msg.supporter .role{color:var(--ok)} .msg.adjudicator .role{color:var(--drift)}
.msg.guardrail .role{color:var(--warn)}
.verdict{display:inline-block;border-radius:6px;padding:1px 8px;font-size:12px;font-weight:700}
.verdict.accept{background:var(--ok-soft);color:var(--ok)}
.verdict.reject{background:var(--bad-soft);color:var(--bad)}
.verdict.rewrite{background:var(--drift-soft);color:var(--drift)}
.verdict.fallback{background:var(--warn-soft);color:var(--warn)}
.expchip{font-size:11px;color:var(--ink2)}
.expchip.ok{color:var(--ok)} .expchip.no{color:var(--bad);font-weight:700}
.vio{font-size:12.5px;color:var(--bad);margin:4px 0;font-variant-numeric:tabular-nums}
.empty{color:var(--ink2);font-size:13px}
.hist{max-height:230px;overflow-y:auto;font-variant-numeric:tabular-nums}
.hentry{display:flex;gap:10px;align-items:center;font-size:12.5px;padding:5px 10px;
  border-radius:8px;margin:3px 0;border-left:3px solid var(--ok);background:var(--ok-soft);
  cursor:pointer}
.hentry.badh{border-left-color:var(--bad);background:var(--bad-soft)}
.hentry.holdh{border-left-color:var(--line);background:var(--chip)}
.hentry .st{color:var(--ink2);flex:0 0 46px}
.hentry .lb{flex:1}
.hentry:hover{filter:brightness(.97)}
footer{margin-top:18px;color:var(--ink2);font-size:12px}
footer code{background:var(--chip);padding:1px 6px;border-radius:5px;
  font:12px ui-monospace,monospace}
#tooltip{position:fixed;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;padding:8px 12px;font-size:12px;box-shadow:0 6px 18px #00000030;
  pointer-events:none;z-index:100;max-width:260px;display:none}
#tooltip b{font-size:13px}
#tooltip .k{color:var(--ink2)}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>🗺️ LongSafe 平面复盘台 <small>__SID__</small></h1>
  <span class="livebadge" id="livebadge" style="display:none">LIVE</span>
</header>
<div class="chips" id="chips"></div>
<div class="card statusbar" id="statusbar"></div>
<div class="cols">
  <div class="card mapcard">
    <div class="tabs" id="tabs" role="tablist"></div>
    <div class="floors" id="floors"></div>
    <div class="controls">
      <button id="play" aria-label="play">▶</button>
      <input type="range" id="scrub" min="0" value="0" step="1" aria-label="tick">
      <span class="ticklab" id="ticklab"></span>
    </div>
    <div class="legend" id="legend"></div>
  </div>
  <div class="card">
    <div class="panel"><h3>🎯 任务</h3><div id="task"></div></div>
    <div class="panel"><h3>📜 授权契约(实时活性)</h3><div id="contract"></div></div>
    <div class="panel"><h3>💬 Forum(流式,当前步可见)</h3><div id="posts"></div></div>
    <div class="panel"><h3>⚖️ 本步决策与辩论</h3><div id="decision" aria-live="polite"></div></div>
    <div class="panel"><h3>🚨 违规日志(累计)</h3><div id="vios"></div></div>
  </div>
</div>
<div class="card" style="margin-top:14px">
  <div class="panel" style="margin:0"><h3>📝 决策历史(点击跳转)</h3>
  <div class="hist" id="hist"></div></div>
</div>
<footer>由 <code>python -m eval.visualize</code>(回放)/ <code>python -m eval.play</code>(交互)生成
 · 完整轨迹在 results/&lt;tag&gt;/trajectories/ · live 模式下若浏览器禁用本地轮询请手动刷新</footer>
</div>
<div id="tooltip"></div>
<script type="application/json" id="state-data">__DATA__</script>
<script>
"use strict";
let P = JSON.parse(document.getElementById("state-data").textContent);
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const KIND_EMOJI = {corridor:"",ward:"🛏️",icu:"🏥",pharmacy:"💊",stair:"🪜",
  checkpoint:"🛂",retail:"🛍️",gate:"🛫",vip:"🛋️",lobby:"🏛️"};
const ROBOT = {wheeled:"🤖",quadruped:"🐕"}[P.meta.embodiment] || "🤖";
const VNAME = {accept:"accept",reject:"reject",rewrite:"rewrite",fallback:"fallback"};

let agents = Object.keys(P.agents);
let cur = agents[agents.length-1];
let idx = P.mode === "live" ? frames().length-1 : 0;
let timer = null;

function frames(){ return P.agents[cur].frames; }
function zinfo(){ const m={}; P.zones.forEach(z=>m[z.id]=z); return m; }
let Z = zinfo();
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---------------- meta chips / legend(静态)
function renderChips(){
  const m = P.meta;
  $("chips").innerHTML = [
    `场景 <b>${esc(m.scenario_id)}</b>`, `bucket <b>${esc(m.bucket)}</b>`,
    `域 <b>${esc(m.domain)}</b>`, `本体 <b>${ROBOT} ${esc(m.embodiment)}</b>`,
    `horizon <b>${m.horizon}</b>`,
    m.drift_type ? `drift <b>${esc(m.drift_type)}</b>` : "",
    m.drift_type ? `labeled <b>${m.drift_target_labeled}</b>` : "",
    m.drift_inject_step != null ? `注入step <b>${m.drift_inject_step}</b>` : "",
    `约束 <b>${m.n_constraints}</b> 条`,
  ].filter(Boolean).map(s=>`<span class="chip">${s}</span>`).join("");
  $("legend").innerHTML = [
    ["var(--ok)","契约允许集(绿框)"],["var(--bad)","禁区/受限(红框)"],
    ["var(--drift)","漂移目标(紫虚线)"],["var(--door)","门(共墙开口)"],
  ].map(([c,t])=>`<span><span class="sw" style="border-color:${c};border-width:2px"></span>${t}</span>`).join("")
  + `<span>${ROBOT} 机器人</span><span>📦 物品</span><span>🧑 行人</span>`
  + `<span>🎯 送达点</span><span>📍 巡视点</span><span>🚩 护送终点</span>`
  + `<span>🐾 仅四足通行</span><span>↕ 跨层通道</span>`;
  if (P.mode === "live") $("livebadge").style.display = "";
}

// ---------------- tabs
function renderTabs(){
  const t = $("tabs"); t.innerHTML = "";
  agents.forEach(a=>{
    const s = P.agents[a].summary || {};
    const b = document.createElement("button");
    b.className = "tab" + (a===cur?" on":""); b.setAttribute("role","tab");
    b.innerHTML = `${esc(a)} <span class="mini ${s.violations? "badm":"good"}">`
      + `${s.success?"✓":"✗"}·${s.violations||0}违规</span>`;
    b.onclick = ()=>{ cur=a; idx=Math.min(idx,frames().length-1); renderTabs(); render(); };
    t.appendChild(b);
  });
}

// ---------------- 楼层画布
const PAD = 12;
function cellOf(){
  const avail = Math.max(360, ($("floors").clientWidth || 880) - 2*PAD - 6);
  return Math.max(18, Math.min(34, Math.floor(avail / P.layout.width)));
}
function buildFloors(){
  const box = $("floors"); box.innerHTML = "";
  const cell = cellOf();
  [...P.layout.floors].sort((a,b)=>b.floor-a.floor).forEach(fl=>{
    const div = document.createElement("div"); div.className = "floor";
    div.innerHTML = `<h4>📍 ${fl.floor}F</h4>`;
    const cv = document.createElement("canvas");
    cv.dataset.floor = fl.floor;
    const wpx = fl.width*cell+2*PAD, hpx = fl.height*cell+2*PAD;
    const dpr = window.devicePixelRatio || 1;
    cv.width = wpx*dpr; cv.height = hpx*dpr;
    cv.style.width = wpx+"px"; cv.style.height = hpx+"px";
    div.appendChild(cv); box.appendChild(div);
    cv.addEventListener("mousemove", e=>hover(e,cv,fl,cell));
    cv.addEventListener("mouseleave", ()=>{$("tooltip").style.display="none";});
  });
}
function rectPx(r, cell){ return [PAD+r.x*cell, PAD+r.y*cell, r.w*cell, r.h*cell]; }
function zoneRect(fl, zid){ return fl.zones.find(z=>z.id===zid); }
function centerPx(zid, cell){
  for (const fl of P.layout.floors){
    const r = zoneRect(fl, zid);
    if (r) return {fl:fl.floor, x:PAD+(r.x+r.w/2)*cell, y:PAD+(r.y+r.h/2)*cell};
  }
  return null;
}
function roundRect(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
}
function violZones(upTo){
  const s = new Set();
  frames().slice(0, upTo+1).forEach(f=>f.vio.forEach(v=>s.add(v.zone)));
  return s;
}
function drawFloor(cv, fl, frame, cell){
  const ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const W = fl.width*cell+2*PAD, H = fl.height*cell+2*PAD;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = cssVar("--chip"); roundRect(ctx,0,0,W,H,10); ctx.fill();
  // 网格点
  ctx.fillStyle = cssVar("--grid");
  for (let gx=0; gx<=fl.width; gx++) for (let gy=0; gy<=fl.height; gy++)
    ctx.fillRect(PAD+gx*cell-.5, PAD+gy*cell-.5, 1, 1);
  // 同层连线(不共墙的边)
  ctx.strokeStyle = cssVar("--line"); ctx.lineWidth = 1.4;
  fl.links.forEach(l=>{
    const a = zoneRect(fl,l.a), b = zoneRect(fl,l.b);
    if (!a || !b) return;
    const [ax,ay] = [PAD+(a.x+a.w/2)*cell, PAD+(a.y+a.h/2)*cell];
    const [bx,by] = [PAD+(b.x+b.w/2)*cell, PAD+(b.y+b.h/2)*cell];
    ctx.setLineDash(l.narrow ? [3,3] : [6,4]);
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();
  });
  ctx.setLineDash([]);
  const vz = violZones(idx);
  // zone 房间
  fl.zones.forEach(zr=>{
    const z = Z[zr.id]; if (!z) return;
    const [x,y,w,h] = rectPx(zr, cell);
    const st = P.zstatus;
    const isDeny = st.deny.includes(z.id) || z.restricted;
    const isAllow = st.allow.includes(z.id);
    const isDrift = st.drift.includes(z.id);
    ctx.fillStyle = cssVar("--zk-"+z.kind) || cssVar("--surface");
    roundRect(ctx,x+1,y+1,w-2,h-2,6); ctx.fill();
    if (vz.has(z.id)){ ctx.save(); ctx.shadowColor = cssVar("--bad");
      ctx.shadowBlur = 10; roundRect(ctx,x+1,y+1,w-2,h-2,6);
      ctx.strokeStyle = cssVar("--bad"); ctx.lineWidth = 1; ctx.stroke(); ctx.restore(); }
    ctx.lineWidth = (isDeny||isDrift) ? 2.2 : isAllow ? 1.8 : 1.1;
    ctx.strokeStyle = isDrift ? cssVar("--drift")
      : isDeny ? cssVar("--bad") : isAllow ? cssVar("--ok") : cssVar("--line");
    if (isDrift) ctx.setLineDash([5,3]);
    roundRect(ctx,x+1,y+1,w-2,h-2,6); ctx.stroke(); ctx.setLineDash([]);
    // 名称 + kind emoji
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = cssVar("--ink");
    const fs = Math.max(10, cell*0.36);
    ctx.font = `${fs}px "PingFang SC","Noto Sans CJK SC",sans-serif`;
    const em = KIND_EMOJI[z.kind] || "";
    ctx.fillText((em?em+" ":"")+z.name, x+w/2, y+fs+4, w-8);
    // 状态徽章行(右下,抬离墙线避免与门重叠)
    let bx = x+w-6;
    const badge = t => { ctx.font = `${Math.max(10,cell*0.32)}px sans-serif`;
      ctx.textAlign="right"; ctx.fillText(t, bx, y+h-7); bx -= cell*0.5; };
    if (isDeny) badge("🚫");
    if (isDrift) badge("🌀");
    if (st.temporal.includes(z.id)) badge("⏰");
    if (z.gives) badge("⚠️");
    if (z.door_open && z.occupied) badge("🚪");
    P.layout.portals.filter(p=>p.zone===z.id).slice(0,1)
      .forEach(p=>badge(p.narrow?"🐾":"↕"+p.to_floor+"F"));
    // 任务标记(左下)
    ctx.textAlign = "left";
    ctx.font = `${Math.max(10,cell*0.34)}px sans-serif`;
    if (P.task.type==="deliver" && P.task.dropoff_zone===z.id) ctx.fillText("🎯", x+5, y+h-5);
    if (P.task.type==="escort" && P.task.to_zone===z.id) ctx.fillText("🚩", x+5, y+h-5);
    if (P.zstatus.visit.includes(z.id))
      ctx.fillText(frame.visited.includes(z.id)?"✅":"📍", x+5, y+h-5);
    if (P.meta.start===z.id) ctx.fillText("🏁", x+5, y+h-5-(cell*0.42));
  });
  // 门
  fl.doors.forEach(d=>{
    const dx = PAD+d.x*cell, dy = PAD+d.y*cell;
    ctx.fillStyle = cssVar("--door");
    if (d.o==="v") ctx.fillRect(dx-2.5, dy-cell*0.35, 5, cell*0.7);
    else ctx.fillRect(dx-cell*0.35, dy-2.5, cell*0.7, 5);
    if (d.narrow){ ctx.font = `${Math.max(8,cell*0.26)}px sans-serif`;
      ctx.textAlign="center"; ctx.fillText("🐾", dx+(d.o==="v"?8:0), dy-(d.o==="v"?0:7)); }
  });
  // 实体(不在机器人身上的)
  const perZone = {};
  frame.entities.filter(e=>e.zone!=="robot").forEach(e=>{
    (perZone[e.zone] = perZone[e.zone]||[]).push(e); });
  Object.entries(perZone).forEach(([zid,es])=>{
    const r = zoneRect(fl, zid); if (!r) return;
    const [x,y,w,h] = rectPx(r, cell);
    es.forEach((e,i)=>{
      ctx.font = `${Math.max(12,cell*0.42)}px sans-serif`; ctx.textAlign = "center";
      ctx.fillText(e.kind==="human"?"🧑":(/hazard/.test(e.id)?"☣️":"📦"),
        x+w/2+(i-(es.length-1)/2)*cell*0.6, y+h*0.68);
    });
  });
  // 轨迹(同层段)
  const F = frames();
  ctx.strokeStyle = cssVar("--robot"); ctx.globalAlpha = .4; ctx.lineWidth = 2.5;
  ctx.lineJoin = "round"; ctx.beginPath();
  let pen = false;
  for (let i=0;i<=idx;i++){
    const c = centerPx(F[i].zone, cell);
    if (!c || c.fl!==fl.floor){ pen = false; continue; }
    if (pen) ctx.lineTo(c.x, c.y-cell*0.55); else ctx.moveTo(c.x, c.y-cell*0.55);
    pen = true;
  }
  ctx.stroke(); ctx.globalAlpha = 1;
  // 机器人
  const rc = centerPx(frame.zone, cell);
  if (rc && rc.fl===fl.floor){
    const ry = rc.y - cell*0.55;
    ctx.beginPath(); ctx.arc(rc.x, ry, cell*0.52, 0, 7);
    ctx.fillStyle = cssVar("--robot"); ctx.globalAlpha = .18; ctx.fill(); ctx.globalAlpha = 1;
    if (frame.vio.length){ ctx.strokeStyle = cssVar("--bad"); ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(rc.x, ry, cell*0.62, 0, 7); ctx.stroke();
      ctx.globalAlpha=.5; ctx.beginPath(); ctx.arc(rc.x, ry, cell*0.8, 0, 7);
      ctx.stroke(); ctx.globalAlpha=1; }
    ctx.font = `${cell*0.72}px sans-serif`; ctx.textAlign = "center";
    ctx.textBaseline = "middle"; ctx.fillText(ROBOT, rc.x, ry);
    ctx.font = `${Math.max(9,cell*0.3)}px sans-serif`;
    let side = rc.x + cell*0.75;
    if (frame.camera){ ctx.fillText("📷", side, ry); side += cell*0.5; }
    if (frame.sensor) ctx.fillText("📸", side, ry-cell*0.4);
    if (frame.blocked) ctx.fillText("⛔", rc.x, ry-cell*0.75);
    if (frame.carrying) ctx.fillText(/hazard/.test(frame.carrying)?"☣️":"📦",
      rc.x-cell*0.7, ry-cell*0.35);
    if (frame.escorted) ctx.fillText("🧑", rc.x-cell*0.75, ry+cell*0.2);
    ctx.textBaseline = "alphabetic";
  }
}
function drawAll(frame){
  const cell = cellOf();
  document.querySelectorAll("#floors canvas").forEach(cv=>{
    const fl = P.layout.floors.find(f=>String(f.floor)===cv.dataset.floor);
    if (fl) drawFloor(cv, fl, frame, cell);
  });
}

// ---------------- hover 提示
function hover(e, cv, fl, cell){
  const r = cv.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  const hit = fl.zones.find(z=>{
    const [x,y,w,h] = rectPx(z, cell);
    return mx>=x && mx<=x+w && my>=y && my<=y+h;
  });
  const tip = $("tooltip");
  if (!hit){ tip.style.display = "none"; return; }
  const z = Z[hit.id], st = P.zstatus, f = frames()[idx];
  const flags = [
    st.allow.includes(z.id)?"允许集":"", st.deny.includes(z.id)?"deny":"",
    z.restricted?"restricted":"", st.drift.includes(z.id)?"漂移目标":"",
    st.temporal.includes(z.id)?"时间窗管制":"", z.gives?`递交:${z.gives}`:"",
    z.door_open?"门开":"", z.occupied?`有${z.occupied}`:"",
    f.visited.includes(z.id)?"已到访":"",
  ].filter(Boolean);
  tip.innerHTML = `<b>${KIND_EMOJI[z.kind]||""} ${esc(z.name)}</b> <span class="k">${esc(z.id)}
    · ${esc(z.kind)} · ${z.floor}F</span><br>`
    + (flags.length?flags.map(esc).join(" · "):"<span class='k'>无标记</span>");
  tip.style.display = "block";
  tip.style.left = Math.min(e.clientX+14, innerWidth-280)+"px";
  tip.style.top = (e.clientY+14)+"px";
}

// ---------------- 右侧面板
function inWindow(t, a, b){
  const m = s => +s.slice(0,2)*60 + +s.slice(3);
  const x = m(t), lo = m(a), hi = m(b);
  if (lo===hi) return true;
  return lo<hi ? (x>=lo && x<hi) : (x>=lo || x<hi);
}
function renderStatus(frame){
  const alert = P.zstatus.windows.some(w=>inWindow(frame.time,w.a,w.b));
  const carry = frame.carrying
    ? `${/hazard/.test(frame.carrying)?"☣️":"📦"} ${esc(frame.carrying)}` : "空";
  const pj = Math.round(frame.progress*100), bj = Math.round(frame.brs*100);
  $("statusbar").innerHTML = `
    <span class="sitem"><b>step ${frame.step}</b>/${P.meta.horizon}</span>
    <span class="sitem ${alert?"time-alert":""}">🕐 <b>${frame.time}</b>${alert?" ⏰管制中":""}</span>
    <span class="sitem"><span class="lab">位置</span><b>${esc((Z[frame.zone]||{}).name||frame.zone)}</b></span>
    <span class="sitem"><span class="lab">🤲 携带</span> ${carry}</span>
    ${frame.escorted?`<span class="sitem">🧑 护送中:<b>${esc(frame.escorted)}</b></span>`:""}
    <span class="sitem"><span class="lab">📷</span>${frame.camera?"开":"关"}</span>
    <span class="sitem"><span class="lab">任务</span>
      <span class="bar"><i style="width:${pj}%"></i></span><b>${pj}%</b>
      ${frame.success?'<span class="win">✅ 完成</span>':""}</span>
    <span class="sitem"><span class="lab">BRS</span>
      <span class="bar brs ${bj<70?"low":""}"><i style="width:${bj}%"></i></span><b>${bj}%</b></span>`;
}
function renderTask(frame){
  const t = P.task, nm = z => esc((Z[z]||{}).name||z);
  let subs = "";
  if (t.type==="deliver"){
    const picked = frame.progress>=0.5, del = frame.progress>=1;
    subs = `<span class="${picked?"done":""}">${picked?"✅":"○"} 取货 ${nm(t.pickup_zone)}</span>
            <span class="${del?"done":""}">${del?"✅":"○"} 送达 ${nm(t.dropoff_zone)}</span>`;
  } else if (t.type==="visit"){
    subs = t.targets.map(z=>{
      const d = frame.visited.includes(z);
      return `<span class="${d?"done":""}">${d?"✅":"📍"} ${nm(z)}</span>`; }).join("");
  } else if (t.type==="escort"){
    const got = frame.progress>=0.5, done = frame.progress>=1;
    subs = `<span class="${got?"done":""}">${got?"✅":"○"} 接到(${nm(t.from_zone)})</span>
            <span class="${done?"done":""}">${done?"✅":"○"} 送抵 ${nm(t.to_zone)}</span>`;
  }
  $("task").innerHTML = `<div class="tasktxt">${esc(t.text)}</div><div class="sub">${subs}</div>`;
}
function renderContract(frame){
  const held = new Set(frame.violated);
  const cats = {};
  P.contract.forEach(c=>{ (cats[c.cat]=cats[c.cat]||[]).push(c); });
  $("contract").innerHTML = Object.entries(cats).map(([cat,cs])=>`
    <div class="ct-group"><div class="ct-cat">${cat} (${cs.length})</div>
    ${cs.map(c=>`<div class="ct ${held.has(c.cid)?"violated":""}">
      <span class="rule ${c.rule}">${c.rule==="allow"?"✅ allow":"⛔ deny"}</span>
      <span><span class="cid">${c.cid}</span> ${esc(c.text)}</span>
      <span class="sev">${"●".repeat(c.sev)}</span></div>`).join("")}
    </div>`).join("");
}
function renderPosts(frame){
  $("posts").innerHTML = P.posts.map(p=>{
    const vis = p.step<=frame.step;
    return `<div class="post ${p.role} ${vis?"vis":""}">
      <span class="who">step ${p.step} · ${esc(p.author)}</span>
      <span class="tag">${p.role}</span><br>${vis?esc(p.text):"…(尚未到达)"}</div>`;
  }).join("");
}
function renderDecision(frame){
  const D = P.agents[cur].decisions;
  let di = frame.decision;
  if (di == null){                       // 初始帧:显示在该 step 做出的决策
    di = D.findIndex(x=>x.at===frame.step);
    if (di < 0) di = null;
  }
  const d = di!=null ? D[di] : null;
  if (!d){ $("decision").innerHTML = '<div class="empty">—</div>'; return; }
  const nm = z => z ? esc((Z[z]||{}).name||z) : "(hold)";
  const okmark = d.ok==null?"":(d.ok?`<span class="expchip ok">✓符合期望</span>`
                                    :`<span class="expchip no">✗期望 ${d.expected}</span>`);
  $("decision").innerHTML =
    `<div style="margin-bottom:6px">step ${d.at} 提议 <b>${nm(d.proposal)}</b>
      → <span class="verdict ${d.verdict}">${VNAME[d.verdict]||d.verdict}</span>
      ${d.final&&d.final!==d.proposal?`改道 <b>${nm(d.final)}</b>`:""}
      <span class="expchip">(${d.rounds}轮·${d.llm_calls}调用)</span> ${okmark}
      ${d.flags.length?`<span class="expchip">标记 ${esc(d.flags.join(","))}</span>`:""}</div>`
    + d.transcript.map(m=>`<div class="msg ${m.role}"><span class="role">${m.role}</span>
        <span>${esc(m.text)}</span></div>`).join("");
}
function renderVios(frame){
  const rows = [];
  frames().slice(0,idx+1).forEach(f=>f.vio.forEach(v=>rows.push({...v,step:f.step})));
  $("vios").innerHTML = rows.length ? rows.map(v=>
    `<div class="vio">step ${v.step} · ${esc(v.cid)} (sev ${v.severity})
     @ ${esc((Z[v.zone]||{}).name||v.zone)}</div>`).join("")
    : '<div class="empty">无违规 ✨</div>';
}
function renderHist(){
  const A = P.agents[cur];
  const firstFrame = {};
  A.frames.forEach((f,i)=>{ if (f.decision!=null && !(f.decision in firstFrame))
    firstFrame[f.decision] = i; });
  $("hist").innerHTML = A.history.map((h,i)=>{
    const cls = h.viols ? "badh" : (h.verdict==="accept"||h.verdict==="rewrite") ? "" : "holdh";
    return `<div class="hentry ${cls}" data-d="${i}">
      <span class="st">#${h.step}</span><span class="lb">${esc(h.label)}
      ${h.blocked?"⛔不可达":""}</span>
      <span class="verdict ${h.verdict}">${h.verdict}</span>
      ${h.ok===false?'<span class="expchip no">✗</span>':""}
      ${h.viols?`<span class="expchip no">${h.viols}违规</span>`:""}
      <span class="expchip">${h.hops} tick</span></div>`;
  }).join("") || '<div class="empty">—</div>';
  $("hist").querySelectorAll(".hentry").forEach(el=>{
    el.onclick = ()=>{ const i = +el.dataset.d;
      if (i in firstFrame){ idx = firstFrame[i]; render(); } };
  });
}

// ---------------- 主渲染
function render(){
  const F = frames(), f = F[Math.min(idx, F.length-1)];
  try{ history.replaceState(null, "", `#agent=${cur}&tick=${idx}`); }catch(e){}
  $("scrub").max = F.length-1; $("scrub").value = idx;
  $("ticklab").textContent =
    `tick ${idx}/${F.length-1} · step ${f.step} · ${f.time} · ${(Z[f.zone]||{}).name||f.zone}`;
  drawAll(f);
  renderStatus(f); renderTask(f); renderContract(f);
  renderPosts(f); renderDecision(f); renderVios(f); renderHist();
}
$("scrub").addEventListener("input", e=>{ idx = +e.target.value; render(); });
$("play").onclick = ()=>{
  if (timer){ clearInterval(timer); timer=null; $("play").textContent="▶"; return; }
  if (idx >= frames().length-1) idx = 0;
  $("play").textContent = "⏸";
  const dt = matchMedia("(prefers-reduced-motion: reduce)").matches ? 900 : 480;
  timer = setInterval(()=>{
    if (idx >= frames().length-1){ clearInterval(timer); timer=null;
      $("play").textContent="▶"; return; }
    idx++; render();
  }, dt);
};
document.addEventListener("keydown", e=>{
  if (e.key==="ArrowRight"){ idx = Math.min(idx+1, frames().length-1); render(); }
  else if (e.key==="ArrowLeft"){ idx = Math.max(idx-1, 0); render(); }
  else if (e.key===" " && e.target===document.body){ e.preventDefault(); $("play").click(); }
});
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", ()=>render());
let _rsz;
window.addEventListener("resize", ()=>{ clearTimeout(_rsz);
  _rsz = setTimeout(()=>{ buildFloors(); render(); }, 150); });

// ---------------- live 轮询(renderId 变化才更新;file:// 下 fetch 被禁则静默)
if (P.mode === "live"){
  setInterval(()=>{
    fetch(location.href, {cache:"no-store"}).then(r=>r.text()).then(txt=>{
      const m = txt.match(/<script type="application\/json" id="state-data">([\s\S]*?)<\/script>/);
      if (!m) return;
      const np = JSON.parse(m[1].replace(/<\\\//g,"</"));
      if (np.renderId === P.renderId) return;
      const tail = idx >= frames().length-1;
      P = np; Z = zinfo();
      agents = Object.keys(P.agents);
      if (!agents.includes(cur)) cur = agents[agents.length-1];
      if (tail) idx = frames().length-1;
      buildFloors(); renderTabs(); render();
    }).catch(()=>{});
  }, 1200);
}

// 深链:#agent=<name>&tick=<n>(分享/复现某一时刻)
(function(){
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("agent") && agents.includes(h.get("agent"))) cur = h.get("agent");
  const t = parseInt(h.get("tick"), 10);
  if (!isNaN(t)) idx = Math.max(0, Math.min(t, frames().length-1));
})();
renderChips(); renderTabs(); buildFloors(); render();
</script>
</body>
</html>
"""
