"""Teacher-briefing visualization for SelfEvolvingHarnessTS.

Visual grammar is taken from two in-lab references:

- TSQualityAgent ``run_logger.py`` / ``training/synthesis/visualize.py``:
  one self-contained HTML, series overlay, chips, collapsible reasoning chain.
- TimeClaw README / assets/framework: architecture as three coupled
  subsystems around a frozen LLM, plus an auditable trajectory.

Numbers are read from already-exposed artifacts. No new evaluation.

  python evaluation/functional/run_teacher_briefing_viz.py
"""
from __future__ import annotations

import html as _html
import json
import math
import struct
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_HTML = PROJECT_ROOT / "artifacts" / "visualization" / "teacher_briefing.html"
NOAA_STATION = "72203812897"
NOAA_ORIGIN = 6552
NOAA_CONTEXT = 192


def _e(s: Any) -> str:
    return _html.escape(str("" if s is None else s))


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npy_f8(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"not npy: {path}")
    hlen = int.from_bytes(raw[8:10], "little")
    data = raw[10 + hlen :]
    n = len(data) // 8
    return struct.unpack("<%dd" % n, data)


def _polyline(xs: Sequence[float], ys: Sequence[float]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def svg_series(values: Sequence[float], title: str, missing_note: str) -> str:
    w, h = 980, 220
    left, right, top, bottom = 44, 16, 28, 28
    pw, ph = w - left - right, h - top - bottom
    finite = [v for v in values if v is not None and not math.isnan(v)]
    lo, hi = min(finite), max(finite)
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad

    def xv(i: int) -> float:
        return left + i / max(len(values) - 1, 1) * pw

    def yv(v: float) -> float:
        return top + (hi - v) / (hi - lo) * ph

    segs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    miss_x: list[float] = []
    for i, v in enumerate(values):
        if v is None or math.isnan(v):
            miss_x.append(xv(i))
            if cur:
                segs.append(cur)
                cur = []
            continue
        cur.append((xv(i), yv(v)))
    if cur:
        segs.append(cur)

    paths = "".join(
        f'<polyline fill="none" stroke="#3a80b8" stroke-width="1.6" '
        f'points="{_polyline(*zip(*seg))}"/>'
        for seg in segs
    )
    ticks = "".join(
        f'<line x1="{x:.1f}" y1="{top + ph + 2:.1f}" x2="{x:.1f}" '
        f'y2="{top + ph + 10:.1f}" stroke="#b83c28" stroke-width="1.4"/>'
        for x in miss_x
    )
    grid = "".join(
        f'<line x1="{left}" y1="{top + i * ph / 3:.1f}" x2="{w - right}" '
        f'y2="{top + i * ph / 3:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>'
        for i in range(4)
    )
    return f"""
<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_e(title)}">
  <text x="{left}" y="16" fill="#334155" font-size="12" font-weight="600">{_e(title)}</text>
  <text x="{w - right}" y="16" text-anchor="end" fill="#94a3b8" font-size="11">{_e(missing_note)}</text>
  {grid}
  <line x1="{left}" y1="{top + ph}" x2="{w - right}" y2="{top + ph}" stroke="#cbd5e1"/>
  {paths}
  {ticks}
  <text x="{left}" y="{h - 6}" fill="#94a3b8" font-size="10">time step (hourly)</text>
  <text x="{left - 6}" y="{top + 4}" text-anchor="end" fill="#94a3b8" font-size="10">{_fmt(hi, 1)}</text>
  <text x="{left - 6}" y="{top + ph}" text-anchor="end" fill="#94a3b8" font-size="10">{_fmt(lo, 1)}</text>
</svg>"""


def svg_bars(cats: Sequence[str], series: Sequence[tuple[str, str, Sequence[float]]],
             height: int = 230) -> str:
    n, k = len(cats), len(series)
    allv = [0.0]
    for _, _, vals in series:
        allv += [float(v) for v in vals]
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-9:
        hi = lo + 1
    pad = (hi - lo) * 0.18 or 0.1
    lo, hi = lo - pad, hi + pad
    top, bottom, left = 18, 36, 36
    ph = height - top - bottom
    gw = k * 22 + 10
    gap = 28
    width = left + n * (gw + gap) + 8

    def yv(v: float) -> float:
        return top + (hi - v) / (hi - lo) * ph

    parts = [f'<line x1="{left}" y1="{yv(0):.1f}" x2="{width - 6}" y2="{yv(0):.1f}" stroke="#cbd5e1"/>']
    for ci, cat in enumerate(cats):
        x0 = left + ci * (gw + gap)
        for si, (_, color, vals) in enumerate(series):
            v = float(vals[ci])
            x = x0 + si * 22
            yt, yb = min(yv(v), yv(0)), max(yv(v), yv(0))
            parts.append(
                f'<rect x="{x}" y="{yt:.1f}" width="18" height="{max(yb - yt, 0.8):.1f}" '
                f'fill="{color}" rx="3"/>'
            )
            parts.append(
                f'<text x="{x + 9}" y="{yt - 5:.1f}" text-anchor="middle" '
                f'fill="#475569" font-size="11" font-weight="600">{_fmt(v, 3)}</text>'
            )
        parts.append(
            f'<text x="{x0 + gw / 2:.0f}" y="{height - 10}" text-anchor="middle" '
            f'fill="#64748b" font-size="11">{_e(cat)}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def svg_framework() -> str:
    boxes = [
        (24, 40, "① Runtime", "序列留在执行器", "Operator 精确计算", "#1d6fa5"),
        (250, 40, "② Memory", "Episode / Skill 卡", "正负冲突对照检索", "#7c3aed"),
        (476, 40, "③ Evolution", "Support → delayed", "Scope/Risk PATCH", "#b45309"),
        (702, 40, "④ Freeze", "held-out Fast-only", "结果不回流本次", "#0f766e"),
    ]
    rects = []
    for i, (x, y, t, a, b, c) in enumerate(boxes):
        rects.append(
            f'<rect x="{x}" y="{y}" width="200" height="108" rx="10" fill="#fff" '
            f'stroke="{c}" stroke-width="1.6"/>'
            f'<text x="{x + 100}" y="{y + 32}" text-anchor="middle" font-weight="700" '
            f'font-size="14" fill="{c}">{t}</text>'
            f'<text x="{x + 100}" y="{y + 58}" text-anchor="middle" font-size="12" fill="#334155">{a}</text>'
            f'<text x="{x + 100}" y="{y + 80}" text-anchor="middle" font-size="12" fill="#64748b">{b}</text>'
        )
        if i < len(boxes) - 1:
            x2 = boxes[i + 1][0]
            rects.append(
                f'<line x1="{x + 200}" y1="{y + 54}" x2="{x2}" y2="{y + 54}" '
                f'stroke="#94a3b8" stroke-width="2"/>'
                f'<polygon points="{x2},{y + 54} {x2 - 8},{y + 49} {x2 - 8},{y + 59}" fill="#94a3b8"/>'
            )
    return f"""
<svg viewBox="0 0 926 170" xmlns="http://www.w3.org/2000/svg">
  <text x="24" y="22" fill="#64748b" font-size="12">对照 TimeClaw 的三子系统：数值不靠 token 手算 · 经验外置 · 轨迹可审计</text>
  {''.join(rects)}
</svg>"""


def chip(text: str, kind: str) -> str:
    return f'<span class="chip chip-{kind}">{_e(text)}</span>'


def winner_badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{_e(text)}</span>'


def details_block(title: str, inner: str, open_: bool = False) -> str:
    op = " open" if open_ else ""
    return f"""<details{op}>
  <summary><span class="toggle-icon">▶</span><span class="summary-id">{title}</span></summary>
  <div class="sample-body">{inner}</div>
</details>"""


def trace_card(arm: str, steps: list[dict[str, Any]]) -> str:
    rows = []
    for i, st in enumerate(steps, 1):
        plan = st.get("adopted_plan") or {}
        prog = (plan.get("program") if isinstance(plan, dict) else None) or "—"
        mode = st.get("mode") or ""
        sg = st.get("support_aggregate_gain")
        dg = st.get("delayed_aggregate_gain")
        harm = st.get("harmed_eval_series_count")
        kind = "ok" if st.get("adopted_delayed_positive") else ("warn" if mode == "FULL_PRICE_SEARCH" else "mut")
        rows.append(f"""
<div class="dim-card">
  <div class="dim-header"><b>Step {i} · {_e(st.get("step"))}</b>
    {chip(str(mode), "tool" if "SEARCH" in str(mode) else "reason")}
    {chip(str(prog), "target")}
    {winner_badge(f"retrain {st.get('consumer_retrains')}", kind)}
  </div>
  <p class="conclusion">Support {_fmt(sg, 6) if sg is not None else "—"}
     · delayed {_fmt(dg, 6) if dg is not None else "—"}
     · harm series {harm if harm is not None else "—"}
     · LLM {st.get("llm_calls")}</p>
</div>""")
    return f'<div><h3 class="arm-h">{_e(arm)}</h3>{"".join(rows)}</div>'


def build_html() -> str:
    fresh = _load_json(E2 / "fresh_confirmation_v1.json")
    l1 = _load_json(E2 / "l1_ladder_v2_replay_r1.json")
    sa1 = _load_json(E2 / "sa1_minimal_r1.json")
    cap = _load_json(E2 / "capstone_epilepsy2_final.json")
    pooled = fresh["cells"]["pooled"]
    a5, a3 = pooled["A5"], pooled["A3"]

    npy = (
        PROJECT_ROOT
        / "data"
        / "benchmark_noaa_fresh_v1"
        / "series"
        / NOAA_STATION
        / "values.npy"
    )
    series = _load_npy_f8(npy)
    window = list(series[NOAA_ORIGIN : NOAA_ORIGIN + NOAA_CONTEXT])
    n_miss = sum(1 for v in window if math.isnan(v))
    series_svg = svg_series(
        window,
        f"NOAA {NOAA_STATION} · origin={NOAA_ORIGIN} · 192-step context",
        f"missing {n_miss}/{NOAA_CONTEXT} = {n_miss / NOAA_CONTEXT:.1%}  （红竖线）",
    )

    cost_svg = svg_bars(
        ["首个正采纳成本（重训）", "总重训"],
        [
            ("A5", "#1d6fa5", [69, 99]),
            ("A3", "#b83c28", [123, 195]),
        ],
        height=230,
    )
    cls_svg = svg_bars(
        ["无卡 A3 regret", "叶 Scope L1", "承重五轴 SA1 卡"],
        [
            ("regret", "#1d6fa5", [0.7710, 0.5583, 0.0850]),
        ],
        height=220,
    )

    a5_trace = trace_card(
        "A5 · 累积知识 + Target 校准（pooled）",
        a5["trace"],
    )
    a3_trace = trace_card(
        "A3 · 空店冷启动（pooled）",
        a3["trace"],
    )

    l1_v = l1["verdict"]
    sa_h = sa1["headline"]
    sa_sum = sa1["summary"]
    cap_s = cap["score"]

    perceiver = details_block(
        "① 系统在看什么（对照 TSQ Perceiver）",
        f"""
<p><b>Task / Consumer：</b> forecasting · Ridge · sMASE。质量不是绝对干净，而是相对下游效用。</p>
<p><b>部署可见 Context：</b> 缺失几何、周期、Program 作用范围。Dataset 名称不得作为跨域理由。</p>
<p><b>对比对象：</b> {chip("A5 = 累积知识 + held-in 校准", "target")}
{chip("A3 = 去掉跨域积累", "tool")}
{chip("Static = 去掉适应", "reason")}</p>
<p class="muted">TSQ 把质量做成 A vs B 成对裁决；本项目把质量做成 Identity vs Program 的下游增益，并要求 delayed 层独立确认。</p>
""",
        open_=True,
    )

    inspector = details_block(
        "② 一次可审计轨迹（对照 TSQ Inspector / TimeClaw trajectory）",
        f"""
<p class="muted">工件 <code>fresh_confirmation_v1.json</code> · 判词 {_e(fresh["overall_verdict"])}。
NOAA 2024 适应、2025 确认区一次性打开；不是 Fast-only held-out。</p>
<div class="grid">{a5_trace}{a3_trace}</div>
<p><b>读法：</b> A5 在 task_A 一次 FULL_PRICE_SEARCH 就采纳 <code>outlier_mad</code>
（delayed +0.306），此后 DIRECT_RECALL；A3 在 task_A 走 identity，到 task_B 才首次正采纳。
终态 delayed 打平 +0.029688，harm 1 vs 1。主张上限是<strong>降低适应成本，不是提高终态质量</strong>。</p>
""",
        open_=True,
    )

    adjudicator = details_block(
        "③ 分类线：Scope 卡与修订环（development）",
        f"""
<p><b>L1 权限阶梯 v2：</b> {_e(l1_v["headline"])}</p>
<p>v4 尾段无卡 regret 0.7710 → L1 0.5583，改善 {_fmt(l1_v["facts"]["regret_improvement"], 4)}，harm 0。
只转换 1/5 单元（GunPointOldVersusYoung）。</p>
<p><b>SA-1 修订环：</b> 带承重五轴卡相对无卡 regret 差
{chip("+0.6860", "ok")}；修订臂 vs 冻结卡臂 regret 差
{chip("+0.0000", "warn")}。修订买到的是避拒/省探，不是质量。</p>
<table>
<tr><th>臂</th><th>regret</th><th>probes</th><th>supplied</th><th>refused</th><th>harm</th></tr>
<tr><td>A3-reset</td><td>{_fmt(sa_sum["A3-reset"]["cumulative_regret_distinct_units"])}</td>
<td>{sa_sum["A3-reset"]["probes"]}</td><td>0</td><td>0</td><td>0</td></tr>
<tr><td>K0-fixed</td><td>{_fmt(sa_sum["K0-fixed"]["cumulative_regret_distinct_units"])}</td>
<td>{sa_sum["K0-fixed"]["probes"]}</td><td>5</td><td>3</td><td>0</td></tr>
<tr><td>A5-adaptive</td><td>{_fmt(sa_sum["A5-online"]["cumulative_regret_distinct_units"])}</td>
<td>{sa_sum["A5-online"]["probes"]}</td><td>3</td><td>1</td><td>0</td></tr>
</table>
<p class="muted">再遇位：修订后的卡不再供给 PowerCons，冻结卡再供给再挨拒。归因到 narrowing 的省探 = {sa_h["attribution"]["probes_saved_by_narrowing"]}，避拒 = {sa_h["attribution"]["refusals_avoided_by_narrowing"]}。</p>
""",
    )

    final = details_block(
        "④ 密封处女域守卫（Epilepsy2 capstone）",
        f"""
<p>判词 {winner_badge(str(cap_s["verdict"]), "mut")} ·
A5−A3 accuracy = {_fmt(cap_s["delta_a5_minus_a3"], 6)} · harm = {cap_s["harm_a5"]}</p>
<p>三臂全部部署 identity，accuracy 均为 {_fmt(cap_s["acc"]["A5"], 6)}。
族外卡供给 0；A3/A5 各自提出的有害提案（−0.2750 / −0.0500）被 Support 拦住。
这是<strong>条件化与安全</strong>，不是正迁移。</p>
<p class="muted">族内密封正迁移当前 unavailable（公开 UCR 池耗尽），不是方法负结果。见
<code>docs/CLS_LINE_FINAL_REPORT_2026-08-28.md</code>。</p>
""",
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SelfEvolvingHarnessTS 阶段性结果 · 老师汇报可视化</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px; color: #222; font-size: 14px; background: #f8fafc; }}
h1 {{ font-size: 1.25em; margin: 0 0 6px; }}
h2 {{ font-size: 0.95em; font-weight: 700; color: #1a1a1a;
      border-left: 3px solid #4a90d9; padding-left: 10px; margin-top: 28px; }}
h3.arm-h {{ font-size: 0.9em; margin: 8px 0; color: #1d6fa5; }}
.meta {{ color: #64748b; font-size: 0.85em; margin-bottom: 16px; }}
.hero {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; }}
.sum {{ font-size: 1.02em; font-weight: 650; color: #0f172a; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 14px 0 6px; }}
.kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; }}
.kpi .n {{ font-size: 1.35em; font-weight: 700; color: #1d6fa5; }}
.kpi .l {{ font-size: 0.75em; color: #64748b; margin-top: 2px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
@media (max-width: 840px) {{ .kpis, .grid {{ grid-template-columns: 1fr; }} }}
.chip {{ display: inline-block; margin: 2px 3px; padding: 2px 9px; border-radius: 10px;
         font-size: 0.79em; font-weight: 600; }}
.chip-target {{ background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }}
.chip-tool {{ background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }}
.chip-reason {{ background: #f5f3ff; color: #5b21b6; border: 1px solid #ddd6fe; }}
.chip-ok {{ background: #ecfdf5; color: #047857; border: 1px solid #6ee7b7; }}
.chip-warn {{ background: #fff7ed; color: #c2410c; border: 1px solid #fdba74; }}
.badge {{ display: inline-block; padding: 1px 9px; border-radius: 10px; font-weight: 700; font-size: 0.85em; }}
.b-ok {{ background: #dcfce7; color: #166534; }}
.b-warn {{ background: #ffedd5; color: #9a3412; }}
.b-mut {{ background: #e2e8f0; color: #475569; }}
details {{ border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 8px; overflow: hidden; background: #fff; }}
details[open] {{ border-color: #94b4d1; }}
summary {{ display: flex; align-items: center; gap: 10px; padding: 9px 14px; cursor: pointer;
           list-style: none; background: #f8fafc; user-select: none; }}
summary::-webkit-details-marker {{ display: none; }}
.toggle-icon {{ font-size: 0.75em; color: #94a3b8; }}
details[open] .toggle-icon {{ transform: rotate(90deg); }}
.summary-id {{ font-weight: 600; color: #334155; }}
.sample-body {{ padding: 14px 16px 16px; }}
.dim-card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; background: #fafafa; }}
.dim-header {{ margin-bottom: 4px; }}
.conclusion {{ margin: 4px 0 0; color: #333; font-size: 0.9em; }}
.muted, .page-meta {{ color: #64748b; font-size: 0.86em; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.84em; margin: 8px 0; }}
th {{ background: #f4f6f8; text-align: left; padding: 6px 10px; color: #64748b; border-bottom: 2px solid #e2e8f0; }}
td {{ padding: 5px 10px; border-bottom: 1px solid #f1f5f9; }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: #64748b; margin: 6px 0 12px; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
.plotbox {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 10px 4px; }}
blockquote {{ border-left: 3px solid #cbd5e1; margin: 8px 0; padding: 6px 14px; color: #334155; }}
footer {{ color: #94a3b8; font-size: 11px; text-align: center; padding: 18px 0 8px; }}
code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
.warnbox {{ background: #fff7ed; border: 1px solid #fdba74; border-radius: 8px; padding: 10px 14px; }}
</style>
</head>
<body>
<header class="hero">
  <h1>SelfEvolvingHarnessTS · 阶段性结果可视化</h1>
  <p class="meta">给老师汇报用的只读简报 · 可视化语法对照 TSQualityAgent 自包含 HTML 推理链
     与 TimeClaw 框架图 / 可审计轨迹 · 数字全部来自已暴露工件 · 零新评估</p>
  <p class="sum">完整系统主张：累积知识进入新域后，用 held-in 反馈校准，而不是从零搜索。
     目前最硬的 fresh 读数是 NOAA 上<strong>首个正采纳成本 69 vs 123 重训（−43.9%）</strong>，
     终态效用与 harm 打平。</p>
  <div class="kpis">
    <div class="kpi"><div class="n">69 vs 123</div><div class="l">A5 / A3 首正成本（pooled）</div></div>
    <div class="kpi"><div class="n">+0.000</div><div class="l">终态 delayed 差（打平）</div></div>
    <div class="kpi"><div class="n">+0.686</div><div class="l">分类：承重 Scope 卡 vs 无卡</div></div>
    <div class="kpi"><div class="n">harm 0</div><div class="l">密封 Epilepsy2 守卫成立</div></div>
  </div>
</header>

<h2>方法骨架（TimeClaw 式框架图）</h2>
<div class="plotbox">{svg_framework()}</div>
<p class="muted">TimeClaw 把通用 LLM 包在「原生数值 runtime + episodic memory + 能力演化」里。
本项目同一语法，但演化对象是 <b>数据准备 Workflow / Scope / Risk</b>，评价是下游 Consumer 效用，不是预测点数本身。</p>

<h2>老师一眼能看的数据对象（TSQ 式序列图）</h2>
<div class="plotbox">{series_svg}</div>
<p class="muted">真实 NOAA 小时温度，不是示意曲线。缺失用红竖线标出，对应 TSQ 把 missing 画在 B 上的做法。
Harness 的工作不是「把线画得更平滑」，而是决定：这段 Context 该 identity、该修复，还是该弃权。</p>

<h2>A5 vs A3：适应成本（主读数）</h2>
<div class="plotbox">{cost_svg}</div>
<div class="legend"><span><i style="background:#1d6fa5"></i>A5 累积知识</span>
<span><i style="background:#b83c28"></i>A3 空店</span></div>
<p>判词 {winner_badge("FRESH_A5_DELIVERS", "ok")} · 主键是 pooled Consumer。
task_C delayed 两条臂同为 +0.029688，harm 1 vs 1，所以终态质量打平。
<strong>不能说 A5 预测得更准，只能说适应更便宜（69 vs 123 重训，−43.9%）。</strong></p>

{perceiver}
{inspector}

<h2>分类开发线：同一套生命周期，换 Task</h2>
<div class="plotbox">{cls_svg}</div>
<p class="muted">纵轴是 cumulative regret（越低越好）。叶 Scope 回收 +0.2127；换成承重五轴（task×consumer×metric×Pattern×Program）回收 +0.6860。</p>
{adjudicator}
{final}

<h2>主张边界（汇报时必须一起说）</h2>
<div class="warnbox">
<ul>
<li>NOAA 2025 是反馈消耗式确认，<b>不是</b>冻结后的 Fast-only held-out。</li>
<li>分类正效应在 GunPoint 族 development 上；Epilepsy2 密封场是守卫，不是正迁移。</li>
<li>修订环复证的是「避免重复错误」，不是「越改越准」。</li>
<li>n 小、多为单轨迹；Dataset 名不得当成跨域证据。</li>
</ul>
</div>

<blockquote>
向老师可说的一句：我们已经把「经验如何合法进入、如何被当前数据检验、如何在不匹配时保持沉默」做成可审计闭环；
fresh 上证明的是适应成本下降，分类密封场证明的是安全弃权。下一步才是论文级、跨任务的密封正迁移主实验。
</blockquote>

<footer>
工件：<code>artifacts/functional/e2/fresh_confirmation_v1.md</code> ·
<code>l1_ladder_v2_replay_r1.md</code> ·
<code>sa1_minimal_r1.md</code> ·
<code>capstone_epilepsy2_final.md</code> ·
<code>docs/CLS_LINE_FINAL_REPORT_2026-08-28.md</code><br>
生成器：<code>evaluation/functional/run_teacher_briefing_viz.py</code> · 内联 CSS/SVG，离线可打开
</footer>
</body>
</html>
"""


def main() -> int:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"== html -> {OUT_HTML} ({OUT_HTML.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
