"""Self-contained stage briefing for SelfEvolvingHarnessTS (v3).

Content rules for this revision:
- No audience-addressing wording anywhere (the page never names its reader).
- Plain-Chinese phrasing; every term of art gets a short in-place gloss.
- Full research narrative: question -> system -> evidence map -> fresh result ->
  classification line -> controlled mechanisms -> natural & negative results ->
  mechanism facts -> roadmap -> claim boundaries.
- Every block carries its evidence grade, following the repo's four-tier
  discipline (docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md).

Numbers come from already-exposed artifacts. No new evaluation.

  python evaluation/functional/run_teacher_report_viz.py
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
OUT_HTML = PROJECT_ROOT / "artifacts" / "visualization" / "teacher_report.html"
NOAA_STATION = "72203812897"
NOAA_ORIGIN = 6552
NOAA_CONTEXT = 192


# ── helpers ──────────────────────────────────────────────────────────────────
def _e(s: Any) -> str:
    return _html.escape(str("" if s is None else s))


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npy_f8(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"not npy: {path}")
    hlen = int.from_bytes(raw[8:10], "little")
    data = raw[10 + hlen :]
    return struct.unpack("<%dd" % (len(data) // 8), data)


def gp(grade: str) -> str:
    label = {
        "fresh": "FRESH · 密封数据一次性打开",
        "nat": "NATURAL · 自然数据",
        "dev": "DEVELOPMENT · 已曝光数据",
        "pc": "POSITIVE_CONTROL · 人为注入考题",
        "mech": "MECHANISM · 存档重放验证",
        "instr": "INSTRUMENT · 仪器自证",
        "sealed": "SEALED · 从未看过",
    }[grade]
    return f'<span class="gp gp-{grade}">{label}</span>'


def gp_short(grade: str) -> str:
    short = {
        "fresh": "FRESH", "nat": "NATURAL", "dev": "DEVELOPMENT",
        "pc": "POSITIVE_CONTROL", "mech": "MECHANISM", "instr": "INSTRUMENT",
        "sealed": "SEALED",
    }[grade]
    return f'<span class="gp gp-{grade}">{short}</span>'


def chip(text: str, kind: str = "gray") -> str:
    return f'<span class="chip chip-{kind}">{_e(text)}</span>'


def sec_head(no: str, title: str, sub: str, grade: str | None = None) -> str:
    g = f' <span class="sec-grade">{gp_short(grade)}</span>' if grade else ""
    return (f'<div class="sec-head"><span class="sec-no">{no}</span>'
            f'<h2>{title}</h2>{g}</div><p class="sec-sub">{sub}</p>')


# ── SVG builders ────────────────────────────────────────────────────────────
def svg_series(values: Sequence[float]) -> str:
    w, h = 1000, 240
    left, right, top, bottom = 46, 14, 30, 30
    pw, ph = w - left - right, h - top - bottom
    finite = [v for v in values if v is not None and not math.isnan(v)]
    lo, hi = min(finite), max(finite)
    pad = (hi - lo) * 0.14 or 1.0
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

    area = ""
    if segs:
        pts0 = segs[0]
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts0)
        d += f" L {xv(len(values)-1):.2f},{top+ph} L {left},{top+ph} Z"
        area = f'<path d="{d}" fill="url(#gArea)" opacity="0.55"/>'
    lines = "".join(
        f'<polyline fill="none" stroke="#2563eb" stroke-width="1.8" '
        f'stroke-linejoin="round" points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in seg)}"/>'
        for seg in segs
    )
    ticks = "".join(
        f'<line x1="{x:.1f}" y1="{top+ph+2:.1f}" x2="{x:.1f}" y2="{top+ph+11:.1f}" '
        f'stroke="#e11d48" stroke-width="1.6"/>' for x in miss_x
    )
    grid = "".join(
        f'<line x1="{left}" y1="{top+i*ph/3:.1f}" x2="{w-right}" y2="{top+i*ph/3:.1f}" '
        f'stroke="#e8edf5" stroke-dasharray="3,3"/>' for i in range(4)
    )
    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">
<defs><linearGradient id="gArea" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#2563eb" stop-opacity="0.28"/>
<stop offset="1" stop-color="#2563eb" stop-opacity="0.02"/></linearGradient></defs>
{grid}<line x1="{left}" y1="{top+ph}" x2="{w-right}" y2="{top+ph}" stroke="#cbd5e1"/>
{area}{lines}{ticks}
<text x="{left-8}" y="{top+8}" text-anchor="end" fill="#94a3b8" font-size="10">{_fmt(hi,1)}</text>
<text x="{left-8}" y="{top+ph}" text-anchor="end" fill="#94a3b8" font-size="10">{_fmt(lo,1)}</text>
<text x="{left}" y="{h-6}" fill="#94a3b8" font-size="10">时间（小时粒度 · 共 {NOAA_CONTEXT} 步）</text>
</svg>"""


def svg_flip() -> str:
    """C12: same injected block, same program applied once, two consumers."""
    w, h = 900, 196
    zx = 470
    ppu = 360 / 0.4059
    f1, f2 = zx + 0.0648 * ppu, zx + 0.4059 * ppu
    a1, a2 = zx - 0.0455 * ppu, zx - 0.2808 * ppu

    def tk(v: float) -> str:
        x = zx + v * ppu
        lab = f"+{v:.1f}" if v > 0 else (f"−{abs(v):.1f}" if v < 0 else "0")
        return (f'<line x1="{x:.1f}" y1="40" x2="{x:.1f}" y2="158" stroke="#eef2f8"/>'
                f'<text x="{x:.1f}" y="176" text-anchor="middle" fill="#94a3b8" font-size="10">{lab}</text>')

    grid = "".join(tk(v) for v in (0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2))
    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">
{grid}
<text x="14" y="70" fill="#334155" font-size="12" font-weight="700">预测任务</text>
<text x="14" y="86" fill="#94a3b8" font-size="10">延后复核平均改善</text>
<rect x="{f1:.1f}" y="52" width="{max(f2-f1,4):.1f}" height="26" rx="6" fill="#2563eb" opacity="0.9"/>
<text x="{f1-8:.1f}" y="69" text-anchor="end" fill="#1e40af" font-size="11" font-weight="700">+0.0648</text>
<text x="{f2+8:.1f}" y="69" fill="#1e40af" font-size="11" font-weight="700">+0.4059（最强程序）</text>
<text x="14" y="132" fill="#334155" font-size="12" font-weight="700">异常检测</text>
<text x="14" y="148" fill="#94a3b8" font-size="10">平均变化</text>
<rect x="{a2:.1f}" y="114" width="{max(a1-a2,4):.1f}" height="26" rx="6" fill="#e11d48" opacity="0.9"/>
<text x="{a2-8:.1f}" y="131" text-anchor="end" fill="#9f1239" font-size="11" font-weight="700">−0.2808</text>
<text x="{a1+8:.1f}" y="131" fill="#9f1239" font-size="11" font-weight="700">−0.0455</text>
<line x1="{zx}" y1="40" x2="{zx}" y2="158" stroke="#64748b" stroke-width="1.4"/>
<circle cx="{zx}" cy="99" r="3" fill="#64748b"/>
<text x="{zx}" y="30" text-anchor="middle" fill="#64748b" font-size="11" font-weight="700">0（不做任何处理）</text>
</svg>"""


def svg_cost() -> str:
    rows = [
        ("主要口径 · 第一次找到有效方案", 69, 123, "−43.9%"),
        ("主要口径 · 适应全程重训总数", 99, 195, "−49.2%"),
        ("第二口径 · 第一次找到有效方案", 75, 78, "−3.8%"),
        ("第二口径 · 适应全程重训总数", 102, 105, "−2.9%"),
    ]
    w, row_h, top = 900, 68, 6
    bx, bw = 208, 520
    vmax = 195.0
    parts: list[str] = []
    for i, (label, a5, a3, delta) in enumerate(rows):
        y = top + i * row_h
        w5, w3 = bw * a5 / vmax, bw * a3 / vmax
        parts.append(f'<text x="14" y="{y+16:.0f}" fill="#334155" font-size="12" font-weight="700">{_e(label)}</text>')
        parts.append(f'<rect x="{bx}" y="{y+8}" width="{w5:.1f}" height="15" rx="4" fill="#2563eb"/>')
        parts.append(f'<text x="{bx+w5+8:.1f}" y="{y+20}" fill="#1e40af" font-size="11.5" font-weight="700">{a5}</text>')
        parts.append(f'<rect x="{bx}" y="{y+27}" width="{w3:.1f}" height="15" rx="4" fill="#e11d48"/>')
        parts.append(f'<text x="{bx+w3+8:.1f}" y="{y+39}" fill="#9f1239" font-size="11.5" font-weight="700">{a3}</text>')
        parts.append(f'<text x="876" y="{y+26}" text-anchor="end" fill="#059669" font-size="13" font-weight="800">{delta}</text>')
    parts.append(f'<text x="{bx}" y="{top+4*row_h+6}" fill="#94a3b8" font-size="10">0</text>')
    parts.append(f'<text x="{bx+bw}" y="{top+4*row_h+6}" text-anchor="end" fill="#94a3b8" font-size="10">195 次重训（重训 = 下游模型在处理后数据上重新训练一次）</text>')
    return f'<svg viewBox="0 0 {w} {top+4*row_h+14}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def svg_donut(pct: float, center: str, sub: str) -> str:
    r = 56
    c = 2 * math.pi * r
    dash = c * pct / 100
    return f"""<svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" role="img">
<circle cx="80" cy="80" r="{r}" fill="none" stroke="#e8edf5" stroke-width="14"/>
<circle cx="80" cy="80" r="{r}" fill="none" stroke="#059669" stroke-width="14"
 stroke-linecap="round" stroke-dasharray="{dash:.1f} {c:.1f}" transform="rotate(-90 80 80)"/>
<text x="80" y="78" text-anchor="middle" font-size="24" font-weight="800" fill="#065f46">{_e(center)}</text>
<text x="80" y="98" text-anchor="middle" font-size="10.5" fill="#64748b">{_e(sub)}</text>
</svg>"""


def svg_slope() -> str:
    w, h = 900, 268
    xs = [170, 450, 730]
    vals = [0.7710, 0.5583, 0.0850]
    names = ["无经验（对照）", "窄范围经验卡", "五轴范围经验卡"]
    subs = ["全程累计误差 0.7710", "第一步省 0.2127（通过线 0.0885）", "第二步再省 0.4733 · 合计 0.6860"]

    def yv(v: float) -> float:
        return 34 + (0.85 - v) / 0.85 * 150

    parts = [f'<line x1="60" y1="{yv(0):.1f}" x2="{w-30}" y2="{yv(0):.1f}" stroke="#e8edf5" stroke-dasharray="3,3"/>',
             f'<text x="{w-30}" y="{yv(0)-6:.1f}" text-anchor="end" fill="#94a3b8" font-size="10">0</text>']
    for i in range(2):
        x1, x2 = xs[i], xs[i + 1]
        y1, y2 = yv(vals[i]), yv(vals[i + 1])
        parts.append(f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" stroke="#c3d4f5" stroke-width="3"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<rect x="{mx-88:.0f}" y="{my-30:.0f}" width="176" height="22" rx="11" fill="#ecfdf5" stroke="#6ee7b7"/>')
        parts.append(f'<text x="{mx:.0f}" y="{my-15:.0f}" text-anchor="middle" fill="#047857" font-size="11" font-weight="700">{_e(subs[i+1])}</text>')
    for i, (x, v, name) in enumerate(zip(xs, vals, names)):
        y = yv(v)
        parts.append(f'<circle cx="{x}" cy="{y:.1f}" r="9" fill="#2563eb" stroke="#fff" stroke-width="3"/>')
        parts.append(f'<text x="{x}" y="{y+34:.1f}" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="800">{_fmt(v)}</text>')
        parts.append(f'<text x="{x}" y="{y+52:.1f}" text-anchor="middle" fill="#334155" font-size="12" font-weight="600">{_e(name)}</text>')
    parts.append('<text x="60" y="24" fill="#64748b" font-size="11">累计误差 regret（与"事后最优选择"相比差多少，越低越好）</text>')
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def svg_rings() -> str:
    r, c = 50, 2 * math.pi * 50
    d3 = c * 0.875
    return f"""<svg viewBox="0 0 420 150" xmlns="http://www.w3.org/2000/svg" role="img">
<circle cx="95" cy="66" r="{r}" fill="none" stroke="#e8edf5" stroke-width="12"/>
<circle cx="95" cy="66" r="{r}" fill="none" stroke="#059669" stroke-width="12" stroke-linecap="round"
 transform="rotate(-90 95 66)" stroke-dasharray="{c:.1f}"/>
<text x="95" y="72" text-anchor="middle" font-size="21" font-weight="800" fill="#065f46">1.0</text>
<text x="95" y="136" text-anchor="middle" font-size="12" fill="#334155" font-weight="700">带经验（A5）</text>
<circle cx="300" cy="66" r="{r}" fill="none" stroke="#e8edf5" stroke-width="12"/>
<circle cx="300" cy="66" r="{r}" fill="none" stroke="#e11d48" stroke-width="12" stroke-linecap="round"
 transform="rotate(-90 300 66)" stroke-dasharray="{d3:.1f} {c:.1f}"/>
<text x="300" y="72" text-anchor="middle" font-size="21" font-weight="800" fill="#9f1239">0.875</text>
<text x="300" y="136" text-anchor="middle" font-size="12" fill="#334155" font-weight="700">从零开始（A3）</text>
<text x="197" y="70" text-anchor="middle" font-size="15" font-weight="800" fill="#475569">+0.125</text>
<text x="197" y="88" text-anchor="middle" font-size="10" fill="#94a3b8">适应过程得分之差</text>
</svg>"""


def svg_w48() -> str:
    vals = [("Coffee", 0.32143), ("ECG200", 0.19000), ("FordA", -0.00227), ("GunPoint", 0.52000)]
    w, h, top, bottom = 640, 210, 26, 40
    ph = h - top - bottom
    vmax, vmin = 0.52, -0.01
    y0 = top + vmax / (vmax - vmin) * ph

    def yv(v: float) -> float:
        return top + (vmax - v) / (vmax - vmin) * ph

    bw, gap = 74, 46
    x0 = 56
    parts = [f'<line x1="40" y1="{y0:.1f}" x2="{w-16}" y2="{y0:.1f}" stroke="#64748b" stroke-width="1.2"/>']
    for i, (name, v) in enumerate(vals):
        x = x0 + i * (bw + gap)
        y, bh = min(yv(v), y0), max(abs(yv(v) - y0), 2)
        color = "#7c3aed" if v >= 0 else "#e11d48"
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{bh:.1f}" rx="5" fill="{color}" opacity="0.92"/>')
        ly = y - 6 if v >= 0 else y + bh + 14
        parts.append(f'<text x="{x+bw/2:.0f}" y="{ly:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="#334155">{_fmt(v, 3)}</text>')
        parts.append(f'<text x="{x+bw/2:.0f}" y="{h-12}" text-anchor="middle" font-size="11.5" fill="#64748b">{_e(name)}</text>')
    parts.append('<text x="40" y="16" fill="#64748b" font-size="11">考试数据一次性打开 · 只凭经验与数据形态做的修复决策带来的改善</text>')
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def svg_framework() -> str:
    boxes = [
        (26, "① 执行器 Runtime", "原始序列留在服务器端", "数学算子精确计算，不靠模型口算", "#2563eb"),
        (258, "② 经验库 Memory", "经验记录 / 经验卡", "成功、失败、矛盾案例都能查到", "#7c3aed"),
        (490, "③ 进化 Evolution", "近窗试算 + 延后复核", "过关才采纳，并修订适用范围", "#d97706"),
        (722, "④ 冻结 Freeze", "冻结后只读部署", "最后统一阅卷一次，成绩不回流", "#0f766e"),
    ]
    parts = ['<text x="26" y="22" fill="#64748b" font-size="11.5">同一个循环：带着旧知识进新域 → 练习区适应 → 冻结 → 零反馈考场验收</text>']
    for i, (x, t, a, b, c) in enumerate(boxes):
        parts.append(
            f'<rect x="{x}" y="46" width="204" height="112" rx="12" fill="#fff" stroke="{c}" stroke-width="1.8"/>'
            f'<text x="{x+102}" y="78" text-anchor="middle" font-weight="800" font-size="14" fill="{c}">{t}</text>'
            f'<text x="{x+102}" y="104" text-anchor="middle" font-size="11.5" fill="#334155">{a}</text>'
            f'<text x="{x+102}" y="126" text-anchor="middle" font-size="11" fill="#64748b">{b}</text>'
        )
        if i < 3:
            x2 = boxes[i + 1][0]
            parts.append(
                f'<line x1="{x+204}" y1="102" x2="{x2}" y2="102" stroke="#94a3b8" stroke-width="2"/>'
                f'<polygon points="{x2},102 {x2-9},97 {x2-9},107" fill="#94a3b8"/>'
            )
    parts.append(
        '<path d="M 824 158 L 824 196 L 360 196 L 360 162" fill="none" stroke="#94a3b8" '
        'stroke-width="1.8" stroke-dasharray="5,4"/>'
        '<polygon points="360,158 355,168 365,168" fill="#94a3b8"/>'
        '<text x="592" y="188" text-anchor="middle" font-size="11" fill="#64748b">'
        '已打开的证据只进入下一轮知识，不回头修改本次成绩</text>'
    )
    return f'<svg viewBox="0 0 926 210" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


# ── HTML fragment builders ──────────────────────────────────────────────────
def timeline(steps: list[dict[str, Any]], accent: str, headline: str) -> str:
    mode_label = {
        "FULL_PRICE_SEARCH": ("chip-amber", "完整搜索（每个候选都试）"),
        "DIRECT_RECALL": ("chip-green", "直接复用已学方案"),
        "OUT_OF_SELECTION_PROBE": ("chip-gray", "范围外抽检"),
    }
    prog_label = {
        "identity": ("gray", "不处理"),
        "outlier_mad": ("blue", "离群点修复 outlier_mad"),
        "repair_level_shift": ("blue", "台阶修复 repair_level_shift"),
    }
    items = []
    for i, st in enumerate(steps, 1):
        cls, lab = mode_label.get(str(st.get("mode")), ("chip-gray", str(st.get("mode"))))
        plan = st.get("adopted_plan") or {}
        prog = (plan.get("program") if isinstance(plan, dict) else None) or "—"
        pcls, plab = prog_label.get(str(prog), ("blue", str(prog)))
        sg, dg = st.get("support_aggregate_gain"), st.get("delayed_aggregate_gain")
        harm = st.get("harmed_eval_series_count")
        ok = bool(st.get("adopted_delayed_positive"))
        if ok:
            verdict = chip("采纳（延后复核为正）", "green")
        elif st.get("mode") == "FULL_PRICE_SEARCH":
            verdict = chip("本轮没有找到可采纳方案", "red")
        else:
            verdict = chip("未采纳", "gray")
        items.append(f"""
<div class="tl-item{' a3' if accent == 'a3' else ''}">
  <div class="tl-card">
    <div class="tl-top"><b>第 {i} 步 · {_e(st.get('step'))}</b>
      {chip(lab, cls)}{chip(plab, pcls)}{verdict}</div>
    <div class="tl-metrics">
      <span>近窗试算 <b>{_fmt(sg, 6) if sg is not None else '—'}</b></span>
      <span>延后复核 <b>{_fmt(dg, 6) if dg is not None else '—'}</b></span>
      <span>受害序列 <b>{harm if harm is not None else '—'}</b></span>
      <span>模型调用 <b>{st.get('llm_calls')}</b></span>
      <span class="retrain">重训 <b>{st.get('consumer_retrains')}</b></span>
    </div>
  </div>
</div>""")
    return f"""
<div class="tl-col">
  <div class="tl-head" style="color:{'#1e40af' if accent == 'a5' else '#9f1239'}">{headline}</div>
  <div class="tl">{''.join(items)}</div>
</div>"""


def verdict_banner(verdict: str, grade: str, text: str) -> str:
    return f"""<div class="banner">
  <div class="banner-v">{gp_short(grade)}<b class="v">{_e(verdict)}</b></div>
  <div class="banner-t">{text}</div>
</div>"""


# ── CSS / JS (plain strings, no f-string) ───────────────────────────────────
CSS = """
:root{--bg:#f5f7fb;--card:#fff;--ink:#0f172a;--mut:#5b6b7f;--line:#e3e9f1;
--brand:#2563eb;--a5:#2563eb;--a3:#e11d48;--ok:#059669;--warn:#d97706}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.68;font-size:14.5px;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
nav{position:sticky;top:0;z-index:60;background:rgba(255,255,255,.9);backdrop-filter:blur(10px);
border-bottom:1px solid var(--line)}
nav .wrap{max-width:1180px;margin:0 auto;display:flex;align-items:center;padding:0 14px;overflow-x:auto}
nav .brand{font-weight:800;font-size:13px;color:var(--ink);padding:10px 12px 10px 0;white-space:nowrap}
nav a{padding:10px 11px;color:var(--mut);text-decoration:none;font-size:12.5px;font-weight:650;
white-space:nowrap;border-bottom:2px solid transparent}
nav a:hover{color:var(--ink)}nav a.active{color:var(--brand);border-bottom-color:var(--brand)}
.hero{background:linear-gradient(135deg,#0b1220,#111d38 55%,#182748);color:#e8eefc;
padding:58px 20px 42px;position:relative;overflow:hidden}
.hero::after{content:"";position:absolute;inset:0;background:
radial-gradient(620px 250px at 86% 8%,rgba(124,58,237,.22),transparent 60%),
radial-gradient(540px 230px at 10% 92%,rgba(37,99,235,.20),transparent 60%);pointer-events:none}
.hero-inner{max-width:1180px;margin:0 auto;position:relative;z-index:1}
.eyebrow{letter-spacing:.16em;font-size:12px;font-weight:800;color:#93b4ff;text-transform:uppercase;margin:0 0 10px}
.hero h1{font-size:30px;line-height:1.32;margin:0 0 12px;font-weight:800}
.hero h1 em{font-style:normal;background:linear-gradient(90deg,#7dd3fc,#c4b5fd);
-webkit-background-clip:text;background-clip:text;color:transparent}
.hero .sub{color:#aebcd8;max-width:920px;margin:0 0 24px;font-size:14.5px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.kpi{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);border-radius:14px;padding:14px 16px}
.kpi .n{font-size:25px;font-weight:800;color:#fff;font-variant-numeric:tabular-nums}
.kpi .n small{font-size:13px;font-weight:650;color:#9fb3d9}
.kpi .l{font-size:12px;color:#9fb3d9;margin-top:3px;line-height:1.5}
.gradepills{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px;align-items:center}
details.gloss{margin-top:14px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14);
border-radius:12px;color:#dbe6ff}
details.gloss summary{cursor:pointer;padding:10px 16px;font-size:13px;font-weight:700;list-style:none}
details.gloss summary::-webkit-details-marker{display:none}
details.gloss .body{padding:0 16px 14px;font-size:12.5px;color:#b9c6e2;line-height:1.9}
details.gloss b{color:#fff}
main{max-width:1180px;margin:0 auto;padding:6px 16px 44px}
section{margin:36px 0;scroll-margin-top:64px}
.sec-head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.sec-no{font-size:12px;font-weight:800;color:var(--brand);letter-spacing:.1em}
h2{font-size:20px;margin:0;font-weight:800}
h3{font-size:15.5px;margin:0 0 8px;font-weight:800}
.sec-grade{margin-left:auto}
.sec-sub{color:var(--mut);font-size:13px;margin:4px 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;
box-shadow:0 1px 2px rgba(15,23,42,.04)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
@media(max-width:920px){.grid2,.grid3,.grid4,.kpis{grid-template-columns:1fr}}
.gp{display:inline-block;font-size:10.5px;font-weight:800;padding:2px 10px;border-radius:999px;
letter-spacing:.04em;vertical-align:middle}
.gp-fresh{background:#d1fae5;color:#065f46}.gp-nat{background:#ccfbf1;color:#115e59}
.gp-dev{background:#dbeafe;color:#1e40af}.gp-pc{background:#ede9fe;color:#5b21b6}
.gp-mech{background:#fef3c7;color:#92400e}.gp-instr{background:#f1f5f9;color:#475569}
.gp-sealed{background:#ffe4e6;color:#9f1239}
.chip{display:inline-block;margin:1px 3px;padding:1px 9px;border-radius:999px;font-size:11.5px;font-weight:650}
.chip-amber{background:#fef3c7;color:#92400e}.chip-green{background:#d1fae5;color:#065f46}
.chip-gray{background:#f1f5f9;color:#475569}.chip-blue{background:#dbeafe;color:#1e40af}
.chip-violet{background:#ede9fe;color:#5b21b6}.chip-red{background:#ffe4e6;color:#9f1239}
.banner{display:flex;gap:16px;align-items:center;background:linear-gradient(90deg,#ecfdf5,#f0fdfa);
border:1px solid #a7f3d0;border-radius:14px;padding:14px 18px;margin:14px 0}
.banner-v{display:flex;align-items:center;gap:10px;flex-shrink:0}
.banner-v .v{font-size:17px;font-weight:800;color:#065f46}
.banner-t{font-size:13px;color:#334155}
.tl-cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:920px){.tl-cols{grid-template-columns:1fr}}
.tl-head{font-weight:800;font-size:13.5px;margin-bottom:10px}
.tl{position:relative;padding-left:26px}
.tl::before{content:"";position:absolute;left:8px;top:8px;bottom:8px;width:2px;background:var(--line)}
.tl-item{position:relative;margin:0 0 10px}
.tl-item::before{content:"";position:absolute;left:-24px;top:12px;width:11px;height:11px;border-radius:50%;
background:var(--a5);border:2px solid #fff;box-shadow:0 0 0 2px var(--a5)}
.tl-item.a3::before{background:var(--a3);box-shadow:0 0 0 2px var(--a3)}
.tl-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:9px 12px}
.tl-top{margin-bottom:5px}
.tl-metrics{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--mut)}
.tl-metrics b{color:var(--ink);font-variant-numeric:tabular-nums}
.tl-metrics .retrain{margin-left:auto;background:#f1f5f9;border-radius:6px;padding:0 8px;font-weight:700}
table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
th{background:#f4f7fb;text-align:left;padding:8px 11px;color:#64748b;font-weight:700;
border-bottom:2px solid var(--line);white-space:nowrap}
td{padding:8px 11px;border-bottom:1px solid #eef2f8;vertical-align:top}
tr:last-child td{border-bottom:none}
.rbar{height:9px;border-radius:5px;background:#e8edf5;overflow:hidden;margin-top:4px}
.rbar i{display:block;height:100%;border-radius:5px;background:linear-gradient(90deg,#2563eb,#7c3aed)}
.stepper{display:flex;flex-wrap:wrap;gap:10px}
.step{flex:1;min-width:230px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.step .no{display:inline-flex;width:24px;height:24px;border-radius:50%;background:#f1f5f9;color:#475569;
align-items:center;justify-content:center;font-weight:800;font-size:12px;margin-right:8px}
.step b{font-size:13.5px}.step p{margin:6px 0 0;font-size:12.5px;color:var(--mut)}
.step.current{border:2px solid var(--brand);background:#f8faff}
.step.current .no{background:var(--brand);color:#fff}
.warnbox{background:#fffbeb;border:1px solid #fcd34d;border-radius:14px;padding:16px 20px}
.warnbox ul{margin:6px 0 0;padding-left:20px}.warnbox li{margin:5px 0;font-size:13.5px}
.quote{background:linear-gradient(135deg,#0b1220,#16233f);color:#dbe6ff;border-radius:16px;
padding:24px 28px;font-size:15.5px;line-height:1.8;margin:30px 0}
.quote b{color:#fff}
footer{max-width:1180px;margin:0 auto;padding:10px 16px 40px;color:#94a3b8;font-size:12px}
footer code{background:#f1f5f9;padding:1px 5px;border-radius:4px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:.92em}
.note{font-size:12.5px;color:var(--mut)}
.chain{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:8px 0}
.chain .cv{background:#fff;border:1px solid var(--line);border-radius:10px;padding:6px 12px;font-size:12px}
.chain .cv b{color:#5b21b6}
.chain .arr{color:#94a3b8;font-weight:800}
.mini{font-size:12.5px;color:var(--mut);margin-top:8px}
.kv{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.kv .pill{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:4px 10px;font-size:12px}
.kv .pill b{font-variant-numeric:tabular-nums}
@media print{nav{display:none}.hero{padding:30px 20px}}
"""

JS = """
(function(){
  var els=document.querySelectorAll('[data-count]');
  els.forEach(function(el){
    var target=parseFloat(el.getAttribute('data-count'));
    var dec=parseInt(el.getAttribute('data-dec')||'0',10);
    var pre=el.getAttribute('data-prefix')||'';var suf=el.getAttribute('data-suffix')||'';
    var t0=null,dur=950;
    function fmt(v){var s=v.toFixed(dec);return pre+s+suf}
    el.textContent=fmt(0);
    function step(ts){if(!t0)t0=ts;var p=Math.min((ts-t0)/dur,1);
      var eased=1-Math.pow(1-p,3);el.textContent=fmt(target*eased);
      if(p<1)requestAnimationFrame(step);else el.textContent=fmt(target);}
    requestAnimationFrame(step);
  });
  var links=document.querySelectorAll('nav a[data-sec]');
  var secs=[];
  links.forEach(function(a){var s=document.getElementById(a.getAttribute('data-sec'));if(s)secs.push([s,a]);});
  function onScroll(){var y=window.scrollY+90,cur=null;
    secs.forEach(function(p){if(p[0].offsetTop<=y)cur=p[1];});
    links.forEach(function(a){a.classList.toggle('active',a===cur);});}
  window.addEventListener('scroll',onScroll,{passive:true});onScroll();
})();
"""


# ── assemble page ───────────────────────────────────────────────────────────
def build_html() -> str:
    fresh = _load_json(E2 / "fresh_confirmation_v1.json")
    sa1 = _load_json(E2 / "sa1_minimal_r1.json")
    cap = _load_json(E2 / "capstone_epilepsy2_final.json")
    pooled = fresh["cells"]["pooled"]

    npy = (PROJECT_ROOT / "data" / "benchmark_noaa_fresh_v1" / "series"
           / NOAA_STATION / "values.npy")
    window = list(_load_npy_f8(npy)[NOAA_ORIGIN : NOAA_ORIGIN + NOAA_CONTEXT])
    n_miss = sum(1 for v in window if math.isnan(v))

    nav_items = [
        ("motivation", "研究问题"), ("system", "系统设计"), ("map", "数据盘点"),
        ("fresh", "核心结果"), ("cls", "第二任务"), ("ctrl", "受控验证"),
        ("neg", "自然数据与未通过项"), ("mech", "规律发现"), ("road", "下一步"), ("limit", "使用边界"),
    ]
    nav = "".join(f'<a href="#{i}" data-sec="{i}">{t}</a>' for i, t in nav_items)

    hero = f"""
<header class="hero"><div class="hero-inner">
  <p class="eyebrow">SelfEvolvingHarnessTS · 阶段性研究简报 · 2026-08-31</p>
  <h1>会积累、会校准、<br>知道什么时候该住手的<em>时序数据准备系统</em></h1>
  <p class="sub">时序数据没有统一的"干净"标准：同一个修复操作，在预测任务里是改善，在异常检测任务里可能是破坏。
本项目把<b>积累经验</b>和<b>适应新数据</b>做成同一个循环：先在旧数据上攒下经过审计的"经验卡"，
到新数据上先用练习区反馈校准，最后把整个系统冻结，在从未看过的数据上只考一次。
本页所有数字都来自已经打开过的实验记录，没有为了做图新跑任何实验。</p>
  <div class="kpis">
    <div class="kpi"><div class="n"><span data-count="43.9" data-dec="1" data-prefix="−" data-suffix="%">−43.9%</span></div>
      <div class="l">全新数据上的适应成本：找到第一个有效方案，带经验 69 次 vs 从零 123 次模型重训</div></div>
    <div class="kpi"><div class="n"><span data-count="0.686" data-dec="3" data-prefix="−">−0.686</span></div>
      <div class="l">第二个任务上：带适用范围的经验卡把累计误差从 0.7710 压到 0.0850（两次独立重跑同向）</div></div>
    <div class="kpi"><div class="n"><span data-count="1.0" data-dec="1">1.0</span><small> vs 0.875</small></div>
      <div class="l">受控对照实验的适应过程得分：带经验 vs 从零开始，且全程零事故</div></div>
    <div class="kpi"><div class="n"><span data-count="41" data-dec="0">41</span><small> 条</small></div>
      <div class="l">特意保留、从未看过的数据，留给系统冻结后的最终考试</div></div>
  </div>
  <div class="gradepills">
    <span style="color:#9fb3d9;font-size:12px;font-weight:700;margin-right:4px">证据等级（越高越硬）：</span>
    {gp('fresh')}{gp('nat')}{gp('dev')}{gp('pc')}{gp('mech')}{gp('instr')}{gp('sealed')}
  </div>
  <details class="gloss"><summary>名词与读法速览（点开）</summary><div class="body">
    <b>重训</b>：下游模型在处理后的数据上重新训练一次；适应成本主要用它衡量，越少越省。
    <b>近窗试算 / 延后复核</b>：每个候选方案先在时间靠前的窗口试（Support），
    过关后必须在时间上靠后的独立窗口再确认一次（delayed），两道都过才算数。
    <b>经验卡</b>：一条带"适用范围"的处理经验；范围对不上时必须保持沉默，宁可不动作。
    <b>不处理（identity）</b>：判断这段数据不需要修复，原样放行——这本身也是一个正式决策。
    <b>累计误差 regret</b>：每一步与"事后最优选择"的差距累加，越低越好。
    <b>A5 / A3 / Static</b>：A5 = 完整系统（带历史经验 + 现场适应）；A3 = 同一系统清空历史经验（对照）；
    Static = 经验冻结不用、也不适应（对照）。
  </div></details>
</div></header>"""

    s1 = f"""
<section id="motivation">
{sec_head("01", "研究问题：'干净'没有统一定义，它取决于谁来用这份数据",
"为什么必须做成自适应系统，而不是一条固定的清洗流水线。", "pc")}
<div class="grid2">
  <div class="card">
    <h3>同一个修复，两个任务，一好一坏</h3>
    <p style="margin:0">我们做了一个严格对照：同一份被注入缺陷的训练数据、同一个修复程序只执行一次、
两个下游任务消费<b>完全相同</b>的字节。结果四种离群修复程序<b>全部同向翻转</b>——
预测任务的延后复核平均改善 {chip("+0.0648 ~ +0.4059", "blue")}，
异常检测任务的平均变化 {chip("−0.0455 ~ −0.2808", "red")}，反例 <b>0 例</b>。
其中最强的那个程序（winsorize），在预测任务 +0.4059，在异常检测上却让关键指标掉了 0.1672。</p>
  </div>
  <div class="card">
    <h3>这决定了系统的三条设计约束</h3>
    <ul style="margin:6px 0 0;padding-left:18px">
      <li>数据好不好，只能看<b>下游任务的实际效果</b>，不能只看数据本身；</li>
      <li>旧经验不能盲目搬到新数据，也不能永远不用——需要一条<b>合法通道 + 检验关卡</b>；</li>
      <li>判断"新数据像不像旧数据"时，只能用<b>数据形态 × 处理方式 × 任务类型</b>这些看得见的特征，
<b>禁止</b>拿数据集名字说事。</li>
    </ul>
  </div>
</div>
<div class="card" style="margin-top:14px">{svg_flip()}
<p class="note">说明：横轴是相对"完全不处理"的变化，右侧为改善、左侧为变差。这些是<b>人为注入已知答案的考题</b>
（POSITIVE_CONTROL），用来确认整条测量链路确实能读到"任务不同、好坏不同"这个现象；现象本身由真实数据实验交叉印证。
（工件：t1_flip_control_v1 · t1b_training_flip_v3）</p>
</div>
</section>"""

    s2 = f"""
<section id="system">
{sec_head("02", "系统设计：积累经验和适应新数据，是同一个循环",
"四个部件 + 一条纪律：打开过的证据只进入下一轮，绝不回头改本次成绩。")}
<div class="card">{svg_framework()}</div>
<div class="grid2" style="margin-top:14px">
  <div class="card">
    <h3>练习区（held-in）与考场（held-out）</h3>
    <p style="margin:0"><b>练习区</b>：允许反复试、反复学的数据，反馈预算事先定死、不能超。
每个候选方案都要过两道门：先在近处窗口试算，再在时间上靠后的独立窗口复核。<br>
<b>考场</b>：进场前把整个系统冻结，答题时只靠已经学会的策略，<b>不许</b>再学习、再试错；
最后由外部统一阅卷一次，成绩带不回来。</p>
  </div>
  <div class="card">
    <h3>经验进系统的唯一通道</h3>
    <p style="margin:0"><code>原始经验记录 → 确定性整理 → 审计 → 冻结成经验卡 → 部署时只读使用
→ 新数据的现场反馈决定：确认 / 修订 / 否决</code>。<br>
三种角色：{chip("A5 完整系统", "blue")}{chip("A3 清空经验对照", "gray")}{chip("Static 不适应对照", "gray")}——
A3 和 Static 只用来算清"经验值多少、适应值多少"，A5 与它们比时<b>反馈预算完全相同</b>。</p>
  </div>
</div>
</section>"""

    map_rows = [
        ("预测任务三组真实数据（电力 / 交通 / 气象）", "每组 12 条训练 + 8 条评估 × 2 种下游模型", gp_short("dev"), "已用过、已打开；可回归复查，不能再当新考场"),
        ("NOAA 全球小时气象（本页核心结果）", "12 训练 + 4 评估；2024 练习区 + 2025 考试区", gp_short("fresh"), "2025 已按规程一次性打开；更后面还封着一段从未动过"),
        ("NAB 异常检测源数据（4 个来源）", "31 个文件 · 40 条经验记录", gp_short("nat"), "来源数量到此为止，不再扩"),
        ("NAB 异常检测目标数据", "6 个文件", gp_short("dev"), "已打开；只能复查"),
        ("Yahoo S5 异常检测", "65 条序列：前 24 条已用，41 条密封", gp_short("sealed"), "41 条留给冻结后的最终考试，之前绝不读取"),
        ("SMD 多传感器数据", "28 台设备 × 38 通道", gp_short("instr"), "只做过结构勘察；与当前单变量协议不匹配，已关闭"),
    ]
    map_html = "".join(
        f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td class='note'>{d}</td></tr>"
        for a, b, c, d in map_rows
    )
    s3 = f"""
<section id="map">
{sec_head("03", "数据盘点：哪些用过、哪些还封着",
"缺的从来不是数据总量，而是&ldquo;匹配任务 + 能预先定好分界 + 考试部分从未被看过&rdquo;的干净考场。")}
<div class="card"><table>
<tr><th style="width:32%">数据</th><th>规模</th><th>状态</th><th>使用约束</th></tr>
{map_html}
</table></div>
</section>"""

    a5, a3 = pooled["A5"], pooled["A3"]
    tl5 = timeline(a5["trace"], "a5",
                   f"A5 · 带经验 ｜ 第 1 步就找到有效方案（累计 69 次重训）｜ 只用 {a5['llm_calls']} 次模型调用")
    tl3 = timeline(a3["trace"], "a3",
                   f"A3 · 从零开始 ｜ 第 2 步才找到（累计 123 次重训）｜ 用了 {a3['llm_calls']} 次模型调用")
    s4 = f"""
<section id="fresh">
{sec_head("04", "核心结果：全新数据上，带经验的适应成本省一半",
"2024 年数据练、2025 年数据考，考试区按规程一次性打开；所有规则在开考前冻结；自动深度修改全程关闭。", "fresh")}
{verdict_banner("带经验一方在全新数据上胜出（两个评分口径一致）", "fresh",
   f"整场考试只用了 {fresh['llm_call_count']} / {fresh['llm_call_budget']} 次模型调用，"
   f"下游模型共重训 {fresh['consumer_retrains_total']} 次，用时约 {fresh['wall_seconds']/60:.0f} 分钟。")}
<div class="grid2">
  <div class="card">
    <h3>适应成本对比：蓝 = 带经验，红 = 从零开始</h3>
    {svg_cost()}
    <div class="kv">
      <span class="pill">第一次找到有效方案 <b>69 vs 123</b> 次重训</span>
      <span class="pill">全程重训总数 <b>99 vs 195</b></span>
      <span class="pill">最终成绩两方<b>完全相同</b></span>
      <span class="pill">受害序列 <b>1 / 1</b>（相同）</span>
    </div>
    <p class="mini">第二评分口径（按通道分别评分）：75 vs 78 / 102 vs 105，受害 0 / 0，
结论为"打平偏正"（细节见 fresh_confirmation_v1 裁定附录）。</p>
  </div>
  <div class="card" style="display:flex;flex-direction:column;align-items:center;justify-content:center">
    {svg_donut(43.9, "−43.9%", "第一次找到有效方案的成本（主要口径）")}
    <p class="mini" style="text-align:center;max-width:360px">注意边界：两种做法<b>最终成绩一样</b>、伤害一样——
所以这份证据说明的是<b>适应过程更省</b>，不能说"预测更准了"。</p>
  </div>
</div>
<div class="card" style="margin-top:14px">
  <h3>系统实际看到的数据（示例片段）</h3>
  {svg_series(window)}
  <p class="note">NOAA 气象站 {NOAA_STATION} 的真实温度序列，从第 {NOAA_ORIGIN} 小时起取 {NOAA_CONTEXT} 步，
红色竖线是缺失值（这段里 {n_miss} 个，占 {n_miss/NOAA_CONTEXT:.1%}）。
系统要做判断的不是"把线画平滑"，而是：这段数据<b>不需要处理、需要修复、还是该弃权</b>——
而且每个判断都要在时间上靠后的独立窗口复核过才算数。</p>
</div>
<div class="card" style="margin-top:14px">
  <h3>完整决策过程（逐步，全部有记录可查）</h3>
  <div class="tl-cols">{tl5}{tl3}</div>
  <p class="note">白话读法：带经验的一方<b>第一次就找对了药方</b>（离群点修复，延后复核 +0.306），
之后两个新窗口都直接复用，几乎不花代价；从零开始的一方第一次没试出来（选了"不处理"），
第二次才靠完整搜索找到，第三次还要重新再搜一遍。两者最后落点相同——
<b>经验省掉的是中间的反复试错</b>。</p>
</div>
<div class="warnbox" style="margin-top:14px"><b>如实说明：</b>
这次考试是"练完就考"的一次性打开，系统在考试时仍消耗了反馈，还不是"完全冻结、零反馈"的最终形态；
第二评分口径只是打平。两项都有明确的后续安排（见第 09 节路线图）。</div>
</section>"""

    sa_sum = sa1["summary"]

    def _rbar(v: float, vmax: float) -> str:
        w = max(3.0, v / vmax * 100)
        return f'<div class="rbar"><i style="width:{w:.0f}%"></i></div>'

    sa_rows = f"""
<tr><td><b>无经验（对照）</b></td><td><b>{_fmt(sa_sum['A3-reset']['cumulative_regret_distinct_units'])}</b>{_rbar(sa_sum['A3-reset']['cumulative_regret_distinct_units'], 0.8)}</td>
<td>{sa_sum['A3-reset']['probes']}</td><td>0</td><td>0</td><td>0</td></tr>
<tr><td><b>固定版经验卡</b></td><td><b>{_fmt(sa_sum['K0-fixed']['cumulative_regret_distinct_units'])}</b>{_rbar(sa_sum['K0-fixed']['cumulative_regret_distinct_units'], 0.8)}</td>
<td>{sa_sum['K0-fixed']['probes']}</td><td>{sa_sum['K0-fixed']['supplied_in_pool']}</td><td>{sa_sum['K0-fixed']['supplied_refused']}</td><td>0</td></tr>
<tr><td><b>会自我修订的经验卡</b></td><td><b>{_fmt(sa_sum['A5-online']['cumulative_regret_distinct_units'])}</b>{_rbar(sa_sum['A5-online']['cumulative_regret_distinct_units'], 0.8)}</td>
<td>{sa_sum['A5-online']['probes']}</td><td>{sa_sum['A5-online']['supplied_in_pool']}</td><td>{sa_sum['A5-online']['supplied_refused']}</td><td>0</td></tr>"""
    cap_preds = "".join(
        f"<tr><td><b>{p['id']}</b></td><td>{_e(p['claim'])}</td>"
        f"<td>{chip('与预测一致', 'green') if p['held'] else (chip('与预测不符', 'red') if p['held'] is False else chip('未评分', 'gray'))}</td>"
        f"<td class='note'>{_e(p['observed'])}</td></tr>"
        for p in cap["predictions"]
    )
    s5 = f"""
<section id="cls">
{sec_head("05", "换个任务再验一遍：分类任务上的完整故事",
"预测任务（第 04 节）之外，把同一套系统原封不动搬到分类任务。以下四行结论里，前两行已各重跑一次同向复现。", "dev")}
<div class="card">
<table>
<tr><th>#</th><th>结论</th><th>等级</th><th>关键数字</th></tr>
<tr><td>①</td><td>带适用范围的经验卡带来端到端收益</td><td>{gp_short('dev')} 已复现</td>
<td>五轴范围卡 vs 完全无卡：累计误差 <b>0.7710 → 0.0850</b>（差 0.6860）；窄范围卡同底对照差 0.2127</td></tr>
<tr><td>②</td><td>经验卡能随新证据自我修订</td><td>{gp_short('dev')} 已复现</td>
<td>卡版本 v0→v3/v4；第二次遇到同类数据：会修订的卡<b>保持沉默</b>，固定版<b>再次推荐、再次被拦</b>；两次重跑的修订内容逐字节一致</td></tr>
<tr><td>③</td><td>在完全陌生且经验不匹配的数据上，系统知道住手</td><td>{gp_short('fresh')} 密封已证</td>
<td>经验卡全程沉默（推荐 0 次）；双方自己提的方案都有害，全部被近窗试算拦下；零事故</td></tr>
<tr><td>④</td><td>密封且范围匹配时经验能否<b>加分</b></td><td>{gp_short('sealed')} 暂不可考</td>
<td>公开数据里找不到第二个符合全部条件的新考场（已反证排查）——是资源约束，不是方法失败</td></tr>
</table>
</div>
<div class="grid2" style="margin-top:14px">
  <div class="card">
    <h3>累计误差三级台阶：经验卡的范围定得越准，省得越多</h3>
    {svg_slope()}
    <p class="note">这是提前写下预测清单、再开卷对答案的实验：12 条预测<b>中了 7 条</b>。
没中的几条同样有价值——经验卡实际只匹配到 4 个预期单元里的 1 个，
总误差也没降到预设的 0.20 以下。范围表达能力的边界就这样被量出来了。</p>
  </div>
  <div class="card">
    <h3>"同一个错误不犯第二次"：修订环的关键对照</h3>
    <table>
      <tr><th>做法</th><th>累计误差</th><th>试探</th><th>进候选池</th><th>被安全门拦</th><th>事故</th></tr>
      {sa_rows}
    </table>
    <div class="chain">
      <span class="cv"><b>v0</b> 初始卡</span><span class="arr">→</span>
      <span class="cv"><b>v1</b> 依正面证据修订</span><span class="arr">→</span>
      <span class="cv"><b>v2</b> 再依正面证据修订</span><span class="arr">→</span>
      <span class="cv"><b>v3</b> 依"被拦"证据收窄范围<br><span style="color:#9f1239">新增排除：周期变化极低的片段不适用</span></span>
    </div>
    <p class="mini">如实归因：修订直接避免被拦 1 次；省下的 2 次试探来自候选池构成、
<b>不能</b>记在"收窄"头上；两张卡（会修订 vs 固定）的累计误差完全相同 +0.0000——
所以这轮实验的结论是<b>修订买到的是"不重复犯错"，还不是"越改越好"</b>，
预注册时也只按"不更差"来验收，没有多说一句。</p>
  </div>
</div>
<div class="card" style="margin-top:14px">
  <h3>守卫测试：在完全陌生、且经验对不上的数据上，会不会帮倒忙？{gp_short("fresh")}</h3>
  <div class="kv" style="margin:4px 0 10px">
    <span class="pill">结果：<b>中性（没帮上忙，也没帮倒忙）</b></span>
    <span class="pill">三种做法都选了"不处理"，准确率同为 <b>0.5336</b></span>
    <span class="pill">零事故</span>
    <span class="pill">数据封条核验 <b>11/11 通过</b></span>
  </div>
  <p style="margin:6px 0">经验卡因为一个适用条件对不上（要求的局部偏移等级与实测不符），<b>全程沉默</b>：
推荐 0 次。与此同时，两个会自主提案的做法各自独立提出了两个方案，
试算结果 {chip("−0.2750", "red")} 和 {chip("−0.0500", "red")}，<b>全部被安全门拦下</b>。
也就是说：这一场考的不是"经验能否加分"，而是"经验对不上时系统是否安分"——答案是安分。</p>
  <table><tr><th>提前写下的预测</th><th>内容</th><th>结果</th><th>实测记录</th></tr>{cap_preds}</table>
</div>
</section>"""

    s6 = f"""
<section id="ctrl">
{sec_head("06", "受控验证：经验到底改变了什么",
"注入已知答案的对照实验，把&ldquo;经验在起作用&rdquo;拆成一个个可单独验证的机制。", "pc")}
<div class="grid2">
  <div class="card">
    <h3>对照实验：带经验的适应过程得分</h3>
    {svg_rings()}
    <p class="mini">两个真实数据背景（气象 + 电力负荷），分组规则先定死、结果后打开。
带经验一方每个背景都领先 <b>+0.125</b>，且全程零事故；同样的修复方案
<b>不带适用范围</b>地照搬，在两个背景各造成 {chip("0.5 的事件级事故", "red")}。
结论：经验要带着"什么情况下适用"一起带，才既省试错又不出事。</p>
  </div>
  <div class="card">
    <h3>不看答案的决策：该修的修，不该动的住手</h3>
    {svg_w48()}
    <p class="mini">决策时只给经验与当前数据的形态特征，<b>不给</b>任何正确答案和数据集名。
四个数据集平均改善 <b>+0.257</b>；同一套策略若被强制"一律修复"，
两个数据集出现 {chip("满分事故（准确率掉 1.0）", "red")}，而带范围判断的策略零事故。</p>
  </div>
</div>
<div class="grid2" style="margin-top:14px">
  <div class="card">
    <h3>提案会随任务类型完全分开</h3>
    <p style="margin:0">清空全部经验、屏蔽一切结果泄漏、只保留"任务说明"的文字差异
（两份说明删掉任务名后逐字节一致）的情况下：6 次有效抽样<b>全部</b>按任务分开——
跨任务的方案重合度 9 对全部为 <b>0</b>，同任务内最高 0.5。
预测任务三次都选离群修复，异常检测三次都选不处理。<span class="note">（工件 t3_task_exam_v1）</span></p>
  </div>
  <div class="card">
    <h3>写入系统的经验会真实改变后续判断</h3>
    <p style="margin:0">把 10 条历史经验真实写回系统后重新开考：
预测任务的风险拦截 <b>0/3 → 3/3</b>（与写入的矛盾案例一一对应），
异常检测一侧反向回退——回退原因被定位到"经验卡的表达范围不够"这个具体机制上，
而不是笼统的"系统不稳定"。<span class="note">（工件 t4_conflict_experience_v1/v2）</span></p>
  </div>
</div>
</section>"""

    s7 = f"""
<section id="neg">
{sec_head("07", "自然数据结果，以及没有通过的项目",
"自然数据上的正例很宝贵；没有通过的项目同样重要——它们证明&ldquo;不合格就拦下&rdquo;在真实发生，而不是多试几次总能混过去。", "nat")}
<div class="grid2">
  <div class="card">
    <h3>第一次在纯自然数据上走完整个循环 {gp_short("nat")}</h3>
    <p style="margin:0">没有任何人为注入：系统从零开始自主发现修复方案（离群点修复）→
近窗试算 {chip("+0.0593，3 条序列零受害", "green")} → 延后复核 {chip("+0.0111，为正", "green")}
→ 正式成为本地经验卡。同一条数据上一轮的失败尝试也被如实写回，第二轮据此收敛。</p>
    <p class="mini">同一轮的正式对照还给出一个重要发现，判词直译是
{chip("带上跨域经验：更安全，但没有更快", "amber")}——
来自其他域的 20 张负面经验卡让系统变得<b>过度保守</b>，4 个考核窗口全部只敢选"不处理"。
负面经验被当成了"全局警告"，而它本该是"这类情况别用"的精确提醒。
<b>这是当前最需要修的第一问题</b>，已进入下一步计划（第 09 节）。（工件 t6_nab_evaluate_v2）</p>
  </div>
  <div class="card">
    <h3>为什么把这件事当作正面结果来讲</h3>
    <p style="margin:0">系统没有为了好看把"更保守"包装成"迁移成功"：
判词、双样本注记、失败机制的定位全部如实入档。
<b>负例卡的转译缺陷被识别出来</b>，等在下一步修——这正是这套系统强调"证据先行"的意义：
宁可慢，不能错。</p>
  </div>
</div>
<h3 style="margin:18px 0 8px">没有通过 / 已关闭的项目（按时间）</h3>
<div class="grid3">
  <div class="card"><b>周期参数改绑</b>{chip("假设不成立", "red")}
    <p class="mini">想用改周期参数救回两个受害案例：都没救回来（还差 0.0011 / 0.0215，仍不如不处理）。</p></div>
  <div class="card"><b>短间隙规则重放</b>{chip("不复现", "red")}
    <p class="mini">8 个新案例平均 {chip("−0.0163", "red")}，其中 4 个受害；一个数据源 −0.0507。规则退回"暂定"，未放行任何新查询。</p></div>
  <div class="card"><b>组合策略上线评估</b>{chip("方向对 · 安全不过", "amber")}
    <p class="mini">两个数据源方向都是正的，但整体受害比例 {chip("12/32 = 0.375", "red")} 超过预设上限 0.25 → 拒绝晋升。</p></div>
  <div class="card"><b>用简单信号一票否决</b>{chip("会错杀", "red")}
    <p class="mini">该信号只能保住 68.32% 的正增益（门槛 75%），一个本可 +0.57 的好方案会被它误杀 → 只允许做粗筛，不允许做否决。</p></div>
  <div class="card"><b>范围 × 程序四格诊断</b>{chip("混合结果", "amber")}
    <p class="mini">四种组合全部未过预设材料线（0/4），两个因素谁在起作用还没分清 → 不做任何"修好了"的声明。</p></div>
  <div class="card"><b>课程自举路线</b>{chip("退役", "gray")}
    <p class="mini">三轮全部落空：方案表达方式与准入定价系统性错配。按预定退出机制全停；管线工程被后续路线继承。</p></div>
</div>
</section>"""

    s8 = f"""
<section id="mech">
{sec_head("08", "反复出现的规律发现",
"跨实验重复出现的仪器级发现——它们解释了现象，也直接约束下一步怎么做。", "mech")}
<div class="grid4">
  <div class="card"><b>平均数会掩盖局部故障</b>
    <p class="mini">同一份数据：逐条看读数是"风险缺口"，按平均看却成了"无可行动"——
6 个不同案例、16 条机器记录反复出现。结论：评估粒度必须细到局部。</p></div>
  <div class="card"><b>两道门实测有效</b>
    <p class="mini">"弱晋升 + 当窗强确认 + 二次门"的组合下：6 次经验复用尝试 <b>0 次</b>闯过安全门，
1 次合法放行且保持正向。防线不是纸面的。</p></div>
  <div class="card"><b>嘴上说引用 ≠ 真的遵从</b>
    <p class="mini">系统自称引用了某条经验，与它实际采用的方案之间没有强制关系
（6 条读数的重合字段全部 ≥1，但只是自称）。结论：遵从必须靠机制验证，不能靠自述。</p></div>
  <div class="card"><b>候选清单对抽样运气敏感</b>
    <p class="mini">同窗、同卡、同配置抽 7 次，出现 <b>3 种</b>不同候选清单。
考试设计因此改为"以首选方案为准"，并持续追加稳定性记录。</p></div>
</div>
</section>"""

    s9 = f"""
<section id="road">
{sec_head("09", "下一步：从反馈可靠性开始，走到最终考场",
"当前正在第 1 步；密封数据在整条管线冻结之前保持零读取。")}
<div class="stepper">
  <div class="step current"><span class="no">1</span><b>验证反馈信号可靠（进行中）</b>
    <p>在受控注入的练习数据上：先确认已知正确的修复确实改善独立复核成绩，再测早期的试算信号能否预测这种改善。反馈不可靠，后面一切免谈。</p></div>
  <div class="step"><span class="no">2</span><b>验证任务信息的作用方式</b>
    <p>单独测"任务说明"对系统行为的因果影响，不与第 1 步混在一起考。</p></div>
  <div class="step"><span class="no">3</span><b>双侧闭环</b>
    <p>在正确配置下多轮演练：该修时能形成有效经验卡、不该动时能安分弃权——两件事同场成立。</p></div>
  <div class="step"><span class="no">4</span><b>冻结 → 最终考场</b>
    <p>整条管线冻结后，在 41 条从未看过的序列上做"只读部署"的最终验收（对照从零与不适应两种做法）。</p></div>
</div>
<div class="card" style="margin-top:14px">
  <h3>给最终实验立下的四条规矩（本阶段用教训换来的）</h3>
  <ul style="margin:6px 0 0;padding-left:18px">
    <li><b>反馈量要够</b>：练习数据对半分后，每边至少 20 行样本才允许开考；</li>
    <li><b>缺陷类型要覆盖</b>：至少 2 类独立缺陷 × 各 2 个适用场景，外加 1 个"经验不该匹配"的守卫场景，一次注册、不许逐个追补；</li>
    <li><b>考场要真新</b>：能力级结论必须在从未看过、且范围匹配的数据上给出；</li>
    <li><b>结构先冻结</b>：经验库结构冻结后，只允许调整"决策策略"这一层，其余全部只读。</li>
  </ul>
</div>
</section>"""

    s10 = f"""
<section id="limit">
{sec_head("10", "使用边界：每条结论能说到哪、不能说到哪",
"每一条边界都是自我设限，同时也是下一步设计的输入。", "sealed")}
<div class="warnbox">
<ul>
<li>第 04 节的考试是"练完就考"的一次性打开，系统当场仍消耗了反馈——<b>还不是</b>冻结后零反馈的最终验收；那一场留给 41 条密封序列。</li>
<li>第 05 节的正效应全部来自<b>同一个数据族、已曝光数据</b>；守卫测试证明的是"不帮倒忙"，不是"能加分"；密封且匹配的加分实验暂无可用公开考场。</li>
<li>修订环目前只买到<b>成本</b>（少试探、少被拦），还没买到<b>质量</b>（误差与固定版完全相同）；"越改越准"未验证，留给最终实验。</li>
<li>"发现有害就撤权"这条规则目前只有离线重放背书，还没有自然发生的现场案例（现场零事故，也不制造事故来凑覆盖）。</li>
<li>冷启动阶段的发现召回率偏低（三个案例七处产例位，命中约 29%），已作为量化边界记录在案。</li>
<li>样本量小、多为单条轨迹；任何"后端模型无关"的说法目前都没有证据支持。</li>
</ul>
</div>
<div class="quote">一段话总结：<b>我们已经把"经验怎么合法进入、怎么被当前数据检验、对不上时怎么保持沉默"
做成了有记录、可复查的完整闭环</b>。全新数据上适应成本省 43.9%；陌生且不匹配的数据上零事故；
自然数据上第一次走完全循环，也第一次诚实地暴露了"负面经验被过度放大"的问题。
下一步修掉它，然后在冻结考场上完成真正的零反馈终态验收。</div>
</section>"""

    body = f"""
{hero}
<nav><div class="wrap"><span class="brand">SelfEvolvingHarnessTS</span>{nav}</div></nav>
<main>
{s1}{s2}{s3}{s4}{s5}{s6}{s7}{s8}{s9}{s10}
</main>
<footer>
数据来源（全部为已打开的实验工件，未为新图运行任何实验）：
<code>fresh_confirmation_v1.md</code> ·
<code>l1_ladder_v2_replay_r1.md</code> ·
<code>sa1_minimal_r1.md</code> ·
<code>capstone_epilepsy2_final.md</code> ·
<code>t1_flip_control_v1</code> ·
<code>t1b_training_flip_v3</code> ·
<code>t3_task_exam_v1</code> ·
<code>t4_conflict_experience_v1/v2</code> ·
<code>t6_nab_evaluate_v2</code> ·
<code>docs/CLS_LINE_FINAL_REPORT_2026-08-28.md</code> ·
<code>docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md</code> ·
<code>docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md</code><br>
生成器：<code>evaluation/functional/run_teacher_report_viz.py</code> · 单文件内联样式与图表，离线可打开，可直接打印
</footer>
<script>{JS}</script>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SelfEvolvingHarnessTS · 阶段性研究简报</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def main() -> int:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"== html -> {OUT_HTML} ({OUT_HTML.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
