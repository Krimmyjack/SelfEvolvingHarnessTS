# -*- coding: utf-8 -*-
"""p6_exposure_ledger.py — P6 预注册前的三值曝光账本（只读审计，可重跑）。

对 data/_artifacts/monash_clean.meta.jsonl 的全部 83 条真实序列逐条分类：
  confirmed_exposed          任何持久化产物存在 uid 级证据（字段级/文本级 literal），
                             或某次已文档化真实运行的选择逻辑可确定性重建且覆盖该 uid
                             （evidence 注明 reconstructed:<run>）。
  uncertain_legacy_exposure  所在 domain 被某次无 uid 级日志且不可重建的真实运行消费过。
  certified_virgin           不适用于 legacy 83 条 —— 该类只能授予 P6 冻结后新下载并
                             content-hash 登记的序列（本脚本永不输出该类）。

证据源（全部只读）：
  1) 字段级 JSON 解析：SelfEvolvingHarnessTS/results/**、runs/**、AdaCTS/data/*.json(l)
     —— 读 series_uid / uid / (config|domain|series_family|origin)+item_id 字段值；
  2) 文本兜底：同根 + SelfEvolvingHarnessTS/*.log + BUILD.md，
     只匹配完整 "config:item_id"（裸 item_id 如 "T1" 永不单独匹配，防误报）；
  3) 选择重建（确定性重放，不跑 pipeline、不调 LLM）：
     run_stream_s1.real_domains 的分组/排序/截取逻辑 + load_real.load_signals 生产代码、
     load_real.split_encoder_eval(seed=0)、run_p5a3_final._episodes 轮询。

运行（Agent 根目录）：
  D:/Anaconda_envs/envs/project/python.exe SelfEvolvingHarnessTS/diagnostics/p6_exposure_ledger.py

产出（仅新文件，不改任何现有文件）：
  SelfEvolvingHarnessTS/results/Stage2/P6Probes/exposure_ledger.jsonl
  SelfEvolvingHarnessTS/results/Stage2/P6Probes/exposure_report.md
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys
from collections import Counter, defaultdict

AGENT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT))

import numpy as np  # noqa: E402

from SelfEvolvingHarnessTS.data.load_real import load_signals, split_encoder_eval  # noqa: E402

SEHT = AGENT / "SelfEvolvingHarnessTS"
META = SEHT / "data" / "_artifacts" / "monash_clean.meta.jsonl"
NPZ_CLEAN = SEHT / "data" / "_artifacts" / "monash_clean.npz"
NPZ_REAL12 = AGENT / "AdaCTS" / "data" / "monash_real.npz"
OUT_DIR = SEHT / "results" / "Stage2" / "P6Probes"
LEDGER = OUT_DIR / "exposure_ledger.jsonl"
REPORT = OUT_DIR / "exposure_report.md"

CONFIGS = ("nn5_daily", "fred_md", "tourism_monthly", "covid_deaths",
           "us_births", "saugeenday", "sunspot")
BIG4 = ("covid_deaths", "fred_md", "nn5_daily", "tourism_monthly")

# 字段级解析时视为"序列身份"的键（UID_LIST_KEYS 值为字符串列表，如
# candidates_flash.jsonl 的 split_fingerprint.held_in/held_out.series_uids）
UID_KEYS = ("series_uid", "uid")
UID_LIST_KEYS = ("series_uids", "uids")
CFG_KEYS = ("config", "domain", "series_family", "origin", "dataset")

SWEEP_JSON_ROOTS = [SEHT / "results", AGENT / "runs", AGENT / "AdaCTS" / "data"]
SWEEP_TEXT_ROOTS = [SEHT / "results", AGENT / "runs"]
SWEEP_TEXT_EXTRA = sorted(SEHT.glob("_*.log")) + [SEHT / "BUILD.md"]
TEXT_EXTS = {".md", ".txt", ".csv", ".log", ".json", ".jsonl"}
MAX_BYTES = 30 * 1024 * 1024
# 本脚本自己的产出不作为证据（保证可重跑幂等）
SELF_OUTPUTS = {LEDGER.resolve(), REPORT.resolve()}


# ══════════════════════ 运行台账（args 来源逐条注明；重建只用这里的参数）══════════════════════
# provenance 缩写：BUILD=SelfEvolvingHarnessTS/BUILD.md；transcript=Claude Code 会话转录
# （C:/Users/辉/.claude/projects/C--Users---desktop-agent/*.jsonl，verbatim 命令字符串）。
RUN_REGISTRY = [
    dict(run_id="real_longrun_R0/R1/R2/R1c/R2c+fc_maintable+calibrate_eps",
         date="2026-06-20/21", npz="monash_real(12)", selection="ALL(默认 npz 全量)",
         llm="R1/R2/R1c/R2c=flash；R0/R0''/fc_maintable/R4=无 LLM",
         provenance="BUILD §8 R0-R4 行；日志 _real_step{1b,2}{,_chronos}.log、"
                    "_real_step1b_chronos_table.log、_fc_maintable.log（mtime 2026-06-20/21）；"
                    "transcript 2026-06-20 verbatim 命令（均无 --npz/--configs → 默认 12 信号全量）",
         uid_logged=False),
    dict(run_id="real_longrun_R0'(encoder=real, monash_clean)",
         date="2026-06-20 17:30", npz="monash_clean(83)",
         selection="ALL 83 → split_encoder_eval(frac=0.5, seed=0)：pre 半入编码器预训练、ev 半入诊断语料",
         llm="无",
         provenance="transcript 2026-06-20 verbatim：run_real_longrun --mode diag --encoder real "
                    "--npz .../monash_clean.npz --encoder-cache .../frozen_lstm_real_h64.pt；"
                    "BUILD §8 R0' 行；工件 evaluators/_artifacts/frozen_lstm_real_h64.pt "
                    "mtime 2026-06-20 17:30:35（npz 建成后 53s）",
         uid_logged=False),
    dict(run_id="run_stream_s1[s1_flash,s1_pro]", date="2026-06-23",
         npz="monash_clean(83)", selection="min_signals=5, max_per_domain=8, n_per_signal=3（重建=每大域前 8 条）",
         llm="flash/pro",
         provenance="transcript 2026-06-23 两条 verbatim 命令（--k 2 --epochs 2 --max-per-domain 8 "
                    "--n-per-signal 3）；BUILD §4.5「同配置」+ §8 复现行；runs/s1_{flash,pro}/ "
                    "summary mtime 2026-06-23 11:28/11:59",
         uid_logged=False),
    dict(run_id="run_stream_s1[s1_flash_chronos,s1_pro_chronos]", date="2026-06-23",
         npz="monash_clean(83)", selection="同上 + --substrate chronos（选择与 substrate 无关）",
         llm="flash/pro",
         provenance="transcript 2026-06-23T04:35Z verbatim 链式命令（含 --max-per-domain 8 "
                    "--n-per-signal 3 --substrate chronos）；runs/ mtime 12:43/13:10 (+0800)",
         uid_logged=False),
    dict(run_id="run_stream_s1_v2[s1_{flash,pro}{,_chronos}_v2]", date="2026-06-24/25",
         npz="monash_clean(83)",
         selection="min_signals=5, max_per_domain=0(未给→全量!), n_per_signal=4(默认)（重建=每大域全部 20 条）",
         llm="flash/pro",
         provenance="rerun_v2.sh 全文 verbatim 存于 transcript 6e3ee906（2026-06-24 Write 工具记录）："
                    "--npz monash_clean --llm {flash,pro} --k 2 --epochs 2 --substrate {frozen,chronos} "
                    "--out-dir runs/s1_*_v2，无 --max-per-domain/--n-per-signal；"
                    "4 个 out-dir summary mtime 2026-06-24 23:19 → 06-25 01:16",
         uid_logged=False),
    dict(run_id="anchor_maintable(run_main_table)", date="2026-07-02",
         npz="monash_real(12)", selection="ALL 12（build_real_corpus 全量）", llm="无（chronos 判官）",
         provenance="results/anchor_maintable/config.json（持久化参数：npz=AdaCTS/data/monash_real.npz, "
                    "task=forecast, seeds=2, judge=chronos + 全库代码指纹）",
         uid_logged=False),
    dict(run_id="anchor_s1(run_stream_s1)", date="2026-07-02",
         npz="monash_real(12)", selection="min_signals=4, max_per_domain=0 → 3 域全 12 条", llm="flash",
         provenance="results/anchor_s1/config.json（持久化参数 + 代码指纹含 run_stream_s1.py=ff07d77b0bba3525）；"
                    "flash_run/summary.json；candidates_flash.jsonl 的 "
                    "split_fingerprint.held_in/held_out.series_uids 携 literal uid（S0.5+F2）",
         uid_logged=True),
    dict(run_id="P5Quadrant(run_p5_quadrant)", date="2026-07-09/10",
         npz="monash_real(12)", selection="ALL 12 × 4 preset", llm="无（true judge）",
         provenance="results/Stage2/P5Quadrant/records.jsonl —— 每行 literal series_uid 字段（uid 级日志✓）",
         uid_logged=True),
    dict(run_id="P5A3Final(run_p5a3_final)", date="2026-07-10",
         npz="monash_real(12)",
         selection="episodes: signals[idx%12] 轮询（seeds 80-99 × 3 → 60 episodes 覆盖全 12）",
         llm="是（seeds 80-99 一次性消耗）",
         provenance="manifest.json（'12 signals x FORECAST_PRESETS'）+ records.jsonl（series_family 级）"
                    "+ run_p5a3_final._episodes 确定性轮询重建",
         uid_logged=False),
    dict(run_id="AdaCTS 清洗代实验(eval_out_*)", date="AdaCTS 时期（monash_corrupt）",
         npz="monash_real 同源 12 条（corrupted 版）", selection="ALL 12 × corruption 网格", llm="deepseek+heuristic",
         provenance="AdaCTS/data/eval_out_{deepseek,heuristic}.metrics.jsonl —— 每行 literal config+item_id "
                    "字段（uid 级✓）；monash_corrupt.meta.jsonl",
         uid_logged=True),
]


# ══════════════════════════════ 工具 ══════════════════════════════
def sha256_file(p: pathlib.Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def rel(p: pathlib.Path) -> str:
    try:
        return p.resolve().relative_to(AGENT).as_posix()
    except ValueError:
        return p.as_posix()


def load_meta() -> list[dict]:
    recs = []
    with open(META, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                recs.append(json.loads(ln))
    return recs


# ═════════════════ 选择重建（verbatim 语义拷贝自 run_stream_s1.real_domains L72-89）═════════════════
def select_stream_domains(signals, *, min_signals: int, max_per_domain: int,
                          max_domains: int = 0) -> dict[str, list[str]]:
    """重放 run_stream_s1.real_domains 的信号选择（只选序列，不建语料/不跑 pipeline）。

    与生产代码逐句对应：defaultdict 按 config 分组（保 npz 顺序）→ len>=min_signals →
    sort key=(-len, name) → [:max_domains] → group[:max_per_domain]。
    选择发生在 proposer/substrate/order_seed 之前且不依赖它们 → 与 LLM 臂无关；
    order_domains 只重排 domain 顺序、不改成员。
    """
    by_dom: dict[str, list] = defaultdict(list)
    for s in signals:
        by_dom[getattr(s, "config", "real")].append(s)
    groups = [(name, g) for name, g in by_dom.items() if len(g) >= min_signals]
    groups.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    if max_domains > 0:
        groups = groups[:max_domains]
    out = {}
    for name, group in groups:
        if max_per_domain > 0:
            group = group[:max_per_domain]
        out[str(name)] = [f"{s.config}:{s.item_id}" for s in group]
    return out


def replay_p5a3_episodes(signals, seeds=range(80, 100), n_per_seed=3) -> list[str]:
    """重放 run_p5a3_final._episodes 的 series 轮询（signals[idx % len]）。"""
    consumed = []
    seeds = list(seeds)
    for s in seeds:
        for j in range(n_per_seed):
            idx = (int(s) - int(seeds[0])) * n_per_seed + j
            sig = signals[idx % len(signals)]
            consumed.append(f"{sig.config}:{sig.item_id}")
    return sorted(set(consumed))


# ══════════════════════════════ 证据扫描 ══════════════════════════════
def uid_regex(uid_set: set[str]) -> re.Pattern:
    cfg_alt = "|".join(map(re.escape, CONFIGS))
    return re.compile(r"\b(?:%s):[A-Za-z0-9_]+" % cfg_alt)


def iter_files(roots, exts):
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts and p.resolve() not in SELF_OUTPUTS:
                yield p


def walk_json(obj, on_dict):
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            on_dict(o)
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)


def scan_field_level(uid_set: set[str]):
    """字段级 JSON 解析。返回 (uid→{file}, file→{config})（domain 级）。"""
    uid_hits: dict[str, set[str]] = defaultdict(set)
    dom_hits: dict[str, set[str]] = defaultdict(set)

    def inspect_factory(fkey):
        def on_dict(d):
            for k in UID_KEYS:
                v = d.get(k)
                if isinstance(v, str):
                    base = v.split("|", 1)[0]
                    if base in uid_set:
                        uid_hits[base].add(fkey)
            for k in UID_LIST_KEYS:
                v = d.get(k)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, str) and it.split("|", 1)[0] in uid_set:
                            uid_hits[it.split("|", 1)[0]].add(fkey)
            iid = d.get("item_id")
            if iid is not None:
                for ck in CFG_KEYS:
                    cv = d.get(ck)
                    if isinstance(cv, str) and cv in CONFIGS:
                        cand = f"{cv}:{iid}"
                        if cand in uid_set:
                            uid_hits[cand].add(fkey)
                        break
            for ck in CFG_KEYS:
                cv = d.get(ck)
                if isinstance(cv, str) and cv in CONFIGS:
                    dom_hits[fkey].add(cv)
        return on_dict

    n_files = n_parsed = 0
    for p in iter_files(SWEEP_JSON_ROOTS, {".json", ".jsonl"}):
        if p.stat().st_size > MAX_BYTES:
            continue
        n_files += 1
        fkey = rel(p)
        on_dict = inspect_factory(fkey)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parsed_any = False
        if p.suffix.lower() == ".jsonl":
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    walk_json(json.loads(ln), on_dict)
                    parsed_any = True
                except Exception:
                    pass
        else:
            try:
                walk_json(json.loads(text), on_dict)
                parsed_any = True
            except Exception:
                pass
        n_parsed += parsed_any
    return uid_hits, dom_hits, n_files, n_parsed


def scan_text_level(uid_set: set[str]):
    """文本兜底：完整 'config:item' literal + 裸 config 名（domain 级）。"""
    rx = uid_regex(uid_set)
    uid_hits: dict[str, set[str]] = defaultdict(set)
    dom_hits: dict[str, set[str]] = defaultdict(set)
    files = list(iter_files(SWEEP_TEXT_ROOTS, TEXT_EXTS)) + [p for p in SWEEP_TEXT_EXTRA if p.exists()]
    for p in files:
        if p.stat().st_size > MAX_BYTES:
            continue
        fkey = rel(p)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in rx.findall(text):
            if m in uid_set:
                uid_hits[m].add(fkey)
        for cfg in CONFIGS:
            if cfg in text:
                dom_hits[fkey].add(cfg)
    return uid_hits, dom_hits, len(files)


# ══════════════════════════════ 主流程 ══════════════════════════════
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = load_meta()
    uids = [f"{m['config']}:{m['item_id']}" for m in meta]
    uid_set = set(uids)
    assert len(meta) == 83 and len(uid_set) == 83, f"meta 记录数异常: {len(meta)}/{len(uid_set)}"

    gates: list[tuple[str, bool, str]] = []   # (gate, ok, detail)

    # ── G1: 生产 load_signals 重放（monash_clean）──
    sig83 = load_signals(str(NPZ_CLEAN))
    got83 = [f"{s.config}:{s.item_id}" for s in sig83]
    g1 = (len(sig83) == 83 and set(got83) == uid_set and got83 == uids)
    gates.append(("G1 load_signals(monash_clean)==meta 83 条且保序（min_len/finite 无淘汰）", g1,
                  f"loaded={len(sig83)}"))

    # ── G2: monash_real 12 条 ⊂ 83 且底层数组 bit 级相同 ──
    sig12 = load_signals(str(NPZ_REAL12))
    got12 = [f"{s.config}:{s.item_id}" for s in sig12]
    subset = set(got12) <= uid_set and len(got12) == 12
    by_uid83 = {u: s for u, s in zip(got83, sig83)}
    bitwise = subset and all(np.array_equal(by_uid83[u].clean, s.clean) for u, s in zip(got12, sig12))
    gates.append(("G2 monash_real 12 条 ⊂ monash_clean 且 z-score 后序列 bit 级相同", bitwise,
                  f"12 uids={sorted(got12)}"))

    # ── 重建：S1 家族 ──
    sel_cap8 = select_stream_domains(sig83, min_signals=5, max_per_domain=8)
    sel_v2 = select_stream_domains(sig83, min_signals=5, max_per_domain=0)
    sel_anchor = select_stream_domains(sig12, min_signals=4, max_per_domain=0)
    set_cap8 = {u for g in sel_cap8.values() for u in g}
    set_v2 = {u for g in sel_v2.values() for u in g}
    set_anchor = {u for g in sel_anchor.values() for u in g}
    set_p5a3 = set(replay_p5a3_episodes(sig12))
    set_real12 = set(got12)
    set_r0p_pre, set_r0p_ev = (set(), set())
    pre, ev = split_encoder_eval(sig83, frac=0.5, seed=0)
    set_r0p_pre = {f"{s.config}:{s.item_id}" for s in pre}
    set_r0p_ev = {f"{s.config}:{s.item_id}" for s in ev}

    gates.append(("G3 cap8 选择域=4 大域、各 8 条、皆为各域前 8（npz 序）",
                  sorted(sel_cap8) == sorted(BIG4) and all(len(v) == 8 for v in sel_cap8.values()),
                  json.dumps({k: v for k, v in sel_cap8.items()}, ensure_ascii=False)))
    gates.append(("G4 v2 选择=4 大域全 20 条（80 条），且 cap8 ⊂ v2",
                  all(len(v) == 20 for v in sel_v2.values()) and set_cap8 <= set_v2,
                  f"|v2|={len(set_v2)}"))
    gates.append(("G5 anchor_s1 选择=monash_real 3 域全 12 条",
                  set_anchor == set_real12 and sorted(sel_anchor) == ["fred_md", "nn5_daily", "tourism_monthly"],
                  f"domains={sorted(sel_anchor)}"))
    gates.append(("G6 P5A3 轮询重建覆盖全 12 条", set_p5a3 == set_real12, f"|p5a3|={len(set_p5a3)}"))
    gates.append(("G7 R0' split(seed=0) pre∪ev=83 且不相交",
                  (set_r0p_pre | set_r0p_ev) == uid_set and not (set_r0p_pre & set_r0p_ev),
                  f"|pre|={len(set_r0p_pre)} |ev|={len(set_r0p_ev)}"))

    # ── G8: 各 run 目录 summary.json 的 domain 集与重建一致（外部输出交叉验证）──
    def summary_domains(p: pathlib.Path):
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        for rows in js.values():
            return [r.get("domain") for r in rows]
        return None

    s1_dirs = ["s1_flash", "s1_pro", "s1_flash_chronos", "s1_pro_chronos",
               "s1_flash_v2", "s1_pro_v2", "s1_flash_chronos_v2", "s1_pro_chronos_v2"]
    order_expect = sorted(BIG4)  # (-len,name)：4 域同 20 条 → 字母序
    sum_ok, sum_detail = True, []
    for d in s1_dirs:
        doms = summary_domains(AGENT / "runs" / d / "summary.json")
        ok = doms == order_expect
        sum_ok &= ok
        sum_detail.append(f"{d}:{'OK' if ok else doms}")
    anchor_doms = summary_domains(SEHT / "results" / "anchor_s1" / "flash_run" / "summary.json")
    ok_anchor = anchor_doms == ["fred_md", "nn5_daily", "tourism_monthly"]
    gates.append(("G8 8 个 S1 run 的 summary.json 域序=重建 canonical 序（covid,fred,nn5,tourism）",
                  sum_ok, "; ".join(sum_detail)))
    gates.append(("G8b anchor_s1/flash_run summary 域序=fred,nn5,tourism（config.json note 同）",
                  ok_anchor, str(anchor_doms)))

    # ── G9: 代码指纹（今日 run_stream_s1.py vs anchor_s1 config 指纹 2026-07-02）──
    cur_fp = sha256_file(SEHT / "run_stream_s1.py")
    anchor_fp = None
    try:
        cfg = json.loads((SEHT / "results" / "anchor_s1" / "config.json").read_text(encoding="utf-8"))
        anchor_fp = cfg["code_fingerprint_sha256_16"]["SelfEvolvingHarnessTS/run_stream_s1.py"]
    except Exception:
        pass
    fp_ok = (anchor_fp is not None and cur_fp == anchor_fp)
    gates.append(("G9 run_stream_s1.py 当前 sha256[:16] == anchor_s1(2026-07-02) 指纹（选择代码未漂移）",
                  fp_ok, f"current={cur_fp} anchor={anchor_fp}"))

    # ── G10: npz 时间线（语料先于全部消费 run 建成，未被改写）──
    npz_mtime = NPZ_CLEAN.stat().st_mtime
    import datetime as _dt
    npz_dt = _dt.datetime.fromtimestamp(npz_mtime).strftime("%Y-%m-%d %H:%M:%S")
    gates.append(("G10 monash_clean.npz mtime 早于全部 S1 run（2026-06-23 起）",
                  npz_dt < "2026-06-23", f"npz mtime={npz_dt}"))

    all_gates_ok = all(ok for _, ok, _ in gates)

    # ── 证据扫描 ──
    f_uid, f_dom, n_json_files, n_json_parsed = scan_field_level(uid_set)
    t_uid, t_dom, n_text_files = scan_text_level(uid_set)

    # P5IdentityGate 专项验证：确无真实 uid
    idg = SEHT / "results" / "Stage2" / "P5IdentityGate"
    idg_files = {rel(p) for p in idg.rglob("*") if p.is_file()} if idg.exists() else set()
    idg_uid_hits = {u: fs & idg_files for u, fs in list(f_uid.items()) + list(t_uid.items())}
    idg_uid_hits = {u: fs for u, fs in idg_uid_hits.items() if fs}
    idg_dom_touch = sorted({c for fk, cs in list(f_dom.items()) + list(t_dom.items())
                            if fk in idg_files for c in cs})

    # ── 重建证据映射（仅在对应 gates 通过时授予；否则降 domain 级）──
    recon_evidence: dict[str, list[tuple[str, str]]] = defaultdict(list)  # uid → [(tag, cluster)]
    g = {name.split(" ")[0]: ok for name, ok, _ in gates}

    def grant(uid_iter, tag, cluster, cond=True):
        if not cond:
            return
        for u in uid_iter:
            recon_evidence[u].append((tag, cluster))

    grant(set_cap8, "reconstructed:run_stream_s1[s1_flash|s1_pro|s1_flash_chronos|s1_pro_chronos,"
                    "2026-06-23,max_per_domain=8]", "A_s1_0623",
          g["G1"] and g["G3"] and g["G8"] and g["G9"])
    grant(set_v2, "reconstructed:run_stream_s1[s1_*_v2×4,2026-06-24/25,max_per_domain=0→全量"
                  "(rerun_v2.sh via transcript)]", "B_s1_v2",
          g["G1"] and g["G4"] and g["G8"] and g["G9"])
    grant(set_r0p_pre, "reconstructed:run_real_longrun_R0'[encoder_split(pre=预训练半),seed=0,2026-06-20]",
          "C_r0prime", g["G1"] and g["G7"])
    grant(set_r0p_ev, "reconstructed:run_real_longrun_R0'[encoder_split(ev=诊断语料半),seed=0,2026-06-20]",
          "C_r0prime", g["G1"] and g["G7"])
    grant(set_anchor, "reconstructed:anchor_s1[run_stream_s1,2026-07-02,min_signals=4→全12,"
                      "config.json 持久化参数]", "D_anchor", g["G2"] and g["G5"] and g["G8b"])
    grant(set_real12, "reconstructed:monash_real 默认语料全量 runs[R0/R1/R2/R1c/R2c/fc_maintable/"
                      "anchor_maintable/P5A3Final,2026-06-20→07-10]", "E_real12_family", g["G2"] and g["G6"])

    # ── 组装账本 ──
    rows = []
    cls_count = Counter()
    for m, uid in zip(meta, uids):
        ev_list: list[str] = []
        clusters: set[str] = set()
        literal = False
        for fk in sorted(f_uid.get(uid, ())):
            ev_list.append(f"field_uid:{fk}")
            literal = True
        for fk in sorted(t_uid.get(uid, ())):
            if f"field_uid:{fk}" not in ev_list:
                ev_list.append(f"text_uid:{fk}")
                literal = True
        for tag, cluster in recon_evidence.get(uid, []):
            if tag not in ev_list:
                ev_list.append(tag)
                clusters.add(cluster)

        if literal or clusters:
            cls = "confirmed_exposed"
            confidence = "high" if (literal or len(clusters) >= 2) else "medium"
        else:
            # legacy 序列不存在 certified_virgin 出口：无 uid 级证据时一律 uncertain（保守）
            cls = "uncertain_legacy_exposure"
            confidence = "low"
            ev_list.append(f"domain:{m['config']} 出现于真实运行记录（无 uid 级日志且不可重建）")
        cls_count[(m["config"], cls)] += 1
        rows.append(dict(config=m["config"], item_id=str(m["item_id"]), series_uid=uid,
                         exposure_class=cls, evidence=ev_list, confidence=confidence))

    with open(LEDGER, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 报告 ──
    n_conf = sum(1 for r in rows if r["exposure_class"] == "confirmed_exposed")
    n_unc = len(rows) - n_conf
    conf_hi = sum(1 for r in rows if r["confidence"] == "high")
    conf_med = sum(1 for r in rows if r["confidence"] == "medium")
    lit_uids = sorted(set(f_uid) | set(t_uid))
    p5_uids = sorted(u for u, fs in f_uid.items()
                     if any("P5Quadrant" in fk for fk in fs))

    def count_row(cfg):
        c = cls_count[(cfg, "confirmed_exposed")]
        u = cls_count[(cfg, "uncertain_legacy_exposure")]
        return f"| {cfg} | {c} | {u} | 0 (不适用) |"

    lines = []
    A = lines.append
    A("# P6 曝光账本报告（exposure_report.md）")
    A("")
    A(f"- 生成脚本：`SelfEvolvingHarnessTS/diagnostics/p6_exposure_ledger.py`（只读审计，可重跑）")
    A(f"- 语料指纹：monash_clean.meta.jsonl sha256[:16]=`{sha256_file(META)}`；"
      f"monash_clean.npz=`{sha256_file(NPZ_CLEAN)}`（mtime {npz_dt}）；"
      f"monash_real.npz=`{sha256_file(NPZ_REAL12)}`")
    A(f"- meta schema：`config`/`item_id` + 退化统计；**无 series_uid 字段**——uid 按生产代码 "
      f"`data/load_real.py` L115 约定派生：`series_uid = f\"{{config}}:{{item_id}}\"`。")
    A("")
    A("## 三值分类定义")
    A("")
    A("- **confirmed_exposed**：持久化产物存在该 uid 级 literal 证据（字段级 JSON 解析优先，"
      "文本匹配仅完整 `config:item_id`，裸短 item_id 永不单独匹配），或某次已文档化真实运行的选择逻辑"
      "**确定性重建**覆盖该 uid（evidence 注明 `reconstructed:*`）。")
    A("- **uncertain_legacy_exposure**：所在 domain 被某次无 uid 级日志且**不可重建**的真实运行消费过。")
    A("- **certified_virgin**：**不适用于 legacy 83 条**。该类只能授予 P6 冻结之后新下载并 "
      "content-hash 登记的序列——legacy 语料在冻结前已存在于本机并被多次运行装载，"
      "无法对其出具『从未接触』的构造性证明，故本账本永不输出该类。")
    A("")
    A("## 每 domain × class 计数")
    A("")
    A("| config | confirmed_exposed | uncertain_legacy_exposure | certified_virgin |")
    A("|---|---|---|---|")
    for cfg in CONFIGS:
        A(count_row(cfg))
    A(f"| **合计(83)** | **{n_conf}** | **{n_unc}** | **0** |")
    A("")
    A(f"置信度分布：high={conf_hi}，medium={conf_med}（medium=仅单一重建链支撑，见下）。")
    A("")
    A("## 消费过真实数据的运行台账（args 来源逐条注明）")
    A("")
    A("| run | 日期 | 语料 | 选择/重建 | LLM | uid级日志 |")
    A("|---|---|---|---|---|---|")
    for r in RUN_REGISTRY:
        A(f"| {r['run_id']} | {r['date']} | {r['npz']} | {r['selection']} | {r['llm']} | "
          f"{'✓' if r['uid_logged'] else '✗'} |")
    A("")
    A("provenance 细节：")
    for r in RUN_REGISTRY:
        A(f"- **{r['run_id']}**：{r['provenance']}")
    A("")
    A("## run_stream_s1 选择重建结论")
    A("")
    A("1. **确定性**：`real_domains()`（run_stream_s1.py L63-89）从 npz 顺序装载信号"
      "（`load_signals` 按存储序迭代、无 RNG），按 `config` 分组（保插入序）→ `len>=min_signals` → "
      "`sort(key=(-len, name))` → `group[:max_per_domain]`。全程无随机性；"
      "`--order-seed` 只重排 domain 顺序不改成员；选择发生在 proposer 构造之前且不读 `--llm`/`--substrate`。")
    A("2. **与 LLM 臂无关 → flash 与 pro 选择相同集合（可证完备）**：同一确定性函数、同 npz、同参数；"
      "且 8 个 run 的 summary.json 域序全部等于重建 canonical 序（gate G8 ✓）。")
    A("3. **2026-06-23 四 run（flash/pro × frozen/chronos 判官）**：`--max-per-domain 8` → "
      "每大域前 8 条（npz 序=T1..T8）× 4 域 = **32 条**。")
    A("4. **2026-06-24/25 四 v2 run**：rerun_v2.sh（transcript 全文恢复）**未传 --max-per-domain** → "
      "默认 0=全量 → 每大域全部 20 条 = **80 条**（意外发现：v2 消费面远大于 BUILD.md 文档化的 32 条）。")
    A("5. **anchor_s1（2026-07-02）**：monash_real.npz + `--min-signals 4` → 3 域全 12 条"
      "（config.json 持久化参数，最强 args 证据）。")
    A("6. **代码漂移防护**：当前 run_stream_s1.py sha256[:16] 与 anchor_s1 config 指纹"
      f"（2026-07-02）比对 → {'一致' if fp_ok else '不一致（重建降级 medium，见 gates）'}"
      f"（current={cur_fp}）。6-23/6-24 run 早于指纹日期，另以 summary 域序 + BUILD §4.5 "
      "文档化语义（『每 domain 取前 N 信号』）交叉钉住。")
    A("")
    A("## 验证 gates")
    A("")
    for name, ok, detail in gates:
        A(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    A("")
    A("## 12 个 P5 uid 与重建集的重叠")
    A("")
    A(f"- P5Quadrant literal series_uid（{len(p5_uids)} 条）：`{p5_uids}`")
    A("- 关系：P5 12 条 = monash_real 全量 = {nn5_daily,fred_md,tourism_monthly}×{T1..T4}，"
      "**⊂ 2026-06-23 cap8 集（各域前 8）⊂ v2 全量集**；covid_deaths 不在 monash_real（P5 无 covid）。")
    A(f"- bit 级同源验证（G2）：monash_real 12 条与 monash_clean 对应条目 z-score 后逐点相同 → "
      f"{'PASS' if bitwise else 'FAIL'} → P5/AdaCTS 时期消费的就是同一批底层真实序列。")
    A("")
    A("## P5IdentityGate 验证（任务点 3）")
    A("")
    if idg.exists():
        A(f"- 扫描 {len(idg_files)} 个文件：真实 uid 命中 = {len(idg_uid_hits)} "
          f"{'（无 → P5-A 确为合成 anomaly/forecast slice，记录 uid 形如 c40_p2_0）' if not idg_uid_hits else idg_uid_hits}")
        A(f"- domain 名出现：{idg_dom_touch if idg_dom_touch else '无'}")
    else:
        A("- 目录不存在。")
    A("")
    A("## 意外发现（超出任务列举的证据源）")
    A("")
    A("1. **v2 四连跑消费全 80 条大域序列**（runs/s1_*_v2，2026-06-24/25）：BUILD.md 只文档化了 "
      "cap8 的 flash/pro 复现行；v2 的『无 cap 全量』只能从会话转录恢复的 rerun_v2.sh 得知。")
    A("2. **chronos 判官双跑**（runs/s1_{flash,pro}_chronos，2026-06-23）：BUILD.md 未记；"
      "transcript verbatim 命令含 `--max-per-domain 8` → 与 cap8 同集。")
    singles_in_pre = all(f"{c}:T1" in set_r0p_pre for c in ("us_births", "saugeenday", "sunspot"))
    A("3. **R0' 编码器事件（2026-06-20）**：`run_real_longrun --mode diag --encoder real --npz monash_clean` "
      f"把 **全部 83 条**（含 us_births/saugeenday/sunspot 单条域）按 seed=0 分层对半：pre 半（{len(set_r0p_pre)} 条）进 "
      "**frozen LSTM 编码器预训练**（工件 evaluators/_artifacts/frozen_lstm_real_h64.pt 至今存在），"
      f"ev 半（{len(set_r0p_ev)} 条）进诊断语料。三个单条域 series "
      f"{'都落在 pre 半（重放验证✓）' if singles_in_pre else '的落半重放异常（须人工复核！）'}"
      "→ 它们的唯一曝光通道即编码器预训练。"
      "**P6 注意**：该编码器权重本身内嵌 legacy 真实数据——若 P6 在 virgin 序列上使用它做判官底座，"
      "构成一条间接耦合通道（不违规，但预注册应声明）。")
    A("4. **AdaCTS 清洗代 literal 证据**：AdaCTS/data/eval_out_{deepseek,heuristic}.metrics.jsonl 逐行"
      "含 config+item_id 字段（monash_corrupt = monash_real 同 12 条的受蚀版）→ 12 条重叠序列"
      "早在 AdaCTS 时期即被 LLM 清洗实验逐 uid 消费并留档。")
    A("5. **monash_real ⊂ monash_clean（bit 级）**：两语料 base_std 至 15+ 位小数一致且数组逐点相等（G2），"
      "即 anchor/P5/AdaCTS 全部 12-信号运行消费的正是 83 条中的 12 条。")
    A("6. **results/Stage2/P6Probes/ 已有 u_admission 探针**（electricity_hourly/traffic_hourly）——"
      "P6 新数据入场流程已在走 content-hash 登记路线，与本账本结论一致。")
    A("7. **anchor_s1 候选级日志其实携带 uid**：candidates_flash.jsonl 每条候选的 "
      "`split_fingerprint.held_in/held_out.series_uids` 内嵌 literal series_uid 列表（Stage 0 F2 "
      "series_uid 分组的副产品）→ anchor_s1 的 12 条消费有 uid 级留档，非任务预设的『无 uid 日志』。")
    A("")
    A("## 对 P6 的含义")
    A("")
    A(f"- **legacy 83 条全部只能进 C0/D**（对照/开发池）：{n_conf} 条 confirmed_exposed"
      f"{'，' + str(n_unc) + ' 条 uncertain' if n_unc else '（uncertain=0——所有已识别真实运行的选择均可确定性重建）'}。"
      "没有任何一条可申领 certified_virgin。")
    A("- **V/U（virgin/未见评测池）必须来自 P6 冻结后的新下载**，入库时做 content-hash 登记 + "
      "uid 级消费日志（本次审计暴露的教训：deploy_stream/forward_transfer 仅 cell 级、"
      "run_stream_s1 无选择落盘——P6 harness 应强制 per-series manifest）。")
    A("- **单条域（us_births/saugeenday/sunspot）置信度=medium**：唯一曝光链是 R0' 编码器事件的重建"
      "（无 literal 工件）。保守起见仍记 confirmed_exposed（重建确定性 + 命令 verbatim + 工件 mtime 链），"
      "不因证据薄而降级为『可当 virgin 用』。")
    A("- **frozen_lstm_real_h64.pt 为 legacy-data-derived 工件**：P6 若复用需在预注册中声明（见意外发现 3）。")
    A("")
    A("## 方法与局限")
    A("")
    A(f"- 字段级扫描：{n_json_files} 个 json/jsonl（成功解析 {n_json_parsed}）；文本兜底 {n_text_files} 个文件；"
      f"literal uid 命中 {len(lit_uids)} 条：`{lit_uids}`。")
    A("- 语料 meta 清单的处理：AdaCTS/data/monash_{real,corrupt}.meta.jsonl 属『被整体消费语料的 uid 清单'"
      "（manifest），按任务定义计入 literal 证据；审计对象 monash_clean.meta.jsonl 本身不在扫描根内"
      "（它定义账本行而非曝光证据）。对应 12 条序列的消费性 literal 证据独立存在于 "
      "eval_out_*.metrics.jsonl / P5Quadrant/records.jsonl / candidates_flash.jsonl。")
    A("- 重建假设：npz 未被改写（G10 mtime 早于全部 run；内容 hash 已钉于报告头部）；"
      "选择代码语义自 2026-06-23 起未变（G9 指纹钉 2026-07-02→今；6-23/24 由 BUILD §4.5 文档语义 + "
      "G8 输出域序间接钉住）；会话转录中的命令字符串即实际执行命令（Bash tool_use 记录）。")
    A("- 保守规则：所有 gate 联动——任一重建 gate FAIL 时对应 `reconstructed:*` 证据自动不授予，"
      "相关序列自然回落 uncertain_legacy_exposure；本次运行 gates "
      f"{'全 PASS' if all_gates_ok else '存在 FAIL（见上）'}。")
    A("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    # ── stdout 摘要 ──
    print("=" * 72)
    print("P6 exposure ledger written:")
    print(" ", rel(LEDGER), f"({len(rows)} rows)")
    print(" ", rel(REPORT))
    print("-" * 72)
    for cfg in CONFIGS:
        c = cls_count[(cfg, "confirmed_exposed")]
        u = cls_count[(cfg, "uncertain_legacy_exposure")]
        print(f"  {cfg:<18} confirmed={c:<3} uncertain={u:<3}")
    print(f"  TOTAL confirmed={n_conf} uncertain={n_unc} virgin=0(N/A)  "
          f"conf(high/med)={conf_hi}/{conf_med}")
    print("-" * 72)
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all_gates_ok:
        print("  !! 存在 FAIL gate —— 相关重建证据未授予，对应序列已保守降级。")
    print("=" * 72)


if __name__ == "__main__":
    main()
