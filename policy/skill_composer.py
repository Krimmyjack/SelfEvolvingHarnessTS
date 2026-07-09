"""policy/skill_composer.py — skill-conditioned LLM 块决策（prereg_skill_slice.md §4）。

两段式 tool-mediated 协议：stage 1 给核心视图（structure/mask/skills/policy），LLM 可按名
请求 {window, period, decomp} 或直接给最终决策；stage 2 提供所请求视图并要求最终 JSON。
输出=条件小政策（default + 可观察 cell-bin overrides），由 compile_block_policy 逐 uid
确定性编译到冻结动作池。解析/编译失败 → None（调用方回退 frozen，计 llm_failure）。

views_used 逐块落盘 = Pattern v2 特征发现仪（切片双重身份①）。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..llm import extract_json
from .dataview import CORE_VIEWS, REQUESTABLE_VIEWS
from .skills import SKILLS_V1, compile_skill

_SYS = (
    "You are a time-series data-readiness advisor. A block of ~40 univariate series "
    "(same structural family, mixed degradation levels) arrives at a preprocessing router. "
    "Your job: choose a cleaning SKILL POLICY for this block that minimizes downstream "
    "forecast loss of a frozen judge. You see standardized data views, the skill menu with "
    "risk rules distilled from prior verified experiments, and the frozen incumbent's picks. "
    "Heed the risk rules; prefer the safest skill consistent with the evidence. Output JSON only.")

_DECIDE_INSTR = (
    'FINAL ANSWER format (JSON only):\n'
    '{"default": {"skill": "<name>", "param": <number or null>},\n'
    ' "overrides": [{"when": {"snr": "low|high", "miss": "none|some"},'
    ' "skill": "<name>", "param": <number or null>}],\n'
    ' "rationale": "<=40 words"}\n'
    f'Skills: {", ".join(SKILLS_V1)}. Overrides are optional (max 4); '
    '"when" keys refer to observable SNR bin and missingness bin of each series.')

_STAGE1_TAIL = (
    "\nYou may request extra views before deciding: reply "
    '{"request_views": ["window"|"period"|"decomp", ...]} — or give the FINAL ANSWER now.\n'
    + _DECIDE_INSTR)


def _valid_decision(spec) -> Optional[dict]:
    if not isinstance(spec, dict) or "default" not in spec:
        return None
    d = spec["default"]
    if not isinstance(d, dict) or compile_skill(d.get("skill"), d.get("param")) is None:
        return None
    ovr = []
    for o in (spec.get("overrides") or [])[:4]:
        if not isinstance(o, dict) or not isinstance(o.get("when"), dict):
            continue
        if compile_skill(o.get("skill"), o.get("param")) is None:
            continue
        w = o["when"]
        if w.get("snr") in ("low", "high", None) and w.get("miss") in ("none", "some", None):
            ovr.append(dict(when=dict(snr=w.get("snr"), miss=w.get("miss")),
                            skill=o["skill"], param=o.get("param")))
    return dict(default=dict(skill=d["skill"], param=d.get("param")), overrides=ovr,
                rationale=str(spec.get("rationale", ""))[:400])


def decide_block(views: Dict[str, str], llm, tag: str) -> dict:
    """→ {decision|None, views_used, n_calls, raw}。确定性（temperature 0 + nonce 0 + 缓存）。"""
    core = "\n\n".join(f"[{v}]\n{views[v]}" for v in CORE_VIEWS)
    raw: List[str] = []
    out1 = llm(_SYS, f"BLOCK {tag}\n\n{core}\n{_STAGE1_TAIL}", nonce=0)
    raw.append(out1)
    spec = extract_json(out1)
    views_used = list(CORE_VIEWS)
    n_calls = 1
    if isinstance(spec, dict) and "request_views" in spec:
        req = [v for v in spec.get("request_views", []) if v in REQUESTABLE_VIEWS][:3]
        views_used += req
        extra = "\n\n".join(f"[{v}]\n{views[v]}" for v in req) if req else "(no valid views requested)"
        out2 = llm(_SYS, f"BLOCK {tag}\n\n{core}\n\nRequested views:\n{extra}\n\n"
                         f"Now give the FINAL ANSWER.\n{_DECIDE_INSTR}", nonce=0)
        raw.append(out2)
        spec = extract_json(out2)
        n_calls = 2
    return dict(decision=_valid_decision(spec), views_used=views_used,
                n_calls=n_calls, raw=raw)


_NORM = ("\nIMPORTANT: All summary statistics in the views are NOISY MEASUREMENTS, not ground "
         "truth. Readings tagged [低可靠]/low-reliability, and any conflicts between views, "
         "MUST be resolved in favor of the robust evidence views [period] and [decomp].")

_V2_CORE = ("structure", "mask", "period", "decomp", "skills", "policy")
_V2_REQUESTABLE = ("window",)


def decide_block_v2(views: Dict[str, str], llm, tag: str, mode: str) -> dict:
    """v2 双模式（prereg_skill_slice_v2 §2/§3）：
    mode="v2"     robust 证据（period/decomp）在 core，可再请求 window；
    mode="verify" core=v1 面（structure 已含标注），**强制两段**——stage 2 无条件附
                  period+decomp（隔离"求证行为"与"证据搬运"）；stage 1 直接给决策=违规
                  （记数，仍走 stage 2 要最终答案）。"""
    assert mode in ("v2", "verify")
    sys_p = _SYS + _NORM
    if mode == "v2":
        core = "\n\n".join(f"[{v}]\n{views[v]}" for v in _V2_CORE)
        raw: List[str] = []
        out1 = llm(sys_p, f"BLOCK {tag}\n\n{core}\n\nYou may request extra views before "
                          f"deciding: reply {{\"request_views\": [\"window\"]}} — or give the "
                          f"FINAL ANSWER now.\n{_DECIDE_INSTR}", nonce=0)
        raw.append(out1)
        spec = extract_json(out1)
        views_used = list(_V2_CORE)
        n_calls = 1
        if isinstance(spec, dict) and "request_views" in spec:
            req = [v for v in spec.get("request_views", []) if v in _V2_REQUESTABLE]
            views_used += req
            extra = "\n\n".join(f"[{v}]\n{views[v]}" for v in req) if req else "(none valid)"
            out2 = llm(sys_p, f"BLOCK {tag}\n\n{core}\n\nRequested views:\n{extra}\n\n"
                              f"Now give the FINAL ANSWER.\n{_DECIDE_INSTR}", nonce=0)
            raw.append(out2)
            spec = extract_json(out2)
            n_calls = 2
        return dict(decision=_valid_decision(spec), views_used=views_used,
                    n_calls=n_calls, violation=False, raw=raw)
    # —— verify 模式 ——
    core = "\n\n".join(f"[{v}]\n{views[v]}" for v in CORE_VIEWS)
    out1 = llm(sys_p, f"BLOCK {tag}\n\n{core}\n\nDo NOT decide yet. You MUST first request "
                      f"verification views. Reply ONLY with "
                      f"{{\"request_views\": [\"window\"|\"period\"|\"decomp\", ...]}}.", nonce=0)
    spec1 = extract_json(out1)
    violation = not (isinstance(spec1, dict) and "request_views" in spec1
                     and "default" not in spec1)
    req = ([v for v in spec1.get("request_views", []) if v in REQUESTABLE_VIEWS]
           if isinstance(spec1, dict) else [])
    forced = list(dict.fromkeys(req + ["period", "decomp"]))          # 强制附 robust 证据
    views_used = list(CORE_VIEWS) + forced
    extra = "\n\n".join(f"[{v}]\n{views[v]}" for v in forced)
    out2 = llm(sys_p, f"BLOCK {tag}\n\n{core}\n\nVerification views:\n{extra}\n\n"
                      f"Now give the FINAL ANSWER.\n{_DECIDE_INSTR}", nonce=0)
    return dict(decision=_valid_decision(extract_json(out2)), views_used=views_used,
                n_calls=2, violation=violation, raw=[out1, out2])


def _bins_of(cell: str) -> dict:
    parts = cell.split("|")                        # "forecast|snrLow|miss"
    return dict(snr="low" if "snrLow" in parts[1] else "high",
                miss="none" if parts[2] == "full" else "some")


def compile_block_policy(decision: Optional[dict], rows: List[dict]) -> Optional[Dict[str, str]]:
    """决策 → {uid: action_id}（逐 uid 确定性编译；overrides 先到先得）。None 决策 → None。"""
    if decision is None:
        return None
    out: Dict[str, str] = {}
    for r in rows:
        b = _bins_of(r["cell"])
        pick = decision["default"]
        for o in decision["overrides"]:
            w = o["when"]
            if (w.get("snr") in (None, b["snr"])) and (w.get("miss") in (None, b["miss"])):
                pick = o
                break
        out[r["uid"]] = compile_skill(pick["skill"], pick.get("param"))
    return out
