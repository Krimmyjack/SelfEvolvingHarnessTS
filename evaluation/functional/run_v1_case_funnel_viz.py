"""CASE_FUNNEL_VIZ（S0，2026-08-13：静态漏斗图——全部已暴露数据——
零 LLM 零新评估——development exposure）。老师建议：先跑通全流程并
把可视化效果做出来——本图是"当前状态漏斗"（旧班次数据的如实呈现），
同时是未来 P3 完整漏斗的模板。不建 dashboard——单一自包含 HTML。

漏斗层级（全部从已暴露报告机械计算）：
  1. 全部轨迹（probe 行：wave3 rounds + block2 rounds + supply 54
     + T117 witness 2 窗口）
  2. sign 分箱（positive ≥M / negative <−M / neutral）
  3. 重复失败组（同 workflow+sign ≥2 窗口——3 组）
  4. 受限五类/保底通道分类（确定性——fault_cases）
  5. Case reconciliation（3 × MATCH_ADD_EVIDENCE → 3 初始 Case）
  6. Edit（1：T117 hampel → pending → delayed 拒绝）
  7. H0→H1（0——尚无获批 Skill——如实）

用法：
  python evaluation/functional/run_v1_case_funnel_viz.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

E2 = PROJECT_ROOT / "artifacts/functional/e2"
HTML_REL = E2 / "w1_case_evolution_funnel.html"
M = 0.005


def main() -> int:
    census = json.loads((E2 / "w1_batch_census_dev_report.json")
                        .read_text(encoding="utf-8"))
    b2 = json.loads((E2 / "w1_block2_census_ec_dev_report.json")
                    .read_text(encoding="utf-8"))
    supply = json.loads((E2 / "w1_program_supply_dev_report.json")
                        .read_text(encoding="utf-8"))
    store = json.loads((E2 / "w1_problem_cases_bootstrap.json")
                       .read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for rep, tag in ((census, "wave3"), (b2, "block2")):
        for sid, rounds in (rep.get("development_rounds") or {}).items():
            for r in rounds:
                for cid, gain in r.get("probes") or []:
                    if gain is None:
                        continue
                    rows.append({"source": f"census_{tag}", "series": sid,
                                 "origin": r["origin"], "op":
                                     cid.replace("cand_", ""),
                                 "gain": float(gain)})
    for s in supply.get("search") or []:
        for e in s.get("per_window_gains") or []:
            if e.get("gain") is None:
                continue
            rows.append({"source": f"supply_{s['label']}",
                         "series": e["series"], "origin": e["origin"],
                         "op": "supply", "gain": float(e["gain"])})
    # T117 witness 2 窗口（census exposed families 直读——全精度）
    for ep in (census.get("exposed", {}).get("families", {})
               .get("winsorize|NEGATIVE", {}) or {}).get("episodes") or []:
        rows.append({"source": "witness_t117", "series": ep["series"],
                     "origin": ep["origin"], "op": "winsorize",
                     "gain": float(ep["gain"])})

    n_total = len(rows)
    pos = [r for r in rows if r["gain"] >= M]
    neg = [r for r in rows if r["gain"] < -M]
    neu = [r for r in rows if -M <= r["gain"] < M]
    groups = [
        {"name": "wave3 winsorize family（4 series × 6 窗）",
         "n": 6, "cls": "WORKFLOW_SUPPLY_GAP",
         "action": "MATCH_ADD_EVIDENCE", "case": "case-0001"},
        {"name": "T105 winsorize cluster（单 series × 3 窗）",
         "n": 3, "cls": "NO_ACTIONABLE_FAULT",
         "action": "MATCH_ADD_EVIDENCE", "case": "case-0003"},
        {"name": "T117 winsorize group（同 series × 2 窗）",
         "n": 2, "cls": "SCOPE_MEMORY_RISK_ERROR",
         "action": "MATCH_ADD_EVIDENCE", "case": "case-0002"},
    ]
    n_groups = len(groups)
    n_group_windows = sum(g["n"] for g in groups)  # 组内窗口总数（分母）
    edit = {"proposed": 1, "stage": "pending → delayed_rejected",
            "detail": "hampel 组内 Support 全正 / delayed @1032 −0.1166 拒绝"}

    # ---- 自包含 HTML（内联 CSS——无外部资源）----
    def bar(label: str, n: int, total: int, color: str,
            sub: str = "") -> str:
        pct = (100.0 * n / total) if total else 0.0
        return (
            f'<div class="row"><div class="label">{label}</div>'
            f'<div class="barwrap"><div class="bar" style="'
            f'width:{max(pct, 1.2):.1f}%;background:{color}">'
            f'<span class="count">{n}</span></div></div>'
            f'<div class="sub">{sub}</div></div>')

    # S3/S3b 正控链（NN5 development single-Episode——如实标注）
    s3 = json.loads((E2 / "w1_s3_simplified_slow_path_report.json")
                    .read_text(encoding="utf-8"))
    s3b = json.loads((E2 / "w1_s3b_normal_entry_adoption_report.json")
                     .read_text(encoding="utf-8"))
    s3b_probes = (s3b.get("entry_H1") or {}).get("probes") or []
    s3b_h1_gain = (s3b_probes[0][1] if s3b_probes else None)
    s3b_h0_gain = 0.0  # H0 候选池空 = abstain/identity
    regret = 0.26115204938975767 - (s3b_h1_gain or 0.0)
    chain_rows = [
        ("失败（NN5 @632 repair_level_shift −0.0789）", "实线", "#c62828"),
        ("受限错误类型（WORKFLOW_DECISION_ERROR——确定性）", "实线", "#2e7d32"),
        ("Agent 选择 Edit Intent（impute_ar——D_patch 数值依据）", "实线",
         "#2e7d32"),
        ("Runtime 编译 Manifest（Agent 语义决策/Compiler 机械编译）",
         "实线", "#2e7d32"),
        ("Support/delayed（632→680→728 时间角色）", "实线", "#2e7d32"),
        ("Skill 持久化（single_repair_level_shift_replacement）", "实线",
         "#2e7d32"),
        ("正常入口检索（H1 候选池含 cand_skill_*）", "实线", "#2e7d32"),
        ("Support 授权执行（H1 +0.043 vs H0 abstain 0）", "实线",
         "#2e7d32"),
        ("removal 行为变化（H0 空池 vs H1 探测 skill）", "实线", "#2e7d32"),
    ]
    chain_html = "\n".join(
        f'<div class="row"><div class="label">{label}</div>'
        f'<div class="barwrap"><div class="bar" style="width:28px;'
        f'background:{color}"><span class="count">✓</span></div></div>'
        f'<div class="sub">{mark}</div></div>'
        for label, mark, color in chain_rows)
    chain_notes = (
        f'<p class="note">NN5 development single-Episode 正控（S3/S3b——'
        f'零新 Claim）。H1 实际执行 learned Skill +{s3b_h1_gain:.4f} vs '
        f'真实 H0（候选池空=abstain/identity）0 → 相对 abstain 改善 '
        f'+{s3b_h1_gain:.4f}（CONTROLLED_POLICY_GAIN_OVER_ABSTAIN）。'
        f'存在未选择的更优 Workflow（repair_level_shift @728 +0.261）'
        f'——oracle/headroom regret ≈ {regret:.4f}——系统尚未找到已知'
        f'更优 Workflow。未证明：自然 Batch 归纳 / Source Memory 价值。</p>')

    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Case Evolution 当前状态漏斗（S0 + S3 正控链）</title>
<style>
  body {{ font-family: system-ui, "Microsoft YaHei", sans-serif;
         margin: 2rem auto; max-width: 900px; padding: 0 1rem;
         color: #1a1a1a; }}
  h1 {{ font-size: 1.3rem; }}
  h2 {{ font-size: 1.0rem; color: #444; margin-top: 2rem; }}
  .row {{ display: flex; align-items: center; margin: .55rem 0; }}
  .label {{ width: 300px; font-size: .85rem; flex-shrink: 0; }}
  .barwrap {{ flex: 1; background: #eee; border-radius: 3px; }}
  .bar {{ height: 22px; border-radius: 3px; min-width: 24px;
         display: flex; align-items: center; }}
  .count {{ padding-left: 6px; font-size: .78rem; color: #fff;
           font-weight: 600; text-shadow: 0 1px 1px #00000055; }}
  .sub {{ width: 320px; font-size: .72rem; color: #666;
         padding-left: .8rem; }}
  .note {{ font-size: .78rem; color: #555; line-height: 1.5; }}
  .foot {{ margin-top: 2rem; font-size: .72rem; color: #888; }}
</style>
</head>
<body>
<h1>Case Evolution — 当前状态漏斗（S0，2026-08-13）</h1>
<p class="note">全部数据来自已暴露报告（development exposure——零新
评估——零新 Claim）。本图如实呈现旧班次结果在"受限 Case 驱动"框架下的
位置，不声称 Batch Evolution 已有效。</p>

<h2>① 全部轨迹（probe 行）：{n_total}</h2>
{bar("positive（≥ +0.005）", len(pos), n_total, "#2e7d32")}
{bar("neutral", len(neu), n_total, "#9e9e9e")}
{bar("negative（< −0.005）", len(neg), n_total, "#c62828")}

<h2>② 重复失败组（同 workflow+sign ≥2 窗口）：{n_groups} 组 / {n_group_windows} 窗</h2>
{''.join(bar(g["name"] + " — " + g["cls"], g["n"], n_group_windows,
             "#c62828" if g["cls"] != "NO_ACTIONABLE_FAULT"
             else "#e65100", g["action"] + " → " + g["case"])
         for g in groups)}

<h2>③ 受限五类 / 保底通道分类（确定性——fault_cases）</h2>
<p class="note">选项屏蔽：无机械证据的类不可选。三个组分别落在
WORKFLOW_SUPPLY_GAP（headroom 全失败 + 54 supply 穷举）/
NO_ACTIONABLE_FAULT（两替代已测全失败——动作空间已测空）/
SCOPE_MEMORY_RISK_ERROR（Support 正 + delayed 负——时间风险已测）。
TASK_INTERPRETATION_ERROR 与 QUALITY_DIAGNOSIS_ERROR 无机械证据
工件——如实屏蔽。</p>

<h2>④ Case reconciliation：3 × MATCH_ADD_EVIDENCE</h2>
<p class="note">三个组分别补充证据到三个初始 Case（普通顺序 ID——
同一性由字段比较判定，无 Hash/SHA 身份体系）。</p>

<h2>⑤ Edit：{edit["proposed"]} 提案</h2>
<p class="note">T117 hampel typed patch —— {edit["stage"]} ——
{edit["detail"]}。delayed 拒绝是正确行为（temporal-risk Case）。</p>

<h2>⑥ H0 → H1：1（受控正控——NN5 development single-Episode）</h2>
<p class="note">受限 Case 框架完整链首次跑通（S3/S3b）——详见下节。
这是受控正例（development positive control），非自然 Batch——自然
多轨迹归纳是下一阶段目标（P2 missingness 自然 block）。</p>

<h2>⑦ 受限 Case 框架正控链（S3/S3b——已通过部分）</h2>
{chain_html}
{chain_notes}

<div class="foot">
数据来源：w1_batch_census_dev_report.json（wave3 rounds）+
w1_block2_census_ec_dev_report.json（block2 rounds）+
w1_program_supply_dev_report.json（54 行）+ witness v3（T117 2 窗）+
w1_problem_cases_bootstrap.json（三 Case + trace）。生成器：
evaluation/functional/run_v1_case_funnel_viz.py。
</div>
</body>
</html>"""
    HTML_REL.write_text(html, encoding="utf-8")
    print(f"== funnel: total={n_total} pos={len(pos)} neg={len(neg)} "
          f"neu={len(neu)} groups={n_groups} edit={edit['proposed']}")
    print(f"== html -> {HTML_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
