# HEC-1 v1.1 只读复核报告(主线,非作者;2026-09-03 夜)

范围:`run_hec1.py`(2665 行)、`outer_loop.py`、`restricted_draft.py`、`hec1_contract.py` 当前磁盘字节;只读、0 LLM、
0 fit、未改任何文件、未读 shakedown 效用。测试与回归数字(89/89 聚焦、461/461 主协议、7/7 smoke、125 HEC-1)为作者
自述,主线未复跑。**结论:PASS,附三项 commit 前条件与四项披露。**

## 1. 逐项(CODE FACT 以 file:line 为证)

| 组 | 项 | 结果 | 证据 |
| --- | --- | --- | --- |
| A | 全 census key 去重,范围含已关闭 | PASS | `outer_loop.py:409` `held = held_lineage_keys ∪ ledger.lineage_keys()`;`restricted_draft.py:361-369` `lineage_keys()` 遍历全部 drafts(含 closed);`:396-400` `open_restricted` 对已有 lineage 的 key **raise** |
| A | held 不再只从 bank 推 | PASS | `run_hec1.py:1498/1547/1622` `active_lineage_keys` 由激活事件维护并传入 `held_lineage_keys`;K0 交接携带 `lineage_keys`(`:2241`) |
| B | 缓存键含 arm / cell / face / Consumer / typed Program,不含 Scope | PASS | `run_hec1.py:395-404` |
| B | Scope 只掩码不重拟合;逐位一致前提成立 | PASS | `_build` 一次 `scoped_evaluate(scope=legal)` 存 raw/program 逐序列(`:429-442`);`reading` 用 `where(mask, program, raw)` 重掩码(`:468-473`);前提 = `scoped_serving_evaluator.py:208-212` 两模型与 Scope 无关 |
| B | 退化 context 照原逻辑拒绝 | PASS | `_build` 按 `scoped._prepare` + `_center_scale` 同法算 `degenerate_uids`(`:420-428`);`reading` 对 `selected ∩ degenerate` raise(`:460-466`) |
| B | 校验器拒绝照原逻辑 | PASS | `_build` 存 `verifier_passed`,`reading` raise `WINDOW_VERIFIER_REJECTED`(`:413-415, :457-458`) |
| B | 三本账分记 | **PASS(实现)/ FINDING(接线)** | 缓存对象内 `physical_fits / logical_evaluations / cache_hits`(`:384-386, :432, :448, :456`);但 `Ledgers.cache_hits/misses`(`:135-136`)与之未接——作者已报 shakedown 全程 0/0 |
| C | 未来步预留按 fits 计、最坏成本 | PASS | `reserve_for_future_steps`(`:739-`)Σ_{j>k} j·period × `CACHE_FITS_PER_CELL=2`;docstring 明示"单位换算是全部要点" |
| C | 不加每步上限、不加新优先序、未筛者记录 | PASS | 作者测试"候选序在宽/紧预算下一致";`REPLAY_FITS_BUDGET_SPENT` 条目含 kind / signature / estimate / remaining(`outer_loop.py:750-757`) |
| C+ | 门权威二分 | PASS(待 sol 一句) | `resolve_gate_disagreement`(`:579-617`)分 `AUTHORITY_UPHELD` / `LOST_ACTIVATION`;`classify_authority_breach`(`:620-634`)只在"已激活且权威未批"判 `AUTHORITY_BYPASSED`;`may_activate` 恒等于 P4 `passes`(`:611`) |
| 合同 | 26 / 23 / 19 三量分离;`⌈0.8×23⌉=19`;readout 只计配对点 | PASS(作者自述 + 冻结清单交叉核) | `hec1_scoreability.py` 冻结清单;preflight 独立推导;`assert_frozen` 对 18 与伪造不可评单元均 fail-closed(作者验证) |
| 合同 | D_o ≥ 0.115;P1-only 措辞;validation-search 不进 Harness;穷举诊断不成 K0;Phase F 宏平均 | PASS(作者自述,六项均在 `assert_frozen` 内 fail-closed) | — |
| shakedown | 26/26、165 LLM、3.07 h、无 RunFault;工件降级并移出 readout 白名单;五步外环全"无候选" | 采认为仪器结果 | 0-fit 预扫逐单元命中(3 单元 × 3 臂 = 9 个 `FaceNotEvaluable`,无多无少)→ **N_T_eff=23 为实测** |

## 2. 三项 commit 前条件

1. **sol 一句裁定(gate 3 语义)**——见 §3 推荐文本。
2. **缓存计数接线**:`Ledgers.cache_hits/cache_misses` 与各臂缓存对象的 `cache_hits / logical_evaluations` 合流(misses = logical − hits),入仪器报告;附测试。
3. **召回归因行为锁**:`deployed_via` 的"部署程序 ∈ 单元起始 Active 集"分支加行为测试(作者已列)。
commit message 须点名 19-series 分母行为测试的测试名(sol 机械门 4)。

## 3. gate disagreement 的裁定建议(呈 sol,一行可批)

事实:`online_loop` 的 delayed 准入无覆盖底线,P4 `_gate` 有;二者在 treated < MIN_TREATED 时**必然**不一致;shakedown 3 例
全部为 P4 拒绝、未激活、未绕过。字面读 gate 3 会使三顺序全部降级,实验不可能出结果。

**推荐文本**:
> 科学顺序中,`AUTHORITY_BYPASSED`(任何 Active 增长未经 P4 权威)→ 降级该顺序;`AUTHORITY_UPHELD` 且 P4 失败线
> **仅含 `coverage_floor`** → 计数并披露;`AUTHORITY_UPHELD` 且 P4 失败线**含任何风险线**(aggregate / harmed_fraction /
> single_series_harm)→ 视为两套准入在风险上分歧,**降级该顺序**;`LOST_ACTIVATION` → 计数并披露(对 online 臂不利,
> 不作科学判词)。

理由:第三类是绊线——两套准入若共用 bounded_risk_v1,在风险线上不应分歧;出现即是仪器缺陷。实现上只需在
`audit_hec1_instrument.py:167` 附近按 `p4_gate.failed_lines` 再分一层,不改 runner 语义。

## 4. 四项披露(入课末报告与合同勘误,不阻塞)

1. **replay 预算实际容量**:156 允许额、最坏 2 fits/cell 预留 → 五步各一次 screen 共 150,仅余 6 → **每顺序保证 5 次
   screen、最多再多 1 次**;实际因 raw 跨步复用会更宽裕(第 k 步真实成本 ≈ 5k+5),但预留按最坏计,早期可能拒第二候选。
   "不保证全筛"须如实写。
2. **内环 Support 面的 fits 不进 replay 缓存**(缓存只由 replay 构建)——保守,非缺陷;披露。
3. **`LOST_ACTIVATION` 偏向不利 online 臂**(P4 过而事件未批 → 不激活):计数披露;若频繁,作仪器限制记。
4. **缓存 `_build` 用 `scope=legal` 一次调用**替代原 Static + 带程序两次调用:逐位一致依赖 `scoped_evaluate` 的确定性与
   Scope 无关性(已验证前提;作者以真 KDD cell 测过五情形)。

## 5. 判定

**PASS(条件式)**:满足 §2 三条 → allowlist commit → 发车收据记 commit → runner 断言 HEAD 与干净树 → Phase S-v1.1。
作者自报的"tests define the interface"纪律偏差已自行记录,权威序 AGENTS → 合同 → 测试 → 实现,采认。
