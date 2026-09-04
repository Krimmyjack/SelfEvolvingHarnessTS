# 增量复核包:外环 Slow 的 `harness_view` 接线修复(待非作者复核,2026-09-03 夜)

只读复核对象 = **未提交工作树 vs HEAD `e33f036`**,范围三个文件。作者 = Opus(执行线);复核者 = 非作者(grok 4.6-xhigh)。
本包不含新方法设计;判定只有 **PASS / FAIL**。

## 0. 为什么有这一轮

Forward 首次尝试(`v11_forward_live`,commit `e33f036`)在第 5 单元结束后的第一次外环 Slow 调用崩溃:

```
run_hec1.py OuterSlowAgent.__call__ -> core.run_stage(..., harness_view={})
agent_core.py:179  "instruction": harness_view.instruction
AttributeError: 'dict' object has no attribute 'instruction'
```

主线裁定 `RUN_BLOCKED_NO_VERDICT`(仪器故障,非科学读数),授权修接线 + 补测试 + 新 commit,**Forward 从 0 重跑、不 resume**
(一条顺序一个 commit),**Phase S 与 K0 不重跑但须披露**。

## 1. diff(三文件)

| 文件 | 改了什么 |
| --- | --- |
| `evaluation/main_protocol_p4/run_hec1.py` | `OuterSlowAgent.__init__` 增**必填** `snapshot`;`__call__` 用 `resolve_harness_view(self.snapshot, {}, role="slow")` 作 `harness_view`,`source_snapshot_sha` 由 `""` 改为真 `runtime_bundle_sha`;新增 `Arm.active_snapshot()`(已建则取 `_method._active_snapshot()`,未建则取 `start_snapshot`);`Arm.outer_step` 传该快照;`outer_slow_factory(core, guard_, snapshot)` |
| `tests/main_protocol/test_hec1_v11_amendment.py` | 新增 4 项:NARROW 候选经真实 `core.run_stage`(mock 只在后端回复层)、真实 Harness view 而非空 dict、runner 接线锁(含 `snapshot` 必填)、resume 下仍有快照可问 |
| `evaluation/main_protocol_p4/hec1_contract.py` | 新增 `CODE_PROVENANCE_ERRATUM` 并进 `to_dict()`:Phase S 在 `e33f036`、Phase T 在修复 commit、diff 仅限该路径、Phase S 未到达、Forward 不 resume 的理由、崩溃尝试的留档与开销归类 |

## 2. 复核清单(请逐项给 PASS/FAIL + file:line)

| # | 项 |
| --- | --- |
| A | 修法与 Source-v3 先例一致(`scope_clause_agent.py:257` 用 `resolve_harness_view(snapshot, {}, role="slow")`)——同 role、同空 public features,故没有 Target 观察从这道门漏进 Slow 视野 |
| B | `snapshot` 必填而非默认,同一个类不可能再被"未武装"地构造 |
| C | `Arm.outer_step` 传的是**本臂自己的**活动快照;Static / 非外环臂到不了这里;resume 下退回 `start_snapshot` 是诚实答案而非掩盖 |
| D | 离线脚本路径行为逐位不变(0-LLM 课程记录必须一致) |
| E | 四项新测试真的咬得住:走真 `run_stage` / 真 schema / 真 view,只 mock 回复;非同义反复。**作者已验**:把 `harness_view={}` 注回去,四项以同一 `AttributeError` 失败 |
| F | 没有夹带方法设计:无新 Risk 面、无阈值改动、无新观察特征、无新激活路径。请单独判 `source_snapshot_sha` 由 `""` 改为真 sha 属"应披露的行为变化"还是"本就正确的溯源" |
| G | 工作树 diff 除这三个文件外无其他代码改动(`docs/HEC_EVOLUTION_MAINLINE_PLAN_*` 与账本的新增属主线,**不在本次 commit 内**) |

**不得据以判 FAIL**:留档目录 `forward_v11_attempt1_blocked`;他线未提交 docs;基线既有失败;全量差集尚在跑。

## 3. 请顺带核实一个事实主张

主线"不重跑 Phase S / K0"的依据是"Phase S 从未执行那行"。请从工件自证:

`artifacts/main_protocol/hec1_phase_s_phase_s_v11_live.json`
- `ledgers.llm_outer`
- 两个外环步的 `slow_calls` 与 `candidates`

作者读到的是 `llm_outer = 0`、两步 `slow_calls = 0` / `candidates = 0`。

## 4. 作者已跑

- 新测试 4/4;注入原缺陷后 4/4 失败(同一 `AttributeError`)
- `tests/main_protocol` 全量 **478 passed / 1 skipped**
- `--smoke` **7/7**、0 LLM、4 fits
- `assert_frozen()` clean;`assert_launchable('phase_s'|'phase_t_forward')` 均 True
- 全量 `tests/` 对 `_scratch/pytest_baseline_tests_tree.txt`(UTF-16,55 项 FAILED/ERROR)的差集:**重跑中**,结果补入

## 5. 返回格式

1. 判定:PASS / FAIL
2. A–G 表(PASS/FAIL + file:line)
3. 第 3 节的事实核实(用工件自己的数字)
4. 若 FAIL:只列阻塞项
5. 一段残留披露(不阻塞)
