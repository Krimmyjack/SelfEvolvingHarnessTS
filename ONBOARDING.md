# 开发者守则 — 在 benchmark-v0.2 上做方法

一页纸。读完你就能接一个方法进来而不破坏任何东西。

prereg（`docs/superpowers/specs/2026-07-14-benchmark-v0_2-prereg.md`）是**写给评审看的**——它论证
为什么这些冻结是可信的。**这页纸是写给你看的**：哪些面你可以碰，哪些面碰了就把别人的数字弄脏。

> **一句话版本**：写一个 `prepare()`，输出**同长度、原量纲、不含 inf** 的数组；
> 只在 **Support-A** 上迭代；**Dev-Query 只读不调**；**Support-B 一次性**；**Final-Query 你根本碰不到**。

---

## 1. 方法契约（`benchmark/method_api.py` 是唯一真源）

一个方法就是一个满足 `BenchmarkMethod` Protocol 的对象——**一个方法、一个 `method_id`**：

```python
class MyMethod:
    method_id = "my_method"                        # 非空、无首尾空白

    def prepare(self, series_view: MethodSeriesView,
                task_spec: TaskSpec,
                observed_pattern_spec: Mapping[str, float]) -> PreparedSeries:
        values = do_something(series_view.degraded_inner_train)   # 你能看见的**只有**这个
        return PreparedSeries(
            series_uid=series_view.series_uid,     # 必须原样回传，改了就是 gate 错误
            values=values,
            operators=("impute_linear", "denoise_median"),        # 你实际跑了哪些算子（会被核）
            units="original_units",                # 唯一合法值
        )
```

**你看得见什么**：`MethodSeriesView` 只有 `series_uid` 和 `degraded_inner_train`。
**看不见的**（`PrivateSeriesEpisode` 持有、runner 私有）：`future`（评分靶子）、`regime_tag`、
`split_role`。想绕过去拿 `future` = 作弊，不是聪明。

**`validate_prepared()` 会拒掉的**（`ContractVerdict.code`）：

| code | 含义 |
|---|---|
| `length_changed` | 输出长度 ≠ 输入长度。**分窗/滑窗类变换在这里不合法** |
| `dimensionality_changed` | 不是 1D |
| `units_changed` | `units != "original_units"` |
| `non_finite_output` | 有 `inf`，或全是 `NaN` |
| `unknown_operator` | `operators` 里有 registry 不认识的名字 |
| `forbidden_target_space_transform` | 用了 `changes_target_space=True` 的算子 |

### 为什么归一化算子被禁入

`znorm` / `minmax_norm` / `sliding_window` / `lag_features` / `spectral_features` 的契约里
`changes_target_space=True`——**它们改变下游的目标空间**。benchmark 自己拥有归一化
（`normalization_owner: "benchmark"`，写在冻结 manifest 里）。你在方法里再归一化一次，
预测就落在一个和 `future` 对不上的空间里，而 sMASE 会照算不误，给你一个**看起来正常的错数**。
所以这不是风格偏好，是硬闸：`validate_prepared` 直接判 invalid。

---

## 2. 算子库（`operators/`）

`operators/registry.py` 是**唯一真源**（`OPERATOR_SPECS`）。每个算子带完整契约：

```
allowed_tasks  destructive  preserves_observed  reversible
changes_target_space  requires_dependency  fallback_policy  dependency_policy
```

现有 25 个 canonical 算子，按机制分族（`impute` / `denoise` / `outlier` / `decompose` /
`structural` / `align` / `shape`）。变更历史与每个算子的动机、默认参数出处、已知失效模式
全部在 **`operators/CHANGELOG.md`** —— **写新算子前先读它**，尤其是 `repair_level_shift` 那节
（那是一份关于"合成测试如何骗过你"的完整事故报告）。

### 加新算子的三条硬规矩

1. **契约八字段齐全**，且必须**从第一天**就带着入池（`tests/test_registry_contract.py` 守）。
2. **不许静默回退**。依赖缺失时的行为由 `dependency_policy` 声明：
   `"recorded_fallback"`（记账后降级，如 `denoise_stl` → `denoise_savgol`）或
   `"hard_fail"`（直接 raise，如 `impute_ssm`）。
   ⚠️ 这条纪律有血的来历：`impute_kalman` 这个名字在本仓库历史上指的**一直是 EMA**——
   一个从来没有 Kalman 的 Kalman。台账记着跑了状态空间、实际跑的是指数平滑，
   而 router 从这条假记录里学动作标签。**任何回退都必须过 `operators/_provenance.record()`。**
3. **命名不得复用历史含义**。旧名字（`impute_kalman` / `kalman_filter` / `fill_gaps`）
   保留在 `ALIASES` 里**只为旧 artifact 可重放**，全部标了 `deprecated`。新代码禁用。

### 语义测试三件套（每个新算子都要有）

① 干净输入上**逐位恒等**（不是"近似恒等"）；② 注入缺陷后修复误差显著下降；
③ **反噬测试**——在这个算子最容易帮倒忙的输入上，它必须无害或伤害有界。

第③条最重要。一个算子在它该赢的格子上赢，什么都不证明；**它在不该动的格子上不动**，
才是它没在撒谎的证据。

> 🚩 **并且：合成测试不算数据验证。** `repair_level_shift` 的三件套全绿（干净序列逐位恒等、
> 消除 91% 的注入损伤），然后在真实 roster 上误触发 **82%**——因为那些合成序列
> **恰好就是它设计矩阵假设的那个模型**。任何新算子在被认真使用前，
> **必须在 Support-A discovery 的真实序列上量一遍**。

---

## 3. 你可以碰什么，不可以碰什么

| 面 | 状态 | 说明 |
|---|---|---|
| `operators/` | ✅ **随便加** | 加算子不会移动 v0.2 的数字（有测试证明，见下） |
| `policy/action_spec.py` 的 `action_menu_v2` | ✅ 可扩 | 扩了就是**新菜单版本**（新 SHA），不许原地改旧版 |
| 你自己的方法代码 | ✅ | — |
| `action_menu_v1()` | ❌ **冻结** | P0 冻结动作面，SHA 被测试钉死 |
| `p6/fast_path.py` 的 `det_ladder` / `GRAMMAR_*` | ❌ **冻结** | **H_ref 的候选文法**。它是 v0.2 里被度量的现任者；扩了它 = 被度量的对象换了人，而所有数字还挂在 `h_ref` 这个名字上 |
| `harness/layers.py` 的 `minimal_l2().operator_defaults` | ❌ **一个键都别加** | 它同时是 menu v1 的 meta 字段（加键 → v1 的 SHA 变）**和** P6 `resolve_steps` 的参数来源（加键 → 静默改掉 H_ref 的参数解析）。新算子的参数请**显式写死在 ActionSpec 里** |
| `benchmark/` 的冻结面 | ❌ | registry / split / corruption / metric / harm δ / bootstrap seed / normalization |
| `results/Benchmark_v0_2/*.json` 等冻结产物 | ❌ | 文件 SHA 被 `benchmark_manifest_v0.yaml` 钉着 |

### 关于那个"红着的" pool code pin

`results/Benchmark_v0_2/program_pool.json` 用 SHA 钉了实现池的源文件，**其中包括
`operators/registry.py`**。所以**你加一个算子，那个 pin 就必然对不上**。

这**不是**你弄坏了什么。文件 digest 回答不了"数字有没有动"——**行为 digest 可以**：
`tests/test_frozen_action_surfaces.py` 把池里 8 个 program 在一条固定探针上的输出做**逐字节
SHA256**，证明它们不变 ⇒ v0.2 的每一个读数依然有效。

漂移必须被**显式调和**：`results/Benchmark_v0_2/pool_code_pin_reconciliation.json`
（旧 pin + 新 pin + 行为等价证据）。`test_pool_code_pins_are_intact_or_explicitly_reconciled`
会在你改动池代码后**失败**，直到你更新那份记录并重新证明行为等价。

**正确做法是更新调和记录，不是放宽测试。** 一个总是红着又没人管的检查，
会教会每个人无视红色——那比没有检查更糟。

---

## 4. 三条资源纪律（违反了，你的结果就不能进论文）

roster 共 1867 条合格序列。**角色决定你能用它做什么**（`role_policies` 写在冻结的
`split_manifest.json` 里，不是约定俗成）：

### ① `final_query`（570 条）— **密封，永不读**

`final_query_state: "sealed"`。**不读、不数、不偷看。** 它只在一次性的 Final campaign 里开封
（roster + SHA + 预算 + 顺序全部冻结 → 开封 → 跑完全 roster → 永久关闭）。
开封前必须已有 Support-A dry-run **和** Support-B confirmation（`require_final_eligibility()` 硬闸）。

### ② `support_b`（331 条）— **一次性资源**

`repeatable: false`，`may_confirm_method: true`。它是**代码/配置冻结之后的全局一次
confirmation**，**不是**每个 domain / 每个 episode 可以反复消费的 promotion 资源。

> ⚠️ 这一条被外部分析**接连搞错过三次**。开发期的 promotion 门是
> **Support-A inner-val + Dev-Query**，不是 Support-B。
> `run_support_b_confirmation()` 里 `prior_artifact is not None` 直接 raise：**它是一次性的**。

### ③ `dev_query`（373 条）— **只读，不做任何调参或挑选**

`may_select_best_fixed: false`，`may_train_oracle_transfer: false`。
它是**评估面**，是你看"方法有没有用"的地方。在它上面挑超参、挑算子、挑哪个 dataset 算数，
就是把评估面变成训练面——**null 变成"发现"的最短路径**就是这条。

### 那我在哪里迭代？→ `support_a`（555 条）

`may_select_best_fixed: true`，`may_train_oracle_transfer: true`，`repeatable: true`。
它还有一层固定内部拆分（`support_a_subsplit.json`）：

- **`support_a_discovery`（420 条）** — 搜索、提议、拟合、调参，**随便造**；
- **`support_a_validation`（135 条）** — 你自己的内部把关面。

**反馈信号也只在这里**：`FeedbackAPI.evaluate()` 硬性要求
`handle.split_role == "support_a"` **且** `channel == "closed_form_inner_val"`，
**且有预算**（超了 raise `FeedbackBudgetError`）。任何别的 role 上调用它 → `MethodGateError`。

（另有 `u`（38 条，NOAA weather）= 跨域检查面，harness 不适配它，只做方向性检查。）

---

## 5. 起手式

```bash
# 环境：conda `project`（D:/Anaconda_envs/envs/project/python.exe）
python -m pytest tests/ -q                       # 1018 passed —— 先确认基线是绿的

# 读这三份，按顺序：
#   operators/CHANGELOG.md                        算子库的变更史与事故报告
#   benchmark/method_api.py                       方法契约的唯一真源
#   docs/superpowers/specs/2026-07-14-benchmark-v0_2-prereg.md   （可选）为什么这些冻结可信
```

写完方法后，**先在 Support-A discovery 上跑通契约**（`run_support_dry_run()`），
再谈 Dev-Query 上的读数。
