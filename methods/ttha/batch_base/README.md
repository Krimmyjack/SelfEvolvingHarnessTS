# batch_base — 批级构造公共底座（P / W 两线共用）

状态：2026-09-13 首版，worktree `p-line/batch-base`。对应 `docs/BATCH_FRAMEWORK_COMPARISON_2026-09-13.md` §5「公共底座」与 `docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md` §4 的七个操作。**底座里没有 LLM 提示、没有选择规则、没有哈希。** 两条控制线只调用这里的函数；任何底座改动都是两线共同的公共变更。

验收：`python -m evaluation.main_protocol_p4.batch_base_smoke [--fit]` —— 22 项零拟合检查 + 5 项含一次原种子重训（与 probe 包 `a55_R_s20260918` 权重逐位相等、C_A/C_B/E 逐位相等）。27/27 通过（2026-09-13）。

## 七个操作 → 函数

| 操作 | 调用 | 阶段/权限 |
|---|---|---|
| overview | `ctx = context.open_job(dataset, "a55", run_dir, stage="material"); ctx.overview()` | 只读 T；返回 32 行 `entity_k` 表（`spec.OBS_FIELDS` 20 个字段）+ 分位摘要 + Consumer/动作说明 |
| inspect_data | `ctx.inspect_data([3, 7], kind="hour_profile" \| "daily_means" \| "segment", sub_range=None)` | 只读 T；越界抛 `PermissionError` |
| build_material | `c = policy.compile_policy(policy_json, ctx.overview()["entities"]); ref = materials.build(ctx, c["assignment"], material_id)` | 只读 T；同 assignment → 别名（`ref.alias_of`），不重算 |
| inspect_material | `materials.inspect_material(ctx, ref)` | 只读 T；代理量，不是效用 |
| evaluate | `cells = train.evaluate(ctx, ref, seeds, ledger, repo_root)` | 每 seed 一个子进程，进程内只有 [t−672, t+96)；返回 C_A 分数；缓存命中记 `cache_hits` |
| compare | `feedback.paired(cells_a, cells_b, block="c_a", label_a, label_b)` | 配对差、SD/SE、符号、df=n−1 t 区间、δ=1%×A、实体贡献（诊断，非因果标签） |
| commit | `commit.commit(run_dir, dataset, job, material_id, reason, delivery_seed)` | 只接受本作业已拟合材料；写 `commit.json` 后交付不可改 |
| （commit 之后） | `commit.open_c_b(...)` → `commit.freeze_e(...)` → `commit.score_e(...)` | 文件存在性强制顺序；E 分数只写盘，不返回给控制器使用 |

## 策略语法（Material Plan）

```json
{"default": {"steps": [{"op": "timemixup", "donor_rule": "R", "w": 0.25}]},
 "rules": [{"when": {"all": [{"feature": "r_head", "op": ">=", "value": 0.6},
                             {"feature": "r_gap", "op": "<=", "value": {"quantile": 0.5}}]},
            "steps": [{"op": "timemixup", "donor_rule": "U", "w": 0.25}]}],
 "rationale": "...", "observation_fields_used": ["r_head", "r_gap"]}
```

- 动作与参数集合：`spec.ACTIONS`（timemixup R/U × w{0.1,0.25,0.5}；freqmask mu{0.05,0.1,0.2}；freqmix R/U × mu），≤2 步，`[]`=identity。
- 谓词：`{feature, op, value}` 叶（feature ∈ `spec.OBS_FIELDS`，value 数字或 `{"quantile": q}` 按当前批分布解析）+ all/any/not/const；首个命中规则生效；UNKNOWN（字段缺失）不触发。
- 校验：`policy.validate_policy` fail-closed；rationale 可提 `entity_k`（作业内材料计划），Skill 正文用 `validate_text(..., allow_entity_ids=False)`。
- 控制臂：`policy.random_policy(seed)` 从同一语法均匀抽；`policy.identity_policy()`、`policy.fixed_mixup_policy()`。

## 随机流与可复现

- 增强：`RandomState(900090 + i + 100000*step)`，donor 永远来自原始父对；R/U 规则及 freqmask/freqmix 抽样顺序与 a94/probe 逐位一致（smoke 核过）。
- 训练：`torch.manual_seed(r)` + 父 batch 流 `RandomState(800000+100r+9).randint(0,13856,(2000,64))`；同 seed 各材料共用初始化与 batch。
- 缓存：同 job、同 assignment（材料别名）、同 seed → `train.evaluate` 跳过并记 `cache_hits`。

## 账本

`budget.Ledger(path, max_fit_attempts, max_llm_requests, max_llm_tokens, max_wall_s, max_retries)`：拟合尝试/成功/失败/缓存、LLM 请求/HTTP 尝试/token（未知不补零）、墙钟；超限抛 `BudgetExhausted`。`llm.LiveClient(LLMConfig, ledger)` 沿用 a94 的故障分类与返回模型身份校验；`llm.DryClient` 供测试。

## 目录布局（每次运行）

```
<run_dir>/<job_id>/
  scaler.npz  overview.json
  materials/{index.json, <mid>.npz, <mid>__steps.json}
  cells/<job>__<mid>__s<seed>.json   runs/*.pt   predictions_c_a/*.npz
  commit.json → c_b_scores.json → predictions_e/, e_frozen.json → e_scores.json
```

## 不在底座里（各控制线自带）

Fast/Slow 提示词与解析、工具循环或固定轮次、incumbent 替换/commit 决策规则、Skill 存储与匹配（沿用 `knowledge.py`/`applicability.py` 的既有算法）、运行器与看门狗。

## 已知限制

- `evaluate` 子进程 `python -m methods.ttha.batch_base.train` 需要仓库根目录下的 `SelfEvolvingHarnessTS` 自指 junction（主库有；worktree 用 `mklink /J SelfEvolvingHarnessTS .` 补）。
- 数据集路径写在 `spec.DATASETS`（本机绝对路径）。

## W 主目录接入差集（2026-09-13）

本目录来自 P 工作树已交付首版的源码快照；原 P 文件未修改。W 当前执行费用与并发守卫走
`evaluation/main_protocol_p4/batch_research_runtime.py`，不使用原 evaluate() 的事后拟合计费。
本地必要差集：C_B/E 前置文件检查移到任何标签读取之前；CSV 不再额外解析上界后一行；
评分保留完整 origin×entity NMSE；--no-feedback 拟合只打开 material/T；保存完整频域 mask 索引。
训练算术、原始/派生权重、随机流、参数和主指标不变。这些属于可交回公共底座的修复，
不计为 W 的方法增量。19项 W 接口/真实材料/费用故障检查通过；1次原种子真实重训权重及 C_A 逐位复现。用户授权后W首包已完成：31次拟合（含对齐1次）、12次实验LLM；结果见 `_scratch/dev_batch_research_workflow_v1/REPORT.md`。未支持方法净增益；底座单独score_e入口的全局冻结守卫缺口记录于该报告。

后续W修复：`score_e(..., cohort_root=...)` 在标签读取前核对全局完成/冻结分支和预测文件；W CLI显式启用，直接调用可识别W运行目录标记。原首包三个分支冻结工件通过新守卫只读检查。standalone底座不自动推断未知多分支协议。详见 `_scratch/dev_batch_research_roundtrip_prepare/REPORT.md`。
