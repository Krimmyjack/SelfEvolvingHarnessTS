# TSharness vNext 实施计划 v0.3：Protocol Hardening、M0 Recovery 与可分解的 Harness Evolution

状态：**P0 协议层已实现；M0 已完成 L0–L2，但历史 Windows 环境尚未恢复；Task G 未授权。**

本文完整替代 v0.2。实现位于独立 `vnext/` 包；legacy fast path、P6、H_ref、
`action_menu_v1()`、`minimal_l2().operator_defaults` 与 `results/Benchmark_v0_2`
保持冻结。

## 1. 当前机器状态

```text
M0 = M0_BLOCKED
TASK_G_AUTHORIZED = false
new_holdout_access_count = 0
H0 = NOT_FROZEN
Final = SEALED
```

当前阻塞的接续步骤见 `WINDOWS_M0_HANDOFF.md`。历史 benchmark 环境为 Windows 10、
Python 3.10.19、NumPy 2.2.6；Mac uv 环境仅是 portability shadow，不能签发精确 M0。

当前 `HarnessArtifactV1.h0()` 与 `HarnessArtifactV2.engineering_default()` 都只是兼容期
工程默认，不是研究级 H0。正式 H0 只能由：

```text
Init Corpus（80 Legacy core + 56 Probe-consumed Traffic）
→ operator experience + seed policies + ProgramSpecV1 templates + aggregate memory
→ formal H0
```

产生，并由 `InitCorpusManifestV1`、`InitHarnessPreregV1` 和
`InitHarnessArtifactV1` 唯一绑定。M3a 只选择 runtime supplier control，不改变 H0。

## 2. 权威数据视图与历史暴露

冻结 split/subsplit 不修改。vNext 旁挂以下视图：

- `SA-D/init`：136 条、136 overlap groups，仅用于正式 H0 构造；动作经验、资格、seed
  policy、模板和 memory 都是 H0 的内部组成，不作为独立的后续调参用途；
- `SA-D/search`：284 条、247 groups，用于 M3/M4 nested cross-fit；
- Task G：完整 SA-D 420 条、383 groups 的五折 discovery cross-fit；
- SA-V：135 条、110 groups，一次 vNext candidate 聚合查询；
- Dev：373 条、323 groups，方法冻结后只读一次；
- Support-B、Final、U 保持 sealed/one-shot 边界。

`HistoricalExposureManifestV1` 已证明全部 555 个 Support-A UID 的历史 baseline loss
已公开，SA-V 135/135 UID 也包含在旧逐 UID program/repeat loss 中。因此
`certified_virgin` 只表示未参与旧 H_ref，不表示结果从未暴露。SA-V 的准确名称是：

```text
baseline-outcome-exposed,
vNext-candidate-unqueried promotion gate
```

自动搜索环境不得读取旧逐 UID loss；真正独立确认由 Support-B 承担。

## 3. 生命周期

```text
P0 Protocol Hardening
→ M0 exact asset recovery + shadow raw-to-result reproduction
→ Task G discovery cross-fit
→ M2 minimum vertical slice
→ freeze Init-derived H0
→ M3a runtime supplier identity（H0 不变）
→ M3b equal-budget evolution race + mature supplier swap
→ optional one-shot SA-V
→ method/roster/budget freeze
→ Support-A dry-run
→ PostFreeze-Dev readonly
→ one-shot Support-B
→ Final-ready
→ separately authorized Final
→ U directional report
```

`VNextLifecycle` 将顺序写成 hash-chained、fsync 的机器状态。SA-V 失败只产生
`SAV_CLOSED_FALLBACK_H0`，仍继续确认 H0；Support-B 失败才是主线 terminal。

## 4. 已实现的 P0 协议

### 4.1 访问前 WAL

`OneShotAccessControllerV1` 统一管理 SA-V、Support-B 与 Final bridge：

```text
SEALED → AUTHORIZED → ACCESS_RESERVED → RUNNING
       → CLOSED_PASS/CLOSED_FAIL
       ↘ INTERRUPTED_INFRA → exact resume
```

loader 必须接收 durable `AccessReservationV1`。ledger canonical path 只由 resource
manifest SHA 派生，并具备 exclusive lock、append-only hash chain、fsync、torn-ledger
校验和 terminal result 不可覆盖。timeout、invalid、dependency failure、budget excess
均为方法 terminal；只有同 campaign/code/data/environment/budget/seed/checkpoint 的基础设施
中断可以恢复。

### 4.2 Method 与 evaluator

`MethodInputContractV1` 要求输入一维、非空、无 Inf 且至少一个 finite observation。
空、全 NaN 或 Inf 返回 `METHOD_TERMINAL_INVALID_INPUT`，不得伪造合法 fallback。
合法输入继续使用：

```text
selected program → conservative recovery → raw identity
```

UID 只保留在 audit envelope；改变 UID 不得改变 PatternCard、seed、cache、候选、effect、
PreparedSeries、operator provenance 或 fallback。`observed_pattern_spec` 采用空 allowlist，
dataset/role/scenario/regime 等字段 fail-loud。evaluator 支持 history/inner 两种调用顺序的
一致性测试。

### 4.3 Init Corpus、H0 与 LLM

Init Corpus 固定为 136 条、136 overlap groups：

- Legacy core 80 条：Public Health、Macroeconomics、Cash Demand、Tourism、
  Demography、Hydrology、Solar；
- Probe-consumed extension 56 条：Traffic。

H0 构造必须同时产出 Init-only 的算子经验、seed policies、程序模板和聚合 memory；任一为空
都不得签发正式 H0。Memory 不携带 UID 或 dataset identity。Fresh Support-A、SA-V、Dev、
Support-B、Final 和 Weather U 全部是 H0 构造的禁止输入。

`HarnessArtifactV2` 移除了 `llm_runtime_qualified` 这种混合证据字段。权限拆为：

- `LLMTrialAuthorizationV1`：技术上能否运行 LLM 实验臂；
- `initial_runtime_efficacy`：LLM 在冻结 H0 上是否即时优于 deterministic；
- `evolution_llm_qualified`：LLM edit trajectory 是否优于 deterministic trajectory；
- `mature_runtime_llm_qualified`：成熟 Harness 上 LLM supplier 是否仍有增量。

初始 runtime 失败不会禁止 LLM evolution trial。Harness edit 必须由独立
`HarnessEditAuthorizationArtifactV1` 授权，且 selector/retrieval 只能移动一个冻结网格步，
supplier mix 只能移动一个 slot，risk rule 只能增删一个 predicate。

### 4.4 冻结统计决策

M3a 六臂固定为 `frozen_h0`、deterministic、random、LLM-direct、
LLM-plan-compiler、hybrid。只有 plan-compiler 对 deterministic 是 primary LLM 检验；
direct/hybrid 只作 secondary diagnostic。

Runtime supplier 选择顺序固定为：primary LLM 通过 → random 有增量 → deterministic 安全非劣
→ frozen H0 incumbent。该选择不重写 H0。M3b 四轨迹均从同一 H0 出发，每轨迹最多三轮、每轮三个 slot；
invalid/timeout/duplicate-effect 占 slot且不补发。成熟 endpoint 做 deterministic/LLM 2×2 swap。

SA-V 唯一 comparator 是 H0。promotion 使用 `epsilon=0.02`、`delta=0.05`、CI90、
B=2000、seed 20260713 和 paired overlap-group bootstrap。若 discovery 无合格候选则
abstain，不打开 SA-V。

## 5. M0 Recovery 硬门

M0 使用共享 raw `data/benchmark_v0/raw` 和版本化 derived root
`data/benchmark_v0_2/clean_base`。现有旧 preflight 被降级为 artifact integrity check，
永远不能产生 M0_PASS。

`DataRecoveryManifestV1` 的权威集合是 acquisition manifest、frozen registry source SHA、
METR pins、legacy binding 和 benchmark manifest 的并集。恢复只允许：exact backup byte-copy
→ pinned download 命中同 SHA → 手工官方资产命中同 SHA。所有恢复先进入 quarantine/shadow，
不得覆盖 frozen result。

`CleanBaseIntegrityManifestV1` 覆盖全部 1919 registry records，逐 UID 检查 values、mask、
timestamps、length、source binding、stable slot 和 record sidecar。

环境采用单一 uv 0.11.28 lock，固定 Python 3.10.19、NumPy 2.2.6、SciPy 1.15.2、
statsmodels 0.14.6、PyWavelets 1.8.0、scikit-learn 1.7.2，并绑定 pandas、pyarrow、
Torch、BLAS、硬件、线程、locale/timezone 和 deterministic flags。

raw-to-result 必须在 shadow root 报告 L0 raw SHA 至 L9 headline 的十层证据；Raw、STL、
H_ref、transfer ceiling、ex-COVID、per-UID loss、program provenance、CRN、per-dataset fit 与
fold order 均须匹配，canonical 数值误差不超过 `1e-9`。

最终只有：

```text
M0_PASS
M0_FAIL_PROTOCOL_ERRATUM_REQUIRED
```

不存在 partial pass。

## 6. 当前 M0 阻塞证据

截至 2026-07-15 的机器审计状态：

- 53/53 raw/legacy assets 全部命中 frozen SHA；
- legacy Monash bundle 已恢复为 `f4efc05f…a9d`，原错误版本保存在 quarantine；
- frozen registry 中旧 acquisition receipt 未覆盖的 NOAA SHA 已通过 station ID 补齐
  canonical path provenance；
- `benchmark_v0_1/clean_base` 作为恢复源复制到 v0.2 规范路径，复制前后均为
  1919/1919 通过，路径无关 content SHA 为 `eddbb944…ce966`；
- Mac portability 环境已安装并生成真实 `uv.lock`，80 项 vNext/benchmark 回归测试通过；
- shadow raw→clean-base→registry→split 已完成，L0–L2 与冻结产物一致；
- 历史运行环境现已确认是 `Windows-10-10.0.26200-SP0`、Python 3.10.19、NumPy 2.2.6，
  当前 Mac 环境不能作为历史精确复现环境；
- program-pool 行为探针历史 SHA 为 `9a3049…`，Mac 重建为 `78b9cc…`，开放的
  `ProtocolErratumV1` 在恢复 Windows 环境前阻断 L3–L9；
- 未访问 SA-V、Dev、Support-B、Final 或 U。

当前 readiness 阻塞是历史 Windows execution environment。下一执行顺序固定为：在旧
`D:/Anaconda_envs/envs/project` 捕获 Conda/包/平台指纹 → 行为探针命中 `9a3049…` →
签发 Windows historical lock → shadow L3–L9 复现。Task G 在最终 M0_PASS 前持续关闭；
不能通过更新 golden SHA、修改 UID、registry、依赖版本或阈值来绕过复现。

## 7. 测试与后续执行

测试面覆盖：数据视图计数/互斥、历史暴露、UID substitution、非法输入、call-order、
H0 lineage、LLM static/evolution 分离、typed atomic edit、三 slot accounting、effect dedupe、
mixed-policy retraining、SA-V exact comparator、WAL reserve-before-load、concurrent/double ledger、
exact resume、lifecycle non-feedback，以及 M0 binary verdict。

M0_PASS 后才实现并运行真实 Task G trainers、M3/M4 full-fit scheduler、SA-V aggregate runner、
confirmation roster 和 Final bridge。Final 始终需要单独用户授权。
