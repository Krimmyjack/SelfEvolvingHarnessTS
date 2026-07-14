# operators/ CHANGELOG

算子库的变更台账。**每个算子入库必须在此登记动机与日期**（E-3.3 DoD）。

这份文件不是礼节性的。本仓库的算子事故——`impute_kalman` 从来不是 Kalman、`impute_fft` 曾是
伪装成 imputer 的 denoiser、`v_stl` 曾是"garbage-period 上激进平滑碰巧赢"的伪影——全都是
**算子的声明身份与实际行为脱节**。台账是这条纪律的执行面：一个算子叫什么、为什么存在、
默认参数从哪来、依赖没了会怎样，必须写下来，而不是靠读实现去猜。

---

## 2026-07-14 — E-3.3 R1–R3、R5：动作空间扩充（4 个新算子 + menu v2）

### 为什么现在做

benchmark-v0.2 的 Dev 读数暴露了一件事：**方法侧的动作空间在两个缺陷机制上是空集**。
`benchmark/programs.py: CAPABILITY_GAPS` 在任何 v0.2 数字被读到之前就声明了这一点，
v0.2 实测坐实了它——level_shift 腐蚀格上，池里四个 program 的损失小数点后四位相同。
C1（H\* = f(pattern, task)）的主张是"最优处理取决于条件"，但**没有手的条件化是空谈**：
harness 条件化得再好，也修不了一个它没有算子去碰的缺陷。

时间窗是硬的：TTHA-0 的 prereg 一旦签发，动作面即冻结，中途加算子 = 协议漂移。
所以这四个算子必须在 prereg 签发**之前**全部入列。

### 新增算子

| 算子 | 机制 | 依赖 | 契约要点 |
|---|---|---|---|
| `repair_level_shift` | 结构断层修复（**新机制**） | 无（纯 numpy） | destructive，anomaly 禁用 |
| `hampel_filter` | 局部自适应点式离群修复 | 无（纯 numpy） | destructive，anomaly 禁用 |
| `impute_ssm` | 模型预测族插补（状态空间） | **statsmodels（硬依赖）** | preserves_observed |
| `impute_ar` | 模型预测族插补（双向 AR） | 无（纯 numpy） | preserves_observed |

#### `repair_level_shift`（R1，`operators/s1_structural.py`）

> ### 🚩 结论先行：**本算子不得进入 pool_v3**（详见下方 §"Support-A discovery 检查"）。
> 它作为**方法侧**动作入列（menu v2 的 `v_levelshift`），但它**没有**通过冻结前的真实数据
> 检查：在**未经腐蚀**的真实 roster 序列上它仍会改动 **45.7%** 的序列。把这样一个算子冻进
> benchmark 的 oracle 池，会得到一个建立在"它到处乱动"之上的天花板。

- **动机**：填 `CAPABILITY_GAPS` 里**预先声明过**的 `structural_break` 缺口。先声明缺口、
  后补算子，这个次序是它免疫"事后挑格子"指控的全部依据。
- **算法（当前）**：sup-Chow 扫描 + 贪心前向选择 + **只撤销暂态偏移**。
  设计矩阵 = `[1, t, 连续分段线性(hinge)基底]` ⊕ 季节哑变量 ⊕ 已接受的阶跃列；
  检验量分母用 **batch-means 长程标准差**（非 iid MAD）。
- **未采用 ruptures**：会给冻结池增加第四个必须硬校验的 `requires_dependency`，且
  `assert_pool_dependencies` 的语义是"少一个依赖就拒绝建池"——为一个可以纯 numpy 写出来的
  扫描增加这个脆弱面，不划算。选定即写死：**纯 numpy 路线**。

##### 四个踩过的坑（全部记下来，每一个都改变了实现）

1. **效应量闸放错了层**（合成测试就抓到）。第一版把效应闸放在检测期的 δ̂ 上，②号语义测试
   全灭。δ̂ 是"τ 到序列末尾的**永久阶跃**"的 OLS 系数；而一个宽度 c 的**脉冲型**断层（升上去
   又降回来——benchmark 的 level_shift 正是如此）只能被单个永久阶跃解释掉 c/(n−τ) 的份额：
   实测真实跳高 8.6 的脉冲，第一步贪心的 δ̂ 只有 **0.45**。效应量必须判在**最终分段电平**上。

2. **线性趋势基底 → 模型失配被读成缺陷**（只有真实数据抓得到）。`[1, t]` 假设趋势是线性的；
   `monash:covid_deaths` 是**累计计数**（单调递增的 S 曲线）。失配残差呈系统性 U 形 → sup-Chow
   把它读成一串"电平断层" → 分段对齐把曲线打成垃圾（重建误差 **69σ**）。
   修法是结构性的：趋势基底换成**连续**分段线性（hinge）——**连续基底无法表示跳变**，所以它
   柔到跟得住 S 曲线，而真断点依然留在残差里。柔性与检测力在这里不冲突。

3. **iid 方差假设 → t 统计量恒为显著**（只有真实数据抓得到）。真实序列的残差强自相关，
   后缀和的真实标准误被抬高约 **2.9 倍**（Support-A 实测中位数）。iid 假设下，
   **100% 的干净真实序列**的 max|t| 都越过 3.5。这不是"把阈值 3.0 提到 3.5 就保守掉了"的
   量级问题（v1 的注释正是这么天真地写的），是**分母算错了**。改用 batch-means 长程标准差。

4. **只建日周期 → 周末被读成断层**（只有真实数据抓得到）。hourly 序列的电平主导循环是**周**，
   不是日；只建 period=24 的哑变量，周末那道真实的电平台阶就成了"断层"（traffic_hourly /
   metr_la 误触发 77–100%）。修法：sub-daily 采样改建 7×period 的哑变量——**168 = 7×24，
   周哑变量吞掉日哑变量**（phase mod 168 决定 phase mod 24），不需要两套并存。

##### ⚠️ 最贵的一课：合成测试全绿，是因为它测的是我自己的假设

坑 2/3/4 **在合成测试上一个都暴露不出来**——因为我的合成序列（线性趋势 + 纯正弦 + 白噪）
**恰好就是设计矩阵假设的那个模型**。三件套语义测试全绿、干净序列上逐位恒等、注入断层后消除
91% 的损伤……然后在真实 roster 上误触发 82%、平均损伤 1.78σ。
**拿自己的假设当测试，什么也没测。** 现在 `tests/test_operators_e33.py` 里有一组
`test_r1_reverse_test_nonlinear_trends_are_not_chopped_into_fake_steps`，取自真实 roster
存在的形态（累计计数 S 曲线 / 指数增长 / 饱和凹曲线 / 周末电平差），把这四类失配钉死。

##### Support-A discovery 检查（冻结前必做，420 条真实序列）

**只读 Support-A discovery**，不碰 Dev-Query / Support-B / Final。四个设计变体全测：

| 变体 | 干净真实序列上误触发 | 干净序列损伤 | level_shift 通道：修复后 / 腐蚀损伤 | 被改坏的比例 |
|---|---|---|---|---|
| v1 线性趋势 + iid 方差 + 重基到末段 | 82% | 1.78σ | 1.75σ / 0.106σ | 60% |
| + 样条趋势 + 长程方差 | 77.6% | 1.62σ | 1.66σ / 0.100σ | 64.5% |
| + 周周期哑变量 | 77.6% | 1.62σ | 1.70σ / 0.106σ | 64.5% |
| **+ 暂态偏移语义（采用）** | **45.7%** | **0.56σ** | **0.66σ / 0.100σ** | **36.1%** |

**采用暂态偏移语义**：只撤销"上去又回来"的段（可逆伪影），持续到序列末尾的电平变化
（= 真实 regime 变更）一律保留。它在每个轴上都严格更优，且它才是**数据质量**算子该有的语义。
选择在 Support-A discovery（唯一被许可的调参面）上做出。

**但它仍然不合格。** 四个独立设计变体给出同一个结论：**在这个 roster 上，值域检测器无法把
注入的人工断层与真实的电平结构分开**——因为真实序列里**遍地是真实的电平变化**（传感器掉线
后回来、报送口径变更、电表重新标定），它们和注入的伪影是同一种东西。

##### ⚠️ 度量的诚实说明（不要误读上表）

上表用的是**对干净序列的重建 MAE**，这**不是 benchmark 的度量**。benchmark 打的是**下游预测
sMASE**——算子的输出是**训练输入**，不是重建。铁证：`denoise_stl` 的重建误差是 **0.5075σ**
（它按设计就大改序列），却是 v0.2 里**最好的固定 program**（10.98 vs raw 11.48）。
所以"重建误差大"**不蕴含**"下游变差"。

因此上表**只支持**这一个结论（这已经够判死刑了）：
**本算子分不清干净序列与被腐蚀序列**（在 45.7% 的干净序列上照样动手）。
它**不支持**"它一定会伤害下游"——那需要用 benchmark 自己的度量在 Support-A discovery 上跑一次
下游评估。**在那次评估做完之前，pool_v3 不得冻结它。**

- **默认参数**：`t_threshold=3.5`、`min_jump_sigma=1.0`、`min_segment=10`、`max_breaks=5`。
  **披露**：benchmark 的 level_shift 幅度恰好是 2.0σ，闸值 1.0σ 以 2× 余量放行。1.0 是"一个
  完整的稳健 σ"这个圆整的通用地板，**其选择从未参照过任何 benchmark 结果**。
- **已知失效模式**：贴着序列末尾的断层测不到（末段是每一个判据的右邻）→ 恒等返回 → 不修
  但也不伤。持续型 regime 变更**按设计**不修。

#### `hampel_filter`（R3，`operators/s1_outlier.py`）

- **动机**：现有三个 outlier 算子（winsorize/outlier_iqr/outlier_mad）是**同一族**——全局阈值
  裁剪；`programs.py` 的冻结排除清单已把后两个判为"与 winsorize 机制冗余"。Hampel 是**另一族**：
  阈值由滚动窗口内的中值与 MAD 决定，只替换命中点，趋势/季节被滚动中值跟住。
- **默认参数 = 文献默认**（Hampel 经典半宽 k=3 → `window=7`，`n_sigmas=3.0`；亦即 MATLAB
  `hampel(x)` 的默认）。**刻意不做择优扫参**：它是否胜过 winsorize 是 benchmark 要回答的经验
  问题，先把它调到能赢再拿去量，就是把尺子当靶子。
- **参数敏感性（已扫，如实披露）**：存在真实取舍，不是一个可以两头都要的旋钮——

  | 参数 | spike 召回 | 干净季节序列上的附带损伤 |
  |---|---|---|
  | w=7, 3σ（**采用**，文献默认） | 83% | 0.015σ |
  | w=11, 3σ | 94–100% | 0.026σ |

  窗口相对季节曲率太宽时，波峰处局部 MAD 变小而偏差变大 → 系统性误杀季节峰（w=11 时误杀
  11.8%）。这是 Hampel 的已知弱点，窄窗口是对它的直接防御。
- **反噬性质（真实性质，不藏）**：在**有结构**的干净序列上，hampel 的附带损伤**小于 winsorize**
  （季节 0.015σ vs 0.019σ；随机游走 0.008σ vs 0.022σ）；在**纯白噪声**上反过来（0.102σ vs
  0.040σ）——毫无局部结构可言时，"局部中值"本来就是个差预测。测试如实把不等式的成立范围
  限定在有结构的序列上（roster 落在这一侧，但这是**被声明的**条件，不是被假设的）。

#### `impute_ssm` / `impute_ar`（R2，`operators/s1_impute.py`）

- **动机**：v0.1/v0.2 池里的插补**全是复制/插值族**（linear/ffill/seasonal/ema/fft——填进去的
  值是邻近观测的加权组合）。这两个是**模型预测族**：先对生成过程拟合模型，再用模型预测缺失
  位置。机制上可区分，才值得各占一个动作名额。
- **命名红线**：**不得叫 `impute_kalman`**。那个名字在本仓库历史 trace 里已经指过 `impute_ema`
  （见 `registry.ALIASES`）。复用它会让"同一个名字在旧记录里指 EMA、在新记录里指状态空间"，
  任何跨版本的算子身份审计都会被它骗过去。三个旧 alias 全部标记 `deprecated=True`（保留只为
  旧 artifact 可重放）。
- **`impute_ssm` 模型（冻结）**：`UnobservedComponents(level="local level", seasonal=period,
  concentrate_scale=True)`，MLE(lbfgs, maxiter=50) → Kalman **平滑** → **只写回缺失位置**。

  选型实测（400 点合成，噪声 σ=0.3，period=24）：

  | 配置 | 耗时 | 收敛 | 插补 MAE |
  |---|---|---|---|
  | local level + seasonal（**采用**） | 7.5s | ✅ | 0.213 |
  | local linear trend + seasonal | 15.7s | ❌ 50 步不收敛 | 0.198 |
  | local level（无季节） | 0.1s | ✅ | 0.968 = **线性插补，一点没赚** |

  → 价值**全在季节分量**上；local linear trend 多花一倍时间换 7% MAE、还不收敛 → 不取。

- **⚠️ `impute_ssm` 的成本披露（B2 裁决的决定性输入）**：**~7.5 秒/条序列**，比 `impute_linear`
  慢约 4 个数量级。按 v0.2 的执行单元估算，仅 Dev-Query 一侧就是
  `373 序列 × 17 (scenario,dose,replicate) ≈ 6300 次调用 × 7.5s ≈ 13 小时单线程`——而 v0.2
  **整轮**才跑了 3.79 小时（其中 denoise_stl 已占 99%）。**未经优化，它不是一个可行的
  benchmark 池成员**。作为方法侧动作（TTHA-0 的补丁空间）它完全可用。
- **`impute_ssm` 依赖策略 = `hard_fail`**：statsmodels 缺失 → **抛 ImportError**，绝不降级。
  这是 `impute_kalman` 事故的直接对策：静默换一个算子顶上、名字不变 = 台账记着跑了状态空间、
  实际跑的是指数平滑，而 router 从这条假记录里学动作标签。
  退化**输入**（观测点 < 8、平滑器异常）下允许显式记账回退 `impute_linear`——那是诚实的、
  被台账记下的降级；**永远不回退到 impute_ema/smooth_ema**（测试守）。
- **`impute_ar` 的阶数必须够到季节滞后**（`order=0` = 自动 = `max(8, period)`，上限 n//4）。
  这不是调参偏好，是机制的死活：AR(p) 只能看见 p 步以内的历史，p < period 时它**根本看不见
  季节周期**。实测（period=24、30 点 block 缺口）：

  | | linear | AR(8) | AR(16) | AR(24) | AR(48) |
  |---|---|---|---|---|---|
  | 插补 MAE | 2.00 | 1.33 | 0.40 | **0.34** | 0.34 |

  AR(8) 只砍掉线性插补 1/3 的误差，AR(24) 砍掉 83%。写死 order=8 会让这个算子安静地退化成
  一个平庸的短记忆模型——没有异常、没有报错，只是它不再是它自称的那个东西。
  **`impute_ar` 是纯 numpy 且廉价**——与 `impute_ssm` 的成本处境完全不同。

### 新契约字段：`dependency_policy`

旧契约**说不出**"依赖缺失时该怎么办"：`denoise_stl` 与 `impute_ssm` 的 `requires_dependency`
都是 `"statsmodels"`，但正确行为相反——前者记账回退 savgol，后者必须硬失败。两者在旧契约里
无法区分，于是"不许静默降级"只能靠代码注释和人的记性守，而那正是本项目反复踩的"声明≠执行"。
现在它是一个可被测试机械检查的字段：`"hard_fail" | "recorded_fallback" | None`。

### R5：动作菜单 v2（`policy/action_spec.action_menu_v2`）

新算子进 registry **并不等于它可用**。本项目有三个互不相同的动作枚举面：

1. `policy.action_spec.action_menu_*` —— Router/selector 的版本化动作集。**本次扩的是它**：
   menu v2 = v1 全集（逐位不变）+ 4 个新机制动作（`v_ssm` / `v_ar` / `v_hampel` / `v_levelshift`）。
2. `p6.fast_path.GRAMMAR_* / det_ladder` —— **H_ref 的候选文法，硬编码、已冻结、没有碰**。
3. `fast_path.compose._IMPUTE/_DENOISE/_OUTLIER` —— heuristic 合成的降级链（旧路径，未动：
   新算子经 `usable_ops` 已对 LLM/selector 可见，不需要挤进 heuristic 的优先级表）。

TTHA-0 的 prereg 将 pin 的是 `action_menu_v2().sha256`。

---

## ⚠️ 对 benchmark-v0.2 冻结产物的影响（必读）

注册新算子**必然**改动 `operators/registry.py` 的字节，而 `results/Benchmark_v0_2/program_pool.json`
用 SHA 钉了这个文件。**那个 digest 一定对不上了**——这是 pin 机制的已知局限，`programs.py`
自己的注释早就点破过：

> "Over-pinning -- hashing modules the pool never reaches -- would invalidate a frozen pool for
> edits that provably cannot move a single number, and a freeze that cries wolf gets ignored."

**文件 digest 回答不了"数字有没有动"。行为 digest 可以。**
`tests/test_frozen_action_surfaces.py` 因此对 v0.2 池的 8 个 value-domain program 在一条固定
探针序列上的输出做**逐字节 SHA256**（digest 采于改动 registry **之前**），并锁住：

- 8 个 program 的输出逐字节不变 ⇒ **v0.2 的每一个读数依然有效**；
- 池成员没有增长（新算子**没有**偷偷进池）；
- `p6.fast_path` 的 `det_ladder` 与 `GRAMMAR_*` 逐字节不变 ⇒ **H_ref 的动作空间没有被扩大**
  （它是 v0.2 里被度量的现任者；候选池扩了 = 被度量的对象换了人，而所有数字还挂在 "h_ref"
  这个名字上）；
- `minimal_l2().operator_defaults` **一个键都没加**——它同时是 menu v1 的 meta 字段（加键 →
  v1 的 SHA 变）和 P6 `resolve_steps` 的参数来源（加键 → 静默改掉 H_ref 的参数解析）。
  新算子的参数一律显式写死在 menu v2 的 ActionSpec 里；
- `action_menu_v1().sha256` 不变。

**调和记录已落盘**：`results/Benchmark_v0_2/pool_code_pin_reconciliation.json`
（漂移的 3 个文件 = `_common.py` / `registry.py` / `s1_outlier.py`；`programs.py` 与
`s1_denoise.py` 未动。旧 pin + 新 pin + 行为等价证据全在里面）。
`program_pool.json` **本身没有被改动**——它的文件 SHA 被 `benchmark_manifest_v0.yaml` 的
`program_pool_sha256` 钉着，改它会打破一个**真正的**完整性校验。调和面是旁挂的。

这道闸是**机械执行**的：`test_pool_code_pins_are_intact_or_explicitly_reconciled` 在源文件
digest 漂移而调和记录未更新时**直接失败**。下次改动池代码，请**更新调和记录并重新证明行为等价**，
**不要通过放宽测试来通过测试**——一个总是红着又没人管的检查，会教会每个人无视红色。

→ **pool_v3 的 amendment 里请直接引用该调和记录**（成员变更时重新 pin 一次即可）。

---

## pool_v3 的状态：**被阻塞，不是待写**

用户已裁决 pool_v3 = pool_v2 + `repair_level_shift`（唯一由预声明缺口背书的成员）。
**但 B2 的 amendment 不能签发**——Support-A discovery 检查发现该算子在 45.7% 的**干净**真实
序列上照样动手。把它冻进 oracle 池，就是把天花板建在"它到处乱动"之上。

**解锁 B2 的唯一动作**：用 benchmark **自己的度量**（下游 sMASE，闭式 trainer）在
**Support-A discovery** 上评估 `raw` vs `repair_level_shift`，读两个数——

1. 在 `level_shift` 腐蚀下，它是否胜过 `raw`（以及胜过 `denoise_stl`——那是 v0.2 该通道上的
   现任最优，cell-equal 12.2217 vs raw 12.7598）；
2. 在 `natural` 通道（无腐蚀）下，它伤害多大——这是"它乱动"的代价，oracle 会看见它。

⚠️ **不要用 Dev-Query 做这件事**：那是评估面，不是标定面。也**不要**在 amendment 冻结前
让任何新 program 的 Dev 数字存在——那正是 prereg §2 要防的东西。

顺带纠正一个会被写进 amendment 的**事实错误**（任务书沿用了 v0.1 的叙事）：
**level_shift 格在 pool_v2 下并不是死格**。v0.1 时它是（池里四个 program 全是填充类，
损失小数点后四位相同）；v0.2 加进去了去噪族之后，`denoise_stl` 在该通道上有真实增益
（cell-equal 12.2217 vs raw 12.7598，17 个 cell 里 14 个有 >ε 的增益）。
缺口依然是真的，但它的表现形式是**"天花板被一个处理别的机制的算子定住了"**，
不是"没有任何东西能动这个格子"。amendment 必须这么写。

---

## 未做的事（如实记录，不是遗漏）

- **R4（`denoise_lowpass` Butterworth + 参数化算子的 preset 化）未做**。
  - `denoise_lowpass` 在 `programs.py` 自己的排除逻辑下会被判为"与 denoise_savgol 机制冗余"
    （`smooth_ma`/`smooth_ema` 正是以这个理由被排除的）——加一个注定进不了池的算子，收益为负。
  - **preset 化需要 Support-A discovery 上的标定**，那是一次独立的、要跑数据的任务，
    不是本次"只写 operators/ 与 tests/"的范围。R4 本来就是可选、优先级最低。
- **任何冻结 preset 都尚未标定**。本次所有默认参数来自**文献默认 + 合成序列校验**，
  未消费 Support-A/Dev/Final 的任何一条真实数据。若某个算子将来要进 benchmark 池，
  它的冻结 preset **必须在 Support-A discovery（420 条）上另行标定**——不得沿用这里的合成数默认。
