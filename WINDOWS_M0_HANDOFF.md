# Windows M0 接续手册

更新时间：2026-07-15  
接续分支：`codex/windows-m0-handoff`

## 1. 接续目标

Windows 端的第一目标不是 Task G，也不是构造 H0，而是完成 benchmark-v0.2 的 M0
精确复现。只有生成 `M0_PASS` 后，Task G 才能被授权。

当前必须验证的核心问题是：历史 benchmark 与 program-pool 行为 SHA 均产生于 Windows
环境，而 Mac portability shadow 无法逐字节复现行为探针。

历史环境证据来自 `results/Stage2/P6Freeze/freeze_record.json`、
`results/Stage2/C0Run_A1/wallclock_report.json`、`ONBOARDING.md` 和
`results/Benchmark_v0_2/program_pool.json`：

```text
OS                Windows-10-10.0.26200-SP0
Python executable D:/Anaconda_envs/envs/project/python.exe
Python            3.10.19
NumPy             2.2.6
SciPy             1.15.2
statsmodels       0.14.6
PyWavelets        1.8.0
scikit-learn      1.7.2
Torch             2.12.0+cu126
```

当前 Mac 上安装的 `vnext/environment/.venv` 依赖版本基本一致，但平台是 Darwin/arm64，
只能用于 contract、portability 和普通开发测试，不能签发历史 benchmark 的 `M0_PASS`。

## 2. 当前已完成内容

Mac 端已经完成：

- 53/53 raw/legacy frozen assets 校验；
- 1919/1919 clean-base UID 校验；
- shadow raw → registry → clean-base → split 重建；
- registry、clean-base content、split、Support-A subsplit、dataset manifest、corruption
  grid、METR-LA blocks 与冻结 v0.2 一致；
- vNext/benchmark 80 项回归测试通过；
- Final、Support-B、SA-V、Dev-vNext、U 均未访问；
- Task G 仍为未授权状态。

当前开放问题记录在：

```text
results/vnext/m0/ProtocolErratumV1.json
```

行为探针的关键 SHA：

```text
历史 frozen raw/probe SHA  9a3049d734abb175214c6e6d78d78834cc9985bdb38939319bd0c1cd4a5e
Mac reconstructed SHA       78b9ccc195254ad5cbcbd42f00f8043af9d9c9d3f26000980de91a3d8ba72422
```

由于 `raw` 不执行任何算子，这说明两边首先是探针输入字节不同，不能把 8/8 mismatch
直接解释成算子漂移。

## 3. 数据不会随 Git 分支上传

下列目录被 `.gitignore` 排除，Windows 机器必须已有或从 Mac/备份单独复制：

```text
data/benchmark_v0/raw/
data/benchmark_v0/incoming/                 # 手工许可资产，如存在
data/benchmark_v0_2/clean_base/             # 可重建，但复制后仍必须重新验 SHA
```

不要从不明来源补数据，也不要为了让校验通过修改 manifest。允许的恢复顺序仍是：exact
backup byte-copy → pinned official download 命中原 SHA → 手工官方资产命中原 SHA。

## 4. Windows 首次检查

在仓库根目录打开 PowerShell。优先使用仍然存在的旧 Conda 环境：

```powershell
$Repo = (Get-Location).Path
$Parent = Split-Path $Repo
$Py = "D:\Anaconda_envs\envs\project\python.exe"

$env:PYTHONPATH = $Parent
$env:PYTHONHASHSEED = "20260713"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:TZ = "UTC"
$env:LC_ALL = "C"
$env:LANG = "C"

& $Py -c "import sys,platform,numpy,scipy,statsmodels,pywt,sklearn,torch; print(sys.version); print(platform.platform()); print(platform.machine()); print('numpy',numpy.__version__); print('scipy',scipy.__version__); print('statsmodels',statsmodels.__version__); print('pywt',pywt.__version__); print('sklearn',sklearn.__version__); print('torch',torch.__version__)"
& conda list --explicit
& $Py -m pip freeze
```

把 `conda list --explicit`、`pip freeze` 和平台输出保存到
`results/vnext/m0/windows_environment_capture/`。这一步只采环境，不读任何 holdout。

## 5. 第一硬门：恢复行为探针

先单独运行：

```powershell
& $Py -m pytest tests/test_frozen_action_surfaces.py -q
```

预期结果是 15 项全部通过，其中 `raw` 必须命中 `9a3049…`。同时执行以下只读诊断，记录
探针输入 SHA：

```powershell
& $Py -c "from tests.test_frozen_action_surfaces import _probe_series,_digest; print(_digest(_probe_series()))"
```

判定：

- 命中 `9a3049…`：证明找回了原始数值平台，可继续 M0；
- 仍为 `78b9cc…` 或其他值：停止，不更新 expected SHA，不运行 L3–L9；记录完整环境；
- 只有部分算子失败：探针输入已恢复但算子行为漂移，定位首个失败算子并签发新的 erratum。

禁止用 Windows 新观察值覆盖 `_POOL_OUTPUT_SHA256`。行为 pin 是要验证的历史证据，不是待更新
的 golden file。

## 6. 第二硬门：数据与协议重验

行为 pin 通过后，运行：

```powershell
& $Py -m SelfEvolvingHarnessTS.vnext protocol-audit --root $Repo --out "$Repo\results\vnext\protocol"
& $Py -m SelfEvolvingHarnessTS.vnext m0-audit --root $Repo --out "$Repo\results\vnext\m0"
```

注意：当前 `EnvironmentLockV1` 仍描述 Mac portability shadow，因此 Windows Codex 必须先基于
上一步采集到的真实旧环境签发一个有证据支持的 Windows historical lock，再让 `m0-audit`
通过。不要仅把 `platform_system` 字符串改成 Windows；至少要绑定 Python、包版本、平台、
BLAS/线程信息和 Conda explicit export SHA。

数据门的预期值：

```text
raw/legacy assets                 53/53
clean-base records                1919/1919
clean-base content SHA            eddbb9440fa0362e27d5632805985827cc1218b3b10795948dfc082204fce966
registry SHA                      8b031c189c8064a4b9567d4dd61faa1548e7fb289ee53e326023698638045410
split manifest SHA                5726589c99321d4b510066cc8dbb396f1bab33e0b9df920be5daaf032a2f2542
Support-A subsplit SHA             4e98535abc61c2900dc6bdf1574ac7b6b8d16ccd67193eef6d12547a3edf0512
corruption grid SHA               aad0b2439291a4cf5096a782745146a9835fecc11c2ae7821be86b1975d4a06b
```

## 7. L0–L9 复现要求

正式 shadow 必须是新目录，不能覆盖 `results/Benchmark_v0_2`：

```text
results/vnext/m0/shadow/
```

分层证据：

```text
L0 raw bytes SHA
L1 clean-base arrays SHA
L2 split membership SHA
L3 corruption realization SHA / CRN
L4 per-series prepared input SHA
L5 per-dataset corpus/window counts
L6 per-dataset model parameters or prediction SHA
L7 per-series sMASE digest
L8 cell/regime/dataset fold digest
L9 headline aggregate
```

最终必须重新生成而不是读取旧报告作为 observed：

```text
Raw                         11.481494078943097
best-fixed / denoise_stl    10.97841511927346
H_ref                       11.558342（以冻结报告完整精度比较）
transfer retrained ceiling  10.788125165461718
```

同时比较 ex-COVID Raw/STL/ceiling、per-UID loss digest、program provenance、CRN、
per-dataset fit scope 和固定 fold 顺序。canonical 数值绝对误差不得超过 `1e-9`。

## 8. 允许关闭 erratum 的条件

只有以下条件全部成立，才能把 `ProtocolErratumV1` 从 `OPEN_*` 改成 closed：

1. Windows 行为探针输入命中 `9a3049…`；
2. 8 个 program output SHA 全部命中冻结值；
3. Windows historical environment artifact 绑定 Conda export、包版本和平台；
4. L0–L9 全部通过；
5. headline 与逐 UID/provenance 在预注册门限内一致；
6. 生成正式 `M0ReproductionVerdictV1`，verdict=`M0_PASS`。

在此之前：

```text
TASK_G_AUTHORIZED=false
H0=NOT_FROZEN
SA-V/Dev/Support-B/Final/U=DO_NOT_ACCESS
```

## 9. Windows Codex 建议起始指令

可把下面这段直接交给 Windows Codex：

```text
阅读 WINDOWS_M0_HANDOFF.md、TSharness_vNext_Implementation_Plan.md、
results/vnext/m0/ProtocolErratumV1.json 和 results/Benchmark_v0_2/TD_VERDICT_ADDENDUM.md。
先采集旧 D:/Anaconda_envs/envs/project 环境并运行 frozen action surface 行为 pin。
不得更新 golden SHA，不得读取任何新 holdout，不得启动 Task G。
只有探针命中 9a3049… 后，才继续 shadow L0–L9 raw-to-result reproduction。
把所有环境、层级和 verdict artifact 写入 results/vnext/m0；不要修改
results/Benchmark_v0_2。
```
