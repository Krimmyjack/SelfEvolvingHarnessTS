# CAP-0: Epilepsy2 seal audit + capstone subset freeze

> **SUPERSEDED IN PART (CAP-1, 2026-08-27).**  
> 密封判词 `SEAL_INTACT` 与结构 `MATCH`（TRAIN 80 / TEST 11420 / 476 行 / 98968 点）仍有效。  
> **mod-24 TEST 规则与 CAP-0 开封触发已作废。** 开封与子集以 `cap1_capstone_protocol_freeze_v1` 为准：`sorted(random.Random(20260827).sample(range(11420), 476))` + 完整终考协议。勿在开封时使用 `k=24`。

protocol: `cap0_epilepsy2_capstone_freeze_v1`  
written: 2026-08-27  
HEAD at write: `19c6b227abf1a831bdfdb0808ddb42f74224b48e`

0 LLM / 0 fit / 0 download. No methods or runner edits. No `docs/STAGE_REPORT` write.

## 义务自报

- **零数值/标签读取**: 本书记 D3 只做了 zip 成员名、`ZipInfo` 体积、以及对 `*_TRAIN.ts` / `*_TEST.ts` 的原始字节换行计数；唯一识别的文本是段标记 `@data`。未切分 token、未 `float` 解析、未读类别标签。未打开 `EpilepticSeizures.txt`、`.png`、`val.ts` 内容。
- **密封判词**: `SEAL_INTACT`（证据链见 §1）。
- **结构**: `MATCH`（TRAIN 80 / TEST 11420 数据行）。
- **子集算术**: 最小 `k = 24`，`n_sub = 476`，总点 `556 × 178 = 98968 ≤ 100000`。

## 1. 密封完整性审计 → `SEAL_INTACT`

破封定义：任何 runner / 脚本 / 工件读取过 D3 zip 的**数值或标签**。zip 打开 + 成员名列举 = 已授权的密封核验，不构成破封。

### 1.1 ROSTER.md

`data/ucr_conf_downloaded/ROSTER.md` D3 行：

| 项 | 记录 |
|---|---|
| role | D3_reserve |
| dataset | Epilepsy2 |
| URL | `aeon-toolkit/EpilepticSeizures.zip`（`Epilepsy2.zip` 曾 404.php） |
| downloaded_utc | 2026-08-25T13:26:20Z |
| bytes | 16220082 |
| sealed | yes — sol 2026-08-26 remain sealed |
| values loaded | no（zip open + member names only） |

§ D3_reserve 声明：结构备用；不得因 D1 计算不可行而强用；成员名已列出且含 `val.ts`。

磁盘现状与 ROSTER 一致：目录内仅 `EpilepticSeizures.zip`，16220082 字节，本地 mtime 2026-08-25 21:26:20（UTC+8 = 声明 UTC）。无残留 404 stub。`*.zip` 被 gitignore，zip 从未入仓。

### 1.2 全仓搜证（数值/标签 vs 成员名）

| 来源 | 做了什么 | 是否读数值/标签 |
|---|---|---|
| `_scratch/cls_conf_dl_download.py` | `_members_only` = `ZipFile.namelist()`；首次 D3 拿到 44 字节 404.php，`BadZipFile` | 否 |
| `_scratch/cls_conf_dl_d3_retry.json` | 正式拉取 16220082 字节；记录成员名；`content_not_read: true`，`values_not_loaded: true` | 否 |
| `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:2854-2860` | 唯一 Python 命中 `D3_reserve` / `EpilepticSeizures.zip`：`seal_table` 笔记，`values_loaded: False`。`CONF_DL_DATA_DIR` 仅 D1；`d2_d3_values_not_loaded: True` | 否 |
| `_scratch/convert_ucr_ts_to_txt_zip.py` | ROSTER 写明只转 D1 BinaryHeartbeat | 否 |
| `artifacts/functional/e2/t6_cls_conf_dl.json/.md` | Epilepsy2 的 80 / 11420 / 178 来自下载前 `metadata.csv` census（`audit_gate` 写于 zip 之前） | 否 |
| 后续 e2 工件 / STAGE_REPORT / S1V2 设计稿 | 记 sealed / 未来 capstone，未开 zip | 否 |

结论：仓库内不存在读取 D3 序列值或标签的 loader、oracle、fit 或转换。

### 1.3 下载至今、涉及该路径的 git 提交

跟踪树里该目录只有 `ROSTER.md`。

```
10f9fee503c4596aba8069b827a4cd37310a28f0  2026-08-26 10:07:25 +0800
  record CLS-CONF-dl D1 termination as compute-budget exceeded
```

`git log -S D3_reserve` 与 `-S EpilepticSeizures` 同为这一条。之后文档提到 Epilepsy2 是密封/改道叙述，不是对该 zip 的写入或打开。

### 1.4 判词

**`SEAL_INTACT`**

证据链：ROSTER 密封声明 + 磁盘字节/时间戳吻合 + zip 未入 git + 唯一路径提交只记 roster + 历史打开仅 namelist + 无 runner 读值 + 本书仅行数计数。

## 2. 结构级信息（未读标签/数值）

| 成员 | 操作 | 结果 |
|---|---|---|
| 6 个成员名 | `namelist()` | 与 ROSTER 逐字一致 |
| `EpilepticSeizures_TRAIN.ts` | 字节换行 + `@data` 后行数 | 87 行文件 / **80** 条数据行 |
| `EpilepticSeizures_TEST.ts` | 同上 | 11427 行文件 / **11420** 条数据行 |
| `val.ts` | 只记名与 `file_size=68077` | 内容未打开；ROSTER 已列，不算缺失 |

TRAIN / TEST 成员存在，行数与官方元数据一致 → **`MATCH`**，不触发 `STRUCTURAL_MISMATCH`。

## 3. 确定性子集规则（预注册，行号算术）

记号：TEST 数据行下标 `i = 0 .. N-1`，`N = 11420`，长度 `L = 178`。

- TRAIN：80 行全用。
- TEST：保留 `i ≡ 0 (mod k)`。
- `n_sub(k) = floor((N-1)/k) + 1`。
- 约束：`400 ≤ n_sub ≤ 480` 且 `(80 + n_sub) × 178 ≤ 100000`。取满足两式的**最小** `k`。

算术：

- `100000 / 178 = 561.797…` → `floor = 561` → `n_sub ≤ 481`。
- 带宽上界 480 更严，故有效 `n_sub ≤ 480`，此时总点最多 `560 × 178 = 99680`。
- `n_sub(23) = floor(11419/23)+1 = 497`（带宽外；`577 × 178 = 102706 > 100000`）。
- `n_sub(24) = floor(11419/24)+1 = 476` ∈ [400, 480]；`556 × 178 = 98968 ≤ 100000`。
- `k = 25..28` 也在带宽内，但不是最小 `k`。`k = 29` 起 `n_sub = 394 < 400`。

**原冻结：`k = 24`，TEST 子集 476 行，TRAIN+TEST 556 条 × 178 = 98968 点。**  
**CAP-1：该行号规则 SUPERSEDED；476 / 98968 保留，行号改为种子随机清单。**

## 4. 冻结声明

1. Capstone Target = 密封 D3 `Epilepsy2` / `EpilepticSeizures.zip`。
2. 子集规则：**SUPERSEDED。** 勿用 `i ≡ 0 (mod 24)`。见 CAP-1 种子清单。总点 98968 仍成立。
3. **Static / A3 / A5 三臂必须使用 CAP-1 的同一子集。**
4. **开封触发 SUPERSEDED。** 改为 S1-v2 正序 ×2 出 `S1V2_FORWARD_SIGNAL` 且反序 ×1 确认后，按 CAP-1 协议自动开封，无需再次授权。
5. 开封前禁止对 D3 做 oracle / fit / 标签读取 / 数值解析。
6. 若日后 loader 发现结构不再匹配，记 `STRUCTURAL_MISMATCH` 并停用本冻结，再回到新下载路线。
