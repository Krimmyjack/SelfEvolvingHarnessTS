"""p6/split_manifest.py — P6 sequential split manifest（两级冻结流程的机械层；台账 v2）。

P6_Plan.md §5 拟定算术的机械实现：

1. legacy 分块：4 大域（nn5_daily/fred_md/tourism_monthly/covid_deaths）各 20 条，
   每域内按 sha256(f"p6|{config}|{series_uid}") 十六进制升序排序，
   前 4 → C0、次 8 → D1、再 8 → D2（合计 C0=16、D1=32、D2=32，恰好用尽、两两不交）。
2. 单条域隔离：us_births/saugeenday/sunspot 各 1 条 → 块 C0_qualitative，永不入 D/V/U。
3. virgin 规格（V1/V2/U）：**不物化，只冻结规则**（候选全集=下载时该 config 全部 item_id、
   排除集、hash 前缀、配额；下载时须记录 content_sha）。
   - V1/V2 排除集都含全部 legacy item_id；V2 额外显式声明排除 V1 已选（"V1_selected"）。
   - U：config=traffic_hourly，候选=全部 item_id 减去探针分析过的清单，
     按 sha256(f"p6u|{item_id}") 升序取前 n。
4. 放置规则强制（违反即 raise P6ManifestError）：
   - 所有 legacy 行 exposure_class ∈ {confirmed_exposed, uncertain_legacy_exposure}；
   - 任何 legacy uid 不得出现在 V/U 规格候选中（经排除集等式校验）；
   - 单条域不得出现在 D/V/U；C0/D1/D2 两两不交且并集恰为 4 大域全部 uid。
5. 状态机 SequentialGate v2（崩溃安全一次性台账；prereg_p6 §4 "cycle terminal 事件"+"台账"）：
   - 事件 = precommit / open / verdict / cycle_terminal，全部进 canonical ledger
     （路径由 manifest 派生：ledger_path(manifest, root)；传入其他路径 → raise）；
   - write-ahead：任何状态变更先 durable 落盘（单次 write + fsync）再改内存；
   - 独占文件锁 <ledger>.lock：实例存活期持有，第二实例/进程 → raise；进程死亡由 OS 释放；
   - 事件 hash 链：prev_event_sha + event_sha（canonical JSON），重放全校验，
     断链/篡改/损坏 → raise（日志损坏 = technical abort，不自动修复）；
   - precommit(cycle) payload 必含 {candidate_edit_sha, harness_state_sha, config_digest,
     materialization_sha, code_sha}，一次性（禁止重新选择候选）；cycle2 需 cycle1 terminal；
   - 开箱条件：V1 ← cycle1 precommit；V2 ← cycle1 terminal + cycle2 precommit；
     U ← cycle2 terminal（不要求 V2 verdict——abstain 也是 terminal，修复 abstain→U 死锁）；
   - open/verdict 各一次且 verdict 须先 open；开箱块 verdict ∈ {promote,reject}；
     abstain 只作为 cycle terminal 存在（V 保持 sealed、terminal 照记）；
   - record_cycle_terminal(cycle, verdict ∈ {promote,reject,abstain})：每 cycle 一次、不可逆；
     V sealed → 只能 abstain；V verdict 已记 → 必须一致；V open（pending）→ 禁止 terminal；
   - 崩溃恢复：重放重建状态；open-intent 无后续 verdict = pending（pending_open(block) 查询），
     只允许按同一 precommit resume（同候选/同字节/同 seeds 属 runner 纪律）。
6. manifest_sha：canonical JSON（sort_keys、ASCII、紧凑分隔符、剔除易变字段如时间戳）
   的 sha256。write_manifest/load_manifest 往返 sha 不变，内嵌 sha 校验防篡改。

确定性红线：本模块无任何 RNG——一切顺序来自 sha256；同输入（任意行序）必得同 manifest_sha。
本模块自身不向 results/ 落任何文件；真实 manifest 的落盘由后续冻结步骤调用 write_manifest。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if os.name == "nt":
    import msvcrt
else:
    import fcntl

__all__ = [
    "ALLOWED_LEGACY_EXPOSURE",
    "BIG_DOMAINS",
    "BLOCK_VERDICTS",
    "CYCLES",
    "DEFAULT_QUOTA",
    "LEDGER_SCHEMA",
    "LEGACY_BLOCKS",
    "OPENABLE_BLOCKS",
    "P6ManifestError",
    "P6SplitManifest",
    "P6StateError",
    "PRECOMMIT_REQUIRED_KEYS",
    "QUALITATIVE_BLOCK",
    "SCHEMA_VERSION",
    "SINGLETON_DOMAINS",
    "STATE_OPEN",
    "STATE_SEALED",
    "STATE_VERDICT",
    "SequentialGate",
    "TERMINAL_VERDICTS",
    "U_CONFIG_DEFAULT",
    "V1_SELECTED_MARKER",
    "build_manifest",
    "compute_event_sha",
    "compute_manifest_sha",
    "ledger_path",
    "legacy_sort_key",
    "load_manifest",
    "read_jsonl",
    "select_u_items",
    "select_virgin_items",
    "u_sort_key",
    "validate_manifest",
    "virgin_sort_key",
    "write_manifest",
]

SCHEMA_VERSION = "p6-split-manifest/1"

BIG_DOMAINS: Tuple[str, ...] = ("covid_deaths", "fred_md", "nn5_daily", "tourism_monthly")
SINGLETON_DOMAINS: Tuple[str, ...] = ("saugeenday", "sunspot", "us_births")
ALLOWED_LEGACY_EXPOSURE = frozenset({"confirmed_exposed", "uncertain_legacy_exposure"})

LEGACY_BLOCKS: Tuple[str, ...] = ("C0", "D1", "D2")
QUALITATIVE_BLOCK = "C0_qualitative"
OPENABLE_BLOCKS: Tuple[str, ...] = ("V1", "V2", "U")
DEFAULT_QUOTA: Mapping[str, int] = {"C0": 4, "D1": 8, "D2": 8}

U_CONFIG_DEFAULT = "traffic_hourly"
V_PER_CONFIG_DEFAULT = 5
U_N_DEFAULT = 20
V1_SELECTED_MARKER = "V1_selected"

#: 顶层易变字段：不参与 manifest_sha，load 时剔除。
VOLATILE_KEYS = frozenset({"manifest_sha", "created_at", "written_at", "timestamp", "ts"})

STATE_SEALED = "sealed"
STATE_OPEN = "open"
STATE_VERDICT = "verdict_recorded"

_REQUIRED_ROW_FIELDS = ("config", "item_id", "series_uid", "exposure_class")


class P6ManifestError(ValueError):
    """Manifest 构造/校验违规（放置规则、schema、篡改检测）。"""


class P6StateError(RuntimeError):
    """状态机违规（重复 open、越序 verdict、日志与 manifest 不匹配等）。"""


# --------------------------------------------------------------------------
# hash 排序键：一切确定性来自 sha256（无 RNG）
# --------------------------------------------------------------------------

def _sha_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def legacy_sort_key(config: str, series_uid: str) -> str:
    """legacy 分块排序键：sha256(f"p6|{config}|{series_uid}") 十六进制。"""
    return _sha_hex(f"p6|{config}|{series_uid}")


def virgin_sort_key(hash_prefix: str, config: str, item_id: str) -> str:
    """V1/V2 选取排序键：sha256(f"{prefix}|{config}|{item_id}") 十六进制。"""
    return _sha_hex(f"{hash_prefix}|{config}|{item_id}")


def u_sort_key(item_id: str) -> str:
    """U 选取排序键：sha256(f"p6u|{item_id}") 十六进制。"""
    return _sha_hex(f"p6u|{item_id}")


# --------------------------------------------------------------------------
# Manifest 数据结构 + canonical sha
# --------------------------------------------------------------------------

def compute_manifest_sha(payload: Mapping[str, Any]) -> str:
    """canonical JSON（sort_keys、ASCII、紧凑分隔符）剔除顶层易变字段后取 sha256。"""
    clean = {k: payload[k] for k in payload if k not in VOLATILE_KEYS}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class P6SplitManifest:
    """不可变 manifest 载体：payload 为 JSON 原生 dict（无易变字段）。"""

    payload: Dict[str, Any]

    @property
    def manifest_sha(self) -> str:
        return compute_manifest_sha(self.payload)

    @property
    def blocks(self) -> Dict[str, List[str]]:
        return {name: list(uids) for name, uids in self.payload["blocks"].items()}

    def block(self, name: str) -> List[str]:
        try:
            return list(self.payload["blocks"][name])
        except KeyError:
            raise P6ManifestError(f"manifest 无块 {name!r}") from None

    @property
    def virgin_specs(self) -> Dict[str, Any]:
        return copy.deepcopy(self.payload["virgin_specs"])

    @property
    def legacy_rows(self) -> Dict[str, Dict[str, str]]:
        return copy.deepcopy(self.payload["legacy_rows"])

    @property
    def big_domains(self) -> Tuple[str, ...]:
        return tuple(self.payload["big_domains"])

    @property
    def singleton_domains(self) -> Tuple[str, ...]:
        return tuple(self.payload["singleton_domains"])


# --------------------------------------------------------------------------
# 构造
# --------------------------------------------------------------------------

def _normalize_rows(ledger_rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen: set = set()
    for idx, raw in enumerate(ledger_rows):
        if not isinstance(raw, Mapping):
            raise P6ManifestError(f"ledger 第 {idx} 行不是 mapping: {type(raw).__name__}")
        missing = [f for f in _REQUIRED_ROW_FIELDS if raw.get(f) in (None, "")]
        if missing:
            raise P6ManifestError(f"ledger 第 {idx} 行缺字段 {missing}")
        row = {f: str(raw[f]) for f in _REQUIRED_ROW_FIELDS}
        uid = row["series_uid"]
        if uid in seen:
            raise P6ManifestError(f"series_uid 重复: {uid!r}")
        seen.add(uid)
        rows.append(row)
    return rows


def build_manifest(
    ledger_rows: Iterable[Mapping[str, Any]],
    u_excluded_item_ids: Iterable[str],
    *,
    big_domains: Sequence[str] = BIG_DOMAINS,
    singleton_domains: Sequence[str] = SINGLETON_DOMAINS,
    quota: Optional[Mapping[str, int]] = None,
    v_per_config: int = V_PER_CONFIG_DEFAULT,
    u_config: str = U_CONFIG_DEFAULT,
    u_n: int = U_N_DEFAULT,
) -> P6SplitManifest:
    """从曝光账本行 + U 探针排除清单构造 sequential split manifest。

    违反任何放置规则即 raise P6ManifestError；返回前会跑完整 validate_manifest。
    """
    big = tuple(sorted({str(c) for c in big_domains}))
    singles = tuple(sorted({str(c) for c in singleton_domains}))
    if not big:
        raise P6ManifestError("big_domains 不能为空")
    overlap = set(big) & set(singles)
    if overlap:
        raise P6ManifestError(f"big_domains 与 singleton_domains 重叠: {sorted(overlap)}")

    quota_map = dict(DEFAULT_QUOTA if quota is None else quota)
    if sorted(quota_map) != sorted(LEGACY_BLOCKS):
        raise P6ManifestError(f"quota 键必须恰为 {LEGACY_BLOCKS}，got {sorted(quota_map)}")
    quota_map = {b: int(quota_map[b]) for b in LEGACY_BLOCKS}
    if any(v <= 0 for v in quota_map.values()):
        raise P6ManifestError(f"quota 必须为正: {quota_map}")
    per_domain_total = sum(quota_map.values())

    if int(v_per_config) <= 0:
        raise P6ManifestError("v_per_config 必须为正")
    if int(u_n) <= 0:
        raise P6ManifestError("u_n 必须为正")
    u_config = str(u_config)
    if u_config in big or u_config in singles:
        raise P6ManifestError(
            f"U 域 {u_config!r} 与 legacy/单条域冲突：legacy/单条域不得出现在 U 规格"
        )

    rows = _normalize_rows(ledger_rows)
    by_config: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        cfg = row["config"]
        if cfg not in big and cfg not in singles:
            raise P6ManifestError(f"未知 config {cfg!r}（uid={row['series_uid']!r}）不在大域/单条域清单")
        if row["exposure_class"] not in ALLOWED_LEGACY_EXPOSURE:
            raise P6ManifestError(
                f"legacy 行 {row['series_uid']!r} exposure_class={row['exposure_class']!r} "
                f"∉ {sorted(ALLOWED_LEGACY_EXPOSURE)}（virgin 类不得进 legacy 账本）"
            )
        by_config.setdefault(cfg, []).append(row)

    # legacy 分块：每域内 sha256 升序 → C0/D1/D2 恰好用尽
    blocks: Dict[str, List[str]] = {b: [] for b in LEGACY_BLOCKS}
    for cfg in big:
        got = by_config.get(cfg, [])
        if len(got) != per_domain_total:
            raise P6ManifestError(
                f"大域 {cfg!r} legacy 行数 {len(got)} != C0+D1+D2 配额 {per_domain_total}"
            )
        ordered = sorted(got, key=lambda r: (legacy_sort_key(cfg, r["series_uid"]), r["series_uid"]))
        uids = [r["series_uid"] for r in ordered]
        c0_n, d1_n = quota_map["C0"], quota_map["D1"]
        blocks["C0"].extend(uids[:c0_n])
        blocks["D1"].extend(uids[c0_n : c0_n + d1_n])
        blocks["D2"].extend(uids[c0_n + d1_n :])

    # 单条域隔离
    qualitative: List[str] = []
    for cfg in singles:
        got = by_config.get(cfg, [])
        if len(got) != 1:
            raise P6ManifestError(f"单条域 {cfg!r} 行数 {len(got)} != 1")
        qualitative.append(got[0]["series_uid"])

    legacy_rows_payload = {
        r["series_uid"]: {
            "config": r["config"],
            "item_id": r["item_id"],
            "exposure_class": r["exposure_class"],
        }
        for r in rows
    }
    legacy_items_by_cfg = {
        cfg: sorted(r["item_id"] for r in by_config[cfg]) for cfg in big
    }

    def _v_spec(name: str, prefix: str, exclude_v1: bool) -> Dict[str, Any]:
        return {
            "block": name,
            "hash_prefix": prefix,
            "configs": list(big),
            "per_config_n": int(v_per_config),
            "candidate_universe": "download-time full item_id set of the config",
            "exclusions": {cfg: list(legacy_items_by_cfg[cfg]) for cfg in big},
            "additional_exclusions": [V1_SELECTED_MARKER] if exclude_v1 else [],
            "rule": (
                "candidates = universe(config) - exclusions[config]"
                + (f" - {V1_SELECTED_MARKER}" if exclude_v1 else "")
                + f"; sort by sha256('{prefix}|{{config}}|{{item_id}}') hex asc;"
                + f" take first {int(v_per_config)} per config"
            ),
            "materialized": False,
            "requires_content_sha_at_download": True,
        }

    u_exclusions = sorted({str(x) for x in u_excluded_item_ids})
    virgin_specs: Dict[str, Any] = {
        "V1": _v_spec("V1", "p6v1", exclude_v1=False),
        "V2": _v_spec("V2", "p6v2", exclude_v1=True),
        "U": {
            "block": "U",
            "hash_prefix": "p6u",
            "config": u_config,
            "n": int(u_n),
            "candidate_universe": f"download-time full item_id set of {u_config}",
            "exclusions": u_exclusions,
            "additional_exclusions": [],
            "rule": (
                f"candidates = universe({u_config}) - exclusions;"
                " sort by sha256('p6u|{item_id}') hex asc;"
                f" take first {int(u_n)}"
            ),
            "materialized": False,
            "requires_content_sha_at_download": True,
        },
    }

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "big_domains": list(big),
        "singleton_domains": list(singles),
        "quota": quota_map,
        "blocks": {
            "C0": blocks["C0"],
            "D1": blocks["D1"],
            "D2": blocks["D2"],
            QUALITATIVE_BLOCK: qualitative,
        },
        "legacy_rows": legacy_rows_payload,
        "virgin_specs": virgin_specs,
        "counts": {
            "C0": len(blocks["C0"]),
            "D1": len(blocks["D1"]),
            "D2": len(blocks["D2"]),
            QUALITATIVE_BLOCK: len(qualitative),
            "legacy_total": len(rows),
        },
    }
    manifest = P6SplitManifest(payload)
    validate_manifest(manifest)
    return manifest


# --------------------------------------------------------------------------
# 校验（build 出口与 load 入口共用；对 payload 全量重建比对，篡改即 raise）
# --------------------------------------------------------------------------

def validate_manifest(manifest: P6SplitManifest) -> None:
    p = manifest.payload
    if p.get("schema_version") != SCHEMA_VERSION:
        raise P6ManifestError(f"schema_version 不符: {p.get('schema_version')!r}")
    for key in ("big_domains", "singleton_domains", "quota", "blocks",
                "legacy_rows", "virgin_specs", "counts"):
        if key not in p:
            raise P6ManifestError(f"manifest 缺字段 {key!r}")

    big = [str(c) for c in p["big_domains"]]
    singles = [str(c) for c in p["singleton_domains"]]
    if big != sorted(set(big)) or singles != sorted(set(singles)):
        raise P6ManifestError("big_domains/singleton_domains 必须排序去重")
    if set(big) & set(singles):
        raise P6ManifestError("big_domains 与 singleton_domains 重叠")

    quota = p["quota"]
    if sorted(quota) != sorted(LEGACY_BLOCKS):
        raise P6ManifestError(f"quota 键必须恰为 {LEGACY_BLOCKS}")
    q_c0, q_d1, q_d2 = (int(quota[b]) for b in LEGACY_BLOCKS)
    if min(q_c0, q_d1, q_d2) <= 0:
        raise P6ManifestError("quota 必须为正")
    per_domain_total = q_c0 + q_d1 + q_d2

    # legacy 行：exposure_class 白名单 + config 归属
    legacy_rows: Mapping[str, Mapping[str, str]] = p["legacy_rows"]
    by_cfg: Dict[str, List[Tuple[str, str]]] = {}
    for uid, info in legacy_rows.items():
        cfg = str(info["config"])
        exp = str(info["exposure_class"])
        if exp not in ALLOWED_LEGACY_EXPOSURE:
            raise P6ManifestError(
                f"legacy {uid!r} exposure_class={exp!r} ∉ {sorted(ALLOWED_LEGACY_EXPOSURE)}"
            )
        if cfg not in big and cfg not in singles:
            raise P6ManifestError(f"legacy {uid!r} config={cfg!r} 不在大域/单条域清单")
        by_cfg.setdefault(cfg, []).append((str(uid), str(info["item_id"])))

    # 重建 legacy 分块（顺序敏感比对：任何放置篡改必被抓）
    expected: Dict[str, List[str]] = {b: [] for b in LEGACY_BLOCKS}
    for cfg in big:
        got = by_cfg.get(cfg, [])
        if len(got) != per_domain_total:
            raise P6ManifestError(f"大域 {cfg!r} legacy 行数 {len(got)} != {per_domain_total}")
        uids = [u for u, _ in sorted(got, key=lambda t: (legacy_sort_key(cfg, t[0]), t[0]))]
        expected["C0"].extend(uids[:q_c0])
        expected["D1"].extend(uids[q_c0 : q_c0 + q_d1])
        expected["D2"].extend(uids[q_c0 + q_d1 :])
    blocks = p["blocks"]
    for name in LEGACY_BLOCKS:
        if [str(u) for u in blocks.get(name, [])] != expected[name]:
            raise P6ManifestError(
                f"块 {name} 与 sha256 排序重建不一致（放置违规或篡改）"
            )

    expected_qual: List[str] = []
    for cfg in singles:
        got = by_cfg.get(cfg, [])
        if len(got) != 1:
            raise P6ManifestError(f"单条域 {cfg!r} 行数 {len(got)} != 1")
        expected_qual.append(got[0][0])
    if [str(u) for u in blocks.get(QUALITATIVE_BLOCK, [])] != expected_qual:
        raise P6ManifestError(
            f"{QUALITATIVE_BLOCK} 必须恰为单条域全部 uid（单条域永不入 D/V/U）"
        )

    # 显式不交/并集断言（双保险，错误信息更直白）
    s_c0, s_d1, s_d2 = set(expected["C0"]), set(expected["D1"]), set(expected["D2"])
    if (s_c0 & s_d1) or (s_c0 & s_d2) or (s_d1 & s_d2):
        raise P6ManifestError("C0/D1/D2 必须两两不交")
    big_uids = {u for u, info in legacy_rows.items() if str(info["config"]) in big}
    if (s_c0 | s_d1 | s_d2) != big_uids:
        raise P6ManifestError("C0∪D1∪D2 必须恰为 4 大域全部 legacy uid（恰好用尽）")
    if set(expected_qual) & (s_c0 | s_d1 | s_d2):
        raise P6ManifestError("单条域 uid 不得出现在 C0/D1/D2")

    # virgin 规格：排除集必须恰含全部 legacy item_id（legacy 不得进 V/U）
    specs = p["virgin_specs"]
    for name in OPENABLE_BLOCKS:
        if name not in specs:
            raise P6ManifestError(f"virgin_specs 缺 {name}")
    for name, prefix in (("V1", "p6v1"), ("V2", "p6v2")):
        spec = specs[name]
        if spec.get("hash_prefix") != prefix:
            raise P6ManifestError(f"{name} hash_prefix 必须为 {prefix!r}")
        if [str(c) for c in spec.get("configs", [])] != big:
            raise P6ManifestError(f"{name} configs 必须恰为大域清单 {big}")
        if int(spec.get("per_config_n", 0)) <= 0:
            raise P6ManifestError(f"{name} per_config_n 必须为正")
        excl = spec.get("exclusions", {})
        if sorted(excl) != big:
            raise P6ManifestError(f"{name} exclusions 键必须恰为大域清单")
        for cfg in big:
            legacy_items = sorted(item for _, item in by_cfg[cfg])
            if [str(x) for x in excl[cfg]] != legacy_items:
                raise P6ManifestError(
                    f"{name}/{cfg} 排除集 != 该域全部 legacy item_id"
                    "（legacy uid 不得出现在 V 规格候选中）"
                )
        if spec.get("materialized") is not False:
            raise P6ManifestError(f"{name} 规格不得物化（materialized 必须为 False）")
        if spec.get("requires_content_sha_at_download") is not True:
            raise P6ManifestError(f"{name} 必须要求下载时记录 content_sha")
        extra = [str(x) for x in spec.get("additional_exclusions", [])]
        if name == "V1" and extra:
            raise P6ManifestError("V1 不得声明额外排除")
        if name == "V2" and extra != [V1_SELECTED_MARKER]:
            raise P6ManifestError(
                f"V2 必须显式声明额外排除 {V1_SELECTED_MARKER!r}（V2 候选须排除 V1 已选）"
            )

    u_spec = specs["U"]
    if u_spec.get("hash_prefix") != "p6u":
        raise P6ManifestError("U hash_prefix 必须为 'p6u'")
    u_cfg = str(u_spec.get("config", ""))
    if not u_cfg or u_cfg in big or u_cfg in singles:
        raise P6ManifestError(
            f"U 域 {u_cfg!r} 非法：不得为空、不得为 legacy 大域或单条域"
        )
    if int(u_spec.get("n", 0)) <= 0:
        raise P6ManifestError("U n 必须为正")
    u_excl = [str(x) for x in u_spec.get("exclusions", [])]
    if u_excl != sorted(set(u_excl)):
        raise P6ManifestError("U exclusions 必须排序去重")
    if u_spec.get("materialized") is not False:
        raise P6ManifestError("U 规格不得物化（materialized 必须为 False）")

    counts = p["counts"]
    expected_counts = {
        "C0": len(expected["C0"]),
        "D1": len(expected["D1"]),
        "D2": len(expected["D2"]),
        QUALITATIVE_BLOCK: len(expected_qual),
        "legacy_total": len(legacy_rows),
    }
    got_counts = {k: int(counts.get(k, -1)) for k in expected_counts}
    if got_counts != expected_counts:
        raise P6ManifestError(f"counts 不一致: got {got_counts}, want {expected_counts}")


# --------------------------------------------------------------------------
# virgin 规则的确定性应用（供后续物化/单测；不产生任何落盘副作用）
# --------------------------------------------------------------------------

def select_virgin_items(
    manifest: P6SplitManifest,
    block: str,
    config: str,
    candidate_item_ids: Iterable[str],
    v1_selected: Optional[Iterable[str]] = None,
) -> List[str]:
    """对候选全集应用 V1/V2 冻结规则，返回选中 item_id（hash 升序前 n）。

    V2 的规则显式包含 "排除 V1 已选"：必须提供 v1_selected；V1 则禁止提供。
    候选不足 n → raise（宁缺毋滥，不降配额）。
    """
    if block not in ("V1", "V2"):
        raise P6ManifestError(f"select_virgin_items 只接受 V1/V2（U 用 select_u_items），got {block!r}")
    spec = manifest.payload["virgin_specs"][block]
    config = str(config)
    if config not in [str(c) for c in spec["configs"]]:
        raise P6ManifestError(f"config {config!r} 不在 {block} 规格 configs 中")
    exclusions = {str(x) for x in spec["exclusions"][config]}
    if block == "V2":
        if v1_selected is None:
            raise P6ManifestError("V2 规则显式排除 V1 已选：必须提供 v1_selected")
        exclusions |= {str(x) for x in v1_selected}
    elif v1_selected is not None:
        raise P6ManifestError("V1 规则不含 V1_selected 排除：不得传 v1_selected")
    candidates = sorted({str(x) for x in candidate_item_ids} - exclusions)
    n = int(spec["per_config_n"])
    if len(candidates) < n:
        raise P6ManifestError(
            f"{block}/{config} 候选不足：需 {n}，排除后仅剩 {len(candidates)}"
        )
    prefix = str(spec["hash_prefix"])
    candidates.sort(key=lambda item: (virgin_sort_key(prefix, config, item), item))
    return candidates[:n]


def select_u_items(
    manifest: P6SplitManifest,
    candidate_item_ids: Iterable[str],
) -> List[str]:
    """对 U 域候选全集应用冻结规则：减探针排除清单，sha256 升序取前 n。"""
    spec = manifest.payload["virgin_specs"]["U"]
    exclusions = {str(x) for x in spec["exclusions"]}
    candidates = sorted({str(x) for x in candidate_item_ids} - exclusions)
    n = int(spec["n"])
    if len(candidates) < n:
        raise P6ManifestError(f"U 候选不足：需 {n}，排除后仅剩 {len(candidates)}")
    candidates.sort(key=lambda item: (u_sort_key(item), item))
    return candidates[:n]


# --------------------------------------------------------------------------
# 落盘 / 读回（往返 sha 不变；内嵌 sha + 全量重建双重防篡改）
# --------------------------------------------------------------------------

def write_manifest(manifest: P6SplitManifest, path) -> str:
    """写 manifest JSON（含内嵌 manifest_sha），返回 sha。落盘路径由调用方负责。"""
    validate_manifest(manifest)
    sha = manifest.manifest_sha
    doc = dict(manifest.payload)
    doc["manifest_sha"] = sha
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha


def load_manifest(path) -> P6SplitManifest:
    """读回 manifest；剔除易变字段、校验内嵌 sha 与全部放置规则。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise P6ManifestError("manifest 文件顶层必须是 JSON object")
    embedded = doc.get("manifest_sha")
    payload = {k: v for k, v in doc.items() if k not in VOLATILE_KEYS}
    manifest = P6SplitManifest(payload)
    if embedded is not None and str(embedded) != manifest.manifest_sha:
        raise P6ManifestError(
            f"manifest_sha 校验失败：内嵌 {embedded} != 重算 {manifest.manifest_sha}"
        )
    validate_manifest(manifest)
    return manifest


def read_jsonl(path) -> List[Dict[str, Any]]:
    """读 jsonl（utf-8），跳过空行。供账本/元数据解析复用。"""
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# 状态机 v2：precommit → open → verdict → cycle terminal（崩溃安全一次性台账）
# --------------------------------------------------------------------------

LEDGER_SCHEMA = "p6-gate-ledger/2"

#: cycle terminal 判决白名单（prereg_p6 §4 步骤 6；其余一律 raise）。
TERMINAL_VERDICTS: Tuple[str, ...] = ("abstain", "promote", "reject")
#: 开箱块（V1/V2/U）verdict 白名单：abstain 不是开箱块判决——
#: abstain = V 保持 sealed + cycle terminal 记 "abstain"。
BLOCK_VERDICTS: Tuple[str, ...] = ("promote", "reject")
#: precommit payload 必备键（prereg_p6 §4 步骤 4；缺任一/空白值 → raise）。
PRECOMMIT_REQUIRED_KEYS: Tuple[str, ...] = (
    "candidate_edit_sha",
    "code_sha",
    "config_digest",
    "harness_state_sha",
    "materialization_sha",
)
CYCLES: Tuple[int, ...] = (1, 2)
_V_BLOCK_OF_CYCLE: Mapping[int, str] = {1: "V1", 2: "V2"}
_CYCLE_OF_V_BLOCK: Mapping[str, int] = {"V1": 1, "V2": 2}
_EVENT_HEADER_FIELDS: Tuple[str, ...] = (
    "schema", "seq", "ts", "manifest_sha", "event", "prev_event_sha", "event_sha",
)


def ledger_path(manifest, root) -> Path:
    """canonical ledger 路径：root / f"p6_gate_ledger_{manifest_sha}.jsonl"。

    文件身份由 manifest 派生——同一 manifest 在同一目录下只有一本台账，
    消灭"任意 log_path 第二本台账"。接受 P6SplitManifest 或 64 位十六进制 sha。
    """
    sha = manifest.manifest_sha if isinstance(manifest, P6SplitManifest) else str(manifest)
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise P6StateError(f"ledger_path 需要 P6SplitManifest 或 64 位 hex sha，got {sha!r}")
    return Path(root) / f"p6_gate_ledger_{sha}.jsonl"


def compute_event_sha(event: Mapping[str, Any]) -> str:
    """事件自身 sha：剔除 event_sha 字段后的 canonical JSON（sort_keys/ASCII/紧凑）的 sha256。"""
    clean = {k: event[k] for k in event if k != "event_sha"}
    try:
        canonical = json.dumps(clean, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise P6StateError(f"事件不可 canonical JSON 序列化: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _genesis_sha(manifest_sha: str) -> str:
    """hash 链创世 prev：绑定 manifest 身份，防止跨 manifest 移植台账字节。"""
    return _sha_hex(f"p6-gate-genesis|{manifest_sha}")


def _check_cycle(cycle: Any) -> int:
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle not in CYCLES:
        raise P6StateError(f"cycle 必须 ∈ {list(CYCLES)}，got {cycle!r}")
    return int(cycle)


def _check_terminal_verdict(verdict: Any) -> str:
    if not isinstance(verdict, str) or verdict not in TERMINAL_VERDICTS:
        raise P6StateError(f"verdict {verdict!r} ∉ 白名单 {list(TERMINAL_VERDICTS)}")
    return verdict


def _check_block_verdict(verdict: Any) -> str:
    v = _check_terminal_verdict(verdict)
    if v not in BLOCK_VERDICTS:
        raise P6StateError(
            "'abstain' 不是开箱块 verdict：abstain 路径 = V 保持 sealed + "
            "record_cycle_terminal(cycle, 'abstain')"
        )
    return v


def _check_precommit_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise P6StateError(f"precommit payload 必须是 mapping，got {type(payload).__name__}")
    bad: List[str] = []
    for key in PRECOMMIT_REQUIRED_KEYS:
        val = payload.get(key)
        if not isinstance(val, str) or not val.strip():
            bad.append(key)
    if bad:
        raise P6StateError(f"precommit payload 缺必备键（或值非非空字符串）: {bad}")
    return {str(k): copy.deepcopy(v) for k, v in payload.items()}


def _check_bindings(bindings: Any) -> Dict[str, str]:
    if bindings is None:
        return {}
    if not isinstance(bindings, Mapping):
        raise P6StateError(f"bindings 必须是 mapping，got {type(bindings).__name__}")
    out: Dict[str, str] = {}
    for key, val in bindings.items():
        if not isinstance(key, str) or not key or not isinstance(val, str) or not val.strip():
            raise P6StateError(f"bindings 项 {key!r} 必须为非空 str → 非空 str")
        out[key] = val
    return out


class _ExclusiveLedgerLock:
    """台账独占锁：实例存活期持有 <ledger>.lock 的字节锁；进程死亡由 OS 释放。

    Windows 用 msvcrt.locking(LK_NBLCK)（真互斥：跨进程与同进程第二 fd 均被拒），
    POSIX 用 fcntl.flock(LOCK_EX|LOCK_NB)。锁的是字节区而非文件存在性，
    因此崩溃残留的 lock 文件不会造成重启死锁。
    """

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._fd: Optional[int] = None
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
        fd = os.open(str(self._path), flags)
        try:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise P6StateError(
                f"台账独占锁被占用（另一 SequentialGate 实例/进程持有）: {self._path}"
            ) from exc
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)


class SequentialGate:
    """两级冻结开箱状态机 v2（崩溃安全一次性台账）。

    状态机转移（全部单向不可逆）：
      record_precommit(t) → open_block(V_t) → record_verdict(V_t) → record_cycle_terminal(t)
      abstain 路径：跳过 open/verdict，直接 record_cycle_terminal(t, "abstain")（V_t 保持 sealed）
      U：cycle2 terminal 已记录后 open 一次（不要求 V2 verdict——abstain 也是 terminal）
    台账：canonical 路径（由 manifest 派生）+ 独占文件锁 + write-ahead + 事件 hash 链；
    构造时重放即恢复；open-intent 无后续 verdict = pending，只允许按同一 precommit resume。
    close() 不写任何字节：close 后的文件状态与进程崩溃等价。
    """

    def __init__(self, manifest: P6SplitManifest, ledger) -> None:
        # 先建立可安全 close 的最小属性集（构造中途失败不留锁/fd）。
        self._closed = False
        self._fd: Optional[int] = None
        self._lock: Optional[_ExclusiveLedgerLock] = None
        validate_manifest(manifest)
        self._manifest_sha = manifest.manifest_sha
        if ledger is None:
            raise P6StateError(
                "SequentialGate v2 必须落盘台账：传入 ledger 目录或 canonical 文件路径"
                "（纯内存模式已废除）"
            )
        given = Path(ledger)
        if given.exists() and given.is_dir():
            path = ledger_path(self._manifest_sha, given)
        else:
            canonical = ledger_path(self._manifest_sha, given.parent)
            if given.name != canonical.name:
                raise P6StateError(
                    f"台账路径必须为 canonical（由 manifest 派生）：want {canonical.name!r}，"
                    f"got {given.name!r}——任意 log_path 的第二本台账不被允许"
                )
            path = canonical
        self._log_path: Path = path

        self._states: Dict[str, str] = {b: STATE_SEALED for b in OPENABLE_BLOCKS}
        self._verdicts: Dict[str, str] = {}
        self._precommits: Dict[int, Dict[str, Any]] = {}
        self._terminals: Dict[int, str] = {}
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._seq = 0
        self._tip = _genesis_sha(self._manifest_sha)

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # 先取锁再重放：两实例并发重放+追加被排除在外。
        self._lock = _ExclusiveLedgerLock(self._log_path.with_name(self._log_path.name + ".lock"))
        try:
            if self._log_path.exists():
                self._replay()
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            self._fd = os.open(str(self._log_path), flags)
        except BaseException:
            self.close()
            raise

    # ---- 生命周期 ----

    def close(self) -> None:
        """释放 append fd 与独占锁（幂等）。不写任何字节：文件状态与进程崩溃等价。"""
        if self._closed:
            return
        self._closed = True
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()

    def __enter__(self) -> "SequentialGate":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC 兜底，不作为正常释放路径
        try:
            self.close()
        except Exception:
            pass

    # ---- 查询 ----

    @property
    def manifest_sha(self) -> str:
        return self._manifest_sha

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def chain_tip(self) -> str:
        """当前 hash 链末端（最后事件的 event_sha；无事件时为 genesis）。"""
        return self._tip

    def state(self, block: str) -> str:
        if block not in self._states:
            raise P6StateError(f"未知可开箱块 {block!r}（状态机只管 {OPENABLE_BLOCKS}）")
        return self._states[block]

    def verdict(self, block: str) -> Optional[str]:
        if block not in self._states:
            raise P6StateError(f"未知可开箱块 {block!r}")
        return self._verdicts.get(block)

    def precommit(self, cycle: int) -> Optional[Dict[str, Any]]:
        """cycle 的 precommit 记录 {seq, event_sha, payload}；未落账 → None。"""
        info = self._precommits.get(_check_cycle(cycle))
        return copy.deepcopy(info) if info is not None else None

    def cycle_terminal(self, cycle: int) -> Optional[str]:
        """cycle terminal verdict（promote/reject/abstain）；未记录 → None。"""
        return self._terminals.get(_check_cycle(cycle))

    def pending_open(self, block: str) -> Optional[Dict[str, Any]]:
        """块处于"已开启未判决"（open-intent 无后续 verdict）时返回 open 事件副本，否则 None。

        崩溃恢复语义：pending 块只允许按事件中的 precommit 绑定 resume
        （同候选/同字节/同 seeds 由 runner 纪律保证）；机械层禁止重开、
        重新 precommit（重新选择候选）与直接 terminal。
        """
        if block not in self._states:
            raise P6StateError(f"未知可开箱块 {block!r}")
        info = self._pending.get(block)
        return copy.deepcopy(info) if info is not None else None

    def can_open(self, block: str) -> bool:
        """块当前是否可 open。非 V1/V2/U（含 C0/D1/D2 数据块）恒 False。"""
        if block not in self._states or self._states[block] != STATE_SEALED:
            return False
        if block == "V1":
            return 1 in self._precommits and 1 not in self._terminals
        if block == "V2":
            return 2 in self._precommits and 1 in self._terminals and 2 not in self._terminals
        return 2 in self._terminals  # U ← cycle2 terminal（abstain 也是 terminal）

    # ---- 迁移（全部 write-ahead：先 durable 落盘，再改内存） ----

    def record_precommit(self, cycle: int, payload: Mapping[str, Any]) -> str:
        """cycle precommit 落账（一次性、不可覆盖）。payload 必含 PRECOMMIT_REQUIRED_KEYS。

        **崩溃恢复幂等（F4/finding 34）**：若已存在本 cycle 的 pending precommit（precommit 已
        落账但 cycle 未 terminal）且新五元组逐字段完全一致 → 返回既有记录 event_sha、**不写新
        事件**（允许确定性重算续跑，hash 链不分叉）；不一致 / 已 terminal → 仍拒绝（现行为）。"""
        self._ensure_active()
        cyc = _check_cycle(cycle)
        pay = _check_precommit_payload(payload)
        existing = self._precommits.get(cyc)
        if existing is not None and cyc not in self._terminals:
            same = all(
                existing["payload"].get(k) == pay.get(k) for k in PRECOMMIT_REQUIRED_KEYS
            )
            if same:
                return str(existing["event_sha"])       # 幂等续跑：不追加事件
        self._require_precommit_allowed(cyc)
        return self._commit("precommit", {"cycle": cyc, "payload": pay})

    def open_block(self, block: str, bindings: Optional[Mapping[str, Any]] = None) -> str:
        """开箱（一次性）。V 块绑定自动取自对应 precommit；U 可附带额外 bindings。"""
        self._ensure_active()
        self._require_open_allowed(block)
        fields: Dict[str, Any] = {"block": block}
        if block in _CYCLE_OF_V_BLOCK:
            if bindings is not None:
                raise P6StateError("V 块 open 的绑定只能来自 precommit：不接受额外 bindings")
            cyc = _CYCLE_OF_V_BLOCK[block]
            pc = self._precommits[cyc]
            fields["cycle"] = cyc
            fields["precommit_event_sha"] = pc["event_sha"]
            fields["bindings"] = {k: pc["payload"][k] for k in PRECOMMIT_REQUIRED_KEYS}
        else:
            fields["bindings"] = _check_bindings(bindings)
        return self._commit("open", fields)

    def record_verdict(self, block: str, verdict: str, result_digest: Optional[str] = None) -> str:
        """开箱块判决（一次性、须先 open）。verdict ∈ {promote, reject}。"""
        self._ensure_active()
        v = _check_block_verdict(verdict)
        self._require_verdict_allowed(block)
        fields: Dict[str, Any] = {"block": block, "verdict": v}
        if block in _CYCLE_OF_V_BLOCK:
            cyc = _CYCLE_OF_V_BLOCK[block]
            fields["cycle"] = cyc
            fields["precommit_event_sha"] = self._precommits[cyc]["event_sha"]
        if result_digest is not None:
            if not isinstance(result_digest, str) or not result_digest.strip():
                raise P6StateError("result_digest 必须是非空字符串")
            fields["result_digest"] = result_digest
        return self._commit("verdict", fields)

    def record_cycle_terminal(self, cycle: int, verdict: str) -> str:
        """cycle terminal 事件：每 cycle 只能记录一次、不可逆；verdict ∈ TERMINAL_VERDICTS。

        一致性：V_t sealed → 只能 "abstain"（abstain 路径 V 保持 sealed，terminal 照记）；
        V_t verdict 已记录 → terminal 必须与其一致；V_t open（pending）→ 禁止 terminal。
        """
        self._ensure_active()
        cyc = _check_cycle(cycle)
        v = _check_terminal_verdict(verdict)
        self._require_terminal_allowed(cyc, v)
        vb = _V_BLOCK_OF_CYCLE[cyc]
        pc = self._precommits.get(cyc)
        fields: Dict[str, Any] = {
            "cycle": cyc,
            "verdict": v,
            "v_block": vb,
            "v_block_state": self._states[vb],
            "precommit_event_sha": None if pc is None else pc["event_sha"],
        }
        return self._commit("cycle_terminal", fields)

    # ---- 内部：前置条件（在线与重放共用，不改状态） ----

    def _ensure_active(self) -> None:
        if self._closed:
            raise P6StateError("SequentialGate 已 close：不得再写事件")

    def _require_precommit_allowed(self, cyc: int) -> None:
        if cyc in self._precommits:
            raise P6StateError(
                f"cycle{cyc} precommit 已落账，一次性：崩溃后只能按同一 precommit resume，"
                "禁止重新选择候选"
            )
        if cyc in self._terminals:
            raise P6StateError(
                f"cycle{cyc} 已 terminal（{self._terminals[cyc]!r}）：不得再 precommit"
            )
        if cyc == 2 and 1 not in self._terminals:
            raise P6StateError("cycle2 precommit 需 cycle1 terminal 已记录（cycle 严格顺序）")

    def _require_open_allowed(self, block: str) -> None:
        if block not in self._states:
            raise P6StateError(f"块 {block!r} 不可 open（只有 {OPENABLE_BLOCKS} 有开箱门）")
        st = self._states[block]
        if st != STATE_SEALED:
            raise P6StateError(f"{block} 已处于 {st}：每块只能 open 一次")
        if block == "U":
            if 2 not in self._terminals:
                raise P6StateError(
                    "U 开启需 cycle2 terminal 已记录（abstain 也是 terminal；不要求 V2 verdict）"
                )
            return
        cyc = _CYCLE_OF_V_BLOCK[block]
        if cyc in self._terminals:
            raise P6StateError(
                f"cycle{cyc} 已 terminal（{self._terminals[cyc]!r}）：{block} 永久封存"
            )
        if block == "V2" and 1 not in self._terminals:
            raise P6StateError("V2 开启需 cycle1 terminal 已记录")
        if cyc not in self._precommits:
            raise P6StateError(
                f"{block} 开启需 cycle{cyc} precommit 先落账（precommit 必须先于 open）"
            )

    def _require_verdict_allowed(self, block: str) -> None:
        if block not in self._states:
            raise P6StateError(f"未知可开箱块 {block!r}")
        st = self._states[block]
        if st == STATE_SEALED:
            raise P6StateError(
                f"{block} 尚未 open，不能记录 verdict"
                "（abstain 路径 = 保持 sealed + cycle terminal 记 'abstain'）"
            )
        if st == STATE_VERDICT:
            raise P6StateError(f"{block} verdict 只能记录一次")

    def _require_terminal_allowed(self, cyc: int, v: str) -> None:
        if cyc in self._terminals:
            raise P6StateError(f"cycle{cyc} terminal 只能记录一次（不可逆）")
        if cyc == 2 and 1 not in self._terminals:
            raise P6StateError("cycle2 terminal 需 cycle1 terminal 已记录（cycle 严格顺序）")
        vb = _V_BLOCK_OF_CYCLE[cyc]
        st = self._states[vb]
        if st == STATE_OPEN:
            raise P6StateError(
                f"{vb} 已开启未判决（pending）：只允许按同一 precommit resume 记 verdict，"
                "不得直接 terminal"
            )
        if st == STATE_SEALED and v != "abstain":
            raise P6StateError(f"{vb} 未开箱：cycle{cyc} terminal 只能为 'abstain'")
        if st == STATE_VERDICT and v != self._verdicts[vb]:
            raise P6StateError(
                f"cycle{cyc} terminal {v!r} 与 {vb} verdict {self._verdicts[vb]!r} 不一致"
            )

    # ---- 内部：WAL 提交与状态应用 ----

    def _commit(self, kind: str, fields: Dict[str, Any]) -> str:
        """write-ahead 提交：构造事件 → durable 落盘（单次 write + fsync）→ 才改内存状态。"""
        if self._fd is None:
            raise P6StateError("台账 fd 不可用（gate 未正确初始化）")
        evt: Dict[str, Any] = {
            "schema": LEDGER_SCHEMA,
            "seq": self._seq + 1,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha": self._manifest_sha,
            "event": kind,
            "prev_event_sha": self._tip,
        }
        overlap = set(evt) & set(fields)
        if overlap:
            raise P6StateError(f"事件字段与 header 冲突: {sorted(overlap)}")
        evt.update(fields)
        evt["event_sha"] = compute_event_sha(evt)
        line = json.dumps(evt, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
        view = memoryview(line.encode("utf-8"))
        while view:
            written = os.write(self._fd, view)
            view = view[written:]
        os.fsync(self._fd)
        self._advance_and_apply(evt)
        return str(evt["event_sha"])

    def _advance_and_apply(self, evt: Mapping[str, Any]) -> None:
        """事件已 durable（在线）或已通过校验（重放）后的唯一内存状态入口。"""
        kind = evt["event"]
        if kind == "precommit":
            self._precommits[int(evt["cycle"])] = {
                "seq": int(evt["seq"]),
                "event_sha": str(evt["event_sha"]),
                "payload": copy.deepcopy(dict(evt["payload"])),
            }
        elif kind == "open":
            block = str(evt["block"])
            self._states[block] = STATE_OPEN
            self._pending[block] = copy.deepcopy(dict(evt))
        elif kind == "verdict":
            block = str(evt["block"])
            self._states[block] = STATE_VERDICT
            self._verdicts[block] = str(evt["verdict"])
            self._pending.pop(block, None)
        elif kind == "cycle_terminal":
            self._terminals[int(evt["cycle"])] = str(evt["verdict"])
        else:  # 防御：_commit 只发已知类型，_replay 先经 _verify_event
            raise P6StateError(f"未知事件类型 {kind!r}")
        self._seq = int(evt["seq"])
        self._tip = str(evt["event_sha"])

    # ---- 内部：重放（崩溃恢复 + 完整性校验） ----

    def _replay(self) -> None:
        """重放台账重建状态：hash 链、seq、manifest 绑定、状态机合法性全校验。

        任何断链/篡改/损坏（含撕裂写的半行）→ raise：
        prereg 语义 = 日志损坏 → P6-technical-abort，不自动修复。
        """
        text = self._log_path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise P6StateError(f"台账第 {lineno} 行为空：append-only 单行事件流被破坏")
            try:
                evt = json.loads(line)
            except json.JSONDecodeError as exc:
                raise P6StateError(
                    f"台账第 {lineno} 行损坏（非法 JSON，疑似撕裂写）→ technical abort"
                ) from exc
            if not isinstance(evt, dict):
                raise P6StateError(f"台账第 {lineno} 行不是 JSON object")
            self._verify_event(evt, lineno)
            self._advance_and_apply(evt)

    def _verify_event(self, evt: Mapping[str, Any], lineno: int) -> None:
        for key in _EVENT_HEADER_FIELDS:
            if key not in evt:
                raise P6StateError(f"台账第 {lineno} 行缺字段 {key!r}")
        if evt["schema"] != LEDGER_SCHEMA:
            raise P6StateError(
                f"台账第 {lineno} 行 schema {evt['schema']!r} != {LEDGER_SCHEMA!r}"
            )
        if evt["manifest_sha"] != self._manifest_sha:
            raise P6StateError(f"台账第 {lineno} 行绑定的 manifest_sha 与当前 manifest 不一致")
        if evt["seq"] != self._seq + 1:
            raise P6StateError(f"台账第 {lineno} 行 seq 不连续（append-only 违约）")
        if evt["prev_event_sha"] != self._tip:
            raise P6StateError(f"台账第 {lineno} 行 hash 链断裂（prev_event_sha 不匹配）")
        if compute_event_sha(evt) != evt["event_sha"]:
            raise P6StateError(f"台账第 {lineno} 行事件被篡改（event_sha 校验失败）")

        kind = evt["event"]
        if kind == "precommit":
            cyc = _check_cycle(evt.get("cycle"))
            _check_precommit_payload(evt.get("payload"))
            self._require_precommit_allowed(cyc)
        elif kind == "open":
            block = str(evt.get("block"))
            self._require_open_allowed(block)
            if block in _CYCLE_OF_V_BLOCK:
                cyc = _CYCLE_OF_V_BLOCK[block]
                if evt.get("cycle") != cyc:
                    raise P6StateError(f"台账第 {lineno} 行 open {block} 的 cycle 字段错误")
                pc = self._precommits[cyc]
                if evt.get("precommit_event_sha") != pc["event_sha"]:
                    raise P6StateError(f"台账第 {lineno} 行 open 未链回 cycle{cyc} precommit")
                want = {k: pc["payload"][k] for k in PRECOMMIT_REQUIRED_KEYS}
                if evt.get("bindings") != want:
                    raise P6StateError(f"台账第 {lineno} 行 open 绑定与 precommit 不一致")
            else:
                _check_bindings(evt.get("bindings"))
        elif kind == "verdict":
            block = str(evt.get("block"))
            _check_block_verdict(evt.get("verdict"))
            self._require_verdict_allowed(block)
            rd = evt.get("result_digest")
            if rd is not None and (not isinstance(rd, str) or not rd.strip()):
                raise P6StateError(f"台账第 {lineno} 行 result_digest 非法")
            if block in _CYCLE_OF_V_BLOCK:
                cyc = _CYCLE_OF_V_BLOCK[block]
                pc = self._precommits.get(cyc)
                if pc is None or evt.get("precommit_event_sha") != pc["event_sha"]:
                    raise P6StateError(f"台账第 {lineno} 行 verdict 未链回 cycle{cyc} precommit")
        elif kind == "cycle_terminal":
            cyc = _check_cycle(evt.get("cycle"))
            v = _check_terminal_verdict(evt.get("verdict"))
            self._require_terminal_allowed(cyc, v)
            vb = _V_BLOCK_OF_CYCLE[cyc]
            if evt.get("v_block") != vb or evt.get("v_block_state") != self._states[vb]:
                raise P6StateError(f"台账第 {lineno} 行 cycle_terminal 的 V 块状态快照不一致")
            pc = self._precommits.get(cyc)
            want_sha = None if pc is None else pc["event_sha"]
            if evt.get("precommit_event_sha") != want_sha:
                raise P6StateError(f"台账第 {lineno} 行 cycle_terminal 未正确链回 precommit")
        else:
            raise P6StateError(f"台账第 {lineno} 行未知事件类型 {kind!r}")
