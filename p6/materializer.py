"""p6/materializer.py — P6 virgin materializer（sealed materialization manifest；prereg §2）。

外审 NO-GO 兑现件：两级 manifest 的第二级。selection_rule_manifest（签发时冻结）之下，
C0 阶段把 V1/V2/U 的冻结选取规则应用到"下载时全量 item_id 宇宙"，产出
**sealed_materialization_manifest**（确切 uid、content_sha、length、finite check），
链回规则 manifest sha（rule_manifest_sha）并携带自身 sha（materialization_sha）。
后续所有 open 事件绑定本 sha（prereg §2 条 3；绑定动作在台账侧，非本模块职责）。

依赖注入红线：universe = [(item_id, np.ndarray)] 由调用方提供——真跑时由
AdaCTS.data.load_monash.load_config_series 下载供给（该 loader 返回 NaN 线性填充后的
float 序列），测试用合成 fixture；**本模块自身不做网络、不读 data/、无 RNG**。

选取规则不自铸：排序键复用 split_manifest 的冻结函数（virgin_sort_key / u_sort_key），
tie-break 同为 (sha_key, item_id) 升序——与 select_virgin_items / select_u_items 语义一致。

technical abort（prereg §2："候选不足/下载失败/hash 漂移/长度不足/重复 uid → technical
abort（不得换 config、降 quota、人工补条）"）→ raise P6TechnicalAbort：
  1. 排除后候选不足 quota；
  2. universe 内 item_id 重复；
  3. **选中**条目含非有限值（NaN/±inf——loader 填充后仍应有限）；
  4. **选中**条目长度 < MIN_SERIES_LEN=144（prereg §2 eligibility len≥144）；
  5. load_materialization 时内嵌 sha 校验失败 / schema 与内容重建不一致（hash 漂移/篡改）。
**冻结释义：条 3/4 的检查作用在"hash 规则选中的条目"上，不做预过滤**——若预过滤掉
不合格候选，下一顺位会自动顶替，等价于被禁止的"人工补条"；不合格条目被选中即整体 abort。

V1/V2/U 接口（prereg §2 两级 manifest + split_manifest.virgin_specs 形状）：
  - block_spec = manifest.virgin_specs["V1"|"V2"|"U"] 原样 dict；
  - V1/V2 为多 config 规格 → 逐 config 物化：必须传 keyword `config=`（∈ spec["configs"]），
    排除集 = spec["exclusions"][config]；
  - **V2 的排除集须含 V1 已选：v1_selected 显式传入**（None → ValueError）；
    V1/U 传入 v1_selected → ValueError（接口误用是调用方 bug，非数据故障，不占用 abort 语义）；
  - U 为单 config 规格：quota = spec["n"]，排除集 = spec["exclusions"]（探针消费清单）。

content_sha 口径（与 diagnostics/p6_u_admission_probe_v2._sha256_f64 一致）：
NaN 填充后（= loader 返回状态）、z-score 前的 float64 C-contiguous ravel bytes 的 sha256。

红线：只依赖 stdlib + numpy + split_manifest 的只读排序键；不落盘副作用
（write_materialization 的路径由调用方给）；不修改任何现有文件。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .split_manifest import u_sort_key, virgin_sort_key

__all__ = [
    "MATERIALIZATION_SCHEMA",
    "MIN_SERIES_LEN",
    "P6TechnicalAbort",
    "SealedMaterialization",
    "compute_materialization_sha",
    "content_sha",
    "load_materialization",
    "load_materialization_bound",
    "materialize",
    "validate_materialization",
    "write_materialization",
]

MATERIALIZATION_SCHEMA = "p6-sealed-materialization/1"
MIN_SERIES_LEN = 144                     # prereg §2 eligibility：len ≥ 144
_HEX64 = set("0123456789abcdef")

#: 顶层易变字段：不参与 materialization_sha，load 时剔除（对齐 split_manifest.VOLATILE_KEYS 风格）。
_VOLATILE_KEYS = frozenset({"materialization_sha", "created_at", "written_at", "timestamp", "ts"})

_ITEM_FIELDS = ("uid", "item_id", "config", "content_sha", "length", "finite_ok")


class P6TechnicalAbort(RuntimeError):
    """P6-technical-abort（prereg 顶注）：基础设施/数据完整性故障，不计入任何 claim 分支。"""


def content_sha(x: np.ndarray) -> str:
    """内容指纹：float64 C-contiguous ravel bytes 的 sha256（NaN 填充后、z-score 前状态）。"""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compute_materialization_sha(payload: Mapping[str, Any]) -> str:
    """canonical JSON（sort_keys、ASCII、紧凑分隔符）剔除顶层易变字段后取 sha256。"""
    clean = {k: payload[k] for k in payload if k not in _VOLATILE_KEYS}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SealedMaterialization:
    """不可变 sealed materialization 载体：payload 为 JSON 原生 dict（无易变字段）。"""

    payload: Dict[str, Any]

    @property
    def materialization_sha(self) -> str:
        return compute_materialization_sha(self.payload)

    @property
    def block(self) -> str:
        return str(self.payload["block"])

    @property
    def config(self) -> str:
        return str(self.payload["config"])

    @property
    def rule_manifest_sha(self) -> str:
        return str(self.payload["rule_manifest_sha"])

    @property
    def items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in self.payload["items"]]

    def uids(self) -> List[str]:
        return [str(it["uid"]) for it in self.payload["items"]]

    def item_ids(self) -> List[str]:
        return [str(it["item_id"]) for it in self.payload["items"]]


# --------------------------------------------------------------------------
# spec 归一化（V1/V2 多 config 规格切片；U 单 config 规格）
# --------------------------------------------------------------------------

def _check_rule_sha(rule_manifest_sha: str) -> str:
    sha = str(rule_manifest_sha)
    if len(sha) != 64 or any(c not in _HEX64 for c in sha):
        raise ValueError(
            f"rule_manifest_sha 必须是 64 位十六进制（selection_rule_manifest 的 sha），got {sha!r}"
        )
    return sha


def _normalize_spec(
    block_spec: Mapping[str, Any],
    config: Optional[str],
    v1_selected: Optional[Iterable[str]],
) -> Tuple[str, str, str, int, set, List[str], str]:
    """→ (block, config, hash_prefix, quota, exclusions, v1_selected_sorted, rule_text)。

    接口误用（未知块、缺 config、v1_selected 传错侧）→ ValueError（调用方 bug）；
    数据级不足在 materialize 主体里 abort。
    """
    if not isinstance(block_spec, Mapping):
        raise ValueError(f"block_spec 必须是 mapping，got {type(block_spec).__name__}")
    block = str(block_spec.get("block", ""))
    if block not in ("V1", "V2", "U"):
        raise ValueError(f"未知可物化块 {block!r}（只接受 V1/V2/U 规格）")
    prefix = str(block_spec.get("hash_prefix", ""))

    if block == "U":
        want_prefix = "p6u"
        if prefix != want_prefix:
            raise ValueError(f"U 规格 hash_prefix 必须为 {want_prefix!r}，got {prefix!r}")
        if v1_selected is not None:
            raise ValueError("U 规则不含 V1_selected 排除：不得传 v1_selected")
        cfg = str(block_spec.get("config", ""))
        if not cfg:
            raise ValueError("U 规格缺 config")
        if config is not None and str(config) != cfg:
            raise ValueError(f"config={config!r} 与 U 规格 config={cfg!r} 不一致")
        quota = int(block_spec.get("n", 0))
        if quota <= 0:
            raise ValueError(f"U 规格 n 必须为正，got {block_spec.get('n')!r}")
        exclusions = {str(x) for x in block_spec.get("exclusions", [])}
        v1_list: List[str] = []
    else:
        want_prefix = "p6v1" if block == "V1" else "p6v2"
        if prefix != want_prefix:
            raise ValueError(f"{block} 规格 hash_prefix 必须为 {want_prefix!r}，got {prefix!r}")
        configs = [str(c) for c in block_spec.get("configs", [])]
        if config is None:
            raise ValueError(
                f"{block} 是多 config 规格：materialize 逐 config 物化，必须传 keyword config="
            )
        cfg = str(config)
        if cfg not in configs:
            raise ValueError(f"config {cfg!r} 不在 {block} 规格 configs {configs} 中")
        quota = int(block_spec.get("per_config_n", 0))
        if quota <= 0:
            raise ValueError(f"{block} 规格 per_config_n 必须为正")
        excl_map = block_spec.get("exclusions", {})
        if cfg not in excl_map:
            raise ValueError(f"{block} 规格 exclusions 缺 config {cfg!r}")
        exclusions = {str(x) for x in excl_map[cfg]}
        if block == "V2":
            if v1_selected is None:
                raise ValueError(
                    "V2 规则显式排除 V1 已选（prereg §2）：必须提供 v1_selected"
                )
            v1_list = sorted({str(x) for x in v1_selected})
            exclusions |= set(v1_list)
        else:
            if v1_selected is not None:
                raise ValueError("V1 规则不含 V1_selected 排除：不得传 v1_selected")
            v1_list = []

    rule_text = str(block_spec.get("rule", ""))
    return block, cfg, prefix, quota, exclusions, v1_list, rule_text


def _sort_key(block: str, prefix: str, cfg: str, item_id: str) -> str:
    """冻结排序键（复用 split_manifest；不自铸）：V=sha256(prefix|config|item)、U=sha256(p6u|item)。"""
    if block == "U":
        return u_sort_key(item_id)
    return virgin_sort_key(prefix, cfg, item_id)


# --------------------------------------------------------------------------
# materialize：排除 → hash 排序 → 取 quota → 逐条 {uid, content_sha, length, finite_ok}
# --------------------------------------------------------------------------

def materialize(
    block_spec: Mapping[str, Any],
    universe: Sequence[Tuple[str, np.ndarray]],
    rule_manifest_sha: str,
    *,
    config: Optional[str] = None,
    v1_selected: Optional[Iterable[str]] = None,
) -> SealedMaterialization:
    """把冻结选取规则应用到下载时宇宙，产出 sealed materialization（见模块 docstring）。

    universe：[(item_id, np.ndarray)]（依赖注入——本函数不做网络）。
    abort（P6TechnicalAbort）：重复 item_id / 排除后候选不足 quota /
    选中条目非有限值 / 选中条目长度 < MIN_SERIES_LEN。
    """
    rule_sha = _check_rule_sha(rule_manifest_sha)
    block, cfg, prefix, quota, exclusions, v1_list, rule_text = _normalize_spec(
        block_spec, config, v1_selected
    )

    by_id: Dict[str, np.ndarray] = {}
    for item_id, series in universe:
        iid = str(item_id)
        if iid in by_id:
            raise P6TechnicalAbort(
                f"{block}/{cfg}: universe 内 item_id 重复：{iid!r}（下载完整性故障）"
            )
        by_id[iid] = series

    candidates = sorted(set(by_id) - exclusions)
    if len(candidates) < quota:
        raise P6TechnicalAbort(
            f"{block}/{cfg}: 候选不足 quota：需 {quota}，排除后仅剩 {len(candidates)}"
            "（不得换 config、降 quota、人工补条）"
        )
    candidates.sort(key=lambda item: (_sort_key(block, prefix, cfg, item), item))
    selected = candidates[:quota]

    items: List[Dict[str, Any]] = []
    for iid in selected:
        arr = np.asarray(by_id[iid], dtype=np.float64).ravel()
        finite_ok = bool(np.all(np.isfinite(arr))) and arr.size > 0
        if not finite_ok:
            raise P6TechnicalAbort(
                f"{block}/{cfg}: 选中条目 {iid!r} 含非有限值（NaN/±inf）——"
                "不做预过滤顶替（等价人工补条），整体 technical abort"
            )
        if int(arr.size) < MIN_SERIES_LEN:
            raise P6TechnicalAbort(
                f"{block}/{cfg}: 选中条目 {iid!r} 长度 {arr.size} < {MIN_SERIES_LEN}"
                "（prereg §2 eligibility）——整体 technical abort"
            )
        items.append(
            {
                "uid": f"{cfg}:{iid}",
                "item_id": iid,
                "config": cfg,
                "content_sha": content_sha(arr),
                "length": int(arr.size),
                "finite_ok": finite_ok,
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "block": block,
        "config": cfg,
        "hash_prefix": prefix,
        "quota": int(quota),
        "rule_manifest_sha": rule_sha,
        "rule": rule_text,
        "exclusions_applied": sorted(exclusions),
        "v1_selected": v1_list,
        "universe_size": len(by_id),
        "n_candidates_after_exclusion": len(candidates),
        "items": items,
    }
    sm = SealedMaterialization(payload)
    validate_materialization(sm)
    return sm


# --------------------------------------------------------------------------
# 校验 + 落盘/读回（往返 sha 不变；内嵌 sha + 选取重建双重防篡改）
# --------------------------------------------------------------------------

def validate_materialization(
    sm: SealedMaterialization,
    series_by_uid: Optional[Mapping[str, Any]] = None,
) -> None:
    """schema/一致性校验（materialize 出口与 load 入口共用）；违规 → P6TechnicalAbort。

    内部一致性：quota、字段、finite/length、hash 排序次序、去重。
    **可复算校验（F2/finding 32）**：给出 `series_by_uid`（uid → 提取后 float 序列）时，
    逐条对 float64 字节重算 content_sha（口径同 diagnostics/p6_u_admission_probe_v2._sha256_f64）
    并与记录比对，任一不一致 / 缺数据 → P6TechnicalAbort（content_sha 不再只是"权威指纹"，
    而是可离线复核的绑定）。
    """
    p = sm.payload
    if p.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise P6TechnicalAbort(f"schema_version 不符: {p.get('schema_version')!r}")
    for key in ("block", "config", "hash_prefix", "quota", "rule_manifest_sha",
                "exclusions_applied", "v1_selected", "universe_size",
                "n_candidates_after_exclusion", "items"):
        if key not in p:
            raise P6TechnicalAbort(f"materialization 缺字段 {key!r}")
    block, cfg, prefix = str(p["block"]), str(p["config"]), str(p["hash_prefix"])
    if block not in ("V1", "V2", "U"):
        raise P6TechnicalAbort(f"非法 block {block!r}")
    _check_rule_sha(str(p["rule_manifest_sha"]))
    items = p["items"]
    if len(items) != int(p["quota"]):
        raise P6TechnicalAbort(f"items 数 {len(items)} != quota {p['quota']}")
    excl = {str(x) for x in p["exclusions_applied"]}
    seen: set = set()
    for it in items:
        for f in _ITEM_FIELDS:
            if f not in it:
                raise P6TechnicalAbort(f"item 缺字段 {f!r}: {it}")
        iid = str(it["item_id"])
        if iid in seen:
            raise P6TechnicalAbort(f"items 内 item_id 重复：{iid!r}")
        seen.add(iid)
        if iid in excl:
            raise P6TechnicalAbort(f"排除集条目 {iid!r} 出现在选中 items 中")
        if str(it["uid"]) != f"{cfg}:{iid}":
            raise P6TechnicalAbort(f"uid {it['uid']!r} != '{cfg}:{iid}'")
        if it["finite_ok"] is not True:
            raise P6TechnicalAbort(f"item {iid!r} finite_ok 必须为 True（非有限值应已 abort）")
        if int(it["length"]) < MIN_SERIES_LEN:
            raise P6TechnicalAbort(f"item {iid!r} length {it['length']} < {MIN_SERIES_LEN}")
        sha = str(it["content_sha"])
        if len(sha) != 64 or any(c not in _HEX64 for c in sha):
            raise P6TechnicalAbort(f"item {iid!r} content_sha 非 64 位 hex")
    # 选中次序必须与冻结 hash 规则一致（次序篡改必被抓）
    ordered = sorted(seen, key=lambda item: (_sort_key(block, prefix, cfg, item), item))
    if [str(it["item_id"]) for it in items] != ordered:
        raise P6TechnicalAbort("items 次序与冻结 hash 排序规则不一致（篡改或规则漂移）")

    # 可复算校验（F2）：给出数据即逐条重算 content_sha 比对。
    if series_by_uid is not None:
        for it in items:
            uid = str(it["uid"])
            if uid not in series_by_uid:
                raise P6TechnicalAbort(f"content_sha 复算：缺 uid {uid!r} 的序列数据")
            recomputed = content_sha(np.asarray(series_by_uid[uid], dtype=np.float64))
            if recomputed != str(it["content_sha"]):
                raise P6TechnicalAbort(
                    f"content_sha 复算不一致（uid {uid!r}）：记录 {it['content_sha']} != "
                    f"重算 {recomputed}（数据漂移/篡改）"
                )


def write_materialization(sm: SealedMaterialization, path) -> str:
    """写 sealed materialization JSON（含内嵌 materialization_sha），返回 sha。"""
    validate_materialization(sm)
    sha = sm.materialization_sha
    doc = dict(sm.payload)
    doc["materialization_sha"] = sha
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha


def load_materialization(path) -> SealedMaterialization:
    """读回 sealed materialization；剔除易变字段、校验内嵌 sha 与全部一致性规则。

    内嵌 sha 与重算不一致 = hash 漂移 → P6TechnicalAbort（prereg §2）。
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise P6TechnicalAbort("materialization 文件顶层必须是 JSON object")
    embedded = doc.get("materialization_sha")
    payload = {k: v for k, v in doc.items() if k not in _VOLATILE_KEYS}
    sm = SealedMaterialization(payload)
    if embedded is None or str(embedded) != sm.materialization_sha:
        raise P6TechnicalAbort(
            f"materialization_sha 校验失败（hash 漂移）：内嵌 {embedded!r} != "
            f"重算 {sm.materialization_sha}"
        )
    validate_materialization(sm)
    return sm


def load_materialization_bound(
    source: Any,
    selection_manifest: Any,
    *,
    series_by_uid: Optional[Mapping[str, Any]] = None,
    expected_materialization_sha: Optional[str] = None,
    v1_selected: Optional[Iterable[str]] = None,
) -> SealedMaterialization:
    """manifest-bound loader（F2/finding 32）：把 sealed materialization **绑定**到 selection
    rule manifest 并逐条核验后返回 SealedMaterialization。任一不符 → P6TechnicalAbort。

    校验：
      1. `rule_manifest_sha == selection_manifest.manifest_sha`（不再只做 64-hex 格式检查，
         须与传入 selection manifest 的**实际内容 sha 相等**）；
      2. block/config ∈ manifest 的 virgin_specs；
      3. 逐条：config == 本块 config、uid == f"{config}:{item_id}"、item_id ∉ 该块/config 排除集
         （= "uid ∈ manifest"：物化条目必须落在 selection 规则允许的候选空间内；preset 是下游
         corruption 维、不在两级 manifest 内，故此处只核 config）；
      4. `series_by_uid` 给出 → 逐条 content_sha 复算一致（经 validate_materialization）；
      5. `expected_materialization_sha` 给出 → materialization_sha 相等（= precommit 绑定值）。

    source 可为路径（走 load_materialization 内嵌 sha 校验）或已加载的 SealedMaterialization。
    selection_manifest 只被只读消费（`.manifest_sha` 与 `.payload["virgin_specs"]`）。
    """
    sm = source if isinstance(source, SealedMaterialization) else load_materialization(source)
    validate_materialization(sm, series_by_uid)      # 内部一致性 + 可复算（若给数据）

    man_sha = getattr(selection_manifest, "manifest_sha", None)
    if not isinstance(man_sha, str) or not man_sha:
        raise P6TechnicalAbort("selection_manifest 缺 manifest_sha（须为 P6SplitManifest 或同 API）")
    if sm.rule_manifest_sha != man_sha:
        raise P6TechnicalAbort(
            f"rule_manifest_sha 与 selection manifest 内容 sha 不一致："
            f"{sm.rule_manifest_sha} != {man_sha}（物化未链回本 selection 规则）"
        )

    try:
        specs = selection_manifest.payload["virgin_specs"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise P6TechnicalAbort(f"selection_manifest 无 virgin_specs：{exc}") from exc
    block, cfg = sm.block, sm.config
    if block not in specs:
        raise P6TechnicalAbort(f"block {block!r} 不在 selection manifest virgin_specs")
    spec = specs[block]
    if block == "U":
        want_cfg = str(spec.get("config", ""))
        if cfg != want_cfg:
            raise P6TechnicalAbort(f"U 物化 config {cfg!r} != manifest U config {want_cfg!r}")
        exclusions = {str(x) for x in spec.get("exclusions", [])}
    else:
        configs = [str(c) for c in spec.get("configs", [])]
        if cfg not in configs:
            raise P6TechnicalAbort(f"{block} 物化 config {cfg!r} 不在 manifest configs {configs}")
        exclusions = {str(x) for x in spec.get("exclusions", {}).get(cfg, [])}
        if block == "V2" and v1_selected is not None:
            exclusions |= {str(x) for x in v1_selected}

    for it in sm.items:
        iid = str(it["item_id"])
        if str(it["config"]) != cfg:
            raise P6TechnicalAbort(f"item {iid!r} config {it['config']!r} != 块 config {cfg!r}")
        if str(it["uid"]) != f"{cfg}:{iid}":
            raise P6TechnicalAbort(f"item uid {it['uid']!r} != '{cfg}:{iid}'")
        if iid in exclusions:
            raise P6TechnicalAbort(
                f"物化 item {iid!r} 落在 selection manifest 排除集内（uid ∉ manifest 允许候选空间）"
            )

    if expected_materialization_sha is not None and sm.materialization_sha != str(
        expected_materialization_sha
    ):
        raise P6TechnicalAbort(
            f"materialization_sha 与绑定值不一致：{sm.materialization_sha} != "
            f"{expected_materialization_sha}（precommit 绑定漂移）"
        )
    return sm
