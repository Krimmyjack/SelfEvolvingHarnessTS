"""LLM client 验证：抽取器 + 缓存（离线）+ 一次真实 API smoke（联网）。

运行：  python -m SelfEvolvingHarnessTS.tests.test_llm_client   （cwd=Agent）
"""
from __future__ import annotations

import hashlib
import os

import pytest

from SelfEvolvingHarnessTS.llm import LLMClient, MODELS, extract_code_block, extract_json


# ── 1. 代码块抽取（fast_path compose 用）─────────────────────────────────
def test_extract_code_block():
    text = "blah\n```python\ndef improve_readiness(ts):\n    return ts\n```\ntail"
    blk = extract_code_block(text, fn_name="improve_readiness")
    assert blk is not None and "def improve_readiness" in blk
    assert extract_code_block(text, fn_name="other_fn") is None


# ── 2. JSON 抽取（slow_path proposer 解析 K 候选用）──────────────────────
def test_extract_json():
    # 围栏 list
    fenced = 'here are candidates:\n```json\n[{"op":"set","path":"l2.active_operators.x"}]\n```'
    obj = extract_json(fenced)
    assert isinstance(obj, list) and obj[0]["op"] == "set"
    # 裸 object 夹在散文里（含嵌套大括号 + 字符串内的括号）
    prose = 'I propose: {"edited_layer":"L2","value":{"a":1},"note":"use } carefully"} done.'
    o2 = extract_json(prose)
    assert isinstance(o2, dict) and o2["edited_layer"] == "L2" and o2["value"]["a"] == 1
    # 裸数组
    assert extract_json("result = [1, 2, 3] ok") == [1, 2, 3]
    # 不可解析 → None
    assert extract_json("no json here at all") is None


# ── 3. 缓存命中（离线，不联网）────────────────────────────────────────────
def test_cache_hit_offline():
    c = LLMClient(cache_name="test_offline", temperature=0.0)
    key = hashlib.sha1(f"{c.model}|0.0|0|SYS|USER".encode("utf-8")).hexdigest()
    c._cache[key] = "CACHED"
    assert c("SYS", "USER") == "CACHED"
    assert c.n_hit == 1 and c.n_api == 0          # 命中缓存，未发网络请求


# ── 4. 模型别名 ──────────────────────────────────────────────────────────
def test_model_aliases():
    assert LLMClient(model="flash").model == "deepseek-chat"
    assert LLMClient(model="pro").model == "deepseek-v4-pro"
    assert LLMClient(model="deepseek-chat").model == "deepseek-chat"


# ── 5. 真实 API smoke（联网；确认 key/连通）──────────────────────────────
@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"),
                    reason="live smoke 需要 DEEPSEEK_API_KEY（外评①：fallback key 已移除）")
def test_live_api_smoke():
    llm = LLMClient(cache_name="smoke", temperature=0.0)
    out = llm("You are a terse assistant. Output only what is asked.",
              "Reply with exactly the two characters: OK")
    assert isinstance(out, str) and "OK" in out.upper()
    n_api_after_first = llm.n_api
    out2 = llm("You are a terse assistant. Output only what is asked.",
               "Reply with exactly the two characters: OK")              # 同输入 → 缓存命中
    assert out2 == out and llm.n_hit >= 1 and llm.n_api == n_api_after_first
    print("    live stats:", llm.stats())


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
