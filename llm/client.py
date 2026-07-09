"""llm/client.py — 固定 LLM 客户端（OpenAI 兼容 DeepSeek）+ 代码/JSON 抽取。

移植自 previous/check7-13 + SelfHarnessTS/exploration/llm_client.py 的 CachedLLM 模式
（磁盘缓存 + 重试 + nonce 强制采样）。

配置（previous/check8-10）：
  flash = deepseek-chat   （~3–9 s/调用，默认；fast_path compose / 廉价提议）
  pro   = deepseek-v4-pro （~35–50 s/调用，强；slow_path proposer 开放空间提议更优，check9）
密钥优先取环境变量 DEEPSEEK_API_KEY；缺省回退到 previous 中记录的 key（research 用）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional

import requests

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_KEY_FALLBACK = "sk-8ef4e576d4e64512834cebed5b1023da"   # previous/check7-10；优先用环境变量覆盖
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY") or _KEY_FALLBACK

MODELS = {"flash": "deepseek-chat", "pro": "deepseek-v4-pro"}

_CACHE_DIR = Path(__file__).resolve().parent / "_cache"


class LLMClient:
    """callable client(system, user, nonce) -> str，带磁盘缓存 + 重试。

    nonce 让 temperature>0 时强制取不同样本（绕过缓存碰撞）——proposer 采 K 个互异候选时用。
    """

    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.7,
                 cache_name: str = "default", timeout: int = 180,
                 url: str = DEEPSEEK_URL, key: str = DEEPSEEK_KEY, max_retries: int = 4,
                 max_api_calls: Optional[int] = None, max_api_seconds: Optional[float] = None,
                 max_cost_usd: Optional[float] = None,
                 estimated_cost_per_call_usd: float = 0.0):
        self.model = MODELS.get(model, model)        # 允许传 "flash"/"pro" 别名或全名
        self.temperature, self.timeout = temperature, timeout
        self.url, self.key, self.max_retries = url, key, max_retries
        self.max_api_calls = max_api_calls
        self.max_api_seconds = max_api_seconds
        self.max_cost_usd = max_cost_usd
        self.estimated_cost_per_call_usd = float(estimated_cost_per_call_usd)
        self.estimated_cost_usd = 0.0
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = _CACHE_DIR / f"{cache_name}.json"
        self._cache = {}
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}
        self.n_api = self.n_hit = 0
        self.api_seconds = 0.0

    def _check_budget(self) -> None:
        """Fail closed before an uncached API call would exceed a configured budget."""
        if self.max_api_calls is not None and self.n_api >= self.max_api_calls:
            raise RuntimeError(
                f"LLM budget exceeded: api_calls={self.n_api} max_api_calls={self.max_api_calls}"
            )
        if self.max_api_seconds is not None and self.api_seconds >= self.max_api_seconds:
            raise RuntimeError(
                f"LLM budget exceeded: api_seconds={self.api_seconds:.3f} "
                f"max_api_seconds={self.max_api_seconds}"
            )
        if self.max_cost_usd is not None:
            next_cost = self.estimated_cost_usd + self.estimated_cost_per_call_usd
            if next_cost > self.max_cost_usd + 1e-12:
                raise RuntimeError(
                    f"LLM budget exceeded: estimated cost would be ${next_cost:.6f} "
                    f"> max_cost_usd=${self.max_cost_usd:.6f}"
                )

    def _save(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            pass

    def __call__(self, system: str, user: str, nonce: int = 0) -> str:
        k = hashlib.sha1(
            f"{self.model}|{self.temperature}|{nonce}|{system}|{user}".encode("utf-8")
        ).hexdigest()
        if k in self._cache:
            self.n_hit += 1
            return self._cache[k]
        self._check_budget()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        if self.temperature is not None:                 # 某些模型（如 claude-opus-4-8）拒收 temperature
            payload["temperature"] = self.temperature
        last = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"]
                self.n_api += 1
                self.api_seconds += time.time() - t0
                self.estimated_cost_usd += self.estimated_cost_per_call_usd
                self._cache[k] = out
                self._save()
                return out
            except Exception as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(3.0 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: "
                           f"{type(last).__name__}: {last}")

    def stats(self) -> dict:
        return {"model": self.model, "api_calls": self.n_api, "cache_hits": self.n_hit,
                "avg_latency_s": round(self.api_seconds / max(self.n_api, 1), 2),
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "budget": {"max_api_calls": self.max_api_calls,
                           "max_api_seconds": self.max_api_seconds,
                           "max_cost_usd": self.max_cost_usd,
                           "estimated_cost_per_call_usd": self.estimated_cost_per_call_usd}}


def get_client(model: str = "flash", **kw) -> LLMClient:
    return LLMClient(model=model, **kw)


# ── 输出抽取 ────────────────────────────────────────────────────────────────
_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code_block(text: str, fn_name: Optional[str] = None) -> Optional[str]:
    """抽 ```python``` 代码块；指定 fn_name 时要求块内含 `def fn_name`（fast_path compose 用）。"""
    blocks = _CODE_BLOCK.findall(text)
    candidates = blocks if blocks else [text]
    for blk in candidates:
        if fn_name is None or f"def {fn_name}" in blk:
            return blk.strip()
    return None


def _balanced(s: str, open_ch: str, close_ch: str) -> Optional[str]:
    """从首个 open_ch 起扫到匹配的 close_ch（字符串感知），返回该子串。"""
    start = s.find(open_ch)
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def extract_json(text: str) -> Optional[Any]:
    """从 LLM 输出抽 JSON：优先 ```json``` 围栏，否则扫第一个平衡的 [..] 或 {..}。

    供 slow_path/proposer 把 K 个 EditPatch 候选（JSON）解析出来。失败返回 None。
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates: List[str] = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    # 按「最先出现的括号」取最外层结构（避免把 {"steps":[..]} 误抓成内层数组）
    i_obj, i_arr = text.find("{"), text.find("[")
    order = ([("{", "}"), ("[", "]")] if (i_obj != -1 and (i_arr == -1 or i_obj < i_arr))
             else [("[", "]"), ("{", "}")])
    for oc, cc in order:
        blk = _balanced(text, oc, cc)
        if blk:
            candidates.append(blk)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


if __name__ == "__main__":
    llm = LLMClient(cache_name="smoke", temperature=0.0)
    print("reply:", repr(llm("You are a terse assistant.", "Reply with exactly: OK")))
    print("stats:", llm.stats())

