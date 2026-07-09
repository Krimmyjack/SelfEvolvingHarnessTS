"""llm/ — 固定 LLM 客户端（共享基础设施）。

同一个 fixed LLM 既是 fast_path/compose 的程序合成器，也是 slow_path/proposer 的编辑提议者
（R1：不改权重）。提供带磁盘缓存 + 重试的 OpenAI 兼容 DeepSeek 客户端 + 代码/JSON 抽取。
"""
from .client import (
    LLMClient, MODELS, get_client,
    extract_code_block, extract_json,
)

__all__ = ["LLMClient", "MODELS", "get_client", "extract_code_block", "extract_json"]
