import hashlib

import pytest

from SelfEvolvingHarnessTS.contracts.canonical import (
    CANONICALIZATION_VERSION,
    canonical_json_document_bytes,
    canonical_jsonl_bytes,
    canonical_sha256,
    canonical_text_bytes,
    parse_json_document,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_text_canonicalization_normalizes_bom_newlines_and_unicode():
    composed = "caf\u00e9\n".encode("utf-8")
    decomposed_crlf = b"\xef\xbb\xbfcafe\xcc\x81\r\n\r\n"
    assert CANONICALIZATION_VERSION == "m0-c14n/1"
    assert canonical_text_bytes(composed) == "caf\u00e9\n".encode("utf-8")
    assert _digest(canonical_text_bytes(composed)) == _digest(
        canonical_text_bytes(decomposed_crlf)
    )


def test_json_document_canonicalization_ignores_order_whitespace_and_unicode_form():
    left = '{"b":"caf\\u00e9", "a": 1}'.encode()
    right = b'{\n  "a":1,"b":"cafe\\u0301"\n}'
    assert canonical_json_document_bytes(left) == canonical_json_document_bytes(right)
    assert canonical_sha256(parse_json_document(left)) == canonical_sha256(
        parse_json_document(right)
    )


@pytest.mark.parametrize(
    "document,match",
    [
        (b'{"a":1,"a":2}', "duplicate JSON key"),
        (b'{"x":NaN}', "non-finite"),
        (b'{"x":Infinity}', "non-finite"),
    ],
)
def test_json_document_rejects_ambiguous_or_nonfinite_values(document, match):
    with pytest.raises(ValueError, match=match):
        parse_json_document(document)


def test_text_rejects_nul_and_invalid_utf8():
    with pytest.raises(ValueError, match="NUL"):
        canonical_text_bytes(b"unsafe\x00text")
    with pytest.raises(ValueError, match="UTF-8"):
        canonical_text_bytes(b"\xff")


def test_json_normalization_rejects_key_collisions():
    with pytest.raises(ValueError, match="normalization collision"):
        canonical_json_document_bytes(
            '{"\u00e9":1,"e\u0301":2}'.encode("utf-8")
        )


def test_jsonl_canonicalizes_rows_but_preserves_row_order():
    left = b'{"b":2,"a":1}\r\n\r\n{"id":2}\r\n'
    same = b'{ "a": 1, "b": 2 }\n{"id":2}\n'
    permuted = b'{"id":2}\n{"a":1,"b":2}\n'
    assert canonical_jsonl_bytes(left) == canonical_jsonl_bytes(same)
    assert _digest(canonical_jsonl_bytes(left)) != _digest(
        canonical_jsonl_bytes(permuted)
    )
