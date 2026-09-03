"""P0 v2 报告细胞检查（reason 码——零 LLM）。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT = (PROJECT_ROOT / "artifacts/functional/e2"
          / "w1_p0_batch_evidence_causal_report_v2.json")


def main() -> int:
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    cells = d["cells"]
    print("--- patch cases, arm=full (abstain reasons) ---")
    for c in cells:
        if c["kind"] == "patch" and c["arm"] == "full":
            print(f"case {c['case']} rep {c['rep']}: got={c['decision']} "
                  f"reason={c['reason']} calls={c['calls']}")
    print("--- swap arm (evidence-following cells) ---")
    for c in cells:
        if c["arm"] == "swap" and c["correct"]:
            print(f"case {c['case']} rep {c['rep']}: "
                  f"got={c['decision']} (exp={c['expected']})")
    print("--- protocol error cells ---")
    for c in cells:
        if c["protocol_error"]:
            print(f"case {c['case']} {c['kind']} arm={c['arm']} "
                  f"rep={c['rep']}: reason={c['reason']} "
                  f"retried={c['whole_retried']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
