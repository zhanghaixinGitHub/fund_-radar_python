"""备案声明的第二层原文一致性告警；补充“信息真实”等变体，不修订首次检查。

只有原文明示备案不作真实性保证/认定时才启用。告警用于阻止直接采用，并不
宣称覆盖所有语义；公司自行保证本公告真实与监管机关备案认定须区分。
"""

import argparse
import json
import re
from pathlib import Path

from scripts.local_fund_repair_run import read, save, sha
from scripts.local_fund_semantics import compact
from scripts.local_fund_verified_facts import literal_matches


def filing_confirmation_issues(statement: str, body: str) -> list[dict]:
    """以明确否定原句核对正向确认；正确的投资价值免责不能抵消真实性断言。

    不把任意“真实”词面判错，必须同句涉及备案或证监会且存在确认动作。
    排除直接否定该确认及公司自行声明两种边界；复杂主体歧义仍交原文复核。
    """
    denials = literal_matches(body, r"不(?:表明|代表)[^。；]{0,100}真实性[^。；]{0,55}(?:保证|认定)")
    if not denials:
        return []
    issues = []
    for sentence in re.split(r"[。；\n]", compact(statement)):
        if not re.search(r"备案|证监会", sentence):
            continue
        for affirmative in re.finditer(r"(?:确认|认定|认可|证明)[^。；]{0,48}(?:真实性?|准确性?|完整性?)", sentence):
            prefix = sentence[max(0, affirmative.start() - 24) : affirmative.start()]
            if re.search(r"(?:不代表|不表明|不能|没有|并未|未|不)[^，。；]{0,16}$", prefix):
                continue
            if re.search(r"(?:公司|发行人)(?:自行|自查|声明)?$", prefix):
                continue
            issues.append(
                {
                    "code": "FILING_CONFIRMATION_REQUIRES_SOURCE_REVIEW",
                    "statement": sentence,
                    "matched_affirmation": affirmative[0],
                    "source_denials": denials,
                    "meaning": "明示否定的备案性质与确认真实性表述冲突或缺乏依据，禁止直接采用",
                }
            )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for task in read(args.benchmark / "tasks.json"):
        if task["kind"] != "announcement":
            continue
        for path in sorted((args.benchmark / "raw").glob(task["id"] + "--*.json")):
            if "started" in path.stem:
                continue
            record = read(path)
            issues = []
            for item in record.get("parsed", {}).get("items", []):
                for index, event in enumerate(item["events"]):
                    for field in ("meaning", "mechanism", "caveat"):
                        issues.extend(
                            {"event_index": index, "field": field, **issue}
                            for issue in filing_confirmation_issues(event[field], task["document"]["body"])
                        )
            rows.append({"response": path.name, "response_sha256": sha(path), "issues": issues})
    save(
        args.output,
        {
            "version": "LOCAL_FILING_GUARD_V1",
            "checks": rows,
            "flagged_responses": sum(bool(r["issues"]) for r in rows),
            "old_checks_overwritten": False,
            "model_calls": 0,
            "semantic_verified": False,
        },
    )
    print(json.dumps({"responses": len(rows), "flagged": sum(bool(r["issues"]) for r in rows)}))


if __name__ == "__main__":
    main()
