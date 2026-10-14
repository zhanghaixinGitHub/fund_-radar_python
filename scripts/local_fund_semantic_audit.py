"""首次结果的第二层独立审计：只重放检查，不重试模型，不覆盖第一次状态。

只容忍排版空白的绑定定位与严格逐字校验分开记录。文字、标点、数字、单位、
条件任何非空白改动仍拒绝；合法定位也不证明字段关系或解释正确。
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.local_fund_semantics import amounts, compact, semantic_issues

VERSION = "local-fund-independent-audit-v1"


def whitespace_anchor(source: str, text: str) -> dict | None:
    """将只删除排版空白的连续片段回指原始偏移；多处匹配也不能猜出处。

    不替换逗号、括号、汉字或数字，不删除限定词，不跨引用编号搜索。回填的
    source_text仍是原文原样，包括换行与空格；raw_text保留模型的首次字符串。
    """
    if not isinstance(text, str) or not compact(text):
        return None
    positions = [index for index, char in enumerate(source) if not char.isspace()]
    haystack, needle = compact(source), compact(text)
    offset = haystack.find(needle)
    if offset < 0 or haystack.find(needle, offset + 1) >= 0:
        return None
    start, end = positions[offset], positions[offset + len(needle) - 1] + 1
    return {
        "start": start,
        "end": end,
        "source_text": source[start:end],
        "raw_text": text,
        "match": "EXACT" if text in source else "WHITESPACE_ONLY",
    }


def independent_issues(text: str, evidence: str, body: str) -> list[str]:
    """补充引用缺口和目标值变体；只作拦截，不从全文替错误引用补找证据。"""
    issues = semantic_issues(text, evidence, body)
    claim, full = compact(text), compact(body)
    # 不同字段必须按业务口径检查：未给“未来购买条件”不等于未给“是否已购买”；
    # 可分配利润为正是分红前提，不是累计最低分红计算基数。
    if re.search(r"未(?:提及|披露|说明)(?:未来)?购买(?:条件|机制|时间)", claim):
        issues = [issue for issue in issues if issue != "EXPLICIT_NON_PURCHASE_LOST"]
    if not re.search(r"累计|未来三年.{0,25}分配|现金分红比例不低于", claim):
        issues = [issue for issue in issues if issue != "ANNUAL_AVERAGE_BASIS_MISSING"]
    if re.search(r"亿元美元|万元美元|亿元港元|万元港元", claim):
        issues.append("CURRENCY_UNIT_COMBINATION_UNSUPPORTED")
    if re.search(r"不表明.{0,90}真实性.{0,45}(?:保证|认定)", full):
        # 正确的投资价值免责句不能抵消前一句对材料真实性的错误确认。
        for clause in re.split(r"[，,。；]", claim):
            if re.search(r"备案.{0,25}(?:确认|认可).{0,25}真实性", clause) and not re.search(
                r"不表明|不代表|不能|没有|不保证|不确认", clause
            ):
                issues.append("FILING_TRUTH_CONFIRMATION_UNSUPPORTED")
    if "触发值" in full and "目标值" in full and re.search(r"An[≤<].*?A[<＜].*?Am", full):
        if re.search(r"目标(?:值)?[^。；]{0,12}方可(?:相应|对应)?归属|目标(?:值)?[^。；]{0,8}才(?:能)?归属", claim):
            issues.append("TARGET_REQUIRED_FOR_ANY_VESTING")
    cited = amounts(evidence)
    for _, unit in amounts(text):
        dimension = "股" if unit.endswith("股") else "元"
        if not any(source_unit.endswith(dimension) for _, source_unit in cited):
            issues.append("AMOUNT_WITHOUT_QUOTED_SOURCE")
    return list(dict.fromkeys(issues))


def audit_response(raw: dict, document: dict) -> dict:
    """分别记录绑定定位、引用完整性、文本矛盾；不输出“模型已理解正确”。"""
    from app.schemas.fund_information_analysis import quote_catalog

    catalog = quote_catalog(document["body"])
    checks = []
    for item in raw.get("items", []):
        for index, event in enumerate(item.get("events", [])):
            chosen = event.get("quote_ids", [])
            issues, anchors = [], []
            if not chosen or any(ref not in catalog for ref in chosen):
                issues.append("QUOTE_ID_INVALID")
            evidence = "\n".join(catalog[ref]["text"] for ref in chosen if ref in catalog)
            for binding in event.get("bindings", []):
                ref = binding.get("quote_id")
                anchor = (
                    whitespace_anchor(catalog[ref]["text"], binding.get("text", ""))
                    if ref in chosen and ref in catalog
                    else None
                )
                if anchor is None:
                    issues.append("LITERAL_BINDING_NOT_SUPPORTED")
                else:
                    anchors.append({"kind": binding["kind"], "quote_id": ref, **anchor})
            if not event.get("bindings"):
                issues.append("LITERAL_BINDINGS_MISSING")
            for field in ("meaning", "mechanism", "caveat"):
                issues.extend(
                    field + ":" + issue
                    for issue in independent_issues(event.get(field, ""), evidence, document["body"])
                )
            if event.get("table_status") == "UNVERIFIED":
                issues.append("TABLE_BINDING_REQUIRES_REVIEW")
            if document["body_truncated"] or document["body_scope"] != "FULL_TEXT":
                if not re.search(r"部分|摘录|节选|不完整|截断|未纳入", event.get("caveat", "")):
                    issues.append("PARTIAL_SOURCE_LIMITATION_MISSING")
            checks.append(
                {
                    "event_index": index,
                    "title": event["title"],
                    "anchors": anchors,
                    "issues": list(dict.fromkeys(issues)),
                }
            )
    return {
        "checks": checks,
        "issues": [f"E{i}:{issue}" for i, row in enumerate(checks) for issue in row["issues"]],
        "semantic_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，请指定新文件保留首次审计。")

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    data = read(args.run / "snapshot.json")
    cases = [(f"doc-{i:02}", doc) for i, doc in enumerate(data["documents"], 1)]
    cases += [(f"holdout-{i:02}", doc) for i, doc in enumerate(read(args.run / "holdouts.json")["documents"], 1)]
    results = []
    for name, doc in cases:
        record_path = args.run / "raw" / (name + ".json")
        if not record_path.exists():
            results.append({"job": name, "issues": ["NO_MODEL_RESPONSE"], "checks": []})
            continue
        record = read(record_path)
        if record.get("error") or "parsed" not in record:
            results.append({"job": name, "issues": [record.get("error", "NO_PARSED_RESPONSE")], "checks": []})
            continue
        results.append({"job": name, **audit_response(record["parsed"], doc)})
    original_jobs = {job["job_id"]: job for job in read(args.baseline / "event-jobs.json")}
    docs = {doc["id"]: doc for doc in data["documents"]}
    old = []
    for path in sorted(args.baseline.glob("manual-findings-*.json")):
        findings = read(path)
        for finding in findings if isinstance(findings, list) else [findings]:
            event = read(args.baseline / "events" / (finding["job"] + ".json"))["events"][finding["event"]]
            old.append(
                {
                    "id": finding["id"],
                    "issues": independent_issues(
                        finding["statement"],
                        "\n".join(event["quotes"]),
                        docs[original_jobs[finding["job"]]["document_id"]]["body"],
                    ),
                }
            )
    hashes = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("local_fund_semantics.py"))
    }
    value = {
        "version": VERSION,
        "source_hashes": hashes,
        "model_retries": 0,
        "first_checks_overwritten": False,
        "old_errors": old,
        "old_error_flags": sum(bool(row["issues"]) for row in old),
        "old_error_total": len(old),
        "results": results,
        "semantic_verified": False,
        "warning": "只通过这些检查不等于语义正确；原文表格缺列、引用选择遗漏和未覆盖的关系仍需人工核对。",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "old_error_flags": value["old_error_flags"],
                "old_error_total": len(old),
                "flagged_responses": sum(bool(row["issues"]) for row in results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
