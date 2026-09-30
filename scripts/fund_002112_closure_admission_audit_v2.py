"""对已结束的公司原件任务做完整清单审计；不拟合、不读取答案、不改旧准入。"""

import json
from collections import Counter
from datetime import datetime

from app.services.fund_information_history_v1 import ROOT, read, save, sha

OUT = ROOT / "closure/20260929-v1"
DEST = OUT / "company-bodies"


def run():
    """核实每份原件摘要、请求预算和共享拟合账本，保留全部未准入原因。

    旧审核中的字段或指定页通过仅保留为引用，不提升成整份文档的语义完整。
    该审计不重新提取 PDF，正文逐页复现与语义审核仍须对应的审计证据。
    """
    if (OUT / ".company-bodies.lock").exists() or not (DEST / "collection-result.json").exists():
        raise ValueError("COLLECTION_NOT_CLOSED")
    plan = read(DEST / "plan.json")
    for path, expected in {**plan["source_hashes"], **plan["code_hashes"]}.items():
        if sha(path) != expected:
            raise ValueError("FROZEN_DEPENDENCY_CHANGED")
    expected = {i["row"]["announcementId"] for i in plan["items"]}
    paths = sorted((DEST / "documents").glob("*.json"))
    if len(expected) != len(plan["items"]) or expected != {p.stem for p in paths}:
        raise ValueError("COLLECTION_SCOPE_MISMATCH")
    ledger = ROOT / "information-research/20260928-v1/fit-ledger"
    budget = read(ledger.parent / "budget-authorized.json")
    consumed = sorted(ledger.glob("*.json"))
    # 已登记失败和复现同样占额度；不能使用授权文件里最初的 consumed=0。
    total_fits = budget["previous_actual_fits"] + len(consumed)
    remaining = min(12, budget["authorized"] - len(consumed), 82 - total_fits)
    if remaining < 0 or len(consumed) != 12 or total_fits != 70:
        raise ValueError("FIT_LEDGER_REQUIRES_RECONCILIATION")
    requests = sorted((DEST / "requests").glob("*.json"))
    if len(requests) > plan["maximum_new_requests"]:
        raise ValueError("SOURCE_REQUEST_BUDGET_EXCEEDED")
    originals, rows = {}, []
    for path in paths:
        doc = read(path)
        if doc["row"]["published_date"] >= "2025-01-01":
            raise ValueError("SEALED_SCOPE_NOT_ALLOWED")
        failures = []
        if not doc["body_saved"]:
            failures.append(doc["reason"])
        else:
            receipt = doc["receipt"]
            original = receipt["path"]
            if original not in originals:
                originals[original] = sha(original)
            if originals[original] != receipt["sha256"]:
                raise ValueError("ORIGINAL_BYTES_CHANGED")
            if not doc.get("identity", {}).get("passed"):
                failures.append("AUTO_IDENTITY_NOT_PASSED_PRIOR_PROOFS_REQUIRE_MERGE")
            failures.extend(doc.get("revision_issues", []))
            if not doc.get("semantic_verified"):
                failures.append("FULL_SEMANTIC_COVERAGE_NOT_ESTABLISHED")
        rows.append(
            {
                "document_id": path.stem,
                "company": doc["row"]["secCode"],
                "published_date": doc["row"]["published_date"],
                "category": doc["category"],
                "source_record": str(path),
                "source_record_sha256": sha(path),
                "body_saved": doc["body_saved"],
                "unresolved": failures,
                "training_eligible": False,
            }
        )
    prior = sorted((ROOT / "information-research").glob("*/semantic-review-final.json"))
    prior += [OUT / "v25-review-final.json", OUT / "independent-copies/review-final.json"]
    reviews = {str(p): sha(p) for p in prior if p.exists()}
    current = read(DEST / "collection-result.json")
    result = {
        "at": datetime.now().astimezone().isoformat(),
        "collection": current,
        "planned_set_equals_saved_set": True,
        "original_hashes_verified": len(originals),
        "requests_consumed": len(requests),
        "request_limit": plan["maximum_new_requests"],
        "fit_ledger_entries": len(consumed),
        "cumulative_fits": total_fits,
        "remaining_shared_fits": remaining,
        "new_fits": 0,
        "previous_semantic_reviews_preserved": reviews,
        "fit_ledger_sha256": {str(p): sha(p) for p in consumed},
        "blocking_reasons": dict(
            Counter(
                reason if isinstance(reason, str) else json.dumps(reason, ensure_ascii=False, sort_keys=True)
                for row in rows
                for reason in row["unresolved"]
            )
        ),
        "rows": rows,
        "fit_execution_allowed": False,
        "decision": "完整清单已处理；正文缺失、身份/版本与语义覆盖尚未闭合，不得冻结为合格训练输入。",
        "limits": [
            "旧人工审核保持原有字段和页范围效力，本次未作完整合并。",
            "政策与行业新闻正文及历史覆盖仍待核验；目录完整不能代替这些条件。",
            "没有重试已停止来源、读取封存答案或重新领取拟合预算。",
        ],
    }
    target = OUT / "admission-audit-v2" / (datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    save(target, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"rows", "previous_semantic_reviews_preserved", "fit_ledger_sha256"}
            },
            ensure_ascii=False,
        )
    )
    print(str(target))


if __name__ == "__main__":
    run()
