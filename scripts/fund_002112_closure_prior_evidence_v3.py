"""把原批次的指定身份/字段审核按原件摘要合并；不改旧结论、不请求网络、不读取研究答案。"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from app.services.fund_information_history_v1 import ROOT, read, save, sha

OUT = ROOT / "closure/20260929-v1"
DOCS = OUT / "company-bodies/documents"


def normalized(value):
    """仅去除空白以重放旧文本锚点，不改字形、符号、金额或计量单位。"""
    return re.sub(r"\s+", "", str(value))


def nodes(value, pointer=""):
    """记录 JSON 指针，使每项原结论都能定位回不可变审核文件。"""
    if isinstance(value, dict):
        if "document_id" in value:
            yield pointer, value
        for key, item in value.items():
            yield from nodes(item, pointer + "/" + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from nodes(item, pointer + "/" + str(index))


def identity_passed(value, pointer):
    """只沿用明确的身份判定，不把标题单独通过或字段存在当成完整身份通过。"""
    if any(value.get(key) is True for key in ("identity_passed", "source_identity_verified")):
        return True
    if "identit" not in pointer:
        return False
    if value.get("passed") is True:
        return True
    return any(
        isinstance(value.get(key), dict) and value[key].get("passed") is True
        for key in ("initial", "direct", "supplement")
    )


def anchors_passed(value, pages):
    anchors = value.get("anchors", [])
    if isinstance(value.get("anchor"), dict):
        anchors = [*anchors, value["anchor"]]
    if not anchors and value.get("quote") and value.get("page"):
        anchors = [{"page": value["page"], "text": value["quote"]}]
    for anchor in anchors:
        if not isinstance(anchor, dict) or not isinstance(anchor.get("page"), int) or not anchor.get("text"):
            return False
        page = anchor["page"]
        if not 1 <= page <= len(pages) or normalized(anchor["text"]) not in normalized(pages[page - 1]):
            return False
    # 未提供锚点只表示依赖原审核的来源绑定，不能声称重新审核语义。
    return True


def run():
    if (OUT / ".company-bodies.lock").exists():
        raise ValueError("COLLECTION_STILL_RUNNING")
    audits = sorted((OUT / "admission-audit-v2").glob("*.json"))
    previous = read(audits[-1])
    plan = read(OUT / "company-bodies/plan.json")
    for path, expected in {**plan["source_hashes"], **plan["code_hashes"]}.items():
        if sha(path) != expected:
            raise ValueError("FROZEN_DEPENDENCY_CHANGED")
    batches = sorted((ROOT / "information-research").glob("20260929-earnings-v*"))
    directories = [batch / "documents" for batch in batches]
    directories += [OUT / "v25-documents", DOCS]
    # v25 原审核的正文清单也在原根目录的登记文件中；只检索文档目录，绝不遍历答案目录。
    reviews = [
        p
        for batch in batches
        for p in (batch / "semantic-review-final.json", batch / "prior-identity-resolution.json")
        if p.exists()
    ]
    reviews += [OUT / "v25-review-final.json"]
    evidence, unresolved, hashes = defaultdict(list), [], {}
    originals = {}

    @lru_cache(maxsize=128)
    def document(path):
        return read(path)

    def verify_original(receipt):
        path = receipt.get("path")
        if not path or not receipt.get("sha256") or not Path(path).is_file():
            return False
        if path not in originals:
            originals[path] = sha(path)
        return originals[path] == receipt["sha256"]

    for review in reviews:
        data = read(review)
        hashes[str(review)] = sha(review)
        review_plan = (
            OUT / "v25-review-plan.json"
            if review.name == "v25-review-final.json"
            else review.parent / "semantic-review-plan.json"
        )
        frozen_documents = {}
        if review_plan.exists():
            hashes[str(review_plan)] = sha(review_plan)
            frozen_documents = read(review_plan).get("document_hashes", {})
        for pointer, value in nodes(data):
            doc_id = str(value["document_id"])
            if not re.fullmatch(r"\d+", doc_id) or not (DOCS / (doc_id + ".json")).exists():
                continue
            current = document(DOCS / (doc_id + ".json"))
            if current["row"]["published_date"] >= "2025-01-01":
                raise ValueError("SEALED_DOCUMENT_SCOPE")
            source = value.get("source") or value.get("receipt") or {}
            expected = value.get("source_sha256") or (source.get("sha256") if isinstance(source, dict) else None)
            review_documents = (
                ROOT / "information-research/20260929-earnings-v25/documents"
                if review.name == "v25-review-final.json"
                else review.parent / "documents"
            )
            candidates = [review_documents / (doc_id + ".json")]
            candidates += [directory / (doc_id + ".json") for directory in directories]
            matched = None
            for candidate in dict.fromkeys(candidates):
                if not candidate.exists():
                    continue
                old = document(candidate)
                row = old.get("row", {})
                receipt = old.get("receipt", {})
                if not old.get("body_saved") or not old.get("pages"):
                    continue
                if any(row.get(k) != current["row"].get(k) for k in ("announcementId", "secCode", "published_date")):
                    continue
                if expected and expected != receipt.get("sha256"):
                    continue
                # 无直接摘要的旧身份记录仅绑定它所在批次的原文，不推定另一批同编号文件相同。
                if not expected and candidate != candidates[0]:
                    continue
                frozen_hash = frozen_documents.get(str(candidate))
                if frozen_hash and sha(candidate) != frozen_hash:
                    raise ValueError("PRIOR_REVIEW_DOCUMENT_CHANGED")
                if not verify_original(receipt) or not anchors_passed(value, old["pages"]):
                    continue
                matched = (candidate, old)
                break
            item = {"review": str(review), "review_sha256": hashes[str(review)], "pointer": pointer}
            if matched is None:
                # 汇总性观察、无原件绑定、已记录停止结论照旧保存，不作通过升级。
                unresolved.append({**item, "document_id": doc_id, "reason": "NO_EXACT_SOURCE_AND_ANCHOR_BINDING"})
                continue
            path, old = matched
            evidence[doc_id].append(
                {
                    **item,
                    "source_record": str(path),
                    "source_record_sha256": sha(path),
                    "review_frozen_document_hash_verified": str(path) in frozen_documents,
                    "original_sha256": old["receipt"]["sha256"],
                    "original": old["receipt"]["path"],
                    "identity_passed": identity_passed(value, pointer),
                    "anchors_replayed": bool(value.get("anchors") or value.get("anchor") or value.get("quote")),
                    "field_review_only": bool(value.get("metric") or value.get("money") or value.get("unit")),
                    "full_semantic_admission": False,
                    "prior_training_ready": value.get("training_ready"),
                    "prior_record": value,
                }
            )
    rows = []
    for previous_row in previous["rows"]:
        doc_id = previous_row["document_id"]
        current = document(DOCS / (doc_id + ".json"))
        prior = evidence[doc_id]
        receipt = current.get("receipt") or {}
        same = [p for p in prior if p["original_sha256"] == receipt.get("sha256")]
        if not receipt:
            same = prior  # 独立留存的原件仍可核对，不移除原来源停止登记，也不赋予重新抓取权限。
        identity = bool(current.get("identity", {}).get("passed")) or any(p["identity_passed"] for p in same)
        rows.append(
            {
                **previous_row,
                "original_collection_body_saved": current["body_saved"],
                "prior_original_available": bool(prior),
                "identity_after_exact_merge": identity,
                "prior_identity_restored": not current.get("identity", {}).get("passed", False) and identity,
                "prior_evidence": prior,
                "same_original_field_review_count": sum(p["field_review_only"] for p in same),
                "collection_stop_reason_preserved": current.get("reason"),
                "revision_issues_preserved": current.get("revision_issues", []),
                "full_semantics_verified": False,
                "training_eligible": False,
            }
        )
    result = {
        "at": datetime.now().astimezone().isoformat(),
        "audit_code_sha256": sha(__file__),
        "previous_audit": str(audits[-1]),
        "previous_sha256": sha(audits[-1]),
        "source_reviews": hashes,
        "rows": rows,
        "unbound_prior_records": unresolved,
        "summary": {
            "planned": len(rows),
            "bound_evidence_records": sum(len(r["prior_evidence"]) for r in rows),
            "documents_with_bound_evidence": sum(bool(r["prior_evidence"]) for r in rows),
            "prior_identity_restored": sum(r["prior_identity_restored"] for r in rows),
            "identity_after_exact_merge": sum(r["identity_after_exact_merge"] for r in rows),
            "prior_raw_available_despite_collection_stop": sum(
                not r["original_collection_body_saved"] and r["prior_original_available"] for r in rows
            ),
            "unique_original_hashes_reverified": len(originals),
            "unbound_prior_records": len(unresolved),
            "categories": dict(Counter(read(r["source_record"])["category"] for r in rows)),
        },
        "new_network_requests": 0,
        "new_fits": 0,
        "cumulative_fits": previous["cumulative_fits"],
        "remaining_shared_fits": previous["remaining_shared_fits"],
        "fit_execution_allowed": False,
        "decision": "旧指定身份和字段证据已按原件合并保留；必要历史事实、版本关系、政策新闻覆盖仍未全准入，不能训练。",
        "limits": [
            "本次重新核验原件摘要和已保存文本锚点，不冒称重新目视或全文语义审核。",
            "旧来源停止、资源限制、旧候选失败与封存边界全部保留；可读取旧原件不等于允许重试来源。",
            "字段审核通过不能替代完整历史变化链和会计口径比较，未绑定记录仍指向原审核，不被删除或判作失效。",
        ],
    }
    target = OUT / "admission-audit-v3" / (datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    save(target, result)
    print(json.dumps({"path": str(target), **result["summary"], "fit_execution_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    run()
