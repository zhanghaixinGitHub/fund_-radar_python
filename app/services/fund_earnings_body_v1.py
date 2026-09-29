"""已完成目录的原件补齐：单独登记有限批次，不重启达到上限的旧采集器。

只复用旧目录中确定的缺件顺序，不按预测结果选资料。旧文件、停止记录和
代码均受哈希保护；本模块不导入模型、不读取标签、不改变目录覆盖率。
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pypdfium2 as pdfium

from app.services.fund_earnings_batch_v4 import (
    PYTHON,
    EarningsBatch,
    extract_pdf,
    issuer_identity,
    run_path,
)
from app.services.fund_information_history_v1 import (
    OLD,
    ZONE,
    BoundedPublicReader,
    normalize,
    pdf_revision,
    read,
    save,
    sha,
)


def reconcile_pending(previous):
    """仅允许承接未请求及文档数量上限；网络失败不能借换批次名自动重试。"""
    pending = read(previous / "pending-body-worklist.json")
    original = read(previous / "body-worklist.json")
    expected = []
    for position, row in enumerate(original, 1):
        path = previous / "documents" / (row["announcementId"] + ".json")
        if path.exists():
            document = read(path)
            if document["row"] != row:
                raise ValueError("PREVIOUS_DOCUMENT_METADATA_CHANGED")
            if document["body_saved"]:
                continue
            if document.get("reason") != "DOCUMENT_COUNT_LIMIT":
                raise ValueError("FAILED_SOURCE_REQUIRES_SEPARATE_DECISION")
        expected.append((position, row))
    actual = [(item["position"], item["row"]) for item in pending["items"]]
    if actual != expected or pending["count"] != len(expected):
        raise ValueError("PENDING_WORKLIST_NOT_EXACT_RECONCILIATION")
    if len({row["announcementId"] for _, row in expected}) != len(expected):
        raise ValueError("DUPLICATE_PENDING_DOCUMENT")
    return pending["items"]


def body_purpose(row):
    """审计意见附件保留目录身份，但不当作公司新业绩报告重复计数。"""
    if "审计报告" in row["title_plain"]:
        return "AUDITOR_ATTACHMENT_NOT_COMPANY_EARNINGS_REPORT"
    return None


class BodyContinuation(EarningsBatch):
    def prepare(self, previous_name):
        if (self.out / "plan.json").exists():
            plan = self.check_plan()
            if Path(plan["previous_run"]).name != previous_name:
                raise ValueError("RESUME_PREDECESSOR_CHANGED")
            return {"reused_frozen_plan": True, "limits": plan["limits"]}
        previous = run_path(previous_name)
        if previous == self.out:
            raise ValueError("SELF_PREDECESSOR")
        decision = read(previous / "final-decision.json")
        stop = read(previous / "collection-stop.json")
        continuation = read(previous / "continuation.json")
        if (
            decision["scope_complete"]
            or not stop["do_not_resume_collect"]
            or stop["reason"] != "REGISTERED_MAXIMUM_DOCUMENTS_100_REACHED"
            or decision["new_fits"] != 0
        ):
            raise ValueError("BODY_CONTINUATION_PREREQUISITES_NOT_MET")
        items = reconcile_pending(previous)
        if len(items) > continuation["suggested_next_document_limit"]:
            raise ValueError("BODY_CONTINUATION_SCOPE_EXCEEDED")
        protected = dict(read(previous / "protection-before.json")["files"])
        for path, expected in read(previous / "delivery-manifest.json")["files"].items():
            if path in protected and protected[path] != expected:
                raise ValueError("PROTECTION_MANIFEST_CONFLICT")
            protected[path] = expected
        for path, expected in protected.items():
            if sha(path) != expected:
                raise ValueError("PROTECTED_EVIDENCE_CHANGED:" + path)
        protected.update({str(p): sha(p) for p in previous.rglob("*") if p.is_file()})
        ledger = list((OLD / "fit-ledger").glob("*.json"))
        if len(ledger) != 12 or not all(read(p)["budget_consumed"] for p in ledger):
            raise ValueError("FIT_LEDGER_NEEDS_RECONCILIATION")
        previous_plan = read(previous / "plan.json")
        ancestors = list(dict.fromkeys(previous_plan["ancestor_runs"] + [str(previous)]))
        code = dict(previous_plan["code_hashes"])
        for name in (
            "app/services/fund_earnings_body_v1.py",
            "scripts/fund_002112_earnings_body_v1.py",
            "tests/test_fund_earnings_body_v1.py",
        ):
            code[str(PYTHON / name)] = sha(PYTHON / name)
        plan = {
            "schema": "BOUNDED_BODY_CONTINUATION_V1",
            "created_at": datetime.now(ZONE).isoformat(),
            "scope": "EXISTING_CATALOG_BODIES_ONLY_NO_TRAINING",
            "previous_run": str(previous),
            "ancestor_runs": ancestors,
            "windows": [],
            "items": items,
            "maximum_documents": len(items),
            "limits": {"catalog": 0, "body": continuation["suggested_next_pdf_request_upper_bound"]},
            "maximum_pdf_bytes": 25_000_000,
            "maximum_pdf_pages": 500,
            "purpose_exclusion": "Auditor attachments retained as metadata, not company earnings reports",
            "date_range": [
                min(i["row"]["published_date"] for i in items),
                max(i["row"]["published_date"] for i in items),
            ],
            "baseline_complete": decision["catalog_complete_after"],
            "cumulative_fits": decision["cumulative_fits"],
            "new_fit_budget": 0,
            "old_protocol_remaining_fits": 12,
            "source": "Free static.cninfo.com.cn originals and exact URL verified caches only",
            "selection_basis": "All frozen pending positions, no outcome-based selection",
            "acceptance": [
                "Every position accounted for",
                "Original bytes re-extracted",
                "Identity and PDF dates checked independently",
                "No coverage gain inferred from body downloads",
                "Facts need separate period/unit/anchor review",
            ],
            "stops": [
                "No network failure retry",
                "No budget extension",
                "No old batch mutation",
                "Identity or date conflict blocks source admission",
                "No fitting",
            ],
            "code_hashes": code,
            "source_hashes": {
                str(previous / n): sha(previous / n)
                for n in (
                    "pending-body-worklist.json",
                    "pending-body-cache-assessment.json",
                    "body-worklist.json",
                    "row-coverage.json",
                    "coverage-on-new-folds.json",
                    "company-gap-worklist-merged.json",
                    "collection-stop.json",
                    "continuation.json",
                )
            },
        }
        save(self.out / "protection-before.json", {"files": protected})
        save(self.out / "plan.json", plan)
        return {"documents": len(items), "limits": plan["limits"], "new_fits": 0}

    def collect(self):
        plan = self.check_plan()
        if (self.out / "collection-stop.json").exists():
            raise ValueError("COLLECTION_ALREADY_STOPPED_USE_AUDIT")
        for path, expected in read(self.out / "protection-before.json")["files"].items():
            if sha(path) != expected:
                raise ValueError("PROTECTED_FILE_CHANGED:" + path)
        reader, cached = BoundedPublicReader(self.out, plan["limits"]), self.cached_sources(plan)
        try:
            for item in plan["items"]:
                row = item["row"]
                target = self.out / "documents" / (row["announcementId"] + ".json")
                if target.exists():
                    continue
                reason = body_purpose(row)
                if reason:
                    value = {"row": row, "body_saved": False, "purpose_excluded": True, "reason": reason}
                else:
                    try:
                        url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
                        cache = cached.get(url, [])
                        if cache:
                            if len({r["sha256"] for r in cache}) != 1:
                                raise ValueError("CACHED_DOCUMENT_VERSION_CONFLICT")
                            receipt = cache[0]
                            if sha(receipt["path"]) != receipt["sha256"]:
                                raise ValueError("CACHED_SOURCE_CHANGED")
                            raw = Path(receipt["path"]).read_bytes()
                        else:
                            raw, receipt = reader.fetch(url, group="body", maximum_bytes=plan["maximum_pdf_bytes"])
                        if len(raw) > plan["maximum_pdf_bytes"]:
                            raise ValueError("PDF_SIZE_LIMIT")
                        pages, metadata = extract_pdf(raw, plan["maximum_pdf_pages"])
                        value = {
                            "row": row,
                            "receipt": receipt,
                            "body_saved": True,
                            "reused": bool(cache),
                            "pages": pages,
                            "metadata": metadata,
                            "identity": issuer_identity(row, pages),
                            "revision_issues": pdf_revision(metadata, row["published_date"]),
                            "semantic_verified": False,
                        }
                    except (ValueError, pdfium.PdfiumError) as exc:
                        value = {"row": row, "body_saved": False, "reason": str(exc)}
                save(target, value)
                print(
                    json.dumps(
                        {
                            "position": item["position"],
                            "id": row["announcementId"],
                            "saved": value["body_saved"],
                            "reason": value.get("reason"),
                        }
                    ),
                    flush=True,
                )
        finally:
            reader.close()
        save(
            self.out / "collection-stop.json",
            {
                "reason": "ALL_REGISTERED_BODY_POSITIONS_ACCOUNTED",
                "do_not_resume_collect": True,
                "new_fits": 0,
                "safe_next_command": "audit",
            },
        )
        return self.audit()

    def audit(self):
        """仅重放登记原件及身份，不使用旧 collector、不访问网络、不重新计算标签。"""
        plan = self.check_plan()
        previous = Path(plan["previous_run"])
        if reconcile_pending(previous) != plan["items"]:
            raise ValueError("PENDING_SCOPE_CHANGED")
        protected = read(self.out / "protection-before.json")["files"]
        for path, expected in protected.items():
            if sha(path) != expected:
                raise ValueError("PROTECTED_FILE_CHANGED:" + path)
        used = Counter(read(p)["group"] for p in (self.out / "requests").glob("*.json"))
        if any(k not in plan["limits"] or v > plan["limits"][k] for k, v in used.items()):
            raise ValueError("REQUEST_BUDGET_EXCEEDED")
        paths = sorted((self.out / "documents").glob("*.json"))
        if {p.stem for p in paths} != {i["row"]["announcementId"] for i in plan["items"]}:
            raise ValueError("BODY_POSITIONS_NOT_FULLY_ACCOUNTED")
        docs, identities = [], []
        for item in plan["items"]:
            row = item["row"]
            d = read(self.out / "documents" / (row["announcementId"] + ".json"))
            if d["row"] != row:
                raise ValueError("BODY_METADATA_CHANGED")
            if d.get("purpose_excluded") and d["reason"] != body_purpose(row):
                raise ValueError("PURPOSE_EXCLUSION_CHANGED")
            docs.append(d)
            if not d["body_saved"]:
                continue
            receipt = d["receipt"]
            if (
                receipt["url"] != "https://static.cninfo.com.cn/" + row["adjunctUrl"]
                or sha(receipt["path"]) != receipt["sha256"]
            ):
                raise ValueError("SOURCE_IDENTITY_OR_BYTES_CHANGED")
            pages, metadata = extract_pdf(Path(receipt["path"]).read_bytes(), plan["maximum_pdf_pages"])
            proof = issuer_identity(row, pages)
            if (
                [normalize(p) for p in pages] != [normalize(p) for p in d["pages"]]
                or metadata != d["metadata"]
                or proof != d["identity"]
                or pdf_revision(metadata, row["published_date"]) != d["revision_issues"]
            ):
                raise ValueError("BODY_REPLAY_CHANGED")
            identities.append({"id": row["announcementId"], **proof})
        # 原件补齐没有新增目录日期，复制原来的覆盖事实；不声称多覆盖一条训练记录。
        for name in ("row-coverage.json", "company-gap-worklist-merged.json"):
            save(self.out / name, read(previous / name))
        folds = read(previous / "coverage-on-new-folds.json")
        folds = {**folds, "folds": [{**f, "before_complete": f["after_complete"]} for f in folds["folds"]]}
        save(self.out / "coverage-on-new-folds.json", folds)
        save(self.out / "identity-audit.json", identities)
        result = {
            "positions": len(docs),
            "bodies_saved": len(identities),
            "purpose_excluded": sum(bool(d.get("purpose_excluded")) for d in docs),
            "failed_sources": sum(not d["body_saved"] and not d.get("purpose_excluded") for d in docs),
            "identity_passed": sum(i["passed"] for i in identities),
            "reused_bodies": sum(d.get("reused", False) for d in docs),
            "metadata_conflicts": sum(bool(d.get("revision_issues")) for d in docs),
            "requests": dict(used),
            "request_limits": plan["limits"],
            "protected_files": len(protected),
            "new_catalog_requests": 0,
            "catalog_complete": plan["baseline_complete"],
            "added_catalog_complete_rows": 0,
            "new_fits": 0,
            "cumulative_fits": plan["cumulative_fits"],
            "training_ready": False,
        }
        save(self.out / "collection-and-audit-result.json", result)
        return result
