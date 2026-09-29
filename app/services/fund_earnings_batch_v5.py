"""有限补采第五版：执行已预览的完整缺口组，下载前强制核对旧停止来源。

复用第四版的目录、日期、PDF 和覆盖复算，不修改旧实现。新计划分别记录
目录覆盖沿革和可复用正文来源，定向参考目录不冒充训练历史完整窗口。
"""

import json
from datetime import datetime
from pathlib import Path

import pypdfium2 as pdfium

from app.services.fund_earnings_batch_v4 import (
    OLD,
    PYTHON,
    QUERY,
    ROOT,
    RUNS,
    ZONE,
    BoundedPublicReader,
    EarningsBatch,
    catalog_admissions,
    catalog_rows,
    digest,
    extract_pdf,
    issuer_identity,
    merge_windows,
    pdf_revision,
    read,
    run_path,
    save,
    sha,
)


def window_stop_collisions(windows, entries):
    """已知问题的同公司公开日落入窗口时返回交集，不从覆盖池删除该公司。"""
    return [
        {"window": w, "source": e}
        for w in windows
        for e in entries
        if w["stock"] == e["stock"] and w["start"] <= e["published_date"] <= w["end"]
    ]


def document_stop(row, entries):
    """原件身份和网址任一命中即停止，避免换标题、换编号或复用缓存绕过停止。"""
    url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
    return [e for e in entries if e["document_id"] == row["announcementId"] or e.get("url") == url]


class EarningsBatchV5(EarningsBatch):
    """新版本只扩展来源保护；原请求上限、缺失处理、分类和验证门槛不变。"""

    def prepare(self, previous_name):
        if (self.out / "plan.json").exists():
            plan = self.check_plan()
            if Path(plan["previous_run"]).name != previous_name:
                raise ValueError("RESUME_PREDECESSOR_CHANGED")
            return {"reused_frozen_plan": True, "windows": len(plan["windows"])}
        previous = run_path(previous_name)
        if previous == self.out:
            raise ValueError("SELF_PREDECESSOR")
        decision = read(previous / "final-decision.json")
        if not decision["scope_complete"] or decision["new_fits"] != 0:
            raise ValueError("PREVIOUS_SCOPE_NOT_COMPLETE")
        preview_path = previous / "next-safe-catalog-preview.json"
        preview = read(preview_path)
        if preview["requests_executed"] or preview["new_fits"] or len(preview["windows"]) > 40:
            raise ValueError("INVALID_PREVIEW_SCOPE")
        entries = read(previous / "source-stop-registry.json")["entries"]
        windows = merge_windows(preview["windows"])
        collisions = window_stop_collisions(windows, entries)
        if collisions:
            raise ValueError("PREVIEW_INTERSECTS_PRIOR_STOP")
        # 沿革只纳入兼容的公司完整目录；v18 定向参考仅可复用已保存 PDF。
        owner = Path(decision["coverage_owner_run"])
        if owner.resolve().parent != RUNS.resolve():
            raise ValueError("COVERAGE_OWNER_OUTSIDE_RESEARCH")
        ancestor_plan = read(owner / "plan.json")
        ancestors = list(dict.fromkeys([*ancestor_plan["ancestor_runs"], str(owner)]))
        if any(Path(p).resolve().parent != RUNS.resolve() for p in ancestors):
            raise ValueError("ANCESTOR_OUTSIDE_RESEARCH")
        if read(previous / "row-coverage.json") != read(owner / "row-coverage.json"):
            raise ValueError("REFERENCE_ONLY_PREDECESSOR_COVERAGE_CHANGED")
        protected = dict(read(previous / "protection-before.json")["files"])
        for path, h in read(previous / "delivery-manifest.json")["files"].items():
            if path in protected and protected[path] != h:
                raise ValueError("PROTECTION_MANIFEST_CONFLICT")
            protected[path] = h
        if any(sha(p) != h for p, h in protected.items()):
            raise ValueError("PROTECTED_EVIDENCE_CHANGED")
        for p in previous.rglob("*"):
            if p.is_file():
                protected.setdefault(str(p), sha(p))
        ledger = list((OLD / "fit-ledger").glob("*.json"))
        if len(ledger) != 12 or not all(read(p)["budget_consumed"] for p in ledger):
            raise ValueError("FIT_LEDGER_CHANGED")
        code = dict(read(previous / "semantic-review-plan.json")["code_hashes"])
        for n in [
            "app/services/fund_earnings_batch_v5.py",
            "scripts/fund_002112_earnings_data_v5.py",
            "tests/test_fund_earnings_batch_v5.py",
        ]:
            code[str(PYTHON / n)] = sha(PYTHON / n)
        plan = {
            "schema": "SOURCE_STOP_AWARE_EARNINGS_BATCH_V5",
            "created_at": datetime.now(ZONE).isoformat(),
            "previous_run": str(previous),
            "coverage_owner_run": str(owner),
            "ancestor_runs": ancestors,
            "additional_pdf_cache_runs": [str(previous)],
            "selected": preview["selected"],
            "unmerged_windows": preview["windows"],
            "windows": windows,
            "rejected_groups": preview["rejected_groups"],
            "selection_basis": preview["selection_basis"],
            "baseline_complete": decision["catalog_complete"],
            "source": "Free public CNINFO catalogs and original PDFs only",
            "scope": "DATA_ONLY_NO_TRAINING_PROTOCOL",
            "maximum_target_record_gain_if_complete": preview["maximum_target_record_gain_if_complete"],
            "limits": {"catalog": 150, "body": 100},
            "maximum_documents": 100,
            "maximum_pages_per_window": 10,
            "maximum_pdf_bytes": 25000000,
            "maximum_pdf_pages": 500,
            "new_fit_budget": 0,
            "cumulative_fits": 70,
            "old_protocol_remaining_fits": 12,
            "source_stop_entries": entries,
            "acceptance": [
                "完整分页与目录公开日期核验",
                "旧停止原件不得重新下载或靠缓存绕过",
                "完整30日窗口与原记录键不变，缺失不填零、不缩短",
                "正文身份、修订和语义单独验收",
                "达到请求、原件、大小或页数上限保存停止，不扩额",
                "不拟合、不采用、旧候选保持停止",
            ],
            "code_hashes": code,
            "source_hashes": {
                str(p): sha(p)
                for p in [
                    preview_path,
                    previous / "source-stop-registry.json",
                    previous / "resource-stop.json",
                    previous / "row-coverage.json",
                    owner / "plan.json",
                    OLD / "event-admission/row-coverage.json",
                    ROOT / "supplement/cninfo-stock-map.json",
                ]
            },
        }
        save(self.out / "protection-before.json", {"files": protected})
        save(self.out / "plan.json", plan)
        save(
            self.out / "source-stop-preflight.json",
            {
                "known_stop_entries": len(entries),
                "collisions": [],
                "old_fit_ledger_files": {str(p): sha(p) for p in ledger},
                "new_fits": 0,
                "preview_sha256": sha(preview_path),
                "protected_files": len(protected),
            },
        )
        return {
            "prepared": True,
            "selected": plan["selected"],
            "windows": len(windows),
            "limits": plan["limits"],
            "new_fits": 0,
            "old_stops": len(entries),
        }

    def cached_sources(self, plan):
        cached = super().cached_sources(plan)
        for directory in plan["additional_pdf_cache_runs"]:
            for path in (Path(directory) / "receipts").glob("*.json"):
                r = read(path)
                if r["ok"] and r.get("params") is None:
                    cached[r["url"]].append(r)
        return cached

    def collect(self):
        plan = self.check_plan()
        if (self.out / "collection-stop.json").exists():
            raise ValueError("BATCH_STOPPED_USE_AUDIT")
        stocks = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
        reader = BoundedPublicReader(self.out, plan["limits"])
        cached = self.cached_sources(plan)
        try:
            for i, window in enumerate(plan["windows"], 1):
                target = self.out / "catalogs" / (digest(window) + ".json")
                if target.exists():
                    continue
                pages, receipts = [], []
                try:
                    page, count = 1, 1
                    while page <= count:
                        stock = stocks[window["stock"]]
                        params = {
                            "pageNum": page,
                            "pageSize": 30,
                            "column": "szse",
                            "tabName": "fulltext",
                            "plate": "",
                            "stock": stock["code"] + "," + stock["orgId"],
                            "searchkey": "",
                            "secid": "",
                            "category": "",
                            "trade": "",
                            "sortName": "",
                            "sortType": "",
                            "seDate": window["start"] + "~" + window["end"],
                            "isHLtitle": "true",
                        }
                        raw, receipt = reader.fetch(QUERY, params=params, group="catalog")
                        response = json.loads(raw)
                        receipts.append(receipt)
                        count = max(1, (int(response["totalAnnouncement"]) + 29) // 30)
                        if count > plan["maximum_pages_per_window"]:
                            raise ValueError("WINDOW_PAGE_LIMIT")
                        pages.append((page, response))
                        page += 1
                    rows = catalog_rows(pages, window["stock"], window["start"], window["end"])
                    result = {"window": window, "catalog_complete": True, "rows": rows, "receipts": receipts}
                except (ValueError, KeyError, TypeError) as exc:
                    result = {"window": window, "catalog_complete": False, "reason": str(exc), "receipts": receipts}
                save(target, result)
                print(
                    json.dumps(
                        {
                            "catalog": i,
                            "total": len(plan["windows"]),
                            "passed": result["catalog_complete"],
                            "rows": len(result.get("rows", [])),
                        }
                    ),
                    flush=True,
                )
            work, exclusions = catalog_admissions([read(p) for p in sorted((self.out / "catalogs").glob("*.json"))])
            save(self.out / "title-exclusions.json", exclusions)
            save(self.out / "body-worklist.json", work)
            intersections = [
                {"row": row, "sources": document_stop(row, plan["source_stop_entries"])}
                for row in work
                if document_stop(row, plan["source_stop_entries"])
            ]
            save(self.out / "body-stop-intersections.json", intersections)
            for i, row in enumerate(work, 1):
                target = self.out / "documents" / (row["announcementId"] + ".json")
                if target.exists():
                    continue
                try:
                    if document_stop(row, plan["source_stop_entries"]):
                        raise ValueError("PRIOR_SOURCE_STOP_NO_REFETCH_OR_CACHE_BYPASS")
                    if i > plan["maximum_documents"]:
                        raise ValueError("DOCUMENT_COUNT_LIMIT")
                    url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
                    previous = cached.get(url, [])
                    if previous:
                        if len({r["sha256"] for r in previous}) != 1:
                            raise ValueError("CACHED_DOCUMENT_VERSION_CONFLICT")
                        receipt = previous[0]
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
                        "reused": bool(previous),
                        "pages": pages,
                        "metadata": metadata,
                        "identity": issuer_identity(row, pages),
                        "revision_issues": pdf_revision(metadata, row["published_date"]),
                        "semantic_verified": False,
                    }
                except (ValueError, pdfium.PdfiumError) as exc:
                    value = {"row": row, "body_saved": False, "reason": str(exc), "semantic_verified": False}
                save(target, value)
                print(
                    json.dumps(
                        {
                            "body": i,
                            "total": len(work),
                            "saved": value["body_saved"],
                            "reused": value.get("reused", False),
                        }
                    ),
                    flush=True,
                )
                if value.get("reason", "").startswith(("DOCUMENT_COUNT_LIMIT", "PUBLIC_REQUEST_LIMIT")):
                    break
        finally:
            reader.close()
        save(
            self.out / "collection-stop.json",
            {"do_not_resume_collect": True, "reason": "BOUNDED_SCOPE_COMPLETE_OR_RESOURCE_LIMIT", "new_fits": 0},
        )
        return self.audit()
