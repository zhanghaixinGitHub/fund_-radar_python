"""第四版有限业绩资料驱动：下载前核对用途，保留所有原始目录记录。

只导入资料工具；所有输出排他保存。分离目录、原件身份与语义状态，不读取
封存标签、不调用拟合，也不把旧协议剩余额度自动转给新假设。
"""

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pypdfium2 as pdfium

from app.services.fund_earnings_evidence_v1 import plan_groups
from app.services.fund_earnings_history_v2 import identity, merge_windows
from app.services.fund_earnings_titles_v3 import classify_catalog_row, identity_matching_text
from app.services.fund_information_history_v1 import (
    OLD,
    QUERY,
    ROOT,
    ZONE,
    BoundedPublicReader,
    catalog_rows,
    digest,
    normalize,
    pdf_revision,
    read,
    save,
    sha,
)

RUNS = ROOT / "information-research"
PYTHON = Path(__file__).resolve().parents[2]


def run_path(name):
    """只允许研究根目录下的显式版本名；不能用路径跳转或复用旧轮次目录。"""
    if not re.fullmatch(r"\d{8}-earnings-v[1-9]\d*", name):
        raise ValueError("INVALID_DATA_BATCH_NAME")
    path = (RUNS / name).resolve()
    if path.parent != RUNS.resolve():
        raise ValueError("DATA_BATCH_OUTSIDE_ROOT")
    return path


def issuer_identity(row, pages):
    """正文明确列出另一证券代码时拦截；不能因附件里偶然提到母公司而归错主体。"""
    result = identity(
        {**row, "title_plain": identity_matching_text(row["title_plain"])},
        [identity_matching_text(p) for p in pages],
    )
    result["original_catalog_title"] = row["title_plain"]
    first = "\n".join(pages[:2])
    codes = sorted(set(re.findall(r"(?:证\s*券|股\s*票)\s*代\s*码\s*[：:]?\s*(\d{6})(?!\d)", first)))
    if codes and row["secCode"] not in codes:
        result.update({"passed": False, "reason": "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG"})
    return {**result, "explicit_body_codes": codes, "semantic_verified": False}


def catalog_admissions(catalogs):
    """以公告身份去重并核对目录元数据，再生成可审计的候选与排除清单。

    同一原件跨窗口重复出现时必须同主体、同网址、同标题、同公开日；
    包括被排除的原件也不得因分类差异掩盖元数据冲突。
    """
    rows = {}
    for catalog in catalogs:
        if not catalog["catalog_complete"]:
            continue
        for row in catalog["rows"]:
            key = row["announcementId"]
            old = rows.get(key)
            if old and any(old[k] != row[k] for k in ("secCode", "adjunctUrl", "title_plain", "published_date")):
                raise ValueError("DUPLICATE_DOCUMENT_METADATA_CONFLICT")
            rows[key] = row
    entries, exclusions = [], []
    for key in sorted(rows):
        row = rows[key]
        admission = classify_catalog_row(row)
        if admission["kind"]:
            entries.append({**row, "earnings_kind": admission["kind"], "title_admission": admission})
        elif admission["previous_kind"]:
            exclusions.append({"row": row, "admission": admission})
    order = {"FORECAST": 0, "PRELIMINARY_RESULT": 1, "CORRECTION_NOTICE": 2, "REPORTED_RESULT": 3}
    entries.sort(key=lambda r: (order[r["earnings_kind"]], r["published_date"], r["announcementId"]))
    return entries, exclusions


def day_set(lo, hi):
    start, end = date.fromisoformat(lo), date.fromisoformat(hi)
    return {str(start + timedelta(days=n)) for n in range((end - start).days + 1)}


def extract_pdf(raw, maximum_pages):
    """对实际原件独立提取全文并关闭页资源；页数超限时不截断冒充完整正文。"""
    if not raw.startswith(b"%PDF"):
        raise ValueError("NOT_PDF_CONTENT")
    with pdfium.PdfDocument(raw) as doc:
        if len(doc) > maximum_pages:
            raise ValueError("PDF_PAGE_LIMIT")
        pages, metadata = [], doc.get_metadata_dict()
        for page in doc:
            text = page.get_textpage()
            try:
                pages.append(text.get_text_range())
            finally:
                text.close()
                page.close()
    return pages, metadata


class EarningsBatch:
    """一个实例对应一个冻结目录；前序目录由计划固定，恢复不重新排序或换数据。"""

    def __init__(self, name):
        self.out = run_path(name)

    def prepare(self, previous_name):
        if (self.out / "plan.json").exists():
            saved = self.check_plan()
            if Path(saved["previous_run"]).name != previous_name:
                raise ValueError("RESUME_PREDECESSOR_CHANGED")
            return {"reused_frozen_plan": True, "windows": len(saved["windows"])}
        previous = run_path(previous_name)
        if previous == self.out:
            raise ValueError("SELF_PREDECESSOR")
        decision = read(previous / "final-decision.json")
        if not decision["scope_complete"] or decision["new_fits"] != 0:
            raise ValueError("PREVIOUS_DATA_BATCH_NOT_COMPLETE")
        protected = dict(read(previous / "protection-before.json")["files"])
        for p, expected in read(previous / "delivery-manifest.json")["files"].items():
            if p in protected and protected[p] != expected:
                raise ValueError("PROTECTION_MANIFEST_CONFLICT")
            protected[p] = expected
        for p, expected in protected.items():
            if sha(p) != expected:
                raise ValueError("PROTECTED_EVIDENCE_CHANGED:" + p)
        for p in previous.rglob("*"):
            if p.is_file():
                protected.setdefault(str(p), sha(p))
        ledger = sorted((OLD / "fit-ledger").glob("*.json"))
        if len(ledger) != 12 or not all(read(p)["budget_consumed"] for p in ledger):
            raise ValueError("FIT_LEDGER_NEEDS_RECONCILIATION")
        previous_plan = read(previous / "plan.json")
        ancestors = previous_plan.get("ancestor_runs", previous_plan.get("prior_catalog_runs"))
        if not ancestors:
            raise ValueError("PREVIOUS_CATALOG_LINEAGE_MISSING")
        ancestors = list(dict.fromkeys(ancestors + [str(previous)]))
        if any(Path(p).resolve().parent != RUNS.resolve() for p in ancestors):
            raise ValueError("ANCESTOR_OUTSIDE_RESEARCH_ROOT")
        rows = read(previous / "row-coverage.json")
        plan = plan_groups(read(OLD / "event-admission/row-coverage.json"), rows, 4, 40)
        plan["unmerged_windows"] = plan["windows"]
        plan["windows"] = merge_windows(plan["windows"])
        dependencies = [
            PYTHON / p
            for p in (
                "app/services/fund_earnings_batch_v4.py",
                "scripts/fund_002112_earnings_data_v4.py",
                "tests/test_fund_earnings_batch_v4.py",
                "app/services/fund_earnings_evidence_v1.py",
                "app/services/fund_earnings_history_v2.py",
                "app/services/fund_earnings_semantics_v2.py",
                "app/services/fund_earnings_titles_v3.py",
                "app/services/fund_information_history_v1.py",
            )
        ]
        plan.update(
            {
                "schema": "REUSABLE_EARNINGS_DATA_BATCH_V4",
                "created_at": datetime.now(ZONE).isoformat(),
                "previous_run": str(previous),
                "ancestor_runs": ancestors,
                "baseline_complete": sum(r["catalog_complete"] for r in rows),
                "scope": "DATA_ONLY_NO_TRAINING_PROTOCOL",
                "new_fit_budget": 0,
                "cumulative_fits": decision["cumulative_fits"],
                "old_protocol_remaining_fits": 12,
                "limits": {"catalog": 150, "body": 100},
                "maximum_documents": 100,
                "maximum_pages_per_window": 10,
                "maximum_pdf_bytes": 25_000_000,
                "maximum_pdf_pages": 500,
                "source": "Free public CNINFO catalogs and original PDFs only",
                "selection_basis": "Complete original records per missing company window; no outcome reads",
                "title_scope": (
                    "Frozen V3 excludes meetings/corporate returns/verified different report subjects; "
                    "correction notices remain separate"
                ),
                "document_order": "Forecast, preliminary, correction, reported result; date/id ascending",
                "semantic_scope": "Candidate inventory only unless separately source-anchored and visually reviewed",
                "stops": [
                    "No retry of failed or interrupted requests",
                    "No fills for missing facts",
                    "Identity/date/page conflicts stop the affected source",
                    "Hard request/document/size limits",
                    "Old hypotheses remain stopped; no new fitting or adoption",
                ],
                "code_hashes": {str(p): sha(p) for p in dependencies},
                "source_hashes": {
                    str(p): sha(p)
                    for p in (
                        previous / "row-coverage.json",
                        OLD / "event-admission/row-coverage.json",
                        ROOT / "supplement/cninfo-stock-map.json",
                    )
                },
            }
        )
        save(self.out / "protection-before.json", {"files": protected})
        save(self.out / "plan.json", plan)
        return {"selected": plan["selected"], "windows": len(plan["windows"]), "limits": plan["limits"], "fits": 0}

    def check_plan(self):
        plan = read(self.out / "plan.json")
        for p, expected in {**plan["source_hashes"], **plan["code_hashes"]}.items():
            if sha(p) != expected:
                raise ValueError("FROZEN_DEPENDENCY_CHANGED:" + p)
        return plan

    def cached_sources(self, plan):
        values = defaultdict(list)
        for directory in plan["ancestor_runs"]:
            for path in (Path(directory) / "receipts").glob("*.json"):
                item = read(path)
                if item["ok"] and item.get("params") is None:
                    values[item["url"]].append(item)
        inventory = RUNS / "20260928-history-v1/semantic-inventory.json"
        for item in read(inventory):
            value = item["source"]
            values[value["url"]].append({"ok": True, **value})
        return values

    def collect(self):
        plan = self.check_plan()
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
                    page, total_pages = 1, 1
                    while page <= total_pages:
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
                        total_pages = max(1, (int(response["totalAnnouncement"]) + 29) // 30)
                        if total_pages > plan["maximum_pages_per_window"]:
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
            for i, row in enumerate(work, 1):
                target = self.out / "documents" / (row["announcementId"] + ".json")
                if target.exists():
                    continue
                try:
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
        return self.audit()

    def audit(self):
        """重放每页响应，以自然日集合复算完整窗口；只投影折的训练成员身份。"""
        plan = self.check_plan()
        protected = read(self.out / "protection-before.json")["files"]
        for p, expected in protected.items():
            if sha(p) != expected:
                raise ValueError("PROTECTED_FILE_CHANGED:" + p)
        used = Counter(read(p)["group"] for p in (self.out / "requests").glob("*.json"))
        if any(v > plan["limits"][k] for k, v in used.items()):
            raise ValueError("REQUEST_BUDGET_EXCEEDED")
        for p in (self.out / "receipts").glob("*.json"):
            receipt = read(p)
            if receipt["ok"] and sha(receipt["path"]) != receipt["sha256"]:
                raise ValueError("PUBLIC_RAW_CHANGED")
        paths = sorted((self.out / "catalogs").glob("*.json"))
        if {p.stem for p in paths} != {digest(w) for w in plan["windows"]}:
            raise ValueError("CATALOG_SCOPE_MISMATCH")
        current = [read(p) for p in paths]
        for c in current:
            if c["catalog_complete"]:
                w = c["window"]
                pages = [(r["params"]["pageNum"], read(r["path"])) for r in c["receipts"]]
                if catalog_rows(pages, w["stock"], w["start"], w["end"]) != c["rows"]:
                    raise ValueError("PAGINATION_REPLAY_MISMATCH")
        work, exclusions = catalog_admissions(current)
        if work != read(self.out / "body-worklist.json") or exclusions != read(self.out / "title-exclusions.json"):
            raise ValueError("TITLE_ADMISSION_REPLAY_MISMATCH")
        bundle = read(read(OLD / "event-admission/bundle-manifest.json")["path"])
        spans = {k: list(v) for k, v in bundle["coverage"]["company"].items()}
        catalogs = list(current)
        for directory in plan["ancestor_runs"]:
            p = Path(directory)
            folder = p / ("company-catalogs" if (p / "company-catalogs").exists() else "catalogs")
            catalogs.extend(read(f) for f in folder.glob("*.json"))
        for c in catalogs:
            if c["catalog_complete"]:
                w = c["window"]
                spans.setdefault(w["stock"], []).append([w["start"], w["end"]])
        days = {s: set().union(*(day_set(lo, hi) for lo, hi in intervals)) for s, intervals in spans.items()}
        rows, missing = [], defaultdict(list)
        for row in read(OLD / "event-admission/row-coverage.json"):
            lo, hi = [str(date.fromisoformat(row["target"]) - timedelta(days=n)) for n in (30, 1)]
            needed = day_set(lo, hi)
            gaps = [s for s in row["missing_companies"] if not needed.issubset(days.get(s, set()))]
            rows.append(
                {
                    "fund_code": row["fund_code"],
                    "target": row["target"],
                    "catalog_complete": row["company_coverage_passed"] or (bool(row["missing_companies"]) and not gaps),
                    "remaining_companies": gaps,
                }
            )
            for stock in gaps:
                missing[stock].append([lo, hi])
        save(self.out / "row-coverage.json", rows)
        gaps = []
        for stock, intervals in sorted(missing.items()):
            merged = []
            for lo, hi in sorted(intervals):
                if merged and lo <= str(date.fromisoformat(merged[-1][1]) + timedelta(days=1)):
                    merged[-1][1] = max(hi, merged[-1][1])
                else:
                    merged.append([lo, hi])
            gaps.extend({"stock": stock, "start": lo, "end": hi} for lo, hi in merged)
        save(self.out / "company-gap-worklist-merged.json", gaps)
        before = {(r["fund_code"], r["target"]): r for r in read(Path(plan["previous_run"]) / "row-coverage.json")}
        after = {(r["fund_code"], r["target"]): r for r in rows}
        folds = []
        for f in read(OLD / "early-admission/folds.json"):
            keys = [tuple(k) for k in f["train_ids"]]
            folds.append(
                {
                    "name": f["name"],
                    "train_rows": len(keys),
                    "before_complete": sum(before[k]["catalog_complete"] for k in keys),
                    "after_complete": sum(after[k]["catalog_complete"] for k in keys),
                }
            )
        save(self.out / "coverage-on-new-folds.json", {"folds": folds, "training_ready": False})
        docs = [read(p) for p in sorted((self.out / "documents").glob("*.json"))]
        identities, hints = [], []
        for d in docs:
            if not d["body_saved"]:
                continue
            if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
                raise ValueError("BODY_SOURCE_CHANGED")
            pages, _ = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), plan["maximum_pdf_pages"])
            if [normalize(p) for p in pages] != [normalize(p) for p in d["pages"]]:
                raise ValueError("BODY_REPLAY_MISMATCH")
            proof = issuer_identity(d["row"], pages)
            if proof != d["identity"]:
                raise ValueError("IDENTITY_REPLAY_MISMATCH")
            identities.append({"id": d["row"]["announcementId"], **proof})
            hits = [
                i + 1
                for i, p in enumerate(pages)
                if re.search(r"业绩(?:预告|预增|预减|预盈|预亏)|经营业绩的预计", normalize(p))
            ]
            if hits:
                hints.append(
                    {
                        "id": d["row"]["announcementId"],
                        "stock": d["row"]["secCode"],
                        "title": d["row"]["title_plain"],
                        "pages": hits,
                        "status": "UNREVIEWED_HINT",
                    }
                )
        save(self.out / "identity-audit.json", identities)
        save(self.out / "body-forecast-hints.json", hints)
        ledger = list((OLD / "fit-ledger").glob("*.json"))
        if len(ledger) != 12:
            raise ValueError("OLD_FIT_LEDGER_CHANGED")
        result = {
            "protected_files": len(protected),
            "requests": dict(used),
            "request_limits": plan["limits"],
            "catalog_windows_passed": sum(c["catalog_complete"] for c in current),
            "catalog_windows_attempted": len(current),
            "catalog_rows": sum(len(c.get("rows", [])) for c in current),
            "baseline_complete": plan["baseline_complete"],
            "catalog_complete": sum(r["catalog_complete"] for r in rows),
            "by_fund": dict(Counter(r["fund_code"] for r in rows if r["catalog_complete"])),
            "remaining_gap_companies": len(missing),
            "remaining_gap_intervals": len(gaps),
            "body_worklist": len(work),
            "title_exclusions": len(exclusions),
            "title_admission_replay_passed": True,
            "bodies_saved": len(identities),
            "identity_passed": sum(i["passed"] for i in identities),
            "reused_bodies": sum(d.get("reused", False) for d in docs),
            "metadata_conflicts": sum(bool(d.get("revision_issues")) for d in docs),
            "folds": folds,
            "old_fit_ledger_entries": len(ledger),
            "new_fits": 0,
            "cumulative_fits": plan["cumulative_fits"],
            "training_ready": False,
        }
        save(self.out / "collection-and-audit-result.json", result)
        return result
