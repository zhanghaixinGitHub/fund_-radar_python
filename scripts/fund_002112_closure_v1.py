"""按新授权完成剩余目录的有界采集和独立覆盖重算，不调用拟合。

用法：prepare 固定完整范围与总预算；catalogs 可从已保存的窗口继续。
原件、语义、新协议、模型采用分别验收，不把目录完成当作整个任务完成。
"""

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.services.fund_earnings_batch_v4 import PYTHON, RUNS
from app.services.fund_information_history_v1 import (
    QUERY,
    ROOT,
    BoundedPublicReader,
    catalog_rows,
    digest,
    read,
    save,
    sha,
)
from app.services.fund_research_closure_v1 import admission_status, project_coverage, split_windows

OUT = ROOT / "closure/20260929-v1"
PREVIOUS = RUNS / "20260929-earnings-v25"


def prepare():
    """按实际缺口一次冻结全部公司与日期，防止仅补容易的报告组。"""
    if (OUT / "catalog-plan.json").exists():
        return check_plan()
    result = read(PREVIOUS / "collection-and-audit-result.json")
    if result["new_fits"] or not (PREVIOUS / "collection-stop.json").exists():
        raise ValueError("PREDECESSOR_NOT_READY")
    gaps = read(PREVIOUS / "company-gap-worklist-merged.json")
    windows = split_windows(gaps, maximum_days=90)
    mapping = ROOT / "supplement/cninfo-stock-map.json"
    known = read(mapping)["rows"]
    if any(w["stock"] not in known for w in windows):
        raise ValueError("PUBLIC_SOURCE_IDENTITY_MISSING")
    code = [
        PYTHON / "app/services/fund_research_closure_v1.py",
        PYTHON / "scripts/fund_002112_closure_v1.py",
        PYTHON / "app/services/fund_information_history_v1.py",
        PYTHON / "tests/test_fund_research_closure_v1.py",
    ]
    dependencies = [PREVIOUS / "row-coverage.json", PREVIOUS / "company-gap-worklist-merged.json", mapping]
    protected = dict(read(PREVIOUS / "protection-before.json")["files"])
    for p in PREVIOUS.rglob("*"):
        if p.is_file():
            protected[str(p)] = sha(p)
    save(OUT / "protection-before.json", {"files": protected})
    workspace = []
    for repo in [Path("C:/WebStormProject/workSpace05"), PYTHON, Path("C:/ideaProject/workSpace12")]:
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short", "--untracked-files=all"], text=True, encoding="utf-8"
        ).splitlines()
        workspace.append({"path": str(repo), "head": head, "status": status})
    save(OUT / "workspace-before.json", workspace)
    save(
        OUT / "authorization.json",
        {
            "at": datetime.now().astimezone().isoformat(),
            "user_request": "把那剩余的四件事全部做完再停止",
            "scope": ["补剩余历史资料", "完成事实准入", "登记并验证新实验", "合格后采用及实际使用核验"],
            "interpretation": "允许新假设准备；真实拟合总增量最多12次，首次拟合前另冻结协议",
            "previous_cumulative_real_fits": 70,
            "new_fit_maximum": 12,
            "old_candidate_restarts": False,
            "acceptance_thresholds_unchanged": True,
            "automation_remains_paused": True,
            "paid_services_allowed": False,
            "external_writes_allowed": False,
        },
    )
    plan = {
        "at": datetime.now().astimezone().isoformat(),
        "scope": "ALL_REMAINING_COMPANY_CATALOG_WINDOWS_ONLY",
        "source": "Free public CNINFO catalog; no credentials or external writes",
        "previous": str(PREVIOUS),
        "windows": windows,
        "remaining_rows_before": 5576 - result["catalog_complete"],
        "limits": {"catalog": 8000},
        "maximum_pages_per_window": 50,
        "maximum_query_span_days": 90,
        "source_failures_are_not_retried": True,
        "old_source_stops_preserved_at": str(RUNS / "20260929-earnings-v24/source-stop-registry.json"),
        "stopped_pdf_downloads_allowed": False,
        "body_or_semantic_admission_implied": False,
        "code_hashes": {str(p): sha(p) for p in code},
        "source_hashes": {str(p): sha(p) for p in dependencies},
    }
    save(OUT / "catalog-plan.json", plan)
    return {"windows": len(windows), "limits": plan["limits"], "companies": len({w["stock"] for w in windows})}


def check_plan():
    plan = read(OUT / "catalog-plan.json")
    for path, expected in {**plan["source_hashes"], **plan["code_hashes"]}.items():
        if sha(path) != expected:
            raise ValueError("FROZEN_DEPENDENCY_CHANGED:" + path)
    return plan


def collect_catalogs():
    """串行限速采集全部固定窗口；失败留缺口，成功响应恢复时直接复用。"""
    plan = check_plan()
    if (OUT / "catalog-collection-stop.json").exists():
        return audit()
    mapping = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
    reader = BoundedPublicReader(OUT, plan["limits"])
    try:
        for position, window in enumerate(plan["windows"], 1):
            target = OUT / "catalogs" / (digest(window) + ".json")
            if target.exists():
                continue
            receipts, pages = [], []
            try:
                page, count = 1, 1
                while page <= count:
                    identity = mapping[window["stock"]]
                    params = {
                        "pageNum": page,
                        "pageSize": 30,
                        "column": "szse",
                        "tabName": "fulltext",
                        "plate": "",
                        "stock": identity["code"] + "," + identity["orgId"],
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
                value = {"window": window, "catalog_complete": True, "rows": rows, "receipts": receipts}
            except (ValueError, KeyError, TypeError) as exc:
                value = {"window": window, "catalog_complete": False, "reason": str(exc), "receipts": receipts}
            save(target, value)
            if position % 10 == 0 or not value["catalog_complete"]:
                print(
                    json.dumps(
                        {"window": position, "total": len(plan["windows"]), "passed": value["catalog_complete"]}
                    ),
                    flush=True,
                )
            if "PUBLIC_REQUEST_LIMIT" in value.get("reason", ""):
                break
    finally:
        reader.close()
    save(OUT / "catalog-collection-stop.json", {"new_fits": 0, "do_not_retry_failed_sources": True})
    return audit()


def audit():
    """重读每页原始响应，重算目录覆盖；不声称正文或语义已经通过。"""
    plan = check_plan()
    catalogs = [read(p) for p in sorted((OUT / "catalogs").glob("*.json"))]
    for c in catalogs:
        if c["catalog_complete"]:
            pages = []
            for r in c["receipts"]:
                if sha(r["path"]) != r["sha256"]:
                    raise ValueError("RAW_CATALOG_CHANGED")
                pages.append((r["params"]["pageNum"], read(r["path"])))
            w = c["window"]
            if c["rows"] != catalog_rows(pages, w["stock"], w["start"], w["end"]):
                raise ValueError("CATALOG_REPLAY_MISMATCH")
    before = read(PREVIOUS / "row-coverage.json")
    after = project_coverage(before, catalogs)
    save(OUT / "row-coverage.json", after)
    used = Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json"))
    if used["catalog"] > plan["limits"]["catalog"]:
        raise ValueError("CATALOG_BUDGET_EXCEEDED")
    failures = [c for c in catalogs if not c["catalog_complete"]]
    result = {
        "windows_planned": len(plan["windows"]),
        "windows_completed": len(catalogs),
        "windows_passed": len(catalogs) - len(failures),
        "failed_windows": len(failures),
        "requests": dict(used),
        "rows_total": len(after),
        "catalog_complete_before": sum(r["catalog_complete"] for r in before),
        "catalog_complete_after": sum(r["catalog_complete"] for r in after),
        "by_fund": dict(Counter(r["fund_code"] for r in after if r["catalog_complete"])),
        "new_fits": 0,
        "old_cumulative_fits": 70,
        "admission": admission_status(
            catalog_complete=all(r["catalog_complete"] for r in after),
            body_complete=False,
            semantics_complete=False,
            protocol_frozen=False,
            fit_budget=12,
        ),
    }
    save(OUT / "catalog-failures.json", failures)
    save(OUT / "catalog-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "catalogs", "audit"])
    args = parser.parse_args()
    lock = RUNS / ".earnings-preparation.lock"
    owner = {"pid": os.getpid(), "scope": str(OUT), "command": args.command}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    try:
        result = {"prepare": prepare, "catalogs": collect_catalogs, "audit": audit}[args.command]()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        if read(lock) == owner:
            lock.unlink()


if __name__ == "__main__":
    main()
