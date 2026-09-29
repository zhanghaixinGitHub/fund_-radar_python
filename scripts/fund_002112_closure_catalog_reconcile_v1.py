"""用预先固定的单日切片核对目录分页漂移；保留失败窗口，不重试原查询。"""

import json
import os

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
from app.services.fund_research_closure_v1 import project_coverage, split_windows
from scripts.fund_002112_closure_v1 import OUT

DEST = OUT / "catalog-reconciliation"


def run():
    """只针对已证实总数漂移的区间重建独立目录；完整30日特征窗口保持不变。"""
    failed_path = OUT / "catalog-failures.json"
    failed = read(failed_path)
    if any(f["reason"] != "CATALOG_TOTAL_CHANGED" for f in failed):
        raise ValueError("UNSUPPORTED_FAILURE_REQUIRES_DIFFERENT_EVIDENCE")
    windows = split_windows([f["window"] for f in failed], maximum_days=1)
    plan = {
        "method": "INDEPENDENT_SINGLE_DAY_RECONSTRUCTION_OF_DRIFTING_PAGINATION",
        "windows": windows,
        "limit": len(windows) * 20,
        "maximum_pages_per_day": 20,
        "source_sha256": sha(failed_path),
        "code_sha256": sha(__file__),
        "old_failed_snapshot_retained": True,
        "original_requests_retried": False,
    }
    save(DEST / "plan.json", plan)
    stocks = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
    reader = BoundedPublicReader(DEST, {"catalog": plan["limit"]})
    try:
        for number, window in enumerate(windows, 1):
            target = DEST / "catalogs" / (digest(window) + ".json")
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
                    if count > plan["maximum_pages_per_day"]:
                        raise ValueError("SINGLE_DAY_PAGE_LIMIT")
                    pages.append((page, response))
                    page += 1
                rows = catalog_rows(pages, window["stock"], window["start"], window["end"])
                result = {"window": window, "catalog_complete": True, "rows": rows, "receipts": receipts}
            except (ValueError, KeyError, TypeError) as exc:
                result = {"window": window, "catalog_complete": False, "reason": str(exc), "receipts": receipts}
            save(target, result)
            if number % 10 == 0:
                print(json.dumps({"day": number, "total": len(windows)}), flush=True)
    finally:
        reader.close()
    catalogs = [read(p) for p in (DEST / "catalogs").glob("*.json")]
    for catalog in catalogs:
        if catalog["catalog_complete"]:
            replay = []
            for receipt in catalog["receipts"]:
                if sha(receipt["path"]) != receipt["sha256"]:
                    raise ValueError("RECONCILIATION_RAW_CHANGED")
                replay.append((receipt["params"]["pageNum"], read(receipt["path"])))
            window = catalog["window"]
            if catalog["rows"] != catalog_rows(replay, window["stock"], window["start"], window["end"]):
                raise ValueError("RECONCILIATION_REPLAY_MISMATCH")
    rows = project_coverage(read(OUT / "row-coverage.json"), catalogs)
    save(DEST / "row-coverage.json", rows)
    result = {
        "single_day_windows": len(catalogs),
        "passed": sum(c["catalog_complete"] for c in catalogs),
        "catalog_complete_rows": sum(r["catalog_complete"] for r in rows),
        "remaining_rows": sum(not r["catalog_complete"] for r in rows),
        "target_fund_complete": sum(r["catalog_complete"] and r["fund_code"] == "002112" for r in rows),
        "new_fits": 0,
    }
    save(DEST / "result.json", result)
    return result


if __name__ == "__main__":
    lock = OUT / ".catalog-reconciliation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        print(json.dumps(run(), ensure_ascii=False), flush=True)
    finally:
        if lock.read_text(encoding="utf-8") == str(os.getpid()):
            lock.unlink()
