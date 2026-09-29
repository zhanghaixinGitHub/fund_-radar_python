"""复用原公开栏目快照，补足其全部目录；别名保留，不把同网址计成多条新闻。"""

import re
from collections import defaultdict
from pathlib import Path

from app.services.fund_information_history_v1 import ROOT, BoundedPublicReader, read, save, sha
from app.services.fund_public_catalog_v3 import append_catalog

from scripts.fund_002112_closure_v1 import OUT
from scripts.fund_002112_public_history_v1 import PROXY, config, records

DEST = OUT / "public-catalogs"
OLD = ROOT / "information-research/20260928-history-v1/public-history-v2"


def run():
    """栏目总数使用原保存首页冻结，不重新请求失败分页，日期过滤不读取任何标签。"""
    old = {c: read(OLD / f"catalog-{c}.json") for c in (104, 14, 46)}
    sources = {str(OLD / f"catalog-{c}.json"): sha(OLD / f"catalog-{c}.json") for c in old}
    for catalog in old.values():
        for r in catalog["receipts"]:
            if sha(r["path"]) != r["sha256"]:
                raise ValueError("OLD_PUBLIC_RAW_CHANGED")
            sources[r["path"]] = r["sha256"]
    plan = {
        "columns": [104, 14, 46],
        "range": ["2016-12-24", "2023-12-31"],
        "maximum_new_requests": 128,
        "maximum_groups_per_column": 100,
        "source_hashes": sources,
        "code_hashes": {
            str(Path(__file__).resolve()): sha(__file__),
            str(Path(__file__).resolve().parents[1] / "app/services/fund_public_catalog_v3.py"): sha(
                Path(__file__).resolve().parents[1] / "app/services/fund_public_catalog_v3.py"
            ),
        },
        "same_url_different_title": "Preserve catalog aliases; downstream must not double-count one article",
        "scope_is_registered_columns_not_all_industry_policy_or_news": True,
        "no_claim_about_deleted_historical_items": True,
        "new_fits": 0,
    }
    save(DEST / "plan.json", plan)
    reader = BoundedPublicReader(DEST, {"catalog": 128})
    results = []
    try:
        for column, previous in old.items():
            path = DEST / f"catalog-{column}.json"
            if path.exists():
                results.append(read(path))
                continue
            first = previous["receipts"][0]
            html = Path(first["path"]).read_text(encoding="utf-8")
            params, total = config(html)
            previous_receipts = {r["url"]: r for r in previous["receipts"]}
            all_rows, receipts, issues = [], [first], []
            for group in range(min((total + 44) // 45, 100)):
                try:
                    if group == 0:
                        entries = records(html)
                    else:
                        url = f"{PROXY}?startrecord={group * 45 + 1}&endrecord={(group + 1) * 45}&perpage=15"
                        if url in previous_receipts:
                            receipt = previous_receipts[url]
                            raw = Path(receipt["path"]).read_bytes()
                        else:
                            raw, receipt = reader.fetch(url, params=params, group="catalog")
                        receipts.append(receipt)
                        text = raw.decode("utf-8")
                        match = re.search(r"<totalrecord>(\d+)</totalrecord>", text)
                        if not match or int(match[1]) != total:
                            raise ValueError("FROZEN_PUBLIC_TOTAL_DRIFT")
                        entries = records(text)
                    append_catalog(all_rows, entries, group, total)
                except ValueError as exc:
                    issues.append({"group": group, "reason": str(exc)})
                    break
            aliases = defaultdict(list)
            for r in all_rows:
                aliases[r["url"]].append(r)
            result = {
                "column": column,
                "declared_total": total,
                "records": len(all_rows),
                "complete": not issues and len(all_rows) == total,
                "issues": issues,
                "receipts": receipts,
                "entries_in_range": [r for r in all_rows if plan["range"][0] <= r["display_date"] <= plan["range"][1]],
                "all_entries": all_rows,
                "aliases": [rows for rows in aliases.values() if len(rows) > 1],
                "body_and_semantics_complete": False,
                "new_fits": 0,
            }
            save(path, result)
            results.append(result)
            print({k: result[k] for k in ("column", "records", "complete", "issues")}, flush=True)
    finally:
        reader.close()
    save(DEST / "result.json", {"columns": results, "new_fits": 0, "training_ready": False})


if __name__ == "__main__":
    run()
