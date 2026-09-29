"""旧新闻栏目总数变化后，仅登记一次完整的新快照；不拼接新旧分页。"""

import json
import re
from pathlib import Path

from app.services.fund_information_history_v1 import BoundedPublicReader, read, save, sha
from app.services.fund_public_catalog_v3 import append_catalog

from scripts.fund_002112_closure_v1 import OUT
from scripts.fund_002112_public_history_v1 import PROXY, config, records

DEST = OUT / "public-news-current-snapshot"


def run():
    """独立保存当前栏目全部分页；有任何漂移就停止，不在结果出来后扩大次数。"""
    prior = OUT / "public-catalogs/catalog-14.json"
    if "FROZEN_PUBLIC_TOTAL_DRIFT" not in str(read(prior)["issues"]):
        raise ValueError("REGISTERED_DRIFT_EVIDENCE_REQUIRED")
    plan = {
        "column": 14,
        "maximum_requests": 101,
        "maximum_groups": 100,
        "range": ["2016-12-24", "2023-12-31"],
        "old_snapshot_sha256": sha(prior),
        "code_sha256": sha(__file__),
        "whole_snapshot_refreshes": 1,
        "no_mix_with_previous_snapshot": True,
        "new_fits": 0,
    }
    save(DEST / "plan.json", plan)
    reader = BoundedPublicReader(DEST, {"catalog": 101})
    all_rows, receipts, issues = [], [], []
    total = None
    try:
        raw, receipt = reader.fetch("https://www.nhsa.gov.cn/col/col14/index.html", group="catalog")
        receipts.append(receipt)
        html = raw.decode("utf-8")
        params, total = config(html)
        if (total + 44) // 45 > plan["maximum_groups"]:
            raise ValueError("COLUMN_GROUP_LIMIT")
        save(DEST / "snapshot-header.json", {"params": params, "total": total, "receipt": receipt})
        for group in range((total + 44) // 45):
            if group == 0:
                entries = records(html)
            else:
                url = f"{PROXY}?startrecord={group * 45 + 1}&endrecord={(group + 1) * 45}&perpage=15"
                raw, receipt = reader.fetch(url, params=params, group="catalog")
                receipts.append(receipt)
                text = raw.decode("utf-8")
                match = re.search(r"<totalrecord>(\d+)</totalrecord>", text)
                if not match or int(match[1]) != total:
                    raise ValueError("NEW_SNAPSHOT_TOTAL_DRIFT_STOP")
                entries = records(text)
            append_catalog(all_rows, entries, group, total)
            if group % 10 == 0:
                print(json.dumps({"group": group, "records": len(all_rows), "total": total}), flush=True)
    except ValueError as exc:
        issues.append(str(exc))
    finally:
        reader.close()
    # 原始响应再次回放，确保保存的记录来自同一页集合。
    replay = []
    for group, receipt in enumerate(receipts):
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("NEW_NEWS_RAW_CHANGED")
        if not issues:
            append_catalog(replay, records(Path(receipt["path"]).read_text(encoding="utf-8")), group, total)
    if not issues and replay != all_rows:
        raise ValueError("NEW_NEWS_REPLAY_MISMATCH")
    result = {
        "column": 14,
        "declared_total": total,
        "records": len(all_rows),
        "complete": not issues and len(all_rows) == total,
        "issues": issues,
        "receipts": receipts,
        "all_entries": all_rows,
        "entries_in_range": [r for r in all_rows if plan["range"][0] <= r["display_date"] <= plan["range"][1]],
        "body_and_semantics_complete": False,
        "new_fits": 0,
    }
    save(DEST / "result.json", result)
    print({k: result[k] for k in ("records", "complete", "issues")})


if __name__ == "__main__":
    run()
