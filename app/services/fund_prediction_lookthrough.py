"""联接预测补查已核实目标基金报告，不把联接直接持仓当成底层完整组合。"""

import re
from pathlib import Path

from app.core.config import get_settings
from app.services.fund_exposure_common import now, read
from app.services.fund_materials import snapshot
from app.services.fund_materials_store import versioned_save


def recover_target_reports(material: dict, output: Path) -> list[dict]:
    """只处理本包已由原文核准的穿透链，目录先行、原件按同版本复用。

    原发布入口保留报告ID和历史。取得/解析成功仍需聊天核对新增证券、财务、
    市场和独立用途审阅；不会自动改综合包、批准状态、方向或基金市场目录。
    """
    from app.integrations.market_fund_reports import MarketReportClient
    from app.services.direction_1d_protocol import digest
    from app.services.fund_exposure_runtime import execution_lock
    from app.services.fund_holdings_publication import publish_report
    from app.services.market_holdings_parse import parse_report

    code = material["fund_code"]
    chain = material.get("exposure_chain", [code])
    if not chain or chain[0] != code or len(chain) > 8 or len(chain) != len(set(chain)):
        raise ValueError("PREDICTION_LOOKTHROUGH_CHAIN_INVALID")
    root = Path(get_settings().fund_material_directory) / "prediction-lookthrough"
    client = MarketReportClient(root)
    results = []
    try:
        with execution_lock() as locked:
            if not locked:
                raise ValueError("SHARED_HOLDINGS_LOCK_BUSY")
            for target in chain[1:]:
                row = {"owner_fund_code": code, "target_fund_code": target}
                try:
                    if not re.fullmatch(r"\d{6}", target):
                        raise ValueError("PREDICTION_LOOKTHROUGH_TARGET_CODE")
                    # 必须已留本目标报告，不能仅凭人工写入一个链条扩大资料范围。
                    anchors = [
                        d for d in material["documents"] if d.get("kind") == "REPORT" and target in d.get("codes", [])
                    ]
                    if not anchors:
                        raise ValueError("PREDICTION_LOOKTHROUGH_ORIGINAL_ANCHOR_REQUIRED")
                    anchor = max(anchors, key=lambda d: d["published_date"])
                    if digest(anchor["body"]) != anchor.get("body_hash") or not re.search(
                        r"(?<!\d)" + target + r"(?!\d)", anchor["body"]
                    ):
                        raise ValueError("PREDICTION_LOOKTHROUGH_ORIGINAL_ANCHOR_INVALID")
                    name = re.split(r"20\d{2}\s*年", anchor["title"])[0].strip()
                    path = root / "funds" / (target + ".json")
                    state = read(path) if path.exists() else {"target_fund_code": target, "entries": {}}
                    entries = [e for e in client.catalog(target) if e["published"] <= str(now().date())]
                    if not entries:
                        raise ValueError("PREDICTION_LOOKTHROUGH_EMPTY_REPORT_CATALOG")
                    entry = max(entries, key=lambda e: (e["end"], e["published"], e["id"]))
                    old = state["entries"].get(entry["id"])
                    before = {r["id"] for r in (snapshot(target) or {}).get("reports", [])}
                    fingerprint = digest(entry)
                    if (
                        old
                        and old.get("fingerprint") == fingerprint
                        and old.get("status") == "PUBLISHED"
                        and old.get("report_id") in before
                    ):
                        row.update(
                            status="ORIGINAL_REUSED", entry=entry, source=old["source"], report_id=old["report_id"]
                        )
                    else:
                        source = client.body(
                            target, entry, old if old and old.get("fingerprint") == fingerprint else None
                        )
                        item = {"entry": entry, "fingerprint": fingerprint, "source": source, "status": "DOWNLOADED"}
                        if old:
                            item["first_attempt"] = old
                        state["entries"][entry["id"]] = item
                        versioned_save(path, state)
                        report, issues = parse_report(
                            client, root, {"fund_code": target, "fund_name": name}, entry, source
                        )
                        published = publish_report(
                            target, report, directory=Path(get_settings().fund_material_directory)
                        )
                        if not before <= {r["id"] for r in published["reports"]}:
                            raise ValueError("HISTORICAL_TARGET_REPORT_LOST")
                        item.update(status="PARTIAL" if issues else "PUBLISHED", report_id=report["id"], issues=issues)
                        versioned_save(path, state)
                        row.update(
                            status=item["status"], entry=entry, source=source, report_id=report["id"], issues=issues
                        )
                    row["report"] = next(r for r in snapshot(target)["reports"] if r["id"] == row["report_id"])
                except Exception as error:
                    row.update(status="ACTION_REQUIRED", failure=type(error).__name__ + ":" + str(error)[:300])
                results.append(row)
    finally:
        client.close()
    versioned_save(output / "target-reports.json", {"at": now().isoformat(), "fund_code": code, "items": results})
    return results
