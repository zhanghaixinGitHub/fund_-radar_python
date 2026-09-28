"""对五份新取得的报告另做输入覆盖检查；与仅修本地六份报告的结果分开保存。"""

from collections import Counter

from app.services.fund_002112_peer_coverage_audit import CODES, calendar_days
from app.services.fund_002112_peer_material_repair import OUTPUT, FrozenSources, inspect_input, repair_reports
from app.services.fund_002112_zero_fit_review import digest, file_hash, read_json, save_once


def run():
    sources = FrozenSources(read_json(OUTPUT / "protocol.json"))
    recovered, _ = repair_reports(sources)
    reviews = read_json(OUTPUT / "public-pdf-probe/reviewed-results.json")["reviews"]
    new_reports = {c: [] for c in CODES}
    source_manifest = {}
    for review in reviews:
        if review["status"] != "PARSED_CANDIDATE":
            continue
        receipt = review["raw_receipt"]
        if file_hash(receipt["path"]) != receipt["sha256"]:
            raise ValueError("NEW_PDF_SOURCE_CHANGED")
        report = read_json(review["parsed_file"])
        if report["raw"] != receipt or report["fund_code"] != review["fund_code"]:
            raise ValueError("NEW_REPORT_SOURCE_IDENTITY_CHANGED")
        new_reports[review["fund_code"]].append(report)
        source_manifest[review["parsed_file"]] = file_hash(review["parsed_file"])
        source_manifest[receipt["path"]] = receipt["sha256"]
    save_once(OUTPUT / "public-coverage-sources.json", source_manifest)
    snapshot = sources.load("snapshot")
    sources.indices = snapshot["indices"]
    days = calendar_days(
        [sources.load("calendar_cn_a_share_2015_2020_research_v1"), sources.load("calendar_cn_a_share_2021_2025_v1")]
    )
    before = read_json(OUTPUT / "input-coverage-daily.json")
    maps = {c: {r["date"]: r for r in snapshot["funds"][c]["nav"]["rows"]} for c in CODES}
    rows = []
    for n, item in enumerate(before):
        code, target = item["fund_code"], item["target"]
        fund = snapshot["funds"][code]
        after = inspect_input(
            fund, fund["reports"] + recovered[code] + new_reports[code], maps[code], days, target, sources
        )
        if item["after"]["status"] == "INPUT_COMPLETE" and after != item["after"]:
            raise ValueError("PREVIOUS_COMPLETE_INPUT_CHANGED")
        rows.append({"fund_code": code, "target": target, "before": item["after"], "after": after})
        if n % 300 == 0:
            print(f"公开报告补查覆盖 {n}/{len(before)}，新增拟合 0", flush=True)
    summary = {
        "new_fits": 0,
        "new_parsed_reports": sum(map(len, new_reports.values())),
        "interpretation": "Historical reconstruction only; no predictions, training pack or adoption",
        "input_sha256": digest(rows),
        "funds": {},
    }
    for code in CODES:
        subset = [r for r in rows if r["fund_code"] == code]
        summary["funds"][code] = {
            "before": dict(Counter(r["before"]["status"] for r in subset)),
            "after": dict(Counter(r["after"]["status"] for r in subset)),
            "new_complete_input_dates": [
                r["target"]
                for r in subset
                if r["before"]["status"] != "INPUT_COMPLETE" and r["after"]["status"] == "INPUT_COMPLETE"
            ],
        }
    save_once(OUTPUT / "public-input-coverage-daily.json", rows)
    save_once(OUTPUT / "public-input-coverage-summary.json", summary)
    save_once(OUTPUT / "public-market-sources-verified.json", sources.verified)
    print(
        {
            c: {"after": v["after"], "new_complete": len(v["new_complete_input_dates"])}
            for c, v in summary["funds"].items()
        },
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
