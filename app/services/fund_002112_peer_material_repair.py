"""独立修复六份本地季报并复核参考基金输入覆盖；没有采集、数据库或训练入口。"""

import math
import re
from collections import Counter, OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

from app.integrations.dbfund_reports import parse_text as parse_v1
from app.integrations.fund_report_sections_v2 import PARSER_VERSION, parse_text
from app.services.fund_002112_peer_coverage_audit import CODES, NAMES, calendar_days
from app.services.fund_002112_round3_data import exposure, nav_window, select_report
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-material-repair/20260927-v1"


class FrozenSources:
    """协议白名单文件只读；本轮可以使用的行情还须绑定第三轮的旧来源清单。"""

    def __init__(self, protocol):
        self.specs = protocol["sources"]
        self.verified = {}
        for name, spec in self.specs.items():
            if file_hash(spec["path"]) != spec["sha256"]:
                raise ValueError("PROTOCOL_SOURCE_CHANGED:" + name)
        self.market_files = self.load("third_source_manifest")["files"]
        self.cache = OrderedDict()

    def load(self, name):
        return read_json(self.specs[name]["path"])

    def raw(self, receipt):
        path = Path(self.specs["raw_" + receipt["sha256"]]["path"])
        return path.read_bytes()

    def market_read(self, path):
        path = Path(path).resolve()
        expected = self.market_files.get(str(path))
        if expected is None:
            raise ValueError("QUOTE_NOT_IN_FROZEN_SOURCE_MANIFEST")
        if str(path) not in self.verified:
            if file_hash(path) != expected:
                raise ValueError("MARKET_FROZEN_SOURCE_CHANGED")
            self.verified[str(path)] = expected
        return read_json(path)

    def quotes(self, day):
        """从旧回执完整原文恢复股票；不能把旧聚合子集没列的股票当作没有行情。"""
        if not "2020-01-01" <= day <= "2023-12-31":
            raise ValueError("QUOTE_DATE_OUTSIDE_REPAIR_SCOPE")
        if day not in self.cache:
            path = ROOT / "stock-days" / (day + ".json")
            if not path.exists() or str(path.resolve()) not in self.market_files:
                return {}
            aggregate = self.market_read(path)
            receipt = aggregate["receipt"]
            if (
                receipt.get("expires_at")
                and datetime.fromisoformat(receipt["expires_at"]) <= datetime.now().astimezone()
            ):
                raise ValueError("QUOTE_RECEIPT_EXPIRED")
            raw_path = Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]
            if self.market_files.get(str(raw_path.resolve())) != receipt["sha256"]:
                raise ValueError("QUOTE_RECEIPT_HASH_NOT_FROZEN")
            raw = self.market_read(raw_path)["data"]
            rows = {}
            for item in raw["items"]:
                row = dict(zip(raw["fields"], item, strict=True))
                if row.pop("trade_date") != day.replace("-", ""):
                    raise ValueError("QUOTE_DATE_IDENTITY_CHANGED")
                code = row.pop("ts_code")
                if code in rows:
                    raise ValueError("QUOTE_DUPLICATE")
                rows[code] = row
            if any(rows.get(code) != row for code, row in aggregate["rows"].items()):
                raise ValueError("QUOTE_AGGREGATE_DIFFERS_FROM_RAW")
            self.cache[day] = {"rows": rows}
        self.cache.move_to_end(day)
        while len(self.cache) > 32:
            self.cache.popitem(last=False)
        return self.cache[day]


def repair_reports(sources: FrozenSources) -> tuple[dict, list]:
    """重放九份原正文：六份修复、三份成功对照；保留真实原文及日期来源。"""
    saved, regression = {c: [] for c in CODES}, []
    receipts = {}
    for name in sources.specs:
        if name.startswith("receipt_"):
            r = sources.load(name)
            if "read.php" in r["url"] and r.get("content_hash"):
                receipts[r["url"]] = r
    for probe in sources.load("audit_parser-spacing-probes"):
        code = probe["fund_code"]
        receipt = receipts[probe["url"]]
        soup = BeautifulSoup(sources.raw(receipt), "html.parser")
        title = soup.select_one("h1").get_text(" ", strip=True)
        body = soup.select_one("#sohu_content").get_text("\n", strip=True)
        publication = soup.select_one(".article_info > .txt .c")
        match = re.search(r"20\d{2}-\d{2}-\d{2}", publication.text if publication else "")
        if not match or title != probe["title"] or receipt["sha256"] != probe["raw_sha256"]:
            raise ValueError("BODY_IDENTITY_CHANGED")
        if digest({"title": title, "text": body, "published": match[0]}) != receipt["content_hash"]:
            raise ValueError("BODY_CONTENT_HASH_CHANGED")
        if receipt.get("revised_after_receipt"):
            raise ValueError("BODY_REVISED_HISTORY_UNVERIFIED")
        kwargs = {"fund_code": code, "fund_name": NAMES[code], "master_code": "000480" if code == "017493" else code}
        parsed = parse_text([body], title, **kwargs)
        try:
            old = parse_v1([body], title, **kwargs)
            old_status = "PASSED"
        except ValueError as exc:
            old, old_status = None, str(exc)
        if old:
            assert {k: v for k, v in parsed.items() if k not in ("parser_version", "section_normalization")} == {
                k: v for k, v in old.items() if k != "parser_version"
            }
        elif old_status != "REPORT_HOLDINGS_SECTION_MISSING":
            raise ValueError("UNEXPECTED_ORIGINAL_PARSER_FAILURE")
        entry = next(
            e
            for e in sources.load("manifest_" + code)["catalog"]
            if (e["report_end"], e["report_type"]) == (parsed["report_end"], parsed["report_type"])
        )
        public_day = max(parsed["published_date"], match[0], entry["published_date"])
        if public_day > "2024-12-31":
            raise ValueError("REPORT_DATE_OUTSIDE_SCOPE")
        parsed.update(
            available_at=str(date.fromisoformat(public_day) + timedelta(days=1)) + "T08:00:00+08:00",
            source_publication_date=public_day,
            raw=receipt,
            source=entry,
            document_format="PUBLIC_REPRINT_TEXT",
            original_pdf_saved=False,
            training_eligible=False,
            purpose="INDEPENDENT_MATERIAL_REPAIR_REVIEW",
            availability_basis="LATER_DECLARED_AND_REPRINT_DATE_NEXT_DAY_0800_RECONSTRUCTION",
        )
        regression.append(
            {
                "fund_code": code,
                "report_end": parsed["report_end"],
                "report_type": parsed["report_type"],
                "raw_sha256": receipt["sha256"],
                "old_status": old_status,
                "v2_passed": True,
                "holding_count": parsed["holding_count"],
                "business_equal_for_original_pass": True if old else None,
                "edits": parsed["section_normalization"]["edits"],
            }
        )
        if not old:
            path = OUTPUT / "reports" / f"{code}-{parsed['report_end']}-{parsed['report_type']}.json"
            save_once(path, parsed)
            saved[code].append(parsed)
    if sum(len(v) for v in saved.values()) != 6 or len(regression) != 9:
        raise ValueError("REPORT_REPAIR_COUNT_CHANGED")
    save_once(OUTPUT / "report-regression.json", regression)
    return saved, regression


def inspect_input(fund, reports, mapping, days, target, sources):
    """只核完整 20 项输入，不构造答案、训练权重或历史预测方向。"""
    i = days.index(target)
    if days[i - 1] not in mapping or target not in mapping:
        return {"status": "TARGET_OR_BASE_NAV_MISSING"}
    try:
        window = nav_window(mapping, days, target)
        report = select_report(reports, fund["catalog"], target)
        needed = days[i - 21 : i]
        values = window["x"] + exposure(report, sources.quotes, sources.indices, needed)
    except ValueError as exc:
        reason = str(exc)
        allowed = (
            "NO_PRIOR_PUBLIC_CONTIGUOUS_NAV_WINDOW",
            "NO_PRIOR_PUBLIC_REPORT",
            "LATEST_DISCLOSURE_MISSING_NO_FALLBACK",
            "REPORT_AGE_OR_WEIGHT_INVALID",
            "POSITIVE_HOLDING_QUOTE_MISSING:",
            "AMOUNT_BASE_NOT_POSITIVE",
            "INDEX_WINDOW_MISSING",
        )
        if not any(reason == value or (value.endswith(":") and reason.startswith(value)) for value in allowed):
            raise
        return {"status": reason.split(":")[0], "detail": reason}
    if len(values) != 20 or not all(math.isfinite(v) for v in values):
        raise ValueError("INPUT_VECTOR_INVALID")
    return {
        "status": "INPUT_COMPLETE",
        "nav_window_end": window["S"],
        "nav_lag_sessions": window["lag_sessions"],
        "report_end": report["report_end"],
        "report_sha256": report["raw"]["sha256"],
        "input_vector_sha256": digest(values),
    }


def run(progress=print):
    """保存新解析和逐日覆盖，仍不生成可训练包或准入通过决定。"""
    protocol = read_json(OUTPUT / "protocol.json")
    if protocol["funds"] != list(CODES) or protocol["new_fits"] != 0:
        raise ValueError("REPAIR_SCOPE_CHANGED")
    sources = FrozenSources(protocol)
    (OUTPUT / "reports").mkdir(exist_ok=True)
    recovered, regression = repair_reports(sources)
    progress("六份季报修复、三份原通过报告回归完成；新增拟合 0")
    snapshot = sources.load("snapshot")
    sources.indices = snapshot["indices"]
    days = calendar_days(
        [sources.load("calendar_cn_a_share_2015_2020_research_v1"), sources.load("calendar_cn_a_share_2021_2025_v1")]
    )
    # 指数原文比研究日历早 210 个交易日；仅比较原研究日历的覆盖段，不扩展基金输入日期。
    if any([d for d in sorted(index["rows"]) if d >= days[0]] != days for index in sources.indices.values()):
        raise ValueError("INDEX_CALENDAR_DISAGREEMENT")
    maps = {c: {r["date"]: r for r in snapshot["funds"][c]["nav"]["rows"]} for c in CODES}
    if any(max(mapping) > "2024-12-31" for mapping in maps.values()):
        raise ValueError("NAV_SCOPE_CHANGED")
    targets = [d for d in days if "2021-01-01" <= d <= "2023-12-31"]
    rows = []
    for n, target in enumerate(targets):
        for code in CODES:
            fund = snapshot["funds"][code]
            before = inspect_input(fund, fund["reports"], maps[code], days, target, sources)
            after = inspect_input(fund, fund["reports"] + recovered[code], maps[code], days, target, sources)
            if before["status"] == "INPUT_COMPLETE" and before != after:
                raise ValueError("PREVIOUSLY_COMPLETE_INPUT_CHANGED")
            rows.append({"fund_code": code, "target": target, "before": before, "after": after})
        if n % 100 == 0:
            progress(f"资料覆盖 {n}/{len(targets)} 个目标日，新增拟合 0")
    summary = {
        "parser_version": PARSER_VERSION,
        "new_fits": 0,
        "cumulative_real_fits": 52,
        "repaired_reports": 6,
        "unchanged_passing_controls": 3,
        "funds": {},
    }
    for code in CODES:
        subset = [r for r in rows if r["fund_code"] == code]
        summary["funds"][code] = {
            "target_days": len(subset),
            "before": dict(Counter(r["before"]["status"] for r in subset)),
            "after": dict(Counter(r["after"]["status"] for r in subset)),
            "new_complete_input_dates": [
                r["target"]
                for r in subset
                if r["before"]["status"] != "INPUT_COMPLETE" and r["after"]["status"] == "INPUT_COMPLETE"
            ],
        }
    missing = []
    for code in CODES:
        repaired = {(r["report_end"], r["report_type"]) for r in recovered[code]}
        for r in sources.load("audit_report-coverage")[code]["reports"]:
            if not r["parsed"] and (r["report_end"], r["report_type"]) not in repaired:
                missing.append(
                    {
                        "fund_code": code,
                        **r,
                        "needed": "Full original report, identity/date/version, holdings validation",
                    }
                )
    save_once(OUTPUT / "missing-report-worklist.json", missing)
    save_once(OUTPUT / "nav-publication-worklist.json", sources.load("audit_publication-metadata-audit"))
    save_once(OUTPUT / "input-coverage-daily.json", rows)
    save_once(OUTPUT / "input-coverage-summary.json", summary)
    save_once(OUTPUT / "market-sources-verified.json", sources.verified)
    return summary
