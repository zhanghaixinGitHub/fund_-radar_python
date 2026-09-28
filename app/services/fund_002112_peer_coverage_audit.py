"""两只固定参考基金的资料排除溯源；只读取清单中的旧文件，不联网、入库或拟合。

原训练期为 2021—2023；旧 excluded 还含少量 2024 年净值排除记录，必须另计，
不能将其算成训练损失。正文空格实验仅验证解析原因，不替换旧解析器或训练输入。
"""

import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-coverage-audit/20260927-v1"
CODES = ("017493", "160323")
NAMES = {"017493": "东方红新动力", "160323": "华夏磐泰"}


def normalize_section_spacing(text: str) -> str:
    """只给顶层章节号和明确中文标题之间加空格；不动数值、小节编号或表格行。"""
    return re.sub(r"(?m)^([578]\.[1-4])(?=(?:报告)?期末|按行业分类)", r"\1 ", text)


def calendar_days(definitions: list[dict]) -> list[str]:
    """按原交易所休市定义重建日历，只返回截至 2024 年日期；不读取任何封存标签。"""
    result = []
    for definition in definitions:
        first = date.fromisoformat(definition["coverage_start"])
        last = min(date(2024, 12, 31), date.fromisoformat(definition["coverage_end"]))
        closed = set()
        for year in definition["years"]:
            for a, b in year["closed_ranges"]:
                a, b = date.fromisoformat(a), date.fromisoformat(b)
                closed.update(a + timedelta(days=i) for i in range((b - a).days + 1))
        result.extend(
            str(d)
            for i in range((last - first).days + 1)
            if (d := first + timedelta(days=i)).weekday() < 5 and d not in closed
        )
    if result != sorted(set(result)):
        raise ValueError("DUPLICATE_OR_UNSORTED_CALENDAR")
    return result


def nav_gate(mapping: dict, window: list[str]) -> dict:
    """复核原 61 个输入日加 1 个答案日的首个净值门槛，保留每个缺口/晚公布日。"""
    if len(window) != 62 or window != sorted(set(window)):
        raise ValueError("INVALID_NAV_WINDOW")
    absent = [d for d in window if d not in mapping]
    unknown = [d for d in window if d in mapping and not mapping[d].get("ann_date")]
    late = [
        d for d in window[:-1] if d in mapping and mapping[d].get("ann_date") and mapping[d]["ann_date"] > window[-1]
    ]
    reason = (
        "NAV_GAP" if absent else "NAV_PUBLICATION_UNKNOWN" if unknown else "NAV_NOT_PUBLIC_BY_TARGET" if late else None
    )
    return {"reason": reason, "absent_dates": absent, "unknown_publication_dates": unknown, "late_input_dates": late}


def report_gate(reports: list[dict], catalog: list[dict], target: str) -> dict:
    """还原旧包先查已解析报告、再拒绝退回旧报告的顺序，不把缺报告填成旧持仓。"""
    at = target + "T08:00:00+08:00"
    available = [r for r in reports if r["available_at"] <= at]
    listed = [r for r in catalog if r["available_at"] <= at]

    def key(r):
        return r["report_end"], r["available_at"]

    newest = max(listed, key=key) if listed else None
    selected = max(available, key=lambda r: (*key(r), r.get("raw", {}).get("sha256", ""))) if available else None
    reason = None
    if selected is None:
        reason = "EXPOSURE_NO_AVAILABLE_REPORT"
    elif newest and (
        key(newest) > key(selected)
        or (newest["report_end"] == selected["report_end"] and newest["report_type"] != selected["report_type"])
    ):
        reason = "TRAINING_LATEST_DISCLOSURE_MISSING"

    def identity(value):
        return {k: value[k] for k in ("report_end", "report_type", "available_at")} if value else None

    return {"reason": reason, "latest_listed": identity(newest), "latest_parsed": identity(selected)}


def summarize_dates(rows: list[dict]) -> dict:
    """计数按目标日去重，日期范围不表示区间内每一天都有记录。"""
    days = sorted({r["target"] for r in rows})
    return {"count": len(days), "first": days[0] if days else None, "last": days[-1] if days else None}


def run() -> dict:
    """核验冻结来源后逐日解释原排除，再做正文格式隔离实验；产物排他保存。"""
    protocol = read_json(OUTPUT / "protocol.json")
    if protocol["funds"] != list(CODES) or protocol["new_fits"] != 0:
        raise ValueError("AUDIT_SCOPE_CHANGED")
    sources = protocol["sources"]
    for name, spec in sources.items():
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("FROZEN_SOURCE_CHANGED:" + name)

    def load(name):
        return read_json(sources[name]["path"])

    old, snapshot = load("first_dataset"), load("snapshot")
    if digest(snapshot) != old["source_hash"]:
        raise ValueError("OLD_SNAPSHOT_LINEAGE_CHANGED")
    days = calendar_days(
        [
            load("calendar_cn_a_share_2015_2020_research_v1"),
            load("calendar_cn_a_share_2021_2025_v1"),
        ]
    )
    third, eligibility, folds = load("third_inputs"), load("third_eligibility"), load("third_folds")
    receipts = {}
    for name in sources:
        if name.startswith("receipt_"):
            r = load(name)
            previous = receipts.setdefault(r["url"], r)
            if previous["sha256"] != r["sha256"]:
                raise ValueError("RECEIPT_VERSION_CONFLICT")

    def raw(receipt):
        spec = sources["raw_" + receipt["sha256"]]
        return Path(spec["path"]).read_bytes()

    # 原解析器只调用纯文本解析入口，绝不创建网络客户端或调用采集入口。
    from app.integrations.dbfund_reports import parse_text
    from app.integrations.public_fund_reports import report_period

    result = {"new_fits": 0, "sources_verified": len(sources), "funds": {}}
    daily, report_details, probes = [], {}, []
    for code in CODES:
        fund = snapshot["funds"][code]
        mapping = {r["date"]: r for r in fund["nav"]["rows"]}
        if len(mapping) != len(fund["nav"]["rows"]) or max(mapping) > "2024-12-31":
            raise ValueError("NAV_SCOPE_OR_IDENTITY_CHANGED")
        manifest = load("manifest_" + code)
        catalog = manifest["catalog"]
        compact_catalog = [
            {
                "report_end": e["report_end"],
                "report_type": e["report_type"],
                "available_at": str(date.fromisoformat(e["published_date"]) + timedelta(days=1)) + "T08:00:00+08:00",
            }
            for e in catalog
        ]
        if compact_catalog != fund["catalog"]:
            raise ValueError("CATALOG_DIFFERS_FROM_FROZEN_SNAPSHOT")
        if sorted(r["raw"]["sha256"] for r in fund["reports"]) != sorted(
            r["sha256"] for r in manifest["documents"].values()
        ):
            raise ValueError("PARSED_REPORTS_DIFFER_FROM_SNAPSHOT")

        # 目录只提取报告链接，正文只取 sohu_content；网页侧栏的当前净值不参加审计。
        sohu_links, page_chain = {}, []
        for url, receipt in receipts.items():
            if "notice.php?code=" + code not in url:
                continue
            soup = BeautifulSoup(raw(receipt), "html.parser")
            next_link = next((a for a in soup.select('a[href*="notice.php"]') if "下一页" in a.text), None)
            page_chain.append(
                {
                    "url": url,
                    "sha256": receipt["sha256"],
                    "next": urljoin(url, next_link["href"]) if next_link else None,
                }
            )
            for link in soup.select('a[href*="read.php"]'):
                title = link.get_text(" ", strip=True)
                period = report_period(title)
                if period and "2020-09-30" <= period[0] <= "2024-09-30":
                    if NAMES[code] in re.sub(r"\s+", "", title):
                        sohu_links[period] = {"url": urljoin(url, link["href"]), "title": title}
        chain = {r["url"]: r["next"] for r in page_chain}
        cursor, visited = "https://q.fund.sohu.com/q/notice.php?code=" + code, set()
        while cursor:
            if cursor not in chain or cursor in visited:
                raise ValueError("LOCAL_CATALOG_PAGE_CHAIN_INCOMPLETE")
            visited.add(cursor)
            cursor = chain[cursor]
        if len(visited) != len(chain):
            raise ValueError("DISCONNECTED_CATALOG_PAGE")

        recovered = []
        for url, receipt in receipts.items():
            if "read.php?code=" + code not in url:
                continue
            soup = BeautifulSoup(raw(receipt), "html.parser")
            body = soup.select_one("#sohu_content").get_text("\n", strip=True)
            title = soup.select_one("h1").get_text(" ", strip=True)
            period = report_period(title)
            if period not in sohu_links or sohu_links[period]["url"] != url:
                raise ValueError("BODY_CATALOG_IDENTITY_MISMATCH")
            normalized = normalize_section_spacing(body)
            if re.sub(r"\s", "", normalized) != re.sub(r"\s", "", body):
                raise ValueError("DIAGNOSTIC_CHANGED_NONWHITESPACE")
            probe = {
                "fund_code": code,
                "url": url,
                "raw_sha256": receipt["sha256"],
                "title": title,
                "report_end": period[0],
                "report_type": period[1],
                "changes_only_whitespace": True,
            }
            for stage, text in (("original", body), ("spacing_only", normalized)):
                try:
                    parsed = parse_text(
                        [text],
                        title,
                        fund_code=code,
                        fund_name=NAMES[code],
                        master_code="000480" if code == "017493" else code,
                    )
                    probe[stage] = {
                        "passed": True,
                        **{
                            k: parsed[k]
                            for k in (
                                "holding_count",
                                "stock_nav_pct",
                                "stock_value_cny",
                                "disclosed_nav_pct",
                                "published_date",
                            )
                        },
                    }
                except ValueError as exc:
                    probe[stage] = {"passed": False, "reason": str(exc)}
            if not probe["original"]["passed"] and probe["spacing_only"]["passed"]:
                entry = next(e for e in compact_catalog if (e["report_end"], e["report_type"]) == period)
                publication = soup.select_one(".article_info > .txt .c")
                match = re.search(r"20\d{2}-\d{2}-\d{2}", publication.text if publication else "")
                if not match:
                    raise ValueError("REPRINT_PUBLICATION_MISSING")
                latest_day = max(probe["spacing_only"]["published_date"], match[0])
                # 本批六份的目录、报告送出日和转载日期一致；将这一前提明确核验。
                if str(date.fromisoformat(latest_day) + timedelta(days=1)) != entry["available_at"][:10]:
                    raise ValueError("REPRINT_LATER_THAN_CATALOG_REQUIRES_SEPARATE_REVIEW")
                # 这里只核算报告门槛的覆盖，不生成 x、不查行情、更不生成新训练样本。
                recovered.append(entry)
            probes.append(probe)

        errors = {e["article_id"]: e["reason"] for e in manifest["errors"] if "article_id" in e}
        details = []
        for entry in catalog:
            period = (entry["report_end"], entry["report_type"])
            details.append(
                {
                    "article_id": entry["ID"],
                    "report_end": period[0],
                    "report_type": period[1],
                    "published_date": entry["published_date"],
                    "title": entry["TITLE"],
                    "parsed": entry["ID"] in manifest["documents"],
                    "failure": errors.get(entry["ID"]),
                    "local_second_source": sohu_links.get(period),
                }
            )
        report_details[code] = {"reports": details, "local_catalog_pages": page_chain}

        accepted = {r["target"]: r for r in old["train"] if r["fund_code"] == code}
        excluded = {r["target"]: r for r in old["excluded"] if r.get("fund_code") == code}
        training_days = [d for d in days if "2021-01-01" <= d <= "2023-12-31"]
        first_nav = min(mapping)
        fund_daily = []
        for target in training_days:
            i = days.index(target)
            window = days[i - 61 : i + 1]
            ng = nav_gate(mapping, window)
            rg = report_gate(fund["reports"], fund["catalog"], target)
            reason = ng["reason"] or rg["reason"] or "ACCEPTED"
            expected = "ACCEPTED" if target in accepted else excluded[target]["reason"]
            if reason != expected:
                raise ValueError(f"FIRST_EXCLUSION_REPLAY_MISMATCH:{code}:{target}:{reason}:{expected}")
            category = None
            if ng["absent_dates"]:
                if code == "017493" and target < first_nav:
                    category = "BEFORE_C_SHARE_START"
                elif all(d < first_nav for d in ng["absent_dates"]):
                    category = "C_SHARE_61_INPUT_DAY_WARMUP" if code == "017493" else "SOURCE_START_61_INPUT_DAY_WARMUP"
                else:
                    category = "INTERNAL_NAV_GAP"
            later = report_gate(fund["reports"] + recovered, fund["catalog"], target)
            item = {
                "fund_code": code,
                "target": target,
                "first_exclusion": reason,
                "nav_gate": ng,
                "nav_gap_category": category,
                "report_gate": rg,
                "spacing_only_report_gate": later,
                "report_block_removed_only": bool(not ng["reason"] and rg["reason"] and not later["reason"]),
            }
            fund_daily.append(item)
        if len(accepted) + sum(r["first_exclusion"] != "ACCEPTED" for r in fund_daily) != len(training_days):
            raise ValueError("TRAINING_DATE_ACCOUNTING_FAILED")
        daily.extend(fund_daily)
        training_nav_days = [d for d in days if max(first_nav, "2021-01-01") <= d <= "2023-12-31"]
        reasons = {
            reason: summarize_dates([r for r in fund_daily if r["first_exclusion"] == reason])
            for reason in sorted({r["first_exclusion"] for r in fund_daily})
        }
        missing_by_latest = Counter()
        for r in fund_daily:
            if r["first_exclusion"] in ("EXPOSURE_NO_AVAILABLE_REPORT", "TRAINING_LATEST_DISCLOSURE_MISSING"):
                latest = r["report_gate"]["latest_listed"]
                key = latest["report_end"] + "/" + latest["report_type"] if latest else "NOT_LISTED"
                missing_by_latest[key] += 1
        new_rows = [r for r in third["train"] if r["fund_code"] == code]
        old_rows = list(accepted.values())
        removed = [r for r in eligibility["excluded"] if r["fund_code"] == code]
        if {r["target"] for r in old_rows} - {r["target"] for r in new_rows} != {r["target"] for r in removed}:
            raise ValueError("THIRD_ROUND_REJECTION_MISMATCH")
        lifecycle = re.sub(r"\s+", "", fund["reports"][0]["product_description_excerpt"])
        pattern = r"本基金于2022年12月5日新增C类份额" if code == "017493" else r"基金合同生效日2016年12月26日"
        if not re.search(pattern, lifecycle):
            raise ValueError("LIFECYCLE_EVIDENCE_CHANGED")
        result["funds"][code] = {
            "fund_name": catalog[0]["ShortTitle"],
            "family": fund["family"],
            "lifecycle_evidence": pattern,
            "lifecycle_source_sha256": fund["reports"][0]["raw"]["sha256"],
            "nav_frozen_rows_through_2024": len(mapping),
            "nav_start": first_nav,
            "nav_end": max(mapping),
            "nav_training_sessions_expected_from_source_start": len(training_nav_days),
            "nav_training_sessions_missing_interior": [d for d in training_nav_days if d not in mapping],
            "original_training_scope_sessions": len(training_days),
            "first_reasons": reasons,
            "nav_gap_categories": dict(Counter(r["nav_gap_category"] for r in fund_daily if r["nav_gap_category"])),
            "report_first_rejection_by_latest": dict(sorted(missing_by_latest.items())),
            "original_accepted": summarize_dates(old_rows),
            "third_accepted": summarize_dates(new_rows),
            "third_removed": removed,
            "excluded_records_outside_training_scope": dict(
                Counter(r["reason"] for r in excluded.values() if r["target"] > "2023-12-31")
            ),
            "report_catalog_count_through_2024Q3": len(details),
            "parsed_report_count": len(fund["reports"]),
            "report_failure_counts": dict(Counter(errors.values())),
            "local_second_source_full_reports": len(sohu_links),
            "locally_recovered_report_diagnostics": len(recovered),
            "spacing_only_report_gate_removed_days": summarize_dates(
                [r for r in fund_daily if r["report_block_removed_only"]]
            ),
            "per_fold_rows": {f["name"]: sum(key[0] == code for key in f["train_ids"]) for f in folds},
            "limits": [
                "Source-start gaps are not proof the provider lacks earlier NAV.",
                "Missing local reprint is not proof the issuer never published.",
                "First rejection hides later quote/availability failures; "
                "report-gate relief is not new usable samples.",
                "No predictive accuracy or causal effect can be inferred from this coverage audit.",
            ],
        }
    save_once(OUTPUT / "daily-exclusion-trace.json", daily)
    save_once(OUTPUT / "report-coverage.json", report_details)
    save_once(OUTPUT / "parser-spacing-probes.json", probes)
    save_once(OUTPUT / "summary.json", result)
    return result
