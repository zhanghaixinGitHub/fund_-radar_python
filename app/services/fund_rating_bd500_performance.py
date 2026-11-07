"""博道伍佰智航C：固定同策略七产品的三维表现原件编译候选。

仅使用inputs.sources内经Originals(root)逐次校验的字节；不读取研究目录、旧路径
或预先计算的分数。保留本C份额、727个真实估值日、36/25窗口及现行R5原公式。
基金事项采用逐项审阅的不可变底稿，审阅关闭不等于每项经营事件全部结案。
"""

import json
import re
import unicodedata
from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from statistics import median

from bs4 import BeautifulSoup

from app.core.pdf import PDF_LOCK
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import document_pdfium
from app.services.material_content import document_reader as PdfReader

VERSION = "R5_BD500_ACTIVE_QUANT_PERFORMANCE_ORIGINALS_20261029_2"
MATERIAL_REVIEW_HASH = "8dcab2ba67d8ad2da7bd319a3642f47262350f05c6e5e6cfd718cd5462b83b02"
COHORT_REVIEW_HASH = "91dbbc67a55d5c0b1a76a94f632197ef691719dc1862c135f55efa9f43e4626b"
CALENDAR_SOURCE = "fd01fb54d6b58f134f578b2d2f02fa682a62f423f60e0836eeb93f9d95ecef40"
TARGET = "007832"
ALIASES = {"007831", "007832"}
PEERS = {
    "009874": "009874.OF",
    "501219": "501219.SH",
    "011589": "011589.OF",
    "010120": "010120.OF",
    "009043": "009043.OF",
    "002871": "002871.OF",
    "000978": "000978.OF",
}
PERIODS = [
    "2023-09-30",
    "2023-12-31",
    "2024-03-31",
    "2024-06-30",
    "2024-09-30",
    "2024-12-31",
    "2025-03-31",
    "2025-06-30",
    "2025-09-30",
    "2025-12-31",
    "2026-03-31",
    "2026-06-30",
]
TIME_CONTRACT = {
    "kind": "RETROSPECTIVE_HISTORICAL_QUALITY_RESEARCH",
    "valuation_window_end": "as_of_date",
    "nav_and_market_publication_boundary": "research_on",
    "fundamental_disclosure_boundary": "as_of_date",
    "not_historical_asof_tradable_signal": True,
}
_PAGE_CACHE = OrderedDict()
_CALCULATION_CACHE = OrderedDict()
PERIOD_CONFLICT_SOURCE = "01e6588752d7b6061ccbc941b47001d3b57d09899bc17775740047d494ace792"
TRADE_CAL_SOURCE = "bae441f0bc2b03fdcd26ec71bf3181bd4df389b7842be3839a86d0b35e84fb7e"
DISCOVERY_PAGES = [
    "f0962f8496ec26a448063757197ee7cdb08a68100c944b2b3b3b08a1179e30bd",
    "551a113c0d3463322aa0c79f217b0e4bd68e5185b873e9748319b90ecab72c3e",
    "07b687074adb06f56a32b7fc25a7c5063eaa767d5e530a0fef1713ece44e70d4",
    "21c28d66a016f01851792e9130cea8ee49893598c904718c3b0fdc26a5ef7b08",
    "3a9c140bf0e418616eb3592ca30d92467f36db6be89bd040ed494237e19560c7",
    "fbe04e67e3a0f053bdef7b1be8974ce9de2941962a66e601fce40ea308fe5e1d",
]
DISCOVERY_PLAN = "a8b6ab0ee3f42489da53e70861dc0b6da957d5e83938122f4c470ffcec43668e"
NAV_PLAN = "56f8e556478e3b3a2cd1b31a5b54174052b9623a80a57acbe8bcb11da7a737d1"
DISCOVERY_RECEIPT = "4b288c21a11f7136da2579dd708f17841f384fca0436d072da23e878f608f31f"


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def plain(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _iso(value):
    _require(isinstance(value, str) and re.fullmatch(r"\d{8}|\d{4}-\d{2}-\d{2}", value), "BD500_REAL_DATE_REQUIRED")
    return date.fromisoformat(value).isoformat()


def _json(source, digest):
    return json.loads(source.raw[digest], parse_float=D)


def _originals(inputs, root):
    _require(
        inputs.get("version") == VERSION and inputs.get("time_contract") == TIME_CONTRACT, "BD500_VERSION_TIME_CONTRACT"
    )
    _require(
        inputs.get("fund_code") == TARGET
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30",
        "BD500_TARGET_SHARE_PERIOD",
    )
    _require(
        not set(inputs) & {"score", "scores", "dimensions", "observations", "complete_score"},
        "BD500_RAW_ORIGINALS_REQUIRED",
    )
    research = _iso(inputs.get("research_on"))
    _require(research >= inputs["as_of_date"], "BD500_RESEARCH_DATE")
    prices = {m[k] for m in [inputs["target"]] + inputs["peers"] for k in ["nav_ref", "dividend_ref"]}
    prices.add(inputs["auxiliary_nav_ref"])
    prices.add(inputs["calendar_source_ref"])
    prices.add(inputs.get("exchange_calendar_ref"))
    for digest, ref in inputs["sources"].items():
        _require(ref.get("sha256") == digest and ref.get("acquired_at"), "BD500_SOURCE_DATE_REQUIRED")
        _require(_iso(ref["acquired_at"][:10]) <= research, "BD500_SOURCE_NOT_AVAILABLE_AT_RESEARCH")
        if digest in prices:
            if ref.get("published_on"):
                _require(_iso(ref["published_on"]) <= research, "BD500_PRICE_SOURCE_AFTER_RESEARCH")
        elif ref.get("origin_type") in {"REVIEW_EVIDENCE", "PUBLIC_DISCLOSURE_CATALOG", "DISCOVERY_EVIDENCE"}:
            # 底稿制作/目录取得时点不是基础事实公开日；下游逐条核原公告日期，
            # 不把10月整理的容器伪写成9月已公开的发行人文档。
            _require(not ref.get("published_on"), "BD500_RESEARCH_CONTAINER_NOT_ISSUER_PUBLICATION")
            if ref["origin_type"] == "REVIEW_EVIDENCE":
                _require(digest in {MATERIAL_REVIEW_HASH, COHORT_REVIEW_HASH}, "BD500_UNREVIEWED_RESEARCH_CONTAINER")
            if ref["origin_type"] == "DISCOVERY_EVIDENCE":
                _require(
                    digest in {*DISCOVERY_PAGES, DISCOVERY_PLAN, NAV_PLAN, DISCOVERY_RECEIPT},
                    "BD500_UNREVIEWED_DISCOVERY_CONTAINER",
                )
        else:
            _require(
                ref.get("published_on") and _iso(ref["published_on"]) <= inputs["as_of_date"],
                "BD500_FUTURE_OR_UNDATED_FUNDAMENTAL",
            )
    checked = {
        **inputs,
        "as_of_date": research,
        "sources": {
            h: {**r, **({"published_on": _iso(r["published_on"])} if r.get("published_on") else {})}
            for h, r in inputs["sources"].items()
        },
    }
    return Originals(checked, Path(root))


def _pages(source, digest):
    """每次先重验Originals；同进程仅缓存同SHA解析视图，不缓存来源准入。

    PDFium用于定位原页；仅对资产、净资产和行业等数值页恢复原PDF的列布局。
    PDF字节不改写，金额不改值，未因布局不适配删除任何同行。
    """
    _require(digest in source.raw, "BD500_SOURCE_NOT_ARCHIVED")
    if digest in _PAGE_CACHE:
        return list(_PAGE_CACHE[digest])
    raw = source.raw[digest]
    if raw.startswith(b"%PDF"):
        with PDF_LOCK:
            pages = []
            with document_pdfium(raw) as document:
                for i in range(len(document)):
                    with closing(document[i]) as page, closing(page.get_textpage()) as text:
                        pages.append(text.get_text_range())
            selected = set()
            for i, text in enumerate(pages):
                s = plain(text)
                # 目录的章节名不能触发“财务页+下一页”布局恢复，否则会把
                # 紧随目录的产品表按金额列重排，截断业绩比较基准行名。
                if "目录" in s and len(re.findall(r"\.{3,}", text)) >= 8:
                    continue
                if any(
                    k in s
                    for k in [
                        "期末基金资产组合情况",
                        "主要财务指标",
                        "期末基金资产净值",
                        "期末按行业分类",
                        "报告期末按行业分类",
                    ]
                ) or re.search(r"[67]\.1资产负债表", s):
                    selected |= {i, min(i + 1, len(pages) - 1)}
            # 旧版没有“境内”子标题时，完整股票表是境内范围的实证。只提取
            # 该连续区间的原布局，不能由缺港股章节或前十股票推全仓。
            begins = [
                i
                for i, t in enumerate(pages)
                if "期末按公允价值占基金资产净值比例大小排序的所有股票投资明细" in plain(t)
            ]
            has_mainland = any(re.search(r"\d\.2\.1报告期末按行业分类的(?:境内)?股票投资组合", plain(t)) for t in pages)
            if begins and not has_mainland:
                start = begins[-1]
                ends = [i for i in range(start, len(pages)) if "报告期内股票投资组合的重大变动" in plain(pages[i])]
                if ends:
                    selected.update(range(start, ends[0] + 1))
            reader = PdfReader(raw)
            for i in selected:
                pages[i] = reader.pages[i].extract_text(extraction_mode="layout") or ""
    elif source.sources[digest]["file"].endswith(".json"):
        pages = [_json(source, digest)["data"]["notice_content"]]
    else:
        ref = source.sources[digest]
        pages = [BeautifulSoup(raw.decode(ref.get("encoding", "utf-8")), "html.parser").get_text(" ", strip=True)]
    _PAGE_CACHE[digest] = tuple(pages)
    while len(_PAGE_CACHE) > 128:
        _PAGE_CACHE.popitem(last=False)
    return list(pages)


def _claim(source, claim):
    _require(claim.get("source_hash") in source.raw and claim.get("quote"), "BD500_EVENT_ORIGINAL_QUOTE_REQUIRED")
    body = "\n".join(_pages(source, claim["source_hash"]))
    _require(plain(claim["quote"]) in plain(body), "BD500_EVENT_ORIGINAL_QUOTE_MISMATCH")


def _calendar(inputs, source):
    _require(inputs["calendar_source_ref"] == CALENDAR_SOURCE, "BD500_FROZEN_TARGET_CALENDAR_SOURCE")
    history = _json(source, CALENDAR_SOURCE)
    _require(history["fund_code"] == TARGET, "BD500_CALENDAR_SHARE")
    days = history["history"]["dates"]
    months = history["history"]["month_ends"]
    _require(
        len(days) == 727 and days == sorted(set(days)) and days[0] == "2023-09-28" and days[-1] == inputs["as_of_date"],
        "BD500_FIXED_727_CALENDAR",
    )
    expected = [max(d for d in days if d[:7] == m) for m in sorted({d[:7] for d in days})]
    _require(
        len(months) == 37 and months == expected and inputs["calendar"] == {"dates": days, "month_ends": months},
        "BD500_FIXED_REAL_MONTH_ENDS",
    )
    # 本份额历史只用于日期交叉确认；独立交易所原始日历证明没有漏掉市场交易日。
    _require(inputs.get("exchange_calendar_ref") == TRADE_CAL_SOURCE, "BD500_EXCHANGE_CALENDAR_REQUIRED")
    body = _json(source, TRADE_CAL_SOURCE)
    data = body.get("data", {})
    _require(
        type(body.get("code")) is int and body["code"] == 0 and data.get("has_more") is False,
        "BD500_EXCHANGE_CALENDAR_RESPONSE",
    )
    fields = data.get("fields")
    _require(isinstance(fields, list) and len(fields) == len(set(fields)), "BD500_EXCHANGE_CALENDAR_SCHEMA")
    rows = [dict(zip(fields, r, strict=True)) for r in data.get("items", [])]
    _require(rows and all(r.get("exchange") == "SSE" for r in rows), "BD500_EXCHANGE_CALENDAR_IDENTITY")
    actual = sorted(
        _iso(r["cal_date"]) for r in rows if r.get("is_open") == 1 and days[0] <= _iso(r["cal_date"]) <= days[-1]
    )
    _require(actual == days, "BD500_EXCHANGE_TRADING_DAYS_MISMATCH")
    return days, months


def _material_review(inputs, source):
    digest = inputs.get("material_event_review_ref")
    _require(digest == MATERIAL_REVIEW_HASH and digest in source.raw, "BD500_REVIEWED_MATERIAL_LEDGER_REQUIRED")
    review = _json(source, digest)
    _require(
        review.get("fund_code") == TARGET
        and review.get("share_class") == "C"
        and review.get("as_of_date") == inputs["as_of_date"]
        and review.get("research_on") == inputs["research_on"]
        and review.get("primary_ref") == inputs["primary_ref"]
        and review.get("manager_name") == inputs["manager_name"]
        and review.get("review_closed") is True
        and review.get("blocking_event_for_selected_performance") is False,
        "BD500_MATERIAL_REVIEW_BINDING",
    )
    events = review.get("events")
    _require(
        isinstance(events, list) and len(events) == review["catalog_event_count"], "BD500_ALL_CATALOG_EVENTS_REVIEWED"
    )
    catalog = {}
    for h in review["catalog_refs"]:
        body = _json(source, h)
        _require(body["result"]["status"]["code"] == 0, "BD500_EVENT_CATALOG_RESPONSE")
        for row in body["result"]["data"]["data"] or []:
            _require(_iso(row["fbsj"][:10]) <= inputs["as_of_date"], "BD500_EVENT_CATALOG_FUTURE")
            _require(row["id"] not in catalog or catalog[row["id"]] == row, "BD500_EVENT_CATALOG_CONFLICT")
            catalog[row["id"]] = row
    _require(set(catalog) == {r["id"] for r in events}, "BD500_EVENT_CATALOG_REVIEW_CLOSURE")
    keys = set()
    for event in events:
        _require(
            event.get("id") not in keys
            and event.get("reviewed") is True
            and event.get("blocks_selected_performance") is False
            and event.get("reason")
            and event.get("claims"),
            "BD500_UNKNOWN_OR_BLOCKING_EVENT",
        )
        keys.add(event["id"])
        row = catalog[event["id"]]
        _require(
            event["published_on"] == _iso(row["fbsj"][:10]) and plain(event["title"]) == plain(row["title"]),
            "BD500_EVENT_CATALOG_DATE_TITLE_BINDING",
        )
        for claim in event["claims"]:
            _claim(source, claim)
    for claim in review["manager_and_trustee_claims"]:
        _claim(source, claim)
    _require(inputs.get("material_event_pending") is False, "BD500_MATERIAL_FLAG_CONFLICT")
    return {
        "review_hash": digest,
        "events_reviewed": len(events),
        "review_closed": True,
        "not_all_economic_events_claimed_closed": True,
        "blocking_event_for_selected_performance": review["blocking_event_for_selected_performance"],
    }


def _discovery(inputs, source):
    """从完整分页目录重做事前筛选；不读取任何收益或按历史可得性删除候选。

    目录仅发现候选，实际法律身份与策略仍由十二期原报告核准。
    501219是同法律产品的A类上市代码，须与全部定期报告主代码闭合，
    不是把发现用的014198 C类净值冒充A类历史。
    """
    _require(
        inputs.get("discovery_page_refs") == DISCOVERY_PAGES
        and inputs.get("discovery_plan_ref") == DISCOVERY_PLAN
        and inputs.get("nav_plan_ref") == NAV_PLAN
        and inputs.get("discovery_receipt_ref") == DISCOVERY_RECEIPT,
        "BD500_FROZEN_DISCOVERY_REQUIRED",
    )
    receipt = _json(source, DISCOVERY_RECEIPT)
    plan, nav_plan = _json(source, DISCOVERY_PLAN), _json(source, NAV_PLAN)
    _require(
        receipt["ended_with_short_page"] is True
        and receipt["catalog_count"] == 25230
        and [p["raw_sha256"] for p in receipt["pages"]] == DISCOVERY_PAGES,
        "BD500_DISCOVERY_PAGING_RECEIPT",
    )
    catalog = []
    for number, digest in enumerate(DISCOVERY_PAGES):
        body = _json(source, digest)
        data = body.get("data", {})
        _require(
            type(body.get("code")) is int and body["code"] == 0 and data.get("has_more") is (number < 5),
            "BD500_DISCOVERY_PAGE_CLOSURE",
        )
        fields = data.get("fields")
        _require(isinstance(fields, list) and len(fields) == len(set(fields)), "BD500_DISCOVERY_SCHEMA")
        rows = [dict(zip(fields, r, strict=True)) for r in data["items"]]
        _require(
            len(rows) == (5000 if number < 5 else 230) and receipt["pages"][number]["offset"] == number * 5000,
            "BD500_DISCOVERY_PAGE_COUNT",
        )
        catalog.extend(rows)
    _require(
        len({r["ts_code"] for r in catalog}) == len(catalog) == 25230, "BD500_DISCOVERY_DUPLICATE_OR_MISSING_PRODUCT"
    )
    groups = {}
    for row in catalog:
        benchmark = plain(row.get("benchmark") or "").replace("*", "×")
        if (
            row.get("fund_type") == "股票型"
            and row.get("invest_type") == "股票型"
            and row.get("found_date")
            and row["found_date"] <= "20230928"
            and "QDII" not in row["name"]
            and row["ts_code"][:6] not in ALIASES
            and re.search(r"(?:中证(?:小盘)?500指数(?:收益率)?×95%|95%×中证500)", benchmark)
            and "活期" in benchmark
        ):
            name = re.sub(r"-[ACEY]$", "", row["name"])
            groups.setdefault(name, []).append(row)
    frozen = {r["name"]: r for r in plan["candidates"]}
    _require(set(groups) == set(frozen) and len(groups) == 7, "BD500_DISCOVERY_UNREVIEWED_OR_DROPPED_PRODUCT")
    selected = {}
    for name, aliases in groups.items():
        _require(
            sorted(aliases, key=lambda r: r["ts_code"]) == sorted(frozen[name]["aliases"], key=lambda r: r["ts_code"]),
            "BD500_DISCOVERY_ALIAS_CHANGE",
        )
        representative = min(aliases, key=lambda r: (r["found_date"], r["name"].endswith("-C"), r["ts_code"]))
        _require(representative["ts_code"][:6] == frozen[name]["representative"], "BD500_DISCOVERY_REPRESENTATIVE")
        selected[representative["ts_code"][:6]] = representative["ts_code"]
    _require(
        nav_plan["protocol_sha256"] == DISCOVERY_PLAN
        and nav_plan["representatives"] == {**selected, "014198": "501219.SH", TARGET: "007832.OF"}
        and plan["frozen_at"] < nav_plan["frozen_at"],
        "BD500_NAV_PLAN_NOT_FROZEN",
    )
    _require(
        _json(source, COHORT_REVIEW_HASH)["source_plan_sha256"] == DISCOVERY_PLAN, "BD500_COHORT_DISCOVERY_NOT_BOUND"
    )
    return {
        "paged_rows": len(catalog),
        "complete_pages": len(DISCOVERY_PAGES),
        "legal_candidates": len(groups),
        "nav_representatives": nav_plan["representatives"],
        "performance_used_for_selection": False,
    }


def _verify_reports(inputs, source):
    review_hash = inputs.get("cohort_review_ref")
    _require(review_hash == COHORT_REVIEW_HASH and review_hash in source.raw, "BD500_FIXED_STRATEGY_REVIEW_REQUIRED")
    review = _json(source, review_hash)
    _require(
        review.get("expected_tickers") == PEERS and review.get("target_aliases") == sorted(ALIASES),
        "BD500_FROZEN_LEGAL_PRODUCTS",
    )
    peers = inputs["peers"]
    _require({m["code"]: m["ticker"] for m in peers} == PEERS and len(peers) == 7, "BD500_FIXED_SEVEN_PRODUCTS")
    allmembers = [inputs["target"]] + peers
    by_member = {}
    legal = []
    audit = []
    period_conflicts = []
    contract_audit = []
    for member in allmembers:
        code = member["code"]
        reports = member["reports"]
        frozen = review["members"][code]
        _require(
            member["ticker"] == frozen["ticker"]
            and member["master_code"] == frozen["master_code"]
            and member["legal_name"] == frozen["legal_name"]
            and reports == frozen["reports"],
            "BD500_REVIEWED_MEMBER_SOURCE_BINDING",
        )
        _require(
            len(reports) == 12 and {r["period_end"] for r in reports} == set(PERIODS),
            "BD500_FULL_12_PERIOD_BUDGET:" + code,
        )
        rows = []
        masters = set()
        for ref in reports:
            digest = ref["source_hash"]
            pages = _pages(source, digest)
            text = "\n".join(pages)
            s = plain(text)
            header = s[:25000]
            _require(code in header or member["master_code"] in header, "BD500_REPORT_LEGAL_SHARE:" + code)
            _require(plain(member["legal_name"]) in header, "BD500_REPORT_LEGAL_NAME:" + code)
            actual_end = str(report_period(text)[1])
            if actual_end != ref["period_end"]:
                # 原件009043一季报提示段真实写到12月31日；不能把报告文字
                # 静默改为3月31日。只对此已逐页审阅的固定SHA，核封面、
                # 财务表期间、公开时点及同日原单位净值的独立一致性。
                decision = review.get("period_conflicts", {}).get(digest)
                _require(
                    decision
                    and digest == PERIOD_CONFLICT_SOURCE
                    and code == "009043"
                    and ref["kind"] == "QUARTER"
                    and ref["period_end"] == "2026-03-31"
                    and actual_end == "2026-12-31"
                    and decision["admitted_point_period"] == ref["period_end"]
                    and decision["source_narrative_end"] == actual_end
                    and source.sources[digest]["published_on"] == "2026-04-22",
                    "BD500_UNREVIEWED_REPORT_PERIOD_CONFLICT:" + code,
                )
                for claim in decision["claims"]:
                    _claim(source, claim)
                # 注意不覆盖外层十二期财务观察 rows；NAV仅作独立交叉核验。
                nav_rows = _provider_rows(source, member["nav_ref"], member, "fund_nav")
                own = [r for r in nav_rows if _iso(r["nav_date"]) == "2026-03-31"]
                _require(
                    len(own) == 1
                    and D(str(own[0]["unit_nav"])) == D("1.1807")
                    and "期末基金资产净值12,962,224.05" in s
                    and "期末基金份额净值1.1807" in s,
                    "BD500_REPORT_CONFLICT_FINANCIAL_POINT_NOT_CORROBORATED",
                )
                _require(
                    "2026年第1季度报告2026年3月31日" in s[:3000]
                    and re.search(r"报告期[（(]2026年1月1日[－—-]2026年3月31日[）)]", s),
                    "BD500_REPORT_CONFLICT_FINANCIAL_PERIOD_REQUIRED",
                )
                period_conflicts.append(
                    {
                        "code": code,
                        "source_hash": digest,
                        "admitted_point_period": ref["period_end"],
                        "source_narrative_end": actual_end,
                        "original_bytes_or_words_changed": False,
                        "coordinator_attention_required": True,
                    }
                )
            match = re.search(r"基金主代码(\d{6})", header)
            _require(match is not None, "BD500_REPORT_MASTER_CODE")
            masters.add(match[1])
            _require(match[1] == member["master_code"], "BD500_REPORT_MASTER_CHANGED")
            _require(source.sources[digest]["published_on"] <= inputs["as_of_date"], "BD500_FUTURE_CONTRACT")
            if ref["kind"] != "QUARTER":
                _require(
                    "股票" in member["legal_name"] and "指数增强" not in member["legal_name"] and "量化" in s,
                    "BD500_ACTIVE_QUANT_STRATEGY:" + code,
                )
                contract_audit.append(_strategy_contract(code, digest, s))
            try:
                row = _budget({"pages": pages, "entry": ref}, require_mainland=code != TARGET)
            except (ValueError, IndexError) as error:
                raise ValueError("BD500_BUDGET:" + code + ":" + ref["period_end"] + ":" + str(error)) from error
            rows.append(row)
        _require(len(masters) == 1, "BD500_MULTIPLE_MASTER_PRODUCTS")
        legal.append(next(iter(masters)))
        by_member[code] = {r["period_end"]: r for r in rows}
    _require(len(set(legal)) == 8 and not set(legal[1:]) & ALIASES, "BD500_DUPLICATE_LEGAL_PRODUCTS")
    for member in peers:
        code = member["code"]
        rows = by_member[code]
        gaps = [abs(D(rows[p]["stock_nav_ratio"]) - D(by_member[TARGET][p]["stock_nav_ratio"])) for p in PERIODS]
        _require(sum(gaps) / 12 <= D(".05") and max(gaps) <= D(".10"), "BD500_FROZEN_STOCK_BUDGET_GATE:" + code)
        _require(
            all(D(rows[p]["mainland_stock_fraction"]) >= D(".90") for p in PERIODS),
            "BD500_ACTUAL_MAINLAND_GATE:" + code,
        )
        audit.append(
            {
                "code": code,
                "master_code": member["master_code"],
                "periods": 12,
                "mean_gap_pp": str(sum(gaps) / 12 * 100),
                "max_gap_pp": str(max(gaps) * 100),
                "source_period_conflicts": [r for r in period_conflicts if r["code"] == code],
                "contract_observations": [r for r in contract_audit if r["code"] == code],
            }
        )
    audit.append({"code": TARGET, "contract_observations": [r for r in contract_audit if r["code"] == TARGET]})
    return by_member, audit


def _strategy_contract(code, digest, text):
    """只在当期产品说明中核主动量化策略，避免把经理管理的另一指数基金当本产品。

    目标季度没有地域分表时，另用完整报告财务注释的明确国内投资范围；
    同行仍必须逐期从真实行业表或完整股票账簿证明境内比例。
    """
    starts = list(re.finditer(r"2\.2基金产品说明", text))
    _require(starts, "BD500_PRODUCT_STRATEGY_SECTION:" + code)
    blocks = []
    for start in starts:
        tail = text[start.end() : start.end() + 7000]
        end = re.search(r"2\.3基金管理人", tail)
        if end and all(k in tail[: end.start()] for k in ["投资目标", "投资策略", "业绩比较基"]):
            blocks.append(tail[: end.start()])
    # 排除目录和净值注释中“详见2.2”的交叉引用，不能机械取最后一次命中。
    _require(len(blocks) == 1, "BD500_PRODUCT_STRATEGY_SECTION_UNIQUE:" + code)
    block = blocks[0]
    _require(
        ("量化" in block or "数量化" in block or "定量投资模型" in block)
        and (
            any(k in block for k in ["主动", "积极", "超越"]) or ("灵活应用多种策略" in block and "多因子策略" in block)
        )
        and "指数增强" not in block
        and "被动复制" not in block,
        "BD500_PRODUCT_ACTIVE_QUANT_STRATEGY:" + code,
    )
    _require("契约型开放式" in text[:25000], "BD500_OPEN_END_LEGAL_PRODUCT:" + code)
    benchmark = block.replace("*", "×")
    _require(
        re.search(r"(?:中证(?:小盘)?500指数(?:收益率)?×95%|95%×中证(?:小盘)?500指数)", benchmark)
        and re.search(r"存款.{0,50}×5%|5%×.{0,50}存款", benchmark),
        "BD500_500_95_5_CONTRACT:" + code,
    )
    domestic = None
    if code == TARGET:
        starts = list(re.finditer(r"[67]\.4\.1基金基本情况", text))
        if starts:
            tail = text[starts[-1].end() :]
        else:
            # 当前JSON公开转载在6.2后缺少部分排版和6.4.1标题；不补出丢失
            # 段落，只对这一固定SHA使用实际存在的完整法律范围段及6.4.2闭合。
            # 其余指标不据此声称整份转载逐页无缺，亦不补评其余五维。
            _require(
                digest == "1b9355d3a9d82937bfc6a2a0f05019fe0481d383aaa5aadd87942a7222d397fd",
                "BD500_TARGET_LEGAL_SCOPE_SECTION",
            )
            legal = list(re.finditer("本基金为契约型开放式", text))
            _require(len(legal) == 1, "BD500_TARGET_LEGAL_SCOPE_PARAGRAPH")
            tail = text[legal[0].start() :]
        end = re.search(r"[67]\.4\.2会计报表的编制基础", tail)
        _require(end, "BD500_TARGET_LEGAL_SCOPE_SECTION_END")
        scope = tail[: end.start()]
        match = re.search(r"(?:本基金的?)?投资范围.{0,2500}", scope)
        _require(
            match and re.search(r"国内依法发行.{0,10}上市.{0,8}股票", match[0]) and "港股" not in match[0],
            "BD500_TARGET_EXPLICIT_DOMESTIC_EQUITY_MANDATE",
        )
        domestic = match[0]
    return {
        "code": code,
        "source_hash": digest,
        "product_strategy_quote": block,
        "target_domestic_legal_scope_quote": domestic,
    }


def _provider_rows(source, digest, member, api):
    body = _json(source, digest)
    _require(type(body.get("code")) is int and body["code"] == 0, "BD500_PROVIDER_RESPONSE")
    data = body.get("data", {})
    _require(data.get("has_more") is False, "BD500_PROVIDER_TRUNCATED_OR_UNKNOWN")
    fields = data.get("fields")
    items = data.get("items")
    _require(
        isinstance(fields, list) and len(fields) == len(set(fields)) and isinstance(items, list),
        "BD500_PROVIDER_SCHEMA",
    )
    ref = source.sources[digest]
    _require(
        ref.get("api") == api and ref.get("params", {}).get("ts_code") == member["ticker"],
        "BD500_PROVIDER_REQUEST_SHARE",
    )
    rows = [dict(zip(fields, row, strict=True)) for row in items]
    _require(all(r.get("ts_code") == member["ticker"] for r in rows), "BD500_PROVIDER_ROW_WRONG_SHARE")
    return rows


def _dividend_originals(inputs, source):
    expected = {
        "007832": ("007831007832", "2.922.92", "2023年12月12日", D(".292")),
        "000978": ("000978023853", "2.0102.010", "2025年9月29日", D(".201")),
    }
    result = {}
    for item in inputs["dividend_originals"]:
        code = item["code"]
        _require(code in expected and code not in result, "BD500_DIVIDEND_ORIGINAL_IDENTITY")
        codes, cash, day, amount = expected[code]
        s = plain("\n".join(_pages(source, item["source_hash"])))
        _require(
            "交易代码" + codes in s and re.search("分红方案.*?" + re.escape(cash), s) and "除息日" + day in s,
            "BD500_DIVIDEND_SOURCE_CASH_SHARE",
        )
        result[code] = {("2023-12-12" if code == TARGET else "2025-09-29"): amount}
    _require(set(result) == set(expected), "BD500_REAL_DISTRIBUTION_ORIGINALS_REQUIRED")
    return result


def _series(inputs, source, days, months, member, document_events):
    nav = _provider_rows(source, member["nav_ref"], member, "fund_nav")
    div = _provider_rows(source, member["dividend_ref"], member, "fund_div")
    by = {}
    cash = {}
    late = []
    for row in nav:
        day = _iso(row.get("nav_date"))
        announced = _iso(row.get("ann_date"))
        _require(day <= inputs["as_of_date"] and announced <= _iso(inputs["research_on"]), "BD500_NAV_AVAILABILITY")
        _require(day not in by or by[day] == row, "BD500_NAV_CONFLICT_DUPLICATE")
        by[day] = row
        if day in days and announced > inputs["as_of_date"]:
            late.append({"nav_date": day, "ann_date": announced, "was_known_at_nominal_cutoff": False})
    _require(set(days) <= set(by), "BD500_MISSING_REAL_NAV_DAY:" + member["code"])
    for row in div:
        _require(row.get("div_proc") in {"实施", "预案", "股东大会通过", "终止"}, "BD500_UNKNOWN_DIVIDEND_STATE")
        ex = row.get("ex_date")
        if not ex:
            _require(row["div_proc"] != "实施", "BD500_IMPLEMENTED_DIVIDEND_MISSING_DATE")
            continue
        ex = _iso(ex)
        if not days[0] < ex <= days[-1]:
            continue
        _require(
            row["div_proc"] == "实施" and _iso(row.get("ann_date")) <= days[-1], "BD500_UNKNOWN_WINDOW_DISTRIBUTION"
        )
        # 提供商允许净值除息字段为空（同一实施公告可能重复两行）；除息日
        # 仍必须存在并与本份额发行人公告的法定日期/现金完全闭合。非空冲突拒绝。
        if row.get("net_ex_date"):
            _require(_iso(row["net_ex_date"]) == ex, "BD500_DIVIDEND_NET_EX_DATE_CONFLICT")
        value = D(str(row["div_cash"]))
        _require(value.is_finite() and value >= 0, "BD500_CASH_VALUE")
        _require(ex not in cash or cash[ex] == value, "BD500_DUPLICATE_DIVIDEND_CONFLICT")
        cash[ex] = value
    _require(cash == document_events.get(member["code"], {}), "BD500_API_DIVIDEND_ORIGINAL_CONFLICT")
    growth = [D(1)]
    returns = []
    flags = []
    for prior, day in zip(days, days[1:], strict=False):
        a = D(str(by[prior]["unit_nav"]))
        b = D(str(by[day]["unit_nav"]))
        base = a - cash.get(day, D(0))
        _require(a.is_finite() and b.is_finite() and a > 0 and b > 0 and base > 0, "BD500_NAV_VALUE")
        ratio = b / base
        returns.append(ratio - 1)
        growth.append(growth[-1] * ratio)
        aa, bb = by[prior].get("adj_nav"), by[day].get("adj_nav")
        _require(aa is not None and bb is not None, "BD500_AUXILIARY_EVENT_CROSS_CHECK_MISSING")
        if abs(D(str(bb)) / D(str(aa)) / ratio - 1) > D(".003"):
            flags.append(day)
    auxiliary_review = None
    if flags:
        _require(member["code"] == "002871" and flags == ["2026-06-02"] and not cash, "BD500_UNKNOWN_CORPORATE_ACTION")
        raw = source.raw[inputs["auxiliary_nav_ref"]]
        soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
        actual = {}
        for tr in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if cells and re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]):
                actual[cells[0]] = cells
        _require(len(actual) == 7 and "2026-06-02" in actual, "BD500_002871_PRIMARY_NAV_WINDOW")
        for day, cells in actual.items():
            _require(
                D(cells[1]) == D(str(by[day]["unit_nav"])) and D(cells[2]) == D(str(by[day]["accum_nav"])),
                "BD500_002871_PRIMARY_UNIT_CONFLICT",
            )
        auxiliary_review = _002871_no_split(source)
    index = {d: i for i, d in enumerate(days)}
    one = growth[-1] / growth[index["2025-09-30"]] - 1
    three = growth[-1] ** (D("365.25") / D((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days)) - 1
    return {
        "growth": growth,
        "daily": returns,
        "return_1y": one,
        "return_3y": three,
        "index": index,
        "late_announcements": late,
        "cash": cash,
        "resolved_auxiliary_flags": flags,
        "auxiliary_event_review": auxiliary_review,
    }


def _002871_no_split(source):
    """净值异常附近用真实单位/累计净值双源，并以整期份额流量闭合拆分项。

    破折号不直接赋0：两份额期末－期初－申购＋赎回均精确为0，才证明
    拆分变动没有额外份额。该检查不把错误供应商adj_nav当基金真实增长。
    """
    digest = "a7693953cff04b3e5340ced943077434efca825e73325fb5763d0691180dabc8"
    text = plain("\n".join(_pages(source, digest)))
    _require(
        "华夏智胜价值成长股票型发起式证券投资基金" in text[:3000]
        and str(report_period(text)[1]) == "2026-06-30"
        and "本基金本报告期未进行利润分配" in text
        and "本基金本报告期无利润分配事项" in text,
        "BD500_002871_ISSUER_NO_DISTRIBUTION",
    )
    start, end = text.rfind("§9开放式基金份额变动"), text.rfind("§10重大事件揭示")
    _require(0 <= start < end, "BD500_002871_SHARE_FLOW_SECTION")
    block = text[start:end]
    names = [
        "本报告期期初基金份额总额",
        "本报告期基金总申购份额",
        "减:本报告期基金总赎回份额",
        "本报告期基金拆分变动份额",
        "本报告期期末基金份额总额",
    ]
    positions = [block.find(n) for n in names]
    _require(min(positions) >= 0 and positions == sorted(set(positions)), "BD500_002871_SHARE_FLOW_ROWS")
    values = []
    for n in [0, 1, 2, 4]:
        stop = positions[n + 1] if n < 4 else len(block)
        cells = re.findall(r"[\d,]+\.\d{2}", block[positions[n] + len(names[n]) : stop])
        _require(len(cells) == 2, "BD500_002871_SHARE_FLOW_COLUMNS")
        values.append([D(v.replace(",", "")) for v in cells])
    differences = [values[3][n] - values[0][n] - values[1][n] + values[2][n] for n in range(2)]
    _require(differences == [D(0), D(0)], "BD500_002871_UNRESOLVED_SPLIT_FLOW")
    return {
        "issuer_report_hash": digest,
        "original_share_flow_quote": block,
        "residual_share_changes": [str(v) for v in differences],
        "dash_imputed_zero": False,
        "issuer_disclosed_no_distribution": True,
        "provider_adjusted_series_used_for_returns": False,
    }


def calculate(inputs, *, root):
    source = _originals(inputs, root)
    # 每次仍先重验全部原件字节与来源时点。只缓存相同输入下的重解析结果，
    # 避免prepare后快照校验重复解析94份PDF；不能用缓存绕过原件变化。
    cache_key = (str(Path(root).resolve()), json.dumps(inputs, ensure_ascii=False, sort_keys=True))
    if cache_key in _CALCULATION_CACHE:
        return deepcopy(_CALCULATION_CACHE[cache_key])
    event_audit = _material_review(inputs, source)
    discovery_audit = _discovery(inputs, source)
    days, months = _calendar(inputs, source)
    _require(
        inputs["target"]["code"] == TARGET and inputs["target"]["ticker"] == "007832.OF", "BD500_OWN_C_NAV_REQUIRED"
    )
    primary = plain("\n".join(_pages(source, inputs["primary_ref"])))
    _require(
        inputs["legal_name"] in primary[:20000]
        and inputs["manager_name"] in primary[:20000]
        and TARGET in primary[:20000]
        and inputs["holding_date"] == "2026-06-30",
        "BD500_PRIMARY_REPORT_IDENTITY",
    )
    with localcontext() as context:
        context.prec = 70
        _, cohort_audit = _verify_reports(inputs, source)
        dividends = _dividend_originals(inputs, source)
        series = {
            m["code"]: _series(inputs, source, days, months, m, dividends) for m in [inputs["target"]] + inputs["peers"]
        }
        codes = [m["code"] for m in inputs["peers"]]
        risks = {}
        for code, fund in series.items():
            others = [c for c in codes if c != code]
            bench = [sum(series[c]["daily"][i] for c in others) / len(others) for i in range(len(days) - 1)]
            relative = peak = D(1)
            drawdown = negative = D(0)
            for fr, br in zip(fund["daily"], bench, strict=True):
                relative *= (1 + fr) / (1 + br)
                peak = max(peak, relative)
                drawdown = max(drawdown, 1 - relative / peak)
                negative += min(fr - br, D(0)) ** 2
            risks[code] = {
                "extra_drawdown": drawdown,
                "extra_downside": (negative * 252 / (len(days) - 1)).sqrt(),
                "reference_codes": others,
            }
        refs = list(source.sources)
        expected = ["PUBLIC:" + c for c in codes]
        nodes = {}
        for key in ["return_1y", "return_3y", "extra_drawdown", "extra_downside"]:
            values = series if key.startswith("return") else risks
            nodes[key] = {
                "kind": "rank",
                "direction": "higher" if key.startswith("return") else "lower",
                "target": str(values[TARGET][key]),
                "expected_products": expected,
                "peers": [{"product_id": "PUBLIC:" + c, "value": str(values[c][key])} for c in codes],
                "source_refs": refs,
            }
        for name, lag, n in [("monthly_success", 1, 36), ("annual_success", 12, 25)]:
            rows = []
            for i in range(lag, len(months)):
                start, end = months[i - lag], months[i]

                def growth(c, start=start, end=end):
                    f = series[c]
                    return f["growth"][f["index"][end]] / f["growth"][f["index"][start]] - 1

                rows.append(
                    {
                        "start": start,
                        "end": end,
                        "target": str(growth(TARGET)),
                        "reference": str(median(growth(c) for c in codes)),
                    }
                )
            _require(len(rows) == n, "BD500_FIXED_WINDOW_COUNT")
            nodes[name] = {"kind": "windows", "expected_windows": n, "rows": rows, "source_refs": refs}
        registry = {h: {**r, "path": str(Path(root) / r["file"])} for h, r in source.sources.items()}
        packet = {
            "fund_code": TARGET,
            "basis": ACTIVE_BASIS,
            "as_of_date": inputs["as_of_date"],
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_metrics": {
                "target_product_aliases": ["PUBLIC:" + c for c in sorted(ALIASES)],
                "material_event_pending": event_audit["blocking_event_for_selected_performance"],
                "observations": nodes,
            },
            "r4_evidence_registry": registry,
            "raw_archive": [{"kind": "ORIGINAL", "sha256": h} for h in refs],
            "quality_public": {"dataDates": {"nav": inputs["as_of_date"], "holdings": inputs["holding_date"]}},
        }
        result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"return", "risk", "stability"})
    _require(all(result.get(k) is None for k in ["score", "raw_score", "grade"]), "BD500_NO_COMPLETE_TOTAL")
    answer = {
        "dimensions": result["dimensions"],
        "observations": nodes,
        "complete_score": None,
        "grade": None,
        "standard_replay": result,
        "audit": {
            "cohort": cohort_audit,
            "discovery": discovery_audit,
            "material_review": event_audit,
            "calendar_days": len(days),
            "months": len(months),
            "nav_availability": {c: s["late_announcements"] for c, s in series.items()},
            "auxiliary_event_reviews": {
                c: s["auxiliary_event_review"] for c, s in series.items() if s["auxiliary_event_review"] is not None
            },
            "unit_or_cash_values_changed": False,
        },
    }
    _CALCULATION_CACHE[cache_key] = deepcopy(answer)
    while len(_CALCULATION_CACHE) > 2:
        _CALCULATION_CACHE.popitem(last=False)
    return answer


def check_binding(bundle, inputs):
    """绑定fresh current本C份额、完整主报告、管理人及三类日期，不接受同名A份额。"""
    _require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code") == TARGET
        and bundle.get("share_class") == inputs.get("share_class") == "C"
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date")
        and str(bundle.get("holding_date")) == inputs.get("holding_date")
        and bundle["identity"]["legal_name"] == inputs.get("legal_name")
        and bundle["identity"]["manager_name"] == inputs.get("manager_name")
        and bundle["original_report_archive"]["sha256"] == inputs.get("primary_ref"),
        "BD500_CURRENT_BINDING",
    )


CONFLICTING_PACKAGES = (
    "r5_calculation",
    "passive_components_calculation",
    "passive_performance_calculation",
    "stability_calculation",
    "gold_components_calculation",
    "enhanced_performance_calculation",
    "bosera_performance_calculation",
    "active_performance_calculations",
)


def validate_input_archive(bundle, *, root, archive_reader):
    """独立比对归档输入和内联输入，避免只校验一端后使用另一端。"""
    package = bundle.get("bd500_performance_calculation")
    _require(
        isinstance(package, dict)
        and isinstance(package.get("inputs"), dict)
        and isinstance(package.get("input_archive"), dict),
        "BD500_INPUT_ARCHIVE_REQUIRED",
    )
    _require(archive_reader(package["input_archive"], Path(root)) == package["inputs"], "BD500_INPUT_ARCHIVE_MISMATCH")


def check_snapshot(bundle, dimensions, *, root, calculator=None):
    """快照从原件重编译三维，并比对lower/upper/candidate每一端。"""
    _require(not any(bundle.get(k) for k in CONFLICTING_PACKAGES), "BD500_CONFLICTING_PERFORMANCE_INPUTS")
    inputs = bundle["bd500_performance_calculation"]["inputs"]
    check_binding(bundle, inputs)
    result = (calculator or calculate)(inputs, root=root)
    for key in ("return", "risk", "stability"):
        point = D(str(result["dimensions"][key]))
        _require(point.is_finite() and 0 <= point <= 100, "BD500_INVALID_COMPILED_POINT")
        for bound in ("lower", "upper", "candidate"):
            value = dimensions.get(key, {}).get(bound)
            _require(
                value is not None and D(str(value)).is_finite() and abs(D(str(value)) - point) <= D("1e-55"),
                "BD500_DIMENSION_RECALCULATION:" + key + ":" + bound,
            )
    return result


def prepare(session, request, *, root):
    """父层已canonical sources；这里仅从fresh current和归档root准备三维增量。

    不再读取sources.path/source_root外部路径，不延长verified_at/有效期，不替换管理分。
    """
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    return _prepare(
        session,
        request,
        root=Path(root),
        current_reader=current_inputs,
        archive_writer=archive_json,
        source_validator=validate_sources,
        snapshot_calculator=snapshot_calculate,
        original_calculator=calculate,
    )


def _prepare(
    session,
    request,
    *,
    root,
    current_reader,
    archive_writer,
    source_validator,
    snapshot_calculator,
    original_calculator,
):
    """依赖注入仅供轻量边界测试；正式入口全部绑定真实已有服务。"""
    from app.services.fund_rating_reviewed_snapshot import BASIS

    previous = current_reader(session, [TARGET]).get(TARGET)
    _require(
        request.get("fund_code") == TARGET and previous and str(previous["as_of_date"]) == request["as_of_date"],
        "BD500_CURRENT_PERIOD_REQUIRED",
    )
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == TARGET))
    inputs = deepcopy(request["inputs"])
    check_binding(bundle, inputs)
    _require(
        bundle.get("basis") == BASIS and not any(bundle.get(k) for k in CONFLICTING_PACKAGES),
        "BD500_PARTIAL_CURRENT_REQUIRED",
    )
    result = original_calculator(inputs, root=root)
    doc = bundle["reviewed_document"]
    _require(doc.get("code") == TARGET and doc.get("as_of") == inputs["as_of_date"], "BD500_CURRENT_DOCUMENT_SCOPE")
    for key in ["return", "risk", "stability"]:
        reason = request["explanations"][key].strip()
        _require(len(reason) >= 40, "BD500_USER_EXPLANATION_REQUIRED")
        point = result["dimensions"][key]
        doc["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": reason}
        bundle.setdefault("public_explanations", {})[key] = reason
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.update(
        research_format="single",
        research_archive=archive_writer(Path(root), doc),
        bd500_performance_calculation={"inputs": inputs, "input_archive": archive_writer(Path(root), inputs)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(r for h, r in inputs["sources"].items() if h not in known)
    source_validator(bundle, datetime.now(UTC), root=Path(root))
    checked = snapshot_calculator(bundle, root=root)
    _require(
        checked["status"] == "PARTIAL" and all(checked.get(k) is None for k in ["score", "raw_score", "grade"]),
        "BD500_CANNOT_CLAIM_COMPLETE",
    )
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


def _budget(report, *, require_mainland=True):
    """实际股票/NAV以全部份额闭合；同行须另证实际境内股票比例。

    目标季度仅披露总行业表时，地区份额保持未知，不由前十股或合同填100%。
    目标地区适配依据已审阅的境内股票法律机会集合；同行90%门槛仍逐期实证。
    """
    from app.services.fund_quality_v2_balance_proofs import strict_balance_values
    from app.services.fund_quality_v2_stock_book import periodic_financial_view
    from app.services.fund_rating_passive_portfolio import portfolio

    text = unicodedata.normalize("NFKC", "\n".join(report["pages"]))
    entry = report["entry"]
    p = plain(text)
    start = list(re.finditer("期末基金资产组合情况", text))[-1].end()
    tail = text[start:]
    stop = re.search("[57]\\.2\\s*(?:报告期末按行业|期末按行业)", tail)
    if stop is None:
        stop = re.search("\\d\\.2\\s*(?:报告期末按行业|期末按行业)", tail)
    _require(stop, "资产表结束标题未闭合")
    table = tail[: stop.start()]
    money = "(\\d[\\d,]*\\.\\d{2})"
    equity = re.search("权益投资\\s+" + money, table)
    total = re.search("(?m)^\\s*9\\s+合计\\s+" + money + "\\s+100\\.00\\s*$", table)
    _require(equity and total, "完整资产组合表股票与资产总计")
    stock = D(equity[1].replace(",", ""))
    total = D(total[1].replace(",", ""))
    if entry["kind"] == "QUARTER":
        starts = list(re.finditer("(?m)^\\s*3\\.1\\s*主要财务指标\\s*$", text))
        _require(len(starts) == 1, "季度财务段未唯一识别")
        block = text[starts[0].end() :]
        end = re.search("3\\.2\\s*基金净值表现", block)
        _require(end, "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED")
        block = block[: end.start()]
        match = re.search(
            "(?m)^\\s*(?:4[.、]?\\s*)?期末基金资产净值(?:[（(]元[）)])?\\s+((?:[\\d,]+\\.\\d{2}\\s*)+)\\s*$", block
        )
        if match is None:
            match = re.search("(?m)^\\s*4[.、]\\s*期末基金资产净\\s+((?:[\\d,]+\\.\\d{2}\\s*)+)\\s*值\\s*$", block)
        _require(match, "季度NAV行未唯一识别")
        parts = re.findall("[\\d,]+\\.\\d{2}", match[1])
        _require(1 <= len(parts) <= 8, "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED")
        # 带目录的目标季报在产品概况前已出现3.1标题；以实际独立行标题
        # 的位置取概况，不把目录位置误当财务章节起点。
        basic = plain(text[: starts[0].start()])
        code_match = re.search("下属(?:分级)?基金的交易代码((?:\\d{6})+)", basic)
        if code_match is None:
            code_match = re.search("下属分级基金((?:\\d{6})+)的交易代码", basic)
        if code_match is None:
            code_match = re.search("下属分级基金的交易代((?:\\d{6})+)码", basic)
        if code_match:
            codes = re.findall("\\d{6}", code_match[1])
            _require(len(parts) == len(codes), "净资产列数与份额代码列未闭合")
        else:
            _require(len(parts) == 1, "多份额必须有明确代码列")
            codes = [re.search("基金主代码(\\d{6})", basic)[1]]
        nav = sum(D(v.replace(",", "")) for v in parts)
        proof = {"source_row": match[0], "share_codes": codes, "share_net_assets_cny": parts, "source_table": table}
    else:
        view, trace = periodic_financial_view(report["pages"])
        balance = strict_balance_values(view, observed_on=entry["period_end"])
        _require(balance["point_book"], "完整财报净资产未唯一识别")
        b = balance["point_book"]
        nav = D(b["net_assets"])
        _require(D(b["stock_amount"]) == stock and D(b["total_assets"]) == total, "全文组合/财报不一致")
        proof = {"balance": balance, "financial_view": trace, "source_table": table}
    mainland_start = list(re.finditer("\\d\\.2\\.1\\s*报告期末按行业分类的境内股票投资组合", text))
    if not mainland_start:
        mainland_start = list(re.finditer("\\d\\.2\\.1\\s*报告期末按行业分类的股票投资组合", text))
    if not require_mainland and entry["kind"] == "QUARTER" and not mainland_start:
        return {
            "period_end": entry["period_end"],
            "stock_cny": str(stock),
            "total_assets_cny": str(total),
            "nav_cny": str(nav),
            "mainland_stock_cny": None,
            "mainland_stock_fraction": None,
            "country_unknown_at_quarterly_resolution": True,
            "stock_nav_ratio": str(stock / nav),
            "stock_total_asset_ratio": str(stock / total),
            "proof": proof,
        }
    if mainland_start:
        mainland_tail = text[mainland_start[-1].end() :]
        mainland_end = re.search("\\d\\.2\\.2|\\d\\.3\\s*期末", mainland_tail)
        _require(mainland_end, "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED")
        mainland_table = mainland_tail[: mainland_end.start()]
        mainland_total = re.search("(?m)^\\s*合计\\s+" + money + "\\s+\\d+\\.\\d+\\s*$", mainland_table)
        _require(mainland_total, "境内股票金额完整合计")
        mainland = D(mainland_total[1].replace(",", ""))
        _require(0 < mainland <= stock, "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED")
    else:
        _require(entry["kind"] != "QUARTER", "季度省略子标题不能据前十股推全组合")
        try:
            full = portfolio(report["pages"], holding_date=entry["period_end"])
        except ValueError as error:
            start = list(re.finditer("7\\.3\\s*期末按公允价值占基金资产净值比例大小排序的所有股票投资明细", text))[
                -1
            ].end()
            end = re.search("7\\.4\\s*报告期内股票投资组合", text[start:])
            _require(end, "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED")
            stock_table = text[start : start + end.start()]
            rows = list(
                re.finditer(
                    "(?m)^\\s*(\\d{1,4})[ \\t]+(\\d{6})[ \\t]+(.+?)"
                    "[ \\t]+([\\d,]+(?:\\.\\d+)?)[ \\t]+([\\d,]+\\.\\d{2})[ \\t]+(\\d+\\.\\d{2})[ \\t]*$",
                    stock_table,
                )
            )
            _require(rows and [int(r[1]) for r in rows] == list(range(1, len(rows) + 1)), "完整股票表序号必须连续")
            _require(
                all(re.fullmatch("(?:00[0-3]|30[01]|60[0135]|68[89])\\d{3}", r[2]) for r in rows),
                "必须均为境内交易所证券代码",
            )
            _require(len({r[2] for r in rows}) == len(rows), "股票代码不得重复")
            _require(sum(D(r[5].replace(",", "")) for r in rows) == stock, "原始完整股票明细金额不闭合")
            full = {
                "stock_total": str(stock),
                "positions": [
                    {
                        "stock_code": r[2] + (".SH" if r[2][0] == "6" else ".SZ"),
                        "stock_name": r[3],
                        "market_value_cny": r[5].replace(",", ""),
                        "source_row": r[0],
                    }
                    for r in rows
                ],
                "standard_parser_error": str(error),
                "fallback": "逐行原值、连续序号、完整金额闭合",
            }
        _require(D(full["stock_total"]) == stock and full["positions"], "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED")
        _require(
            all(r["stock_code"].endswith((".SH", ".SZ", ".BJ")) for r in full["positions"]),
            "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED",
        )
        _require(
            sum(D(r["market_value_cny"]) for r in full["positions"]) == stock,
            "BD500_RAW_REPORT_BUDGET_NOT_IDENTIFIED",
        )
        mainland = stock
        mainland_table = {
            "basis": "完整股票账簿逐项市场标识与金额闭合",
            "positions": full["positions"],
            "stock_total": full["stock_total"],
        }
    return {
        "period_end": entry["period_end"],
        "stock_cny": str(stock),
        "total_assets_cny": str(total),
        "nav_cny": str(nav),
        "mainland_stock_cny": str(mainland),
        "mainland_stock_fraction": str(mainland / stock),
        "mainland_table": mainland_table,
        "stock_nav_ratio": str(stock / nav),
        "stock_total_asset_ratio": str(stock / total),
        "proof": proof,
        "mainland_evidence_candidates": [
            p[max(0, m.start() - 50) : m.start() + 240] for m in re.finditer("港股通|境内股票", p)
        ][:15],
    }
