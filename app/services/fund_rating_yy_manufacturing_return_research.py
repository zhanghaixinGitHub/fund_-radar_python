"""018125收益原件研究重放：21固定参照、84检查点与真实净值/分红。

3份旧Word从原始CFB字节只读提取连续正文，不借旧绝对路径/旧得分；019607
仅用已核12季总股票/NAV原字段收窄行业舍入误差，阈值不变，未来季度不使用。
重大事项None保持未知，当前模块不产生正式维度或总分，也没有正式prepare出口。
"""

import json
import re
import unicodedata
from copy import deepcopy
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path
from statistics import median

from bs4 import BeautifulSoup

from app.integrations.public_fund_reports import report_period
from app.services.fund_rating_active_performance import TIME_CONTRACT, _require_time_contract
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text as plain
from app.services.fund_rating_r5 import signed_gap_score
from app.services.fund_rating_yy_manufacturing_legacy_word import extract_body
from app.services.fund_rating_yy_manufacturing_liquidity import require
from app.services.fund_rating_yy_manufacturing_market_liquidity import PERIODS, _pages
from app.services.fund_rating_yy_manufacturing_risk_stability_research import _series
from app.services.material_content import content_digest, source_bytes

VERSION = "R5_018125_RETURN_RESEARCH_ORIGINALS_20261030_1"
PEERS = (
    "001644",
    "007114",
    "007579",
    "008084",
    "010000",
    "010160",
    "011604",
    "011766",
    "012181",
    "013108",
    "013384",
    "013479",
    "014380",
    "014536",
    "014700",
    "014930",
    "015058",
    "016263",
    "017760",
    "018501",
    "019607",
)
QUARTERS = (
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
)


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("research_on") == "2026-10-30"
        and inputs.get("time_contract") == TIME_CONTRACT,
        "YYR_SCOPE",
    )
    require(
        inputs.get("material_event_pending") is None and inputs.get("performance_release_requested") is False,
        "YYR_NO_MATERIAL_DEFAULT_OR_RELEASE",
    )
    require(
        not any(k in inputs for k in ["score", "dimensions", "metrics", "observations", "packet"]),
        "YYR_NO_PRECOMPUTED_SCORE",
    )
    require(
        tuple(inputs["peer_codes"]) == PEERS
        and set(inputs["products"]) == {"018125", *PEERS}
        and set(inputs["exposure_refs"]) == {"018125", *PEERS},
        "YYR_FIXED_COHORT",
    )
    reviews = inputs["admission_reviews"]
    require(
        len(reviews) == 84 and {(r["code"], r["period"]) for r in reviews} == {(c, p) for c in PEERS for p in PERIODS},
        "YYR_REQUIRED_84_CHECKPOINTS",
    )
    c = inputs["boundary_refinement"]
    exclusion = inputs["operation_exclusion"]
    require(
        set(c["aggregate_refs"]) == {"018125", "019607"}
        and set(c["target_quarters"]) == set(QUARTERS)
        and exclusion["fund_code"] == "016118"
        and exclusion["normal_open_end_criterion_unchanged"] is True,
        "YYR_BOUNDARY_OR_OPERATION_SCOPE",
    )
    financial = {*(r["source_hash"] for r in reviews), *c["target_quarters"].values(), exclusion["source_hash"]}
    derived = {
        inputs["calendar_ref"],
        inputs["time_contract_ref"],
        inputs["discovery_ref"],
        *(p["dividend_receipt_ref"] for p in inputs["products"].values()),
    }
    original = {
        *financial,
        *(h for values in inputs["exposure_refs"].values() for h in values),
        *c["aggregate_refs"].values(),
        *c["publication_catalog_refs"],
        *(h for p in inputs["products"].values() for k in ["nav_refs", "dividend_refs"] for h in p[k]),
    }
    root = Path(root).resolve()
    ref = inputs["sources"][inputs["calendar_ref"]]
    path = (root / ref["file"]).resolve()
    require(path.is_relative_to(root / "sources"), "YYR_CALENDAR_PATH")
    raw = source_bytes(path)
    require(content_digest(raw) == inputs["calendar_ref"], "YYR_CALENDAR_INDEX_HASH")
    original.update(s["sha256"] for s in json.loads(raw)["sources"])
    require(set(inputs["sources"]) == original | derived and not original & derived, "YYR_COMPLETE_SOURCE_GRAPH")
    for h, ref in inputs["sources"].items():
        require(
            ref.get("sha256") == h and ref.get("archive_kind") == ("DERIVED" if h in derived else "ORIGINAL"),
            "YYR_SOURCE_ROLE",
        )
        if h in financial:
            require(
                isinstance(ref.get("published_on"), str)
                and date.fromisoformat(ref["published_on"]) <= date(2026, 9, 30),
                "YYR_FINANCIAL_PUBLICATION",
            )
    source = Originals(inputs, root)
    for code, p in inputs["products"].items():
        require(
            p["fund_code"] == code
            and p["ts_code"] == code + ".OF"
            and p["format"] == "TUSHARE_NAV_AND_DIVIDEND_ORIGINALS"
            and p["nav_refs"]
            and p["dividend_refs"],
            "YYR_ACTUAL_C_SHARE",
        )
        for key, api in [("nav_refs", "fund_nav"), ("dividend_refs", "fund_div")]:
            for h in p[key]:
                request = inputs["sources"][h]["request_binding"]
                require(
                    request["api_name"] == api
                    and request["status"] == 200
                    and request["params"]["ts_code"] == code + ".OF",
                    "YYR_ACTUAL_REQUEST_IDENTITY",
                )
                if api == "fund_nav":
                    require(
                        request["params"]["start_date"] == "20230928" and request["params"]["end_date"] == "20260930",
                        "YYR_REQUEST_WINDOW",
                    )
        receipt = source.json(p["dividend_receipt_ref"])
        require(
            receipt["code"] == code and {s["sha256"] for s in receipt["sources"]} == set(p["dividend_refs"]),
            "YYR_DIVIDEND_REQUEST_BINDING",
        )
        for item in receipt["sources"]:
            request = inputs["sources"][item["sha256"]]["request_binding"]
            require(
                all(item[k] == request[k] for k in ["api_name", "params", "status"]), "YYR_DIVIDEND_REQUEST_PROVENANCE"
            )
    return source


def _parts(source, h):
    if source.raw[h].startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        if h not in source.pages_cache:
            source.pages_cache[h] = [extract_body(source.raw[h])]
        return source.pages_cache[h]
    return _pages(source, h)


def _admission(inputs, source):
    """机会集合人工判断绑定原句；12季百分比另由真实回包重算，不由主题文字替代。"""

    def context(s):
        return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", s))

    review_audit = []
    for review in inputs["admission_reviews"]:
        require(
            review["normal_open_end_C_RMB"] is True and len(review["manual_opportunity_set_judgment"]) >= 20,
            "YYR_EXPLICIT_MANUAL_OPERATION_REVIEW",
        )
        parts = _parts(source, review["source_hash"])
        text = plain("\n".join(parts))
        head = plain("\n".join(parts[:3]))
        require(
            review["code"] in plain("\n".join(parts[:15])) and "人民币" in text and "契约型开放式" in context(text),
            "YYR_ORIGINAL_C_AND_OPERATION",
        )
        year = review["period"][:4]
        chinese = "".join("〇一二三四五六七八九"[int(c)] for c in year)
        titles = (
            [year + "年年度报告"]
            if review["period"].endswith("12-31")
            else [year + "年中期报告", year + "年半年度报告"]
        )
        titles.extend(
            t.replace(year, cn)
            for t in list(titles)
            for cn in [chinese, chinese.replace("〇", "0"), chinese.replace("〇", "零")]
        )
        require(any(t in head for t in titles), "YYR_REPORT_PERIOD:" + review["code"] + ":" + review["period"])
        require(
            set(review["anchors"]) == {"theme", "strategy_continuity"} and all(review["anchors"].values()),
            "YYR_THEME_AND_STRATEGY_ANCHORS",
        )
        checks = 0
        for claims in review["anchors"].values():
            for claim in claims:
                number = claim.get("physical_page_or_HTML_part", claim.get("part_or_physical_page"))
                if number is None:
                    quote = claim["full_text_anchor"]
                    require(
                        len(context(quote)) >= 10 and context(quote) in context(text), "YYR_FULL_BODY_ORIGINAL_ANCHOR"
                    )
                else:
                    quote = claim["quote"]
                    require(
                        1 <= number <= len(parts)
                        and len(context(quote)) >= 20
                        and context(quote) in context(parts[number - 1]),
                        "YYR_ORIGINAL_ADMISSION_ANCHOR:" + review["code"] + ":" + review["period"],
                    )
                checks += 1
        review_audit.append(
            {
                "code": review["code"],
                "period": review["period"],
                "original_claims": checks,
                "locator_semantics": review["locator_semantics"],
            }
        )
    excluded = inputs["operation_exclusion"]
    text = plain("\n".join(_parts(source, excluded["source_hash"])))
    require(
        "016118" in text and re.search(r"(?:一|1)年.{0,18}持有期|持有期.{0,18}(?:一|1)年", text),
        "YYR_ONE_YEAR_LOCK_SOURCE",
    )
    return review_audit


def _quarter_exposure(inputs, source):
    result = {}
    for code, refs in inputs["exposure_refs"].items():
        require(len(refs) == 4, "YYR_FOUR_YEAR_EXPOSURE_ORIGINALS")
        quarters = {}
        for h in refs:
            raw = source.json(h)
            require(raw.get("ErrCode") == 0 and raw["Data"]["FundCode"] == code, "YYR_EXPOSURE_IDENTITY")
            for q in raw["Data"]["QuarterInfos"]:
                day = q["JZRQ"]
                if day not in QUARTERS:
                    continue
                rows = q["HYPZInfo"]
                require(len({r["HYDM"] for r in rows}) == len(rows), "YYR_DUPLICATE_INDUSTRY")
                domestic = HK = error_domestic = error_HK = D(0)
                for r in rows:
                    require(r["BZDM"] == code and r["FSRQ"] == day, "YYR_EXPOSURE_ROW_SCOPE")
                    value = D(r["ZJZBL"])
                    require(0 <= value <= 110, "YYR_PERCENT_UNIT")
                    error = D(1).scaleb(value.as_tuple().exponent) / 2
                    if re.fullmatch("[A-Z]", r["HYDM"]):
                        domestic += value
                        error_domestic += error
                    elif re.fullmatch(r"\d+", r["HYDM"]):
                        HK += value
                        error_HK += error
                    else:
                        raise ValueError("YYR_UNKNOWN_REGION")
                total = domestic + HK
                error = error_domestic + error_HK
                require(0 < total <= 110, "YYR_TOTAL_PERCENT")
                quarters[day] = (total, error, max(D(0), domestic - error_domestic) / (total + error))
        require(set(quarters) == set(QUARTERS), "YYR_TWELVE_EXPOSURE_QUARTERS")
        result[code] = quarters
    return result


def _refined_019607(inputs, source):
    """总股票占净比每侧0.005pp原精度，区别于行业逐项相加误差；不改变门槛。"""
    c = inputs["boundary_refinement"]
    values = {}
    for code, h in c["aggregate_refs"].items():
        soup = BeautifulSoup(source.raw[h], "html.parser")
        require(code in soup.get_text(), "YYR_AGGREGATE_PRODUCT")
        tables = [t for t in soup.find_all("table") if "股票占净比" in t.get_text()]
        require(len(tables) == 1, "YYR_AGGREGATE_TABLE")
        rows = {}
        for tr in tables[0].select("tbody tr"):
            cells = [x.get_text(strip=True) for x in tr.find_all("td")]
            if not cells:
                continue
            require(
                re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]) and re.fullmatch(r"\d+\.\d{2}%", cells[1]),
                "YYR_ORIGINAL_PERCENT_PRECISION",
            )
            require(cells[0] not in rows, "YYR_DUPLICATE_AGGREGATE_DAY")
            rows[cells[0]] = D(cells[1][:-1])
        require(set(QUARTERS) <= rows.keys(), "YYR_AGGREGATE_QUARTERS")
        values[code] = rows
    published = {}
    for h in c["publication_catalog_refs"]:
        raw = source.json(h)
        require(raw.get("ErrCode") == 0, "YYR_PUBLICATION_CATALOG")
        for row in raw["Data"]:
            require(row["FUNDCODE"] == "019607", "YYR_PUBLICATION_PRODUCT")
            period = report_period(row["TITLE"])
            if period and period[1] == "QUARTER" and row["PUBLISHDATEDesc"][:10] <= "2026-09-30":
                published[period[0]] = row["PUBLISHDATEDesc"][:10]
    require(set(QUARTERS) <= published.keys(), "YYR_PEER_QUARTERS_NOT_FUTURE_REPORTS")
    for day, h in c["target_quarters"].items():
        text = plain("\n".join(_parts(source, h)))
        quarter = str((int(day[5:7]) - 1) // 3 + 1)
        require("018125" in text and day[:4] + "年第" + quarter + "季度报告" in text, "YYR_TARGET_QUARTER_IDENTITY")
    upper = [abs(values["018125"][d] - values["019607"][d]) + D(".01") for d in QUARTERS]
    return sum(upper) / 12, max(upper)


def _compile(inputs, source, root):
    frozen = source.json(inputs["discovery_ref"])
    require(
        frozen["return_admission"]
        == {"mean_stock_gap_pp": "10", "max_stock_gap_pp": "20", "min_domestic_fraction": "0.8"},
        "YYR_FROZEN_CRITERIA",
    )
    timing = _require_time_contract(inputs, Path(root))
    legal = _admission(inputs, source)
    with localcontext() as ctx:
        ctx.prec = 60
        quarters = _quarter_exposure(inputs, source)
        refined = _refined_019607(inputs, source)
        exposure_audit = []
        for code in PEERS:
            gaps = [
                abs(quarters[code][d][0] - quarters["018125"][d][0]) + quarters[code][d][1] + quarters["018125"][d][1]
                for d in QUARTERS
            ]
            mean, maximum = sum(gaps) / 12, max(gaps)
            if code == "019607":
                mean, maximum = refined
            domestic = min(x[2] for x in quarters[code].values())
            require(mean <= 10 and maximum <= 20 and domestic >= D(".8"), "YYR_RETURN_EXPOSURE_NOT_ADMITTED:" + code)
            exposure_audit.append(
                {
                    "code": code,
                    "mean_gap_upper_pp": str(mean),
                    "max_gap_upper_pp": str(maximum),
                    "domestic_lower": str(domestic),
                    "aggregate_original_refinement_used": code == "019607",
                }
            )
        days, months, data, date_audit = _series(inputs, root)
        exponent = D("365.25") / D((date(2026, 9, 30) - date(2023, 9, 28)).days)
        from app.services.fund_rating_active_performance import _growth

        values = {}
        for code in ["018125", *PEERS]:
            points, events = data[code]
            values[code] = {
                "return_1y": _growth(points, events, days, "20250930", "20260930") - 1,
                "return_3y": _growth(points, events, days, days[0], days[-1]) ** exponent - 1,
                "actual_dividend_events_in_window": sum(days[0] < e["date"] <= days[-1] for e in events),
                "noncalendar_NAV_dates_excluded": sorted(set(points) - set(days)),
            }
        parts = {}
        for name in ["return_1y", "return_3y"]:
            target = values["018125"][name]
            reference = median(values[c][name] for c in PEERS)
            parts[name] = {
                "target": target,
                "peer_median": reference,
                "subscore": signed_gap_score(target - reference, ".10"),
            }
        score = D(".3") * parts["return_1y"]["subscore"] + D(".7") * parts["return_3y"]["subscore"]

    def strings(value):
        if isinstance(value, D):
            return str(value)
        if isinstance(value, dict):
            return {k: strings(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strings(v) for v in value]
        return value

    return strings(
        {
            "research_scores": {"return": score},
            "dimensions": {},
            "complete_score": None,
            "grade": None,
            "performance_release_permitted": False,
            "material_event_pending": None,
            "metrics_by_share": values,
            "return_parts": parts,
            "audit": {
                "calendar_points": len(days),
                "annualization_exponent": exponent,
                "peer_count": 21,
                "legal_checkpoints": legal,
                "exposure_admission": exposure_audit,
                "date_audit": date_audit,
                "time_contract": timing,
                "Word_continuous_text_not_physical_pages": True,
                "future_2026_Q3_exposure_not_used": True,
                "exclusion_016118_due_to_preexisting_one_year_lock_not_performance": True,
            },
        }
    )


@lru_cache(maxsize=1)
def _cached(serialized, root):
    inputs = json.loads(serialized)
    return _compile(inputs, _source(inputs, root), root)


def calculate(inputs, *, root):
    _source(inputs, root)
    return deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))


def prepare(*args, **kwargs):
    raise ValueError("YYR_RESEARCH_ONLY_MATERIAL_REVIEW_NOT_CLOSED")
