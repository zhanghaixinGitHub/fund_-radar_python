"""原件适配：易方达官网逐日基准，只复算主动权益稳定性。

不改变R5公式、36月/25年窗口和30/70预算。官网累计基准只公布两位
百分数，每个窗口必须在全部原精度范围内保持同一分档，否则拒绝点值。
本模块不联网、不写正式归档/数据库、不产生完整总分；通过标准评分入口登记。
"""

import json
import re
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from app.services.direction_1d_protocol import calendar
from app.services.fund_rating_direct_index_math import performance
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen, window_frequency_score
from app.services.fund_rating_r5_rules import check_refs
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import content_digest, source_bytes, source_exists

VERSION = "EFUNDS_ACTIVE_STABILITY_ORIGINALS_CANDIDATE_20261029_1"
EXPRESSION = "沪深300指数收益率×95%+活期存款利率（税后）×5%"


def _load(inputs, root):
    """每次重读源字节、核哈希及截止日，拒绝来源越界和未来报告。"""
    raw = {}
    for digest, ref in inputs["sources"].items():
        path = (root / ref["file"]).resolve()
        if not path.is_relative_to(root / "sources") or not source_exists(path):
            raise ValueError("EFUNDS_STABILITY_SOURCE_SCOPE")
        content = source_bytes(path)
        if len(content) > 50_000_000 or content_digest(content) != digest or ref["sha256"] != digest:
            raise ValueError("EFUNDS_STABILITY_SOURCE_BYTES_CHANGED")
        if ref.get("published_on") and ref["published_on"] > inputs["as_of_date"]:
            raise ValueError("EFUNDS_STABILITY_FUTURE_DOCUMENT")
        raw[digest] = content
    return raw


def _plain(raw):
    return BeautifulSoup(json.loads(raw)["data"]["notice_content"], "html.parser").get_text("\n", strip=True)


def _norm(s):
    return re.sub(r"\s+", "", s)


def calculate(inputs: dict, *, root: Path) -> dict:
    """从原件独立编译窗口并调用标准R5选维复算，返回点值和逐窗精度证明。"""
    root = root.resolve()
    if (
        inputs.get("version") != VERSION
        or inputs.get("basis") != ACTIVE_BASIS
        or inputs.get("share_class") != "C"
        or inputs.get("fund_code") != "010737"
        or inputs.get("scope") != "SINGLE_STABILITY_DIMENSION_ONLY_TOTAL_AND_GRADE_NULL"
    ):
        raise ValueError("EFUNDS_STABILITY_IDENTITY_OR_SCOPE")
    raw = _load(inputs, root)
    refs = inputs["refs"]
    code = inputs["fund_code"]
    asof = inputs["as_of_date"]
    page = raw[refs["efunds-010737-page"]].decode("utf-8-sig")
    ref = inputs["sources"][refs["efunds-010737-page"]]
    if urlsplit(ref["url"]).netloc != "www.efunds.com.cn" or urlsplit(ref["url"]).path != f"/fund/{code}.shtml":
        raise ValueError("EFUNDS_STABILITY_PAGE_IDENTITY")
    if f"/market/2.0/his/{code}_all.js" not in page or "累计净值增长率与业绩比较基准收益率历史走势对比图" not in page:
        raise ValueError("EFUNDS_STABILITY_PAGE_SERIES_MEANING")
    overview = raw[refs["efunds-overview"]].decode("utf-8-sig")
    if not all(s in overview for s in ["benchmark/js/", "item.benchReturnSinceFound", "item.nav"]):
        raise ValueError("EFUNDS_STABILITY_PUBLISHER_FIELD_MEANING")
    contracts = {}
    expected = ["2023-12-31", "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31", "2026-06-30"]
    if sorted(inputs["reports"]) != expected:
        raise ValueError("EFUNDS_STABILITY_HISTORICAL_CONTRACT_GAP")
    for period, digest in inputs["reports"].items():
        doc = json.loads(raw[digest])
        text = _norm(_plain(raw[digest]))
        if (
            code not in text
            or _norm(inputs["legal_name"]) not in text
            or EXPRESSION not in text
            or "本基金为指数增强型股票基金" not in text
        ):
            raise ValueError("EFUNDS_STABILITY_HISTORICAL_IDENTITY_OR_BENCHMARK")
        from app.services.fund_quality_v2_disclosures import report_period

        if str(report_period(_plain(raw[digest]))[1]) != period:
            raise ValueError("EFUNDS_STABILITY_REPORT_PERIOD")
        contracts[period] = {
            "source_hash": digest,
            "expression": EXPRESSION,
            "published_on": doc["data"]["notice_date"][:10],
        }
    provider = json.loads(raw[refs["provider"]])
    h = provider["history"]
    if (
        provider["fund_code"] != code
        or provider.get("source", {}).get("enabled") is not True
        or not provider.get("source", {}).get("authorization_verified_at")
    ):
        raise ValueError("EFUNDS_STABILITY_PROVIDER_CODE_OR_AUTHORIZATION")
    dividend_ref = inputs["sources"][refs["own-dividend"]]
    dividend = json.loads(raw[refs["own-dividend"]])
    if (
        dividend_ref.get("api_name") != "fund_div"
        or dividend_ref.get("params") != {"ts_code": code + ".OF"}
        or dividend.get("code") != 0
        or dividend.get("data", {}).get("items") != []
    ):
        raise ValueError("EFUNDS_STABILITY_DIVIDEND_ORIGINAL_NOT_EMPTY")
    sessions, calendar_hash = calendar()
    days = h["dates"]
    month_ends = h["month_ends"]
    expected_days = [str(d) for d in sessions if days[0] <= str(d) <= asof]
    if h["calendar_hash"] != calendar_hash or days != expected_days or days[0] != "2023-09-28" or days[-1] != asof:
        raise ValueError("EFUNDS_STABILITY_COMMON_CALENDAR")
    own = {}
    for row in h["rows"]:
        day = row["nav_date"]
        if day in own:
            raise ValueError("EFUNDS_STABILITY_DUPLICATE_NAV")
        own[day] = row
    if not set(days) <= own.keys():
        raise ValueError("EFUNDS_STABILITY_MISSING_NAV")
    benchmark_ref = inputs["sources"][refs["efunds-benchmark"]]
    if (
        urlsplit(benchmark_ref["url"]).path != f"/benchmark/js/{code}.js"
        or urlsplit(benchmark_ref["url"]).netloc != "www.efunds.com.cn"
    ):
        raise ValueError("EFUNDS_STABILITY_BENCHMARK_SOURCE")
    rows = json.loads(raw[refs["efunds-benchmark"]])
    observed = {}
    identical_duplicates = []
    for row in rows:
        day = date.fromisoformat(row["date"]).isoformat()
        if not all(re.fullmatch(r"-?\d+\.\d{2}", row[k]) for k in ["nav", "benchReturnSinceFound"]):
            raise ValueError("EFUNDS_STABILITY_PUBLISHER_DATE_OR_PRECISION")
        if day in observed:
            if observed[day] != row:
                raise ValueError("EFUNDS_STABILITY_CONFLICTING_DUPLICATE")
            identical_duplicates.append(day)
        observed[day] = row
    if not set(days) <= observed.keys():
        raise ValueError("EFUNDS_STABILITY_BENCHMARK_CALENDAR_GAP")
    with localcontext() as ctx:
        ctx.prec = 70
        levels = {d: 1 + D(observed[d]["benchReturnSinceFound"]) / 100 for d in days}
        if min(levels.values()) <= D(".00005"):
            raise ValueError("EFUNDS_STABILITY_NONPOSITIVE_LEVEL")
        # 原始累计收益来自本C份额；逐日与既有复权净值核准。没有拼接A或Y。
        if any(D(own[d]["adjusted_nav"]) != 1 + D(observed[d]["nav"]) / 100 for d in days):
            raise ValueError("EFUNDS_STABILITY_OWN_RETURN_CONFLICT")
        if any(D(own[d]["adjusted_nav"]) != D(own[d]["unit_nav"]) for d in days):
            raise ValueError("EFUNDS_STABILITY_ADJUSTMENT_EVENT_REQUIRES_OTHER_ROUTE")
        values = performance(days, month_ends, [own[d]["adjusted_nav"] for d in days], [str(levels[d]) for d in days])
        observations = {}
        precision_audit = {}
        for name, count, step in [("monthly_success", 36, 1), ("annual_success", 25, 12)]:
            u = D(PROTOCOL["parameters"]["annual_return_gap_unit"]) / (12 if step == 1 else 1)
            checked = []
            prepared = []
            for row in values[name]:
                first, last = row["start"], row["end"]
                target = D(row["target"])
                epsilon = D(".00005")
                lower = (levels[last] - epsilon) / (levels[first] + epsilon) - 1
                upper = (levels[last] + epsilon) / (levels[first] - epsilon) - 1
                scores = [window_frequency_score(target - lower, u), window_frequency_score(target - upper, u)]
                if scores[0] != scores[1]:
                    raise ValueError("EFUNDS_STABILITY_PRECISION_AMBIGUOUS:" + last)
                checked.append(
                    {
                        "start": first,
                        "end": last,
                        "target": str(target),
                        "reference_lower": str(lower),
                        "reference_upper": str(upper),
                        "invariant_score": str(scores[0]),
                    }
                )
                prepared.append(
                    {"start": first, "end": last, "target": str(target), "reference": row["contract_reference"]}
                )
            if len(prepared) != count:
                raise ValueError("EFUNDS_STABILITY_WINDOW_GAP")
            observations[name] = {
                "kind": "windows",
                "expected_windows": count,
                "rows": prepared,
                "source_refs": [refs["provider"], refs["efunds-benchmark"], *inputs["reports"].values()],
            }
            precision_audit[name] = checked
    registry = {
        h: {"path": str(root / ref["file"]), "published_on": ref.get("published_on")}
        for h, ref in inputs["sources"].items()
    }
    for node in observations.values():
        check_refs(node, registry, asof)
    packet = {
        "fund_code": code,
        "basis": ACTIVE_BASIS,
        "as_of_date": asof,
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_public": {"dataDates": {"nav": asof, "holdings": inputs["holding_date"]}},
        "quality_metrics": {
            "observations": observations,
            "target_product_aliases": ["010736", "010737", "022914"],
            "material_event_pending": False,
        },
        "r4_evidence_registry": registry,
        "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in inputs["sources"]],
    }
    result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"stability"})
    if any(result[k] is not None for k in ["score", "raw_score", "grade"]):
        raise ValueError("EFUNDS_STABILITY_TOTAL_LEAK")
    return {
        "score": result["dimensions"]["stability"],
        "dimensions": result["dimensions"],
        "complete_score": None,
        "grade": None,
        "calculation_revision": PROTOCOL["revision"],
        "standard_replay": result,
        "observations": observations,
        "audit": {
            "contracts": contracts,
            "precision": precision_audit,
            "common_dates": len(days),
            "month_ends": len(month_ends),
            "first": days[0],
            "last": days[-1],
            "publisher_rows": len(rows),
            "identical_publisher_duplicate_dates": identical_duplicates,
            "own_share_crosschecks": len(days),
            "dividend_original_ref": refs["own-dividend"],
            "corporate_action_check": (
                "成功fund_div原回包无记录；全部727日官方本C累计净值增长与复权净值一致，"
                "且复权与单位净值一致；未假设缺失分红/折算为零。"
            ),
            "excluded_provider_nonvaluation_dates": sorted(set(own) - set(days)),
            "precision_invariant_windows": 61,
            "peer_required_for_stability": False,
            "other_dimensions_calculated": False,
        },
    }
