"""按官方人民币全收益日终序列完成 001021 A 类的一年评分。"""

from datetime import date, datetime
from decimal import localcontext
from pathlib import Path

from app.services import fund_rating_bond_index_nav_partial as nav
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_evidence_preview import performance
from app.services.fund_rating_typed_disclosure import number
from app.services.material_content import content_digest

VERSION = "BOND_INDEX_001021_OFFICIAL_BENCHMARK_20261120_V1"
SOURCE_URL = (
    "https://www.spglobal.com/spdji/en/util/redesign/index-data/"
    "get-performance-data-for-datawidget-redesign.dot?indexId=92484665&language_id=1"
)
INDEX_PAGE = "https://www.spglobal.com/spdji/en/indices/fixed-income/iboxx-abf-china/"
IDENTITY = {
    "ticker": "ABTRCHIN", "isin": "GB00B068N195", "index_code": "IB001365", "index_id": 92484665,
    "currency": "CNY", "return_type": "GROSS OF TAX NOMINAL TOTAL RETURN LOCAL", "frequency": "EOD",
}
SOURCE_ROLE = "PUBLISHER_HISTORICAL_EOD_INDEX_LEVELS"


def benchmark(inputs, root, days):
    sha = inputs["benchmark_ref"]
    source = Originals(inputs, root, lazy=True)
    ref = source.sources[sha]
    raw = source.json(sha)
    if (
        ref.get("kind") != "json" or ref.get("source_date_role") != SOURCE_ROLE
        or ref.get("url") != SOURCE_URL or raw.get("source_url") != SOURCE_URL
        or raw.get("index") != IDENTITY or raw.get("coverage") != [days[0], days[-1]]
    ):
        raise ValueError("001021_BENCHMARK_IDENTITY_OR_WINDOW")
    captured = datetime.fromisoformat(raw["captured_at"])
    if captured.tzinfo is None or captured.date() < date.fromisoformat(days[-1]):
        raise ValueError("001021_BENCHMARK_CAPTURE_DATE")
    observations = raw["observations"]
    dates = [day for day, _ in observations]
    if (dates != sorted(set(dates)) or dates[0] != days[0] or dates[-1] != days[-1]
            or not set(days) <= set(dates)):
        raise ValueError("001021_BENCHMARK_VALUATION_DAYS_INCOMPLETE")
    exact_csv = "".join(f"{day},{level}\n" for day, level in observations).encode("ascii")
    if content_digest(exact_csv) != raw["extract_sha256"]:
        raise ValueError("001021_BENCHMARK_EXTRACT_HASH")
    by_day = {day: number(level) for day, level in observations}
    if min(by_day.values()) <= 0:
        raise ValueError("001021_BENCHMARK_INVALID_INDEX_LEVEL")
    return [by_day[day] for day in days], len(dates)


def calculate(inputs: dict, *, root: Path) -> dict:
    """沿用已核基金净值和冻结换分公式，只从指数原记录补三项。"""
    if (inputs["fund_code"], inputs["share_class"], inputs["version"]) != ("001021", "A", VERSION):
        raise ValueError("001021_BENCHMARK_FUND_SCOPE")
    previous = {**inputs, "version": nav.ROLLING_VERSION}
    result = nav.calculate(previous, root=root)
    if result["blocked"] or any(result["dimensions"][key] is None for key in
                                ("risk", "management", "holdings", "liquidity", "cost")):
        return result
    with localcontext() as context:
        context.prec = 65
        days, ends, values, _ = nav.rebuild(previous, root=root)
        levels, source_count = benchmark(inputs, root, days)
        points, metrics = performance({
            "dates": days, "month_ends": ends, "adjusted_nav": values, "benchmark_index": levels,
        })
        for key in ("return", "efficiency", "stability"):
            result["dimensions"][key] = str(points[key])
            result["facts"]["unscored_dimensions"].pop(key, None)
        by_day = dict(zip(days, zip(values, levels, strict=True), strict=True))
        monthly = []
        for start, end in zip(ends[:-1], ends[1:], strict=True):
            nav_start, ref_start = by_day[start]
            nav_end, ref_end = by_day[end]
            monthly.append({
                "from": start, "through": end, "fund_return": str(nav_end / nav_start - 1),
                "reference_return": str(ref_end / ref_start - 1),
            })
    result["facts"].update(
        compiler_version=VERSION,
        complete_type_validated=True,
        performance={**metrics, "benchmark": IDENTITY["ticker"], "benchmark_points": source_count,
                     "benchmark_start": str(levels[0]), "benchmark_end": str(levels[-1]), "monthly": monthly},
        source_sha256=sorted({*result["facts"]["source_sha256"], inputs["benchmark_ref"]}),
    )
    return result
