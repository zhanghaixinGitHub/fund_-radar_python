"""前一中国交易日的IF期现价差；主力原价逐日映射月合约，禁止当作跨合约收益率。"""

import hashlib
from datetime import date, datetime

import numpy as np

from app.integrations import tushare_sprint_futures as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_fxi_interval as original


def root():
    return base.ROOT / "stock-index-futures-history-v1"


def require(condition, message):
    if not condition:
        raise ValueError("FUTURES_BASIS_" + message)


def reconstruct():
    """从13个原始响应重建价差，验证请求、范围、映射、合约存续期和指数单位。"""
    p = base.read(root() / "plan.json")
    require(
        hashlib.sha256((root() / "acquire.py").read_bytes()).hexdigest() == p["script_sha256"],
        "ACQUISITION_CODE_CHANGED",
    )
    for name, sha in p["code"].items():
        require(hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest() == sha, "PARSER_CHANGED")
    sample_root = base.ROOT / "stock-index-futures-feasibility-v1"
    qualification = base.read(sample_root / "qualification-result.json")
    require(
        base.digest(qualification) == p["sample_qualification_hash"] and qualification["sample_source_qualified"],
        "SAMPLE_CHANGED",
    )
    combined = {api: {} for api in parser.FIELDS}
    receipts = {}
    for q in p["queries"]:
        key = q["key"]
        request, meta, parsed = (
            base.read(root() / sub / (key + ".json")) for sub in ("requests", "responses", "parsed")
        )
        raw = (root() / "raw" / (key + ".json")).read_bytes()
        require(request["query"] == q and request["plan_hash"] == base.digest(p), "REQUEST_CHANGED")
        require(
            meta["request_hash"] == base.digest(request)
            and meta["http_status"] == 200
            and meta["bytes"] == len(raw)
            and meta["sha256"] == hashlib.sha256(raw).hexdigest(),
            "RAW_CHANGED",
        )
        require(
            datetime.fromisoformat(request["at"])
            <= datetime.fromisoformat(meta["received_at"])
            <= datetime.fromisoformat(parsed["at"]),
            "RECEIPT_TIME_CHANGED",
        )
        rows = parser.parse(raw, q["api"], q["params"]["start_date"], q["params"]["end_date"])
        require(rows == parsed["rows"] and parsed["response_hash"] == base.digest(meta), "PARSED_CHANGED")
        require(not set(rows) & set(combined[q["api"]]), "SOURCE_DATE_DUPLICATE")
        combined[q["api"]].update(rows)
        receipts[key] = base.digest(meta)
    spot = base.read(base.ROOT / "round-02/market.json")
    require(
        base.digest(spot) == p["old_spot_hash"]
        and spot["plan_hash"] == base.digest(base.read(base.ROOT / "round-02/plan.json")),
        "OLD_SPOT_CHANGED",
    )
    cash = {day: float(v["close"]) for day, v in spot["indices"]["000300.SH"].items()}
    require(not set(cash) & set(combined["index_daily"]), "SPOT_EXTENSION_OVERLAP")
    cash.update({day: row["close"] for day, row in combined["index_daily"].items()})
    basic = base.read(sample_root / "results/basic.json")
    require(base.digest(basic) == qualification["source_result_hashes"]["basic"], "CONTRACT_METADATA_CHANGED")
    contracts = {row["ts_code"]: row for row in basic["rows"]}
    days = [str(day) for day in base.calendar()[0] if "2021-01-01" <= str(day) <= "2026-09-15"]
    for rows in (combined["fut_daily"], combined["fut_mapping"], cash):
        require(set(rows) == set(days), "CALENDAR_COVERAGE_MISSING_OR_EXTRA")
    result = {}
    for day in days:
        quote, mapping = combined["fut_daily"][day], combined["fut_mapping"][day]
        code = mapping["mapping_ts_code"]
        require(code in contracts, "CONTRACT_UNKNOWN")
        contract = contracts[code]
        require(
            contract["list_date"] <= day.replace("-", "") <= contract["delist_date"]
            and contract["exchange"] == "CFFEX"
            and contract["quote_unit"] == "指数点"
            and contract["multiplier"] == 300,
            "CONTRACT_UNIT_OR_LIFETIME_INVALID",
        )
        result[day] = {
            "main_contract": code,
            "futures_close": quote["close"],
            "spot_close": cash[day],
            "basis_pct": float(100 * (quote["close"] / cash[day] - 1)),
            "calendar_days_to_expiry": (
                datetime.strptime(contract["delist_date"], "%Y%m%d").date() - date.fromisoformat(day)
            ).days,
        }
    for key in ("recent", "history"):
        sample = base.read(sample_root / "results" / f"{key}.json")
        require(base.digest(sample) == qualification["source_result_hashes"][key], "QUOTE_SAMPLE_CHANGED")
        row = next(v for v in sample["rows"] if v["ts_code"] == "IF.CFX")
        day = str(datetime.strptime(row["trade_date"], "%Y%m%d").date())
        require(
            all(combined["fut_daily"][day][k] == row[k] for k in parser.FIELDS["fut_daily"]), "QUOTE_SAMPLE_MISMATCH"
        )
    return {
        "rows": result,
        "receipts": receipts,
        "plan_hash": base.digest(p),
        "calendar_hash": base.calendar()[1],
        "historical_first_publication_verified": False,
    }


def history():
    value = base.read(root() / "qualified-history.json")
    checked = reconstruct()
    require(value["snapshot"] == checked, "QUALIFIED_SOURCE_CHANGED")
    q = base.read(root() / "qualification-result.json")
    require(
        q["history_hash"] == base.digest(value) and q["status"] == "QUALIFIED_LAGGED_BASIS_HISTORY",
        "QUALIFICATION_CHANGED",
    )
    return value


def extend(market, t, u, points):
    """U为预测日、T为上一交易日；特征明确取D=prevCN(T)，不依赖T当日是否及时发布。"""
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    index = days.index(t)
    d = days[index - 1] if index else None
    row = points.get(d)
    feature = {"date": d, "available": row is not None}
    if row is not None:
        require(d < t < u and np.isfinite(row["basis_pct"]), "FEATURE_TIME_OR_VALUE_INVALID")
        feature.update(row | {"basis_pct": float(np.clip(row["basis_pct"], -20, 20))})
    return market | {"futures_basis": feature}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "futures_basis"}}


def dataset():
    rows, proof = original.data.dataset()
    h = history()
    extra = [row | {"market": extend(row["market"], row["t"], row["u"], h["snapshot"]["rows"])} for row in rows]
    require([original_row(row) for row in extra] == rows, "ORIGINAL_ROWS_CHANGED")
    return extra, proof | {"futures_history_hash": base.digest(h)}


scope = original.data.scope
