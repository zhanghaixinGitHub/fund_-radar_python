"""按完整原生港币元f57计算流动性点分，跨源差异完整留审计。

新公开原始字段替代旧万港币展示额；不取旧区间中点、不恢复未公开位数。
股票、参照、日历、FX和换分函数原样延用，不声称提供商逐笔数据完全无误。
"""

import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path

from app.services.fund_rating_direct_index import Originals, _table
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_r5 import exposure_factor, peer_gap_score
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, policy
from app.services.fund_rating_yy_manufacturing_liquidity import calculate as known_calculate
from app.services.fund_rating_yy_manufacturing_liquidity import require
from app.services.fund_rating_yy_manufacturing_market_liquidity import (
    PEERS,
    _admission,
    _books,
    _calendars,
    _halts,
    _HK_primary,
)
from app.services.fund_rating_yy_manufacturing_market_liquidity import (
    _source as legacy_source,
)
from app.services.fund_rating_yy_manufacturing_market_liquidity import (
    check_binding as legacy_binding,
)

VERSION = "R5_018125_PUBLISHED_NATIVE_LIQUIDITY_20261030_1"
METHOD = "37d379c20a5b48b55bf67ee45f6ca17287b075a6964f4e2a42172209dcdef814"
NATIVE = {
    "00179": "dd4259bd40e4005668d699c99d6b4db0ea597ec00b2bf56ae32fc2becac16e93",
    "02507": "19a713bbf4064de7e4eca6c35c053e830e9e3501e2d8566f2e767134900b6f0d",
    "06869": "dd173edc74b11c4db3aa307935b901349ea531ddbdfaa24a2ca28f1ef782f2bc",
    "09880": "e6c48b7ccbc343da2b4898b7951374f32ad1b976cbd87a111240db27bd24f25e",
}


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("share_class") == "C",
        "YYPL_SCOPE",
    )
    require(inputs.get("method_ref") == METHOD and inputs.get("native_refs") == NATIVE, "YYPL_FROZEN_NATIVE_INPUTS")
    require(not any(k in inputs for k in ["score", "dimensions", "K", "observations"]), "YYPL_NO_PRECOMPUTED_VALUES")
    require(
        set(inputs["sources"]) == set(inputs["legacy_inputs"]["sources"]) | {METHOD, *NATIVE.values()},
        "YYPL_SOURCE_GRAPH",
    )
    source = Originals(inputs, Path(root))
    require(
        inputs["sources"][METHOD]["archive_kind"] == "DERIVED"
        and all(inputs["sources"][h]["archive_kind"] == "ORIGINAL" for h in NATIVE.values()),
        "YYPL_SOURCE_ROLES",
    )
    legacy_source(inputs["legacy_inputs"], root)
    method = source.json(METHOD)
    require(
        method["calendar_peers_fx_formula_weights_unchanged"] is True
        and method["not_old_interval_midpoint_or_inferred_amount"] is True,
        "YYPL_METHOD",
    )
    return source


def _compile(inputs, source, root):
    old = inputs["legacy_inputs"]
    c = old["capacity"]
    with localcontext() as ctx:
        ctx.prec = 65
        admission = _admission(source, c)
        books = _books(source, c)
        datesA, datesHK = _calendars(source, c)
        halts = _halts(source, c, datesA[120])
        dailyA = {}
        for day, h in c["A_daily_refs"].items():
            rows = _table(source.json(h))
            require(all(r["trade_date"] == day.replace("-", "") for r in rows), "YYPL_A_DATE")
            require(len({r["ts_code"] for r in rows}) == len(rows), "YYPL_DUPLICATE_A_SECURITY")
            dailyA[day] = {r["ts_code"]: D(str(r["amount"])) * 1000 for r in rows if r.get("amount") is not None}
        for security, days in halts.items():
            for day in days:
                require(dailyA[day].get(security) in {None, D(0)}, "YYPL_HALT_CONFLICT")
                dailyA[day][security] = D(0)
        fx = {}
        for cube in ET.fromstring(source.raw[c["FX_history_ref"]]).iter():
            if not cube.attrib.get("time") or cube.attrib["time"] > "2026-09-30":
                continue
            rates = {r.attrib.get("currency"): r.attrib.get("rate") for r in cube}
            if "CNY" in rates and "HKD" in rates:
                fx[cube.attrib["time"]] = D(rates["CNY"]) / D(rates["HKD"])
        native = {}
        for code, h in NATIVE.items():
            params = inputs["sources"][h]["params"]
            value = source.json(h)
            require(
                params
                == {
                    "secid": "116." + code,
                    "klt": "101",
                    "fqt": "0",
                    "beg": "20260709",
                    "end": "20260930",
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                },
                "YYPL_NATIVE_REQUEST",
            )
            require(value["rc"] == 0 and value["data"]["code"] == code, "YYPL_NATIVE_IDENTITY")
            by_day = {}
            for row in value["data"]["klines"]:
                fields = row.split(",")
                require(len(fields) == 11 and fields[0] not in by_day, "YYPL_DAILY_FIELDS")
                require(
                    re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields[0]) and D(fields[5]) >= 0 and D(fields[6]) >= 0,
                    "YYPL_AMOUNT_AND_SHARES",
                )
                by_day[fields[0]] = {"volume": D(fields[5]), "amount": D(fields[6])}
            require(set(by_day) == set(datesHK[60]), "YYPL_60_NATIVE_DAYS_REQUIRED")
            native[code] = by_day
        checks = []
        for day, h in c["HK_primary_refs"].items():
            for code, truth in _HK_primary(source.raw[h], day).items():
                row = native[code][day]
                require(row["volume"] == truth["shares"], "YYPL_PRIMARY_VOLUME_IDENTITY")
                checks.append(
                    {
                        "code": code,
                        "date": day,
                        "provider_amount_HKD": str(row["amount"]),
                        "exchange_published_integer_HKD": str(truth["amount"]),
                        "delta_HKD": str(row["amount"] - truth["amount"]),
                    }
                )
        frozen = source.json(METHOD)["exchange_checks"]
        expected = {(x["code"], x["date"]): D(x["delta_HKD"]) for x in frozen}
        require(
            len(checks) == 92 and {(x["code"], x["date"]): D(x["delta_HKD"]) for x in checks} == expected,
            "YYPL_NO_SILENT_CROSS_SOURCE_CORRECTION",
        )
        windows = {}
        for n in [60, 30]:
            metrics = {}
            components = {}
            for code, book in books.items():
                k = D(0)
                items = []
                for security, holding in book["holdings"].items():
                    if security.endswith(".HK"):
                        values = [native[security[:5]][day]["amount"] * fx[day] for day in datesHK[n]]
                    else:
                        require(all(security in dailyA[day] for day in datesA[n]), "YYPL_MISSING_A_NOT_ZERO")
                        values = [dailyA[day][security] for day in datesA[n]]
                    require(len(values) == n and all(x >= 0 for x in values), "YYPL_SAME_MARKET_DAYS")
                    adv = sum(values) / n
                    require(adv > 0, "YYPL_POSITIVE_ADV")
                    contribution = holding * holding / book["total"] / adv
                    k += contribution
                    items.append(
                        {
                            "security": security,
                            "holding_CNY": str(holding),
                            "ADV_CNY": str(adv),
                            "K_contribution": str(contribution),
                        }
                    )
                metrics[code] = k
                components[code] = items
            score = peer_gap_score(metrics["018125"], [metrics[p] for p in PEERS]) * exposure_factor(
                "market_capacity", metrics["018125"]
            )
            windows[str(n)] = {
                "K_by_product": {k: str(v) for k, v in metrics.items()},
                "source_components": components,
                "market_capacity_score": str(score),
            }
        known = known_calculate(old["known_liquidity_inputs"], root=root)
        terms = policy(ACTIVE_BASIS)["terms"]["liquidity"]
        require(
            {k: D(w) for k, w in terms}
            == {"market_capacity": D(40), "restricted": D(20), "known_payment": D(25), "holders": D(15)},
            "YYPL_UNCHANGED_FIXED_WEIGHTS",
        )
        point = D(".4") * D(windows["60"]["market_capacity_score"]) + sum(
            D(known["subscores"][k]) * D(w) / 100 for k, w in terms if k != "market_capacity"
        )
    return {
        "dimensions": {"liquidity": {"lower": str(point), "upper": str(point), "candidate": str(point)}},
        "component_scores": {"liquidity": str(point)},
        "complete_score": None,
        "grade": None,
        "windows": windows,
        "audit": {
            "peer_admission": admission,
            "actual_native_HK_rows": 240,
            "exchange_checks": checks,
            "known_liquidity": known["audit"],
            "original_contract_no_hidden_precision_requirement": True,
            "unpublished_precision_reconstructed": False,
            "old_interval_midpoint_used": False,
            "cross_source_differences_preserved": True,
            "120_day_diagnostic_remains_in_original_interval_component": True,
        },
    }


@lru_cache(maxsize=1)
def _cached(serialized, root):
    inputs = json.loads(serialized)
    return _compile(inputs, _source(inputs, root), root)


def calculate(inputs, *, root):
    _source(inputs, root)
    return deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))


def check_binding(bundle, inputs):
    legacy_binding(bundle, inputs["legacy_inputs"])


def replay_snapshot_points(bundle, dimensions, *, root):
    inputs = bundle["yy_manufacturing_published_liquidity_calculation"]["inputs"]
    check_binding(bundle, inputs)
    expected = calculate(inputs, root=root)["dimensions"]["liquidity"]
    require(
        all(
            dimensions["liquidity"][k] is not None and D(dimensions["liquidity"][k]) == D(expected[k])
            for k in ["lower", "upper", "candidate"]
        ),
        "YYPL_SNAPSHOT_POINT_REPLAY",
    )


def prepare(session, request, *, root):
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    root = Path(root).resolve()
    raw = deepcopy(request["inputs"])
    previous = current_inputs(session, ["018125"]).get("018125")
    require(
        previous
        and request["expected_current_ref"] == previous["batch_id"]
        and request["as_of_date"] == str(previous["as_of_date"]),
        "YYPL_CURRENT_CHANGED",
    )
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == "018125"))
    check_binding(bundle, raw)
    require(bundle["basis"] == BASIS and not bundle.get("r5_calculation"), "YYPL_PARTIAL_CURRENT")
    raw["sources"] = {h: canonical_source(root, ref) for h, ref in raw["sources"].items()}
    old = raw["legacy_inputs"]
    old["sources"] = {h: raw["sources"][h] for h in old["sources"]}
    old["known_liquidity_inputs"]["sources"] = {h: raw["sources"][h] for h in old["known_liquidity_inputs"]["sources"]}
    value = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    before = deepcopy(doc["dimensions"])
    reason = request["explanations"]["liquidity"].strip()
    require(len(reason) >= 40, "YYPL_EXPLANATION")
    doc["dimensions"]["liquidity"] = {**value["dimensions"]["liquidity"], "reason": reason}
    bundle.setdefault("public_explanations", {})["liquidity"] = reason
    bundle.get("public_metrics", {}).pop("liquidity", None)
    bundle.pop("yy_manufacturing_market_liquidity_calculation", None)
    bundle.pop("yy_manufacturing_liquidity_calculation", None)
    bundle.update(
        research_archive=archive_json(root, doc),
        research_format="single",
        yy_manufacturing_published_liquidity_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    existing = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in existing)
    require(
        all(doc["dimensions"][k] == v for k, v in before.items() if k != "liquidity"), "YYPL_OTHER_DIMENSIONS_CHANGED"
    )
    validate_sources(bundle, datetime.now(UTC), root=root)
    replay_snapshot_points(bundle, doc["dimensions"], root=root)
    value = snapshot_calculate(bundle, root=root)
    require(value["status"] == "PARTIAL" and value["score"] is None, "YYPL_NO_COMPLETE_SCORE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
