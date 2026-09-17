"""将现有市场来源转换为五输入，不增加供应商请求，也不把基金NAV当预测输入。"""

import hashlib
import math

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_only_data as parent

scope = parent.scope
deadline = parent.deadline


def extend(market, t, u, etf_rows, cnya_rows):
    """原始开盘缺口保留现金分红影响；估值差变化单位为百分点，不能与价格收益混称。"""
    etf = parent.etfs.features(t, u, etf_rows)
    cnya = parent.cnya.features(t, u, cnya_rows)
    if (
        len(market["features"]) != 3
        or market["features"][1:] != [etf[0], cnya[0]]
        or market["etf_available"] != bool(etf[-1])
        or market["cnya_available"] != bool(cnya[-1])
        or market["available"] != bool(etf[-1] and cnya[-1])
    ):
        raise ValueError("MARKET_GAP_DELTA_PARENT_SOURCE_CHANGED")
    if market["available"]:
        if 1 + etf[0] / 100 <= 0 or 1 + etf[1] / 100 <= 0:
            raise ValueError("MARKET_GAP_DELTA_NONPOSITIVE_PRICE_RATIO")
        extra = [100 * ((1 + etf[1] / 100) / (1 + etf[0] / 100) - 1), cnya[1]]
    else:
        # 不完整来源不进入新拟合，原问题仍保留，并由同一个SPX规则给方向。
        extra = [0.0, 0.0]
    result = market | {"features": market["features"] + extra}
    if not all(math.isfinite(v) for v in result["features"]):
        raise ValueError("MARKET_GAP_DELTA_NONFINITE_FEATURE")
    return result


def original_row(row):
    return {k: v for k, v in row.items() if k != "market3"} | {"market": row["market3"]}


def dataset():
    """每次训练重验原行、标签成熟与两项新特征，外层摘要被重写也不能伪造向量。"""
    original, proof = parent.dataset()
    folder = base.ROOT / "market-gap-delta-feasibility-v1"
    plan, result, bundle = (base.read(folder / name) for name in ("plan.json", "result.json", "rows.json"))
    if (
        plan["script_sha256"] != hashlib.sha256((folder / "check.py").read_bytes()).hexdigest()
        or plan["parent_rows_hash"] != proof["rows_hash"]
        or result["plan_hash"] != base.digest(plan)
        or bundle["plan_hash"] != base.digest(plan)
        or result["rows_hash"] != base.digest(bundle)
    ):
        raise ValueError("MARKET_GAP_DELTA_MANIFEST_CHANGED")
    e, c = parent.etfs.history(), parent.cnya.history()
    if bundle["source_market_hashes"] != {"etfs": base.digest(e), "cnya": base.digest(c)}:
        raise ValueError("MARKET_GAP_DELTA_RAW_SOURCE_CHANGED")
    rows, cache = [], {}
    for old, captured in zip(original, bundle["rows"], strict=True):
        if {k: v for k, v in captured.items() if k != "market5"} != old:
            raise ValueError("MARKET_GAP_DELTA_ORIGINAL_ROW_CHANGED")
        key = old["t"], old["u"]
        if key not in cache:
            cache[key] = extend(old["market"], *key, e, c)
        if captured["market5"] != cache[key]:
            raise ValueError("MARKET_GAP_DELTA_HISTORY_VECTOR_CHANGED")
        rows.append(old | {"market3": old["market"], "market": captured["market5"]})
    if len(rows) != 38949:
        raise ValueError("MARKET_GAP_DELTA_HISTORY_SCOPE_CHANGED")
    return rows, proof | {"five_feature_rows_hash": base.digest(bundle), "feature_proof_hash": base.digest(result)}


def live_market(source):
    """从父输入定位并重验当时的原始ETF/发行方响应，每目标日由运行器缓存一次。"""
    if parent.load(source["target"]) != source:
        raise ValueError("MARKET_GAP_DELTA_LIVE_PARENT_CHANGED")
    e, c = parent.etfs.load(source["target"]), parent.cnya.load(source["target"])
    if (
        base.digest(e) != source["etf_hash"]
        or base.digest(c) != source["cnya_hash"]
        or e["base"] != source["base"]
        or c["base"] != source["base"]
    ):
        raise ValueError("MARKET_GAP_DELTA_LIVE_SOURCE_CHANGED")
    return extend(source["market"], source["base"], source["target"], e["rows"], c["rows"])
