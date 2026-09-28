"""原基金名单的持仓关联核查，只读取冻结报告和无涨跌标签的日期索引。

报告文字描述、报告期末持仓和公开日期分开保留。没有训练或预测调用，
不把有限披露当作完整组合，不把跨期重合当作当年已知的训练依据。
"""

import json
import re
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from statistics import median

from app.services.fund_002112_zero_fit_review import file_hash, read_json, save_once

ROOT = Path(__file__).resolve().parents[2] / ".local-runs/fund-exposure-002112"
OUT = ROOT / "holdings-compatibility/20260928-v1"
PREVIOUS = ROOT / "style-training-coverage/20260928-v1"
SNAPSHOT = ROOT / "training-ready/sources/98fb1c51f2582687a09ac0c404d514c2ef74520e4cb63a9a9f89a300ef150611.json"
FOLDS = ROOT / "peer-recovered-experiment/20260928-v1/frozen/folds.json"
DECODER = json.JSONDecoder()


def _space(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _skip_value(text, pos):
    """跳过不需要的 JSON 分支，不把其中的净值、标签或其他值解码成对象。"""
    pos = _space(text, pos)
    if text[pos] == '"':
        return DECODER.raw_decode(text, pos)[1]
    if text[pos] not in "[{":
        match = re.search(r"[,\]}\s]", text[pos:])
        return pos + match.start() if match else len(text)
    stack, quoted, escaped = [], False, False
    for i in range(pos, len(text)):
        char = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or (stack.pop(), char) not in {("[", "]"), ("{", "}")}:
                raise ValueError("JSON_BOUNDARY_INVALID")
            if not stack:
                return i + 1
    raise ValueError("JSON_BRANCH_INCOMPLETE")


def json_subtree(text, path, pos=0):
    """仅解码指定对象键或数组下标；冻结快照中的 nav 分支始终跳过。"""
    pos = _space(text, pos)
    if not path:
        return DECODER.raw_decode(text, pos)[0]
    key, tail = path[0], path[1:]
    if isinstance(key, str) and text[pos] == "{":
        pos += 1
        while True:
            pos = _space(text, pos)
            if text[pos] == "}":
                break
            found, pos = DECODER.raw_decode(text, pos)
            pos = _space(text, pos)
            if text[pos] != ":":
                raise ValueError("JSON_OBJECT_INVALID")
            pos = _space(text, pos + 1)
            if found == key:
                return json_subtree(text, tail, pos)
            pos = _space(text, _skip_value(text, pos))
            if text[pos] == "}":
                break
            if text[pos] != ",":
                raise ValueError("JSON_OBJECT_SEPARATOR_INVALID")
            pos += 1
    elif isinstance(key, int) and key >= 0 and text[pos] == "[":
        pos, index = pos + 1, 0
        while True:
            pos = _space(text, pos)
            if text[pos] == "]":
                break
            if index == key:
                return json_subtree(text, tail, pos)
            pos = _space(text, _skip_value(text, pos))
            if text[pos] == "]":
                break
            if text[pos] != ",":
                raise ValueError("JSON_ARRAY_SEPARATOR_INVALID")
            pos, index = pos + 1, index + 1
    raise KeyError(tuple(path))


def holding_vector(report):
    """返回原披露净值权重；缺值、重复股票或超出舍入误差的差额均停止计算。"""
    total = Decimal(str(report["stock_nav_pct"]))
    if not total.is_finite() or total < 0:
        raise ValueError("INVALID_STOCK_NAV_WEIGHT")
    vector = {}
    for holding in report["holdings"]:
        code = holding["stock_code"]
        weight = Decimal(str(holding["nav_weight_pct"]))
        if not code or code in vector or not weight.is_finite() or weight < 0:
            raise ValueError("INVALID_OR_DUPLICATE_HOLDING")
        vector[code] = weight
    disclosed = sum(vector.values(), Decimal(0))
    tolerance = Decimal("0.005") * (len(vector) + 1)
    full = report["full_stock_disclosure"]
    if not isinstance(full, bool) or disclosed - total > tolerance:
        raise ValueError("DISCLOSURE_TOTAL_CONFLICT")
    if full and abs(total - disclosed) > tolerance:
        raise ValueError("FULL_DISCLOSURE_DOES_NOT_RECONCILE")
    return vector, {
        "stock_nav_pct": float(total),
        "disclosed_nav_pct": float(disclosed),
        "disclosed_stock_coverage_pct": float(disclosed / total * 100) if total else None,
        "full_disclosure": full,
        "holding_count": len(vector),
        "undisclosed_stock_nav_pct": float(max(Decimal(0), total - disclosed)) if not full else 0.0,
        "rounding_residual_pct": float(total - disclosed),
        "rounding_tolerance_pct": float(tolerance),
    }


def compare_holdings(target, peer):
    """重合按相同股票的较小净值权重相加；部分披露保留宽松上界。

    上界只描述按已披露小数能容纳的未知空间，不是置信区间，也不估算当天仓位。
    两份完整报告时没有未披露股票，剩余小差额仅为舍入残差。
    """
    a, am = holding_vector(target)
    b, bm = holding_vector(peer)
    common = sorted(set(a) & set(b))
    lower = sum((min(a[c], b[c]) for c in common), Decimal(0))
    upper = min(
        Decimal(str(am["stock_nav_pct"])),
        Decimal(str(bm["stock_nav_pct"])),
        lower + Decimal(str(am["undisclosed_stock_nav_pct"])) + Decimal(str(bm["undisclosed_stock_nav_pct"])),
    )
    # 分项四舍五入可略超过汇总，保留实算可见值，不产生反向区间。
    upper = max(lower, upper)
    names_a = {h["stock_code"]: h["stock_name"] for h in target["holdings"]}
    names_b = {h["stock_code"]: h["stock_name"] for h in peer["holdings"]}
    return {
        "target": am,
        "peer": bm,
        "same_report_end": target["report_end"] == peer["report_end"],
        "both_full": am["full_disclosure"] and bm["full_disclosure"],
        "common_count": len(common),
        "visible_common_nav_pct": float(lower),
        "possible_common_upper_nav_pct": float(upper),
        "common_target_nav_pct": float(sum((a[c] for c in common), Decimal(0))),
        "common_peer_nav_pct": float(sum((b[c] for c in common), Decimal(0))),
        "common_holdings": [
            {
                "stock_code": c,
                "target_name": names_a[c],
                "peer_name": names_b[c],
                "target_nav_pct": float(a[c]),
                "peer_nav_pct": float(b[c]),
                "common_nav_pct": float(min(a[c], b[c])),
            }
            for c in common
        ],
    }


def _stats(values):
    return {"min": min(values), "median": median(values), "max": max(values)} if values else None


def summarize(rows, pairs):
    matched = [r for r in rows if r["status"] == "MATCHED"]
    ids = sorted({r["pair_id"] for r in matched})
    metrics = [pairs[k]["comparison"] for k in ids]
    full = [m for m in metrics if m["both_full"] and m["same_report_end"]]
    return {
        "rows": len(rows),
        "matched_rows": len(matched),
        "unmatched_rows": len(rows) - len(matched),
        "unique_report_pairs": len(ids),
        "pair_ids": ids,
        "visible_common_nav_pct_by_distinct_pair": _stats([m["visible_common_nav_pct"] for m in metrics]),
        "upper_envelope_nav_pct_by_distinct_pair": _stats([m["possible_common_upper_nav_pct"] for m in metrics]),
        "zero_visible_common_pairs": sum(m["common_count"] == 0 for m in metrics),
        "both_full_same_end_pairs": len(full),
        "both_full_same_end_visible_common_pct": _stats([m["visible_common_nav_pct"] for m in full]),
        "first_target": min((r["target"] for r in matched), default=None),
        "last_target": max((r["target"] for r in matched), default=None),
    }


def run():
    protocol = read_json(OUT / "protocol.json")
    if protocol["fit_budget"] != 0 or protocol["network_requests"] != 0:
        raise ValueError("ZERO_FIT_OFFLINE_SCOPE_CHANGED")
    for path, expected in protocol["source_files"].items():
        if file_hash(path) != expected:
            raise ValueError("SOURCE_CHANGED:" + path)
    code_hash = file_hash(__file__)
    save_once(
        OUT / ("code-freeze-" + code_hash[:12] + ".json"),
        {"code_sha256": code_hash, "protocol_sha256": file_hash(OUT / "protocol.json"), "new_fits": 0},
    )
    reports = {r["raw"]["sha256"]: r for r in read_json(PREVIOUS / "reports.json")}
    needed = {r["source"]["sha256"] for r in protocol["target_context_reports"]}
    # 只解码基金报告分支；不解析固定快照的净值、封存标签或证券行情。
    target_reports = json_subtree(SNAPSHOT.read_text(encoding="utf-8"), ["payload", "funds", "002112", "reports"])
    for report in target_reports:
        h = report["raw"]["sha256"]
        if h in needed:
            if h in reports and reports[h] != report:
                raise ValueError("PARSED_REPORT_VERSION_CONFLICT")
            reports[h] = report
    if not needed.issubset(reports):
        raise ValueError("TARGET_CONTEXT_REPORT_MISSING")
    allocation = []
    for h, report in sorted(reports.items()):
        paths = [p for p, expected in protocol["source_files"].items() if expected == h]
        if not paths or report["fund_code"] not in protocol["funds"]:
            raise ValueError("REPORT_SOURCE_NOT_FROZEN")
        _, metadata = holding_vector(report)
        allocation.append(
            {
                "report_sha256": h,
                "fund_code": report["fund_code"],
                "report_end": report["report_end"],
                "report_type": report["report_type"],
                "published_date": report["published_date"],
                "raw_paths": paths,
                **metadata,
            }
        )
    rows = read_json(PREVIOUS / "row-context.json")
    if len(rows) != 5137 or any(
        set(r) != {"fund_code", "target", "report_sha256", "report_publication", "tag"} for r in rows
    ):
        raise ValueError("ROW_CONTEXT_SCHEMA_CHANGED")
    own = {r["target"]: r for r in rows if r["fund_code"] == "002112"}
    pairs = {}

    def pair(a, b):
        key = a + ":" + b
        if key not in pairs:
            ar, br = reports[a], reports[b]
            pairs[key] = {
                "target_sha256": a,
                "peer_sha256": b,
                "peer_fund": br["fund_code"],
                "target_report_end": ar["report_end"],
                "peer_report_end": br["report_end"],
                "target_report_type": ar["report_type"],
                "peer_report_type": br["report_type"],
                "target_published_date": ar["published_date"],
                "peer_published_date": br["published_date"],
                "comparison": compare_holdings(ar, br),
            }
        return key

    daily = []
    for row in rows:
        if row["fund_code"] == "002112":
            continue
        record = {"fund_code": row["fund_code"], "target": row["target"], "peer_report_sha256": row["report_sha256"]}
        if row["target"] not in own:
            daily.append({**record, "status": "NO_EXACT_TARGET_INPUT_ROW"})
            continue
        target = own[row["target"]]
        for r in (target, row):
            if not r["report_publication"] < r["target"] <= "2023-12-31":
                raise ValueError("ORIGINAL_REPORT_PUBLICATION_CHANGED")
            if reports[r["report_sha256"]]["fund_code"] != r["fund_code"]:
                raise ValueError("REPORT_FUND_MISMATCH")
        daily.append(
            {
                **record,
                "status": "MATCHED",
                "target_context": target["tag"],
                "target_publication": target["report_publication"],
                "peer_publication": row["report_publication"],
                "pair_id": pair(target["report_sha256"], row["report_sha256"]),
            }
        )
    codes = [c for c in protocol["funds"] if c != "002112"]
    summary = {
        c: {
            "all": summarize([r for r in daily if r["fund_code"] == c], pairs),
            "medical_target_context": summarize(
                [r for r in daily if r["fund_code"] == c and r.get("target_context") == "MEDICINE_FOCUS"], pairs
            ),
            "by_year": {
                y: summarize([r for r in daily if r["fund_code"] == c and r["target"].startswith(y)], pairs)
                for y in sorted({r["target"][:4] for r in daily})
            },
        }
        for c in codes
    }
    # 原折只提取名称、训练日期标识；不解析 exam 内的实际涨跌和预测内容。
    fold_text = FOLDS.read_text(encoding="utf-8")
    folds = []
    for i in range(4):
        name, ids = json_subtree(fold_text, [i, "name"]), json_subtree(fold_text, [i, "train_ids"])
        keys = {tuple(k) for k in ids}
        selected = [r for r in daily if (r["fund_code"], r["target"]) in keys]
        folds.append(
            {
                "name": name,
                "train_ids_count": len(keys),
                "by_fund": {
                    c: summarize(
                        [r for r in selected if r["fund_code"] == c and r.get("target_context") == "MEDICINE_FOCUS"],
                        pairs,
                    )
                    for c in codes
                },
            }
        )
    full_pairs = []
    for ah, a in reports.items():
        if a["fund_code"] != "002112" or not a["full_stock_disclosure"]:
            continue
        for bh, b in reports.items():
            if b["fund_code"] != "002112" and b["full_stock_disclosure"] and a["report_end"] == b["report_end"]:
                full_pairs.append(pair(ah, bh))
    context_source = read_json(ROOT / "peer-transfer-evidence/20260928-v1/investment-context.json")
    context_daily = context_source["daily"]
    timeline = []
    for h in sorted(needed, key=lambda k: min(r["target"] for r in context_daily if r["report_sha256"] == k)):
        used = [r for r in context_daily if r["report_sha256"] == h]
        report = reports[h]
        timeline.append(
            {
                "report_sha256": h,
                "report_end": report["report_end"],
                "report_type": report["report_type"],
                "contexts": dict(Counter(r["context"] for r in used)),
                "used_rows": len(used),
                "first_used": min(r["target"] for r in used),
                "last_used": max(r["target"] for r in used),
                "effective_publications": sorted({r["report_publication"] for r in used}),
                "holding_age_days_at_first_use": (
                    date.fromisoformat(min(r["target"] for r in used)) - date.fromisoformat(report["report_end"])
                ).days,
                "top_ten": sorted(report["holdings"], key=lambda r: Decimal(r["nav_weight_pct"]), reverse=True)[:10],
            }
        )
    continuity = [
        {
            "from_sha256": a["report_sha256"],
            "to_sha256": b["report_sha256"],
            "pair_id": pair(a["report_sha256"], b["report_sha256"]),
        }
        for a, b in zip(timeline[:-1], timeline[1:], strict=True)
    ]
    broad_ai = [
        h
        for h, r in reports.items()
        if r["fund_code"] == "002170"
        and (r["report_end"], r["report_type"])
        in {("2023-06-30", "QUARTER"), ("2023-06-30", "HALF"), ("2023-09-30", "QUARTER")}
    ]
    ai_target = [r for r in timeline if "AI_COMPUTE" in r["contexts"]]
    bridge = [
        {
            "pair_id": pair(a["report_sha256"], b),
            "earliest_target_context_use": a["first_used"],
            "status": "CROSS_PERIOD_RETROSPECTIVE_ONLY_NOT_2023_TRAINING_OR_CURRENT_PEER_HOLDINGS",
        }
        for a in ai_target
        for b in sorted(broad_ai)
    ]
    outputs = {
        "reports": list(reports.values()),
        "allocations": allocation,
        "report-pairs": pairs,
        "daily-matches": daily,
        "summary": summary,
        "medical-fold-support": folds,
        "full-same-period-pairs": sorted(full_pairs),
        "target-timeline": timeline,
        "target-continuity": continuity,
        "ai-cross-period-reference": bridge,
    }
    for name, value in outputs.items():
        save_once(OUT / (name + ".json"), value)
    receipt = {
        "reports": len(reports),
        "training_rows": len(rows),
        "peer_rows": len(daily),
        "matched_rows": sum(r["status"] == "MATCHED" for r in daily),
        "report_pairs": len(pairs),
        "full_same_period_pairs": len(full_pairs),
        "target_context_reports": len(timeline),
        "AI_cross_period_pairs": len(bridge),
        "new_fits": 0,
        "cumulative_fits": 58,
        "protocol_sha256": file_hash(OUT / "protocol.json"),
        "code_sha256": code_hash,
    }
    save_once(OUT / "calculation-receipt.json", receipt)
    return receipt
