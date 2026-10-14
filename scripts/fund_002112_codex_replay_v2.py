"""002112 错误复盘与规则准入候选；按时间顺序隔离验证，不修改旧成绩。

旧27个可评日期仅作开发复盘。新区间固定为旧区间之后、截至2026-10-13的
五个交易日；没有参数搜索。方向由预先封存的规则一次生成，宿主已知后续信息
不能修改该方向。缺资料、冲突和极端波动均保存为未判断，单独报告覆盖率。
"""

from __future__ import annotations

import argparse
import json
import math
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from app.db.session import get_engine
from app.services import fund_exposure_common as io
from app.services.direction_1d_protocol import window
from sqlalchemy import text

from scripts import fund_002112_codex_replay_v1 as old

OUT = old.data.PY / ".local-runs/codex-backtest-002112-20261014-v2"
RULES = {
    "id": "TREND_CONFLICT_AND_SHOCK_ADMISSION_V2",
    "shock_sigma": 2.0,
    "trend_noise_sigma": 0.25,
    "minimum_quote_fraction": 0.95,
    "market_noise_pct": 0.25,
    "event_direction_votes": False,
    "confidence": "LOW",
}
ERRORS = {
    "C01": ("历史弱势外推", "最新净值已小幅回升，却由较长区间跌幅推成继续下跌；没有已验证的延续关系。"),
    "C05": ("短中期冲突", "五日反弹与二十日弱势冲突，近期涨幅已大，却仍按修复延续报涨。"),
    "C06": ("短中期冲突", "只放大最新一天及持仓回落，五日仍强、宽基为正的反证没有影响准入。"),
    "C07": ("矛盾证据取舍", "最新净值、五日及旧持仓合计为正，却由宽基和较长区间偏弱报跌。"),
    "C11": ("急涨延续", "把较大涨幅和同日股票、宽基同向视为次日支持，回吐反证只停留在文字。"),
    "C13": ("急跌延续", "急跌后的不稳定性被表达为继续跌；同日净值、股票、指数并非三份独立证据。"),
    "C15": ("财报到次日方向缺桥梁", "五日净值接近持平；季度利润增长缺少相对预期和市场定价证据，仍支持次日报涨。"),
    "C17": ("短中期冲突", "五日下跌、二十日上涨，经营改善也构成反证，却强行选择跌。"),
    "C20": ("短中期冲突", "短期回吐与二十日较大上涨冲突，最新一天的下跌被当作持续压力。"),
    "C25": ("微弱证据外推", "净值、持仓和宽基小幅偏弱即报跌，实际回购的反证没有改变是否报方向。"),
    "C34": ("矛盾证据取舍", "净值一日、五日及二十日均为正，小幅负的旧仓位和指数却否决修复；阅读登记另有纠正记录。"),
    "C36": ("微弱证据外推", "最新净值接近不变，旧仓位和指数同日回升仍被作为次日依据；H股融资进度没有方向桥梁。"),
    "C40": ("急跌延续", "急跌和高度相关的持仓、指数压力被直接外推；反弹可能性没有触发停止判断。"),
}


def network_guard(self, address):
    """只允许只读PostgreSQL连接；HTTP、付费模型、采集与本机业务API均被阻断。"""
    if address[0] not in {"localhost", "127.0.0.1", "::1"} or address[1] != 54329:
        raise RuntimeError("REPLAY_EXTERNAL_NETWORK_FORBIDDEN")
    return ORIGINAL_CONNECT(self, address)


ORIGINAL_CONNECT = socket.socket.connect


def freeze():
    """先冻结方法、开发复盘及保护清单；不读取新测试区间的净值或收益数值。"""
    if (OUT / "protocol.json").exists():
        return checked()
    old.checked()
    rows = old.read(old.OUT / "scored-days.json")
    failures = [row for row in rows if row.get("correct") is False]
    if {row["case"] for row in failures} != set(ERRORS):
        raise ValueError("OLD_FAILURE_SET_CHANGED")
    reviews = []
    for row in failures:
        category, mistake = ERRORS[row["case"]]
        reviews.append(
            {
                "case": row["case"],
                "date": row["target"],
                "original": row["review"],
                "actual": row["answer"],
                "category": category,
                "method_mistake": mistake,
                "market_cause_status": "UNKNOWN_WITHOUT_TARGET_INTERVAL_CAUSAL_EVIDENCE",
                "improvement": "反证触发冲突/噪声/异常波动准入；不能把原错例直接反向作为正确预测",
            }
        )
    old.save(OUT / "error-review.json", reviews)
    old_hashes = {str(path): old.sha(path) for path in sorted(old.OUT.rglob("*")) if path.is_file()}
    code = [Path(__file__), Path(old.__file__), Path(old.legacy.__file__), Path(old.data.__file__)]
    targets = ["2026-09-30", "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]
    protocol = {
        "version": "CODEX_RULE_GATED_REPLAY_V2",
        "created_at": datetime.now(old.data.ZONE).isoformat(),
        "development_period": ["2026-08-04", "2026-09-29"],
        "targets": targets,
        "cutoff": "目标交易日北京时间10:00",
        "rules": RULES,
        "max_parameter_searches": 0,
        "prediction_source": "DETERMINISTIC_PREREGISTERED_RULE_WITH_HOST_METHOD_REVIEW",
        "old_hashes": old_hashes,
        "code_hashes": {str(path): old.sha(path) for path in code},
        "source_hashes": {str(path): old.sha(path) for path in [old.data.SOURCE, old.data.MATERIAL]},
        "private_ledger_before": old.ledger(),
        "source_manifest": {},
        "limits": [
            "新日期位于开发期之后，但只有五天；此前部分日期净值已在聊天出现，不能称完全未知答案的独立盲测",
            "规则先封存后批量生成，不根据新日期答案选择参数或人工改方向；候选未替换现用预测",
            "只读历史接收版本恢复净值、行情；报告公开时间与实际下载时间分别检查",
            "历史新闻公告没有可靠接收时间或原件匹配时不投票；只凭旧财报、计划不产生次日方向",
            "允许不判断，所有日期仍在计划中；条件命中率与覆盖率分开，不能靠跳过难题宣称改进",
        ],
    }
    old.save(OUT / "protocol.json", protocol)
    return protocol


def checked():
    protocol = old.read(OUT / "protocol.json")
    for section in ["old_hashes", "code_hashes", "source_hashes"]:
        if any(old.sha(path) != expected for path, expected in protocol[section].items()):
            raise ValueError("FROZEN_OR_OLD_FILE_CHANGED")
    if protocol["rules"] != RULES:
        raise ValueError("FROZEN_RULE_CHANGED")
    if old.ledger() != protocol["private_ledger_before"]:
        raise ValueError("PRIVATE_LEDGER_CHANGED")
    return protocol


def archive_quotes(day, cutoff, code=None):
    """按历史时刻选择最新已接收原件，而非读取今天覆盖过的行情索引。"""
    candidates = []
    stamp = old.data.moment(cutoff)
    for path in (io.ROOT / "quote-receipt-versions").glob("*.json"):
        receipt = io.read(path)
        if old.data.moment(receipt["received_at"]) > stamp:
            continue
        params = receipt.get("params", {})
        wanted = day.replace("-", "")
        if code is None:
            matched = receipt.get("api") == "daily" and params.get("trade_date") == wanted
        else:
            matched = receipt.get("api") == "index_daily" and params.get("ts_code") == code
        if matched:
            candidates.append((receipt["received_at"], path, receipt))
    for _, path, receipt in sorted(candidates, key=lambda item: item[0], reverse=True):
        raw_path = (io.ROOT / receipt["file"]).resolve()
        if not raw_path.is_relative_to(io.ROOT.resolve()) or old.sha(raw_path) != receipt["sha256"]:
            raise ValueError("ARCHIVED_QUOTE_HASH_MISMATCH")
        body = json.loads(raw_path.read_bytes())["data"]
        rows = {}
        for values in body["items"]:
            row = dict(zip(body["fields"], values, strict=True))
            if row["trade_date"] == day.replace("-", "") and (code is None or row["ts_code"] == code):
                rows[row["ts_code"]] = row
        if rows:
            return rows, {
                "path": str(raw_path),
                "hash": receipt["sha256"],
                "received_at": receipt["received_at"],
                "receipt_path": str(path),
                "receipt_hash": old.sha(path),
                "business_date": day,
            }
    return {}, None


def source_versions(connection, days, cutoff):
    """仅查询基准日及以前、当时已实际存入的净值版本，目标日答案不进入输入。"""
    result = connection.execute(
        text("""
        SELECT business_date,payload,content_hash,received_at,stored_at FROM (
          SELECT *,row_number() OVER(PARTITION BY business_date
             ORDER BY received_at DESC,stored_at DESC,version_id DESC) AS choice
          FROM direction_1d_source_version
          WHERE fund_code='002112' AND kind='NAV' AND business_date=ANY(:days)
            AND source_id=(SELECT source_id FROM source_registry WHERE source_code='TUSHARE_PRO_FUND')
            AND received_at<=:cutoff AND stored_at<=:cutoff
        ) ranked WHERE choice=1 ORDER BY business_date
    """),
        {"days": [datetime.fromisoformat(day).date() for day in days], "cutoff": old.data.moment(cutoff)},
    )
    return [
        {
            "nav_date": str(row["business_date"]),
            "unit_nav": row["payload"]["unit_nav"],
            "received_at": row["received_at"].isoformat(),
            "stored_at": row["stored_at"].isoformat(),
            "source_hash": row["content_hash"],
        }
        for row in result.mappings()
    ]


def validate_packet(value):
    """验证输入中的日期、接收时刻和无答案字段；缺项由准入规则明确处理。"""
    cutoff = old.data.moment(value["as_of"])
    base = value["window"]["base_nav_date"]
    if any(
        row["nav_date"] > base
        or old.data.moment(row["received_at"]) > cutoff
        or old.data.moment(row["stored_at"]) > cutoff
        for row in value["nav"]
    ):
        raise ValueError("FUTURE_NAV_OR_VERSION")
    report = value["report"]
    if old.data.moment(report["available_at"]) > cutoff or old.data.moment(report["raw"]["received_at"]) > cutoff:
        raise ValueError("FUTURE_REPORT")
    for proof in value["quote_proofs"]:
        if proof["business_date"] > base or old.data.moment(proof["received_at"]) > cutoff:
            raise ValueError("FUTURE_QUOTE_VERSION")
    if any(c["quote"] and c["quote"]["date"] > base for c in value.get("companies", [])):
        raise ValueError("FUTURE_HOLDING_QUOTE")
    if any(row["date"] > base for row in value.get("market", {}).values()):
        raise ValueError("FUTURE_MARKET_QUOTE")

    def visit(item):
        if isinstance(item, dict):
            if {"actual", "actual_direction", "target_unit_nav", "label", "y"}.intersection(item):
                raise ValueError("ANSWER_IN_INPUT")
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)


def prepare():
    """规则冻结后只读恢复五份输入；原件摘要随新快照单独保护。"""
    protocol = checked()
    if (OUT / "inputs-manifest.json").exists():
        return old.read(OUT / "inputs-manifest.json")
    seed = old.data.load_sources()
    manifest = {"inputs": [], "files": {}}
    for index, target in enumerate(protocol["targets"], 1):
        case = f"N{index:02d}"
        cutoff = target + "T10:00:00+08:00"
        base, resolved = old.data.prediction_target(seed["sessions"], cutoff)
        if resolved != target:
            raise ValueError("TARGET_CALENDAR_MISMATCH")
        wanted = seed["sessions"][seed["sessions"].index(base) - 60 : seed["sessions"].index(base) + 1]
        with get_engine().connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            nav = source_versions(connection, wanted, cutoff)
            connection.rollback()
        report = old.data.legacy.choose_report(seed["reports"], cutoff)
        prices, stock_proof = archive_quotes(base, cutoff)
        markets, proofs = {}, [stock_proof] if stock_proof else []
        for code in ["000300.SH", "000905.SH"]:
            rows, proof = archive_quotes(base, cutoff, code)
            if code in rows:
                markets[code] = {"date": base, "change_pct": rows[code]["pct_chg"]}
                proofs.append(proof)
        companies = [
            {
                "code": h["stock_code"],
                "name": h["stock_name"],
                "weight_pct": float(h["nav_weight_pct"]),
                "quote": {"date": base, "change_pct": prices[h["stock_code"]]["pct_chg"]}
                if h["stock_code"] in prices
                else None,
            }
            for h in report["holdings"]
        ]
        value = {
            "fund_code": "002112",
            "case": case,
            "as_of": cutoff,
            "window": window(old.data.moment(cutoff)),
            "nav": nav,
            "expected_nav_dates": wanted,
            "report": report,
            "companies": companies,
            "market": markets,
            "quote_proofs": proofs,
            "material_policy": "没有逐条确认历史接收原文和相对预期，公告新闻政策不参与方向投票；缺资料不当作没有事件",
        }
        validate_packet(value)
        path = OUT / "inputs" / (case + ".json")
        old.save(path, value)
        manifest["inputs"].append({"case": case, "target": target, "path": str(path), "hash": old.sha(path)})
        raw_path = io.ROOT / report["raw"]["file"]
        manifest["files"][str(raw_path)] = old.sha(raw_path)
        for proof in proofs:
            manifest["files"][proof["path"]] = proof["hash"]
            manifest["files"][proof["receipt_path"]] = proof["receipt_hash"]
    old.save(OUT / "inputs-manifest.json", manifest)
    return manifest


def sign(value, noise):
    return 1 if value > noise else -1 if value < -noise else 0


def predict(value):
    """反证必须改变准入结果；涨跌完全由封存规则生成，不接受事后人工覆盖。

    波动仅用于噪声和异常准入，不投跌票。持仓与宽基用于否决明显矛盾，不与
    同日净值重复增加支持票。季度增长、回购和融资均无未经验证的次日方向票。
    """
    validate_packet(value)
    if [row["nav_date"] for row in value["nav"]] != value["expected_nav_dates"]:
        return {"state": "UNAVAILABLE", "direction": None, "reason": "NAV_WINDOW_NOT_AVAILABLE"}
    values = [float(row["unit_nav"]) for row in value["nav"]]
    returns = [(b / a - 1) * 100 for a, b in zip(values[:-1], values[1:], strict=True)]
    # 以最新观察之前的20个日收益估计噪声；异常是偏离此前均值，不能把稳定
    # 正漂移本身误判为异常，也不能让待检查的大波动同时抬高自己的准入阈值。
    reference_returns = returns[-21:-1]
    volatility = stdev(reference_returns)
    if volatility <= 0:
        return {"state": "ABSTAIN", "direction": None, "reason": "NO_VOLATILITY_CONTEXT"}
    r1, r5, r20 = returns[-1], (values[-1] / values[-6] - 1) * 100, (values[-1] / values[-21] - 1) * 100
    coverage = sum(c["weight_pct"] for c in value["companies"] if c["quote"])
    disclosed = sum(c["weight_pct"] for c in value["companies"])
    impact = sum(c["weight_pct"] * c["quote"]["change_pct"] / 100 for c in value["companies"] if c["quote"])
    observed = {
        "r1_pct": r1,
        "r5_pct": r5,
        "r20_pct": r20,
        "volatility20_pct": volatility,
        "previous20_mean_pct": mean(reference_returns),
        "holding_impact_ppt": impact,
        "quote_weight_pct": coverage,
        "disclosed_weight_pct": disclosed,
        "market_pct": {code: row["change_pct"] for code, row in value["market"].items()},
    }
    blockers = []
    short = sign(r5, RULES["trend_noise_sigma"] * volatility * math.sqrt(5))
    medium = sign(r20, RULES["trend_noise_sigma"] * volatility * math.sqrt(20))
    if abs(r1 - mean(reference_returns)) >= RULES["shock_sigma"] * volatility:
        blockers.append("EXTREME_PREVIOUS_DAY_MOVE")
    if short == 0 or medium == 0:
        blockers.append("TREND_WITHIN_NOISE_BAND")
    elif short != medium:
        blockers.append("SHORT_MEDIUM_CONFLICT")
    if disclosed <= 0 or coverage < RULES["minimum_quote_fraction"] * disclosed:
        blockers.append("HOLDING_QUOTE_COVERAGE_MISSING")
    if len(value["market"]) != 2:
        blockers.append("MARKET_SOURCE_MISSING_AT_CUTOFF")
    if short and sign(r1, RULES["trend_noise_sigma"] * volatility) not in [0, short]:
        blockers.append("LATEST_NAV_CONFLICT")
    if short and sign(impact, RULES["trend_noise_sigma"] * volatility) not in [0, short]:
        blockers.append("DISCLOSED_HOLDING_CONFLICT")
    if short and len(value["market"]) == 2:
        directions = [sign(row["change_pct"], RULES["market_noise_pct"]) for row in value["market"].values()]
        if all(direction == -short for direction in directions):
            blockers.append("BOTH_MARKET_INDICES_CONFLICT")
    return {
        "state": "ABSTAIN" if blockers else "PREDICTED",
        "direction": None if blockers else "UP" if short > 0 else "DOWN",
        "confidence": "LOW",
        "reason_codes": blockers,
        "observed": observed,
        "evidence_policy": "同向走势为待验证假设，昨日价格不构成独立因果证据；事件无方向投票",
    }


def run():
    """先新增保存全部五天方向或未判断状态；这个阶段完全不读取目标日实际净值。"""
    protocol = checked()
    manifest = old.read(OUT / "inputs-manifest.json")
    for path, expected in manifest["files"].items():
        if old.sha(path) != expected:
            raise ValueError("INPUT_SOURCE_CHANGED")
    for row in manifest["inputs"]:
        if old.sha(row["path"]) != row["hash"]:
            raise ValueError("INPUT_CHANGED")
        path = OUT / "predictions" / (row["case"] + ".json")
        if path.exists():
            continue
        prediction = {
            **predict(old.read(row["path"])),
            "case": row["case"],
            "target": row["target"],
            "method": protocol["rules"]["id"],
            "input_hash": row["hash"],
            "generation_source": protocol["prediction_source"],
            "saved_at": datetime.now(old.data.ZONE).isoformat(),
        }
        old.save(path, prediction)
    hashes = {row["case"]: old.sha(OUT / "predictions" / (row["case"] + ".json")) for row in manifest["inputs"]}
    if not (OUT / "first-predictions-manifest.json").exists():
        old.save(
            OUT / "first-predictions-manifest.json",
            {"hashes": hashes, "sealed_at": datetime.now(old.data.ZONE).isoformat()},
        )
    return {"saved": len(hashes), "first_predictions_sealed": True}


def score_gate():
    protocol = checked()
    if not (OUT / "first-predictions-manifest.json").exists():
        raise ValueError("ALL_FIRST_PREDICTIONS_MUST_BE_SAVED")
    seal = old.read(OUT / "first-predictions-manifest.json")
    if len(seal["hashes"]) != len(protocol["targets"]):
        raise ValueError("INCOMPLETE_PREDICTION_SET")
    if any(old.sha(OUT / "predictions" / (case + ".json")) != expected for case, expected in seal["hashes"].items()):
        raise ValueError("FIRST_PREDICTION_CHANGED")
    return protocol, seal


def score():
    """封存门禁通过后统一读取实际净值；覆盖率、全计划命中和条件准确率分别计算。"""
    protocol, seal = score_gate()
    if (OUT / "summary.json").exists():
        return old.read(OUT / "summary.json")
    manifest = old.read(OUT / "inputs-manifest.json")
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        answers = {
            str(row["nav_date"]): str(row["unit_nav"])
            for row in connection.execute(
                text("""
          SELECT nav_date,unit_nav FROM nav_daily WHERE fund_code='002112'
          AND source_id=(SELECT source_id FROM source_registry WHERE source_code='TUSHARE_PRO_FUND')
          AND nav_date=ANY(:days)
        """),
                {"days": [datetime.fromisoformat(day).date() for day in protocol["targets"]]},
            ).mappings()
        }
        connection.rollback()
    rows = []
    for row in manifest["inputs"]:
        prediction = old.read(OUT / "predictions" / (row["case"] + ".json"))
        value = old.read(row["path"])
        scored = {"case": row["case"], "target": row["target"], "prediction": prediction}
        if len(value["nav"]) == 61 and row["target"] in answers:
            actual = old.label(value["nav"][-1]["unit_nav"], answers[row["target"]])
            scored.update(
                answer=actual,
                previous_direction=old.label(value["nav"][-2]["unit_nav"], value["nav"][-1]["unit_nav"])[
                    "actual_direction"
                ],
            )
            if prediction["state"] == "PREDICTED":
                scored["correct"] = prediction["direction"] == actual["actual_direction"]
        rows.append(scored)
    assessed = [row for row in rows if "correct" in row]
    answered = [row for row in rows if "answer" in row]

    def baselines(items):
        return {
            name: {"correct": sum(predicate(row) for row in items), "count": len(items)}
            for name, predicate in {
                "ALWAYS_UP": lambda row: row["answer"]["actual_direction"] == "UP",
                "ALWAYS_DOWN": lambda row: row["answer"]["actual_direction"] == "DOWN",
                "PREVIOUS_DIRECTION": lambda row: row["previous_direction"] == row["answer"]["actual_direction"],
            }.items()
        }

    hits = sum(row["correct"] for row in assessed)
    summary = {
        "version": protocol["version"],
        "rules": protocol["rules"],
        "period": protocol["targets"],
        "planned": len(rows),
        "states": dict(Counter(row["prediction"]["state"] for row in rows)),
        "correct": hits,
        "assessed": len(assessed),
        "conditional_accuracy": hits / len(assessed) if assessed else None,
        "coverage": len(assessed) / len(rows),
        "correct_and_output_share": hits / len(rows),
        "baselines_on_predictable_dates": baselines(assessed),
        "baselines_all_answered_dates": baselines(answered),
        "first_predictions_sealed_at": seal["sealed_at"],
        "scored_at": datetime.now(old.data.ZONE).isoformat(),
        "new_private_calls": old.ledger()["calls"] - protocol["private_ledger_before"]["calls"],
        "old_files_checked": len(protocol["old_hashes"]),
        "old_files_changed": [],
        "limits": protocol["limits"],
    }
    checked()
    old.save(OUT / "scored-days.json", rows)
    old.save(OUT / "summary.json", summary)
    return summary


def main():
    socket.socket.connect = network_guard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "prepare", "run", "score", "verify"])
    args = parser.parse_args()
    if args.action == "freeze":
        p = freeze()
        result = {"rules_frozen": True, "targets": p["targets"], "old_files": len(p["old_hashes"])}
    elif args.action == "prepare":
        result = {"inputs": len(prepare()["inputs"])}
    elif args.action == "run":
        result = run()
    elif args.action == "score":
        result = score()
    else:
        p, seal = score_gate()
        result = {
            "old_files_unchanged": len(p["old_hashes"]),
            "first_predictions_unchanged": len(seal["hashes"]),
            "private_ledger_unchanged": True,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
