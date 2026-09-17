"""当前一日模型的独立 CORE3 运行链；先预测，再核对，旧实验文件保持不变。

输入只有 SPX、FXI 成交 OHLC、CNYA 发行人估值差。CNYA 成交价没有进入
当前三个特征，因此不作为此独立版本的可用性条件。模型训练和阈值均不改变。
"""

import hashlib
import runpy
from collections import defaultdict
from datetime import datetime

import joblib

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_fund_bias as fund_model
from app.services import direction_1d_sprint_market_kernel_direction as kernel
from app.services import direction_1d_sprint_market_only_data as data
from app.services import direction_1d_sprint_market_only_forward as evidence

SERVICES = (fund_model, kernel)
FIRST_TARGET = "2026-09-17"
CONTRACT = "CORE3_NO_UNUSED_CNYA_TRADE_DEPENDENCY_V1"


def root():
    return base.ROOT / "core-forward-v2"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context():
    """只校验一次完整冻结代码，再直接载入已绑定模型，避免逐层递归重复校验90轮。"""
    p = base.read(root() / "plan.json")
    for name, expected in p["code"].items():
        require(sha(base.PROJECT / name) == expected, "CORE_CODE_CHANGED")
    require(base.calendar()[1] == p["calendar_hash"], "CORE_CALENDAR_CHANGED")
    require(base.digest(data.scope()) == p["scope_hash"], "CORE_SCOPE_CHANGED")
    parent = base.ROOT / "core-market-independent-forward-v1"
    comparison = base.ROOT / "core-model-comparison-forward-v1"
    pp, cp = base.read(parent / "plan.json"), base.read(comparison / "plan.json")
    require(base.digest(pp) == p["parent_plan_hash"], "CORE_PARENT_PLAN_CHANGED")
    require(base.digest(cp) == p["comparison_plan_hash"], "CORE_COMPARISON_PLAN_CHANGED")
    require(sha(parent / "capture.py") == pp["script_sha256"], "CORE_PARENT_HELPER_CHANGED")
    require(sha(comparison / "capture.py") == cp["script_sha256"], "CORE_COMPARISON_HELPER_CHANGED")
    helper = runpy.run_path(str(parent / "capture.py"))
    result, bundle = helper["model_bundle"](pp)
    bundles, results = {"round-82": bundle}, {"round-82": result}
    for service in SERVICES:
        key = service.root().name
        spec = cp["models"][key]
        result = base.read(service.root() / "result.json")
        require(base.digest(result) == spec["result_hash"], "CORE_MODEL_RESULT_CHANGED")
        require(result["plan_hash"] == base.digest(base.read(service.root() / "plan.json")), "CORE_MODEL_PLAN_CHANGED")
        require(
            sha(service.root() / "models.joblib") == spec["model_sha256"] == result["model_sha256"],
            "CORE_MODEL_CHANGED",
        )
        bundles[key] = joblib.load(service.root() / "models.joblib")
        for name in service.CANDIDATES:
            for head in bundles[key][name].values():
                service.validate_head(head, name, p["training_cutoff"])
        results[key] = result
    return {"plan": p, "helper": helper, "bundles": bundles, "results": results}


def answers(market, fund, ctx):
    values = ctx["helper"]["answers"](market, fund, ctx["bundles"]["round-82"])
    for service in SERVICES:
        bundle = ctx["bundles"][service.root().name]
        extra = (
            service.answers(market, fund["group"], bundle, fund["code"])
            if service is fund_model
            else service.answers(market, fund["group"], bundle)
        )
        values.update({name: extra[name] for name in service.CANDIDATES})
    return values


def capture_fxi(t, target, at):
    """复用原 FXI 请求账本；每时槽最多一次，不请求本版本未用的 CNYA 成交价。"""
    etf = data.etfs
    folder = etf.root() / target / "FXI"
    for slot in etf.SLOTS:
        if (folder / f"raw/{slot}.json").exists():
            try:
                _, meta = etf.load_piece(target, t, "FXI", slot)
                return {"slot": slot, "hash": base.digest(meta)}
            except ValueError:
                continue
    slot = etf.slot_for(at)
    if slot is None or data.ended(base.now()) or base.now() >= data.deadline(target):
        return None
    path = folder / f"requests/{slot}.json"
    if path.exists():
        return None
    etf.closed_before(base.now(), t, target)
    request = {
        "at": base.now().isoformat(),
        "source": etf.SOURCE,
        "symbol": "FXI",
        "target": target,
        "base": t,
        "required_dates": etf.required(t, target),
        "url": etf.url_for("FXI"),
    }
    base.save(path, request)
    try:
        raw, headers = etf.fetch_etf("FXI")
        meta = {
            "source": etf.SOURCE,
            "symbol": "FXI",
            "received_at": base.now().isoformat(),
            "request_hash": base.digest(request),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "headers": headers,
        }
        raw_path = folder / f"raw/{slot}.bin"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as output:
            output.write(raw)
        base.save(folder / f"raw/{slot}.json", meta)
        etf.load_piece(target, t, "FXI", slot)
        return {"slot": slot, "hash": base.digest(meta)}
    except Exception as exc:
        base.save(folder / f"failures/{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        return None


def materialize(t, target, spx_ref, fxi_ref, ctx):
    """回读原始响应、公告和真实接收时刻；训练输入公式直接复用已冻结的 CORE3 实现。"""
    days = list(map(str, base.calendar()[0]))
    require(target in days and days.index(target) > 0 and days[days.index(target) - 1] == t, "CORE_ADJACENCY_CHANGED")
    spx, ref = data.spx_observation(target, t, spx_ref["origin"], spx_ref["slot"])
    require(ref == spx_ref, "CORE_SPX_REFERENCE_CHANGED")
    _, meta = data.etfs.load_piece(target, t, "FXI", fxi_ref["slot"])
    require(base.digest(meta) == fxi_ref["hash"], "CORE_FXI_REFERENCE_CHANGED")
    raw = (data.etfs.root() / target / f"FXI/raw/{fxi_ref['slot']}.bin").read_bytes()
    points = data.etfs.parse(raw, "FXI", data.overnight.alignment(t, target)["required_us_dates"][-1])
    cn = data.cnya.load(target)
    require(cn["base"] == t, "CORE_CNYA_BASE_CHANGED")
    return {
        "base": t,
        "target": target,
        "contract": CONTRACT,
        "spx_ref": ref,
        "fxi_ref": fxi_ref,
        "fxi_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "cnya_hash": base.digest(cn),
        "source_received_at": [spx["received_at"], meta["received_at"], cn["at"]],
        "market": ctx["helper"]["market"](spx["parsed"]["rows"], points, cn["rows"], t, target),
        "fund_nav_used_as_predictor": False,
        "cnya_trade_quote_required": False,
    }


def load_input(target, ctx):
    value = base.read(root() / "inputs" / target / "input.json")
    actual = materialize(value["base"], target, value["spx_ref"], value["fxi_ref"], ctx)
    require({k: v for k, v in value.items() if k not in ("at", "plan_hash")} == actual, "CORE_INPUT_CHANGED")
    created = datetime.fromisoformat(value["at"])
    require(
        value["plan_hash"] == base.digest(ctx["plan"]) and created < data.deadline(target),
        "CORE_INPUT_LATE_OR_PLAN_CHANGED",
    )
    require(all(datetime.fromisoformat(t) <= created for t in actual["source_received_at"]), "CORE_INPUT_SOURCE_LATE")
    return value


def capture(ctx):
    at = base.now()
    target = str(at.date())
    days = list(map(str, base.calendar()[0]))
    if target < FIRST_TARGET or target not in days or data.slot_for(at) is None or data.ended(at):
        return None
    if (root() / "inputs" / target / "input.json").exists():
        return load_input(target, ctx)
    t = days[days.index(target) - 1]
    # 各源独立尝试；SPX尚未完整时仍采集其余两源，避免将最后时槽浪费在串行等待。
    parts, errors = {}, {}
    actions = {
        "spx": lambda: data.capture_spx(base.now(), t, target),
        "fxi": lambda: capture_fxi(t, target, base.now()),
        "cnya": lambda: data.cnya.capture(base.now()),
    }
    for name, action in actions.items():
        try:
            parts[name] = action() if base.now() < data.deadline(target) else None
        except Exception as exc:
            errors[name] = base.error_code(exc)
    base.save(
        root() / "capture-status.json",
        {
            "at": base.now().isoformat(),
            "target": target,
            "available": {n: bool(v) for n, v in parts.items()},
            "errors": errors,
        },
        replace=True,
    )
    if errors or not all(parts.get(n) for n in actions) or base.now() >= data.deadline(target):
        return None
    value = materialize(t, target, parts["spx"][1], parts["fxi"], ctx)
    if base.now() >= data.deadline(target):
        return None
    base.save(
        root() / "inputs" / target / "input.json",
        value | {"at": base.now().isoformat(), "plan_hash": base.digest(ctx["plan"])},
    )
    return load_input(target, ctx)


def validate(value, receipt, source, ctx):
    fund = next((f for f in data.scope() if f["code"] == value["code"]), None)
    require(fund and all(value[k] == fund[k] for k in fund), "CORE_FORECAST_SCOPE_CHANGED")
    created, readback = datetime.fromisoformat(value["at"]), datetime.fromisoformat(receipt["readback_at"])
    require(
        datetime.fromisoformat(source["at"]) <= created <= readback < data.deadline(value["u"]), "CORE_FORECAST_LATE"
    )
    require(created < evidence.sprint_end() and value["u"] >= FIRST_TARGET, "CORE_FORECAST_WINDOW_INVALID")
    require(value["t"] == source["base"] and value["u"] == source["target"], "CORE_FORECAST_TARGET_CHANGED")
    require(all(datetime.fromisoformat(r["at"]) <= created for r in ctx["results"].values()), "CORE_MODEL_CREATED_LATE")
    require(receipt["forecast_hash"] == base.digest(value) and receipt["status"] == "VERIFIED", "CORE_RECEIPT_CHANGED")
    require(
        value["input_hash"] == base.digest(source) and value["plan_hash"] == base.digest(ctx["plan"]),
        "CORE_FORECAST_BINDING_CHANGED",
    )
    require(
        value["model_hashes"] == {n: r["model_sha256"] for n, r in ctx["results"].items()},
        "CORE_FORECAST_MODELS_CHANGED",
    )
    require(value["answers"] == answers(source["market"], fund, ctx), "CORE_FORECAST_ANSWER_CHANGED")
    require(value["status"] == "MODEL_NOT_RELEASED" and value["contract"] == CONTRACT, "CORE_FORECAST_POLICY_CHANGED")
    return value


def write_forecasts(source, ctx):
    for fund in data.scope():
        path = root() / "forward" / source["target"] / f"{fund['code']}.json"
        if path.exists():
            continue
        if base.now() >= min(data.deadline(source["target"]), evidence.sprint_end()):
            break
        value = fund | {
            "at": base.now().isoformat(),
            "t": source["base"],
            "u": source["target"],
            "input_hash": base.digest(source),
            "plan_hash": base.digest(ctx["plan"]),
            "model_hashes": {n: r["model_sha256"] for n, r in ctx["results"].items()},
            "answers": answers(source["market"], fund, ctx),
            "status": "MODEL_NOT_RELEASED",
            "contract": CONTRACT,
        }
        base.save(path, value)
        loaded, at = base.read(path), base.now()
        receipt = {
            "readback_at": at.isoformat(),
            "forecast_hash": base.digest(loaded),
            "status": "VERIFIED" if loaded == value and at < data.deadline(value["u"]) else "LATE_OR_INVALID",
        }
        base.save(root() / "receipts" / value["u"] / path.name, receipt)
        validate(value, receipt, source, ctx)


def day16(ctx):
    """核对16日真实保存的两级证据；这里只读，绝不把现在的回读时间充作当时的预测时间。"""
    parent = base.ROOT / "core-market-independent-forward-v1"
    compare = base.ROOT / "core-model-comparison-forward-v1"
    source = base.read(parent / "input.json")
    require(
        {k: v for k, v in source.items() if k not in ("at", "plan_hash")} == ctx["helper"]["inputs"](),
        "CORE_DAY16_SOURCE_CHANGED",
    )
    require(source["plan_hash"] == ctx["plan"]["parent_plan_hash"], "CORE_DAY16_INPUT_PLAN_CHANGED")
    require(
        all(datetime.fromisoformat(t) <= datetime.fromisoformat(source["at"]) for t in source["source_received_at"]),
        "CORE_DAY16_SOURCE_LATE",
    )
    cp = base.read(compare / "plan.json")
    result = []
    for fund in data.scope():
        old = base.read(parent / "forward/2026-09-16" / f"{fund['code']}.json")
        old_receipt = base.read(parent / "receipts/2026-09-16" / f"{fund['code']}.json")
        value = base.read(compare / "forward/2026-09-16" / f"{fund['code']}.json")
        receipt = base.read(compare / "receipts/2026-09-16" / f"{fund['code']}.json")
        expected = answers(source["market"], fund, ctx)
        for record, proof, ph in (
            (old, old_receipt, ctx["plan"]["parent_plan_hash"]),
            (value, receipt, ctx["plan"]["comparison_plan_hash"]),
        ):
            require(all(record[k] == fund[k] for k in fund), "CORE_DAY16_SCOPE_CHANGED")
            require(record["base"] == "2026-09-15" and record["target"] == "2026-09-16", "CORE_DAY16_TARGET_CHANGED")
            created, readback = datetime.fromisoformat(record["at"]), datetime.fromisoformat(proof["readback_at"])
            require(
                datetime.fromisoformat(source["at"]) <= created <= readback < data.deadline("2026-09-16"),
                "CORE_DAY16_LATE",
            )
            require(
                all(datetime.fromisoformat(r["at"]) <= created for r in ctx["results"].values()),
                "CORE_DAY16_MODEL_LATE",
            )
            require(
                proof["forecast_hash"] == base.digest(record) and proof["plan_hash"] == ph == record["plan_hash"],
                "CORE_DAY16_RECEIPT_CHANGED",
            )
            require(
                record["input_hash"] == base.digest(source) and record["status"] == "MODEL_NOT_RELEASED",
                "CORE_DAY16_BINDING_CHANGED",
            )
        require(value["parent_forecast_hash"] == base.digest(old), "CORE_DAY16_PARENT_CHANGED")
        require(
            datetime.fromisoformat(old_receipt["readback_at"]) <= datetime.fromisoformat(value["at"]),
            "CORE_DAY16_PARENT_LATE",
        )
        require(
            old["model_hash"] == ctx["results"]["round-82"]["model_sha256"] and value["models"] == cp["models"],
            "CORE_DAY16_MODELS_CHANGED",
        )
        require(
            old["independent_contract"] == value["independent_input_contract"] == "CORE3_V1_NOT_OLD_PARENT_FORECAST",
            "CORE_DAY16_CONTRACT_CHANGED",
        )
        require(
            old["answers"] == {k: expected[k] for k in ctx["helper"]["NAMES"]} and value["answers"] == expected,
            "CORE_DAY16_ANSWERS_CHANGED",
        )
        # 规范包装只用于结果核对，origin_hash始终绑定真实原记录，原文件不改名、不迁移。
        result.append(
            value
            | {
                "t": value["base"],
                "u": value["target"],
                "origin_hash": base.digest(value),
                "origin": "core-model-comparison-forward-v1",
            }
        )
    return result


def forecasts(ctx):
    valid = day16(ctx)
    inputs = {}
    for path in sorted((root() / "forward").glob("*/*.json")):
        value = base.read(path)
        require(path.stem == value["code"] and path.parent.name == value["u"], "CORE_FORECAST_PATH_CHANGED")
        if value["u"] not in inputs:
            inputs[value["u"]] = load_input(value["u"], ctx)
        receipt = base.read(root() / "receipts" / value["u"] / path.name)
        valid.append(validate(value, receipt, inputs[value["u"]], ctx))
    return valid


def observe(values, at):
    """保留首次实际成熟标签及每个修订版本；仅用实际观察快照，不补造历史预测。"""
    due = [f for f in values if at >= datetime.fromisoformat(f["u"] + "T18:00:00+08:00")]
    first = {}
    for f in due:
        path = root() / "outcomes" / f["u"] / f"{f['code']}.json"
        if path.exists():
            first[f["code"], f["u"]] = base.read(path)
    for path in evidence.observation_paths(due):
        snapshot = base.read(path)
        if datetime.fromisoformat(snapshot["at"]) > at:
            continue
        for f in due:
            pair = evidence.observed_pair(f, snapshot, at)
            if pair is None:
                continue
            key = f["code"], f["u"]
            value = pair | {
                "observation_path": path.relative_to(base.ROOT).as_posix(),
                "observation_hash": base.digest(snapshot),
                "observed_at": at.isoformat(),
                "forecast_hash": base.digest(f),
            }
            if key not in first:
                base.save(root() / "outcomes" / f["u"] / f"{f['code']}.json", value)
                first[key] = value
            elif evidence.pair_version(pair) != evidence.pair_version(first[key]):
                version = base.digest(evidence.pair_version(pair))[:24]
                dest = root() / "revisions" / f["u"] / f["code"] / f"{version}.json"
                if not dest.exists():
                    base.save(dest, value | {"first_outcome_hash": base.digest(first[key])})


def report(values):
    at, paired, cache, revised = base.now(), defaultdict(list), {}, set()
    for f in values:
        path = root() / "outcomes" / f["u"] / f"{f['code']}.json"
        if not path.exists():
            continue
        outcome = base.read(path)
        pair = evidence.validated_outcome(f, outcome, cache, at)
        for revision in (root() / "revisions" / f["u"] / f["code"]).glob("*.json"):
            v = base.read(revision)
            evidence.validated_outcome(f, v, cache, at)
            require(v["first_outcome_hash"] == base.digest(outcome), "CORE_REVISION_PARENT_CHANGED")
            revised.add((f["code"], f["u"]))
        for name, choice in f["answers"].items():
            paired[name].append(f | pair | {"prediction": choice["prediction"]})
    closed = [
        str(day)
        for day in base.calendar()[0]
        if str(day) >= "2026-09-16" and data.deadline(str(day)) <= min(at, evidence.sprint_end())
    ]
    count = sum(f["u"] in closed for f in values)
    matched = len(next(iter(paired.values()), []))
    value = {
        "at": at.isoformat(),
        "verified_forecasts": len(values),
        "branches_per_fund": 8,
        "matched": matched,
        "pending": len(values) - matched,
        "matched_forward_metrics": {n: base.metrics(v) for n, v in paired.items()},
        "closed_targets": closed,
        "missing_due_predictions": len(closed) * 30 - count,
        "eligible_coverage": count / (len(closed) * 30) if closed else None,
        "whole_watchlist_coverage": count / (len(closed) * 43) if closed else None,
        "revised_questions": len(revised),
        "outcome_policy": "FIRST_OBSERVED_RAW_NAV_PAIR_WITH_REVISIONS_RETAINED",
        "legacy_collection": "RETIRED_AFTER_2026_09_16; OLD_ARTIFACTS_RETAINED",
        "new_fits": 0,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(root() / "report.json", value, replace=True)
    return value


def run():
    ctx = context()
    source = capture(ctx)
    if source is not None:
        write_forecasts(source, ctx)
    values = forecasts(ctx)
    observe(values, base.now())
    # 净值刷新在预测之后，并复用原30基金/半小时时槽上限；已匹配的题不再拉取。
    pending = [f for f in values if not (root() / "outcomes" / f["u"] / f"{f['code']}.json").exists()]
    evidence.refresh_if_needed(pending, base.now())
    observe(values, base.now())
    return report(values)


def preflight():
    ctx = context()
    values = forecasts(ctx)
    value = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(ctx["plan"]),
        "verified_existing_forecasts": len(values),
        "answer_checks": sum(len(f["answers"]) for f in values),
        "new_source_requests": 0,
        "new_fits": 0,
        "status": "PASSED",
    }
    base.save(root() / "preflight.json", value, replace=True)
    return value
