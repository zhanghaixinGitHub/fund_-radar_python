"""第93轮复用当前 CORE3 实际提前输入；不重新拉取行情，不调用已退役的旧实验链。"""

from collections import defaultdict
from datetime import datetime

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_block_newton_forward as prior_forward
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_market_block_newton as prior_model
from app.services import direction_1d_sprint_market_futures_basis as model


def validate(value, receipt, parent, source, manifest, bundle, extra, history):
    parent_receipt = base.read(core.root() / "receipts" / parent["u"] / f"{parent['code']}.json")
    model.require(parent_receipt["forecast_hash"] == base.digest(parent), "PARENT_RECEIPT_CHANGED")
    created = datetime.fromisoformat(value["at"])
    model.require(datetime.fromisoformat(parent_receipt["readback_at"]) <= created, "PARENT_READBACK_LATE")
    readback = datetime.fromisoformat(receipt["readback_at"])
    model.require(all(value[k] == parent[k] for k in ("code", "family", "group", "t", "u")), "FORWARD_ID_CHANGED")
    model.require(
        value["u"] >= model.FIRST_TARGET
        and datetime.fromisoformat(parent["at"]) <= created <= readback < core.data.deadline(value["u"]),
        "FORWARD_LATE",
    )
    model.require(
        created < core.evidence.sprint_end() and datetime.fromisoformat(manifest["at"]) <= created,
        "MODEL_OR_SPRINT_LATE",
    )
    model.require(
        value["parent_hash"] == base.digest(parent)
        and value["input_hash"] == parent["input_hash"] == base.digest(source),
        "FORWARD_INPUT_CHANGED",
    )
    model.require(
        value["model_hash"] == manifest["model_sha256"] and value["plan_hash"] == manifest["plan_hash"],
        "FORWARD_MODEL_CHANGED",
    )
    model.require(
        receipt["forecast_hash"] == base.digest(value) and receipt["status"] == "VERIFIED", "FORWARD_RECEIPT_CHANGED"
    )
    model.require(
        value["status"] == "MODEL_NOT_RELEASED" and value["contract"] == core.CONTRACT, "FORWARD_CONTRACT_CHANGED"
    )
    model.require(value["r92_forecast_hash"] == base.digest(extra), "R92_PARENT_CHANGED")
    extra_receipt = base.read(prior_model.root() / "receipts" / parent["u"] / f"{parent['code']}.json")
    model.require(
        extra_receipt["forecast_hash"] == base.digest(extra)
        and datetime.fromisoformat(extra_receipt["readback_at"]) <= created,
        "R92_PARENT_READBACK_LATE_OR_CHANGED",
    )
    model.require(
        value["futures_history_hash"] == base.digest(history) and datetime.fromisoformat(history["at"]) <= created,
        "FUTURES_SOURCE_CHANGED_OR_LATE",
    )
    market = model.data.extend(source["market"], parent["t"], parent["u"], history["snapshot"]["rows"])
    actual = model.answers(market, parent["group"], bundle)
    model.require(value["answers"] == {n: actual[n] for n in model.CANDIDATES}, "FORWARD_ANSWERS_CHANGED")
    return value


def context():
    manifest, bundle = model.models()
    ctx = core.context()
    parents = {(f["code"], f["u"]): f for f in core.forecasts(ctx) if f["u"] >= model.FIRST_TARGET}
    prior_manifest, prior_bundle = prior_model.models()
    history = model.data.history()
    extras, sources = {}, {}
    for key, p in parents.items():
        if p["u"] not in sources:
            sources[p["u"]] = core.load_input(p["u"], ctx)
        extra = base.read(prior_model.root() / "forward" / p["u"] / f"{p['code']}.json")
        receipt = base.read(prior_model.root() / "receipts" / p["u"] / f"{p['code']}.json")
        prior_forward.validate(extra, receipt, p, sources[p["u"]], prior_manifest, prior_bundle)
        extras[key] = extra
    return manifest, bundle, ctx, parents, extras, history


def run():
    # 先完成已有CORE3及R92分支，再在同一实际输入上增加独立价差模型。
    prior_forward.run()
    manifest, bundle, ctx, parents, extras, history = context()
    sources = {}
    for key, p in parents.items():
        path = model.root() / "forward" / p["u"] / f"{p['code']}.json"
        if path.exists() or base.now() >= min(core.data.deadline(p["u"]), core.evidence.sprint_end()):
            continue
        if p["u"] not in sources:
            sources[p["u"]] = core.load_input(p["u"], ctx)
        source, extra = sources[p["u"]], extras[key]
        market = model.data.extend(source["market"], p["t"], p["u"], history["snapshot"]["rows"])
        actual = model.answers(market, p["group"], bundle)
        value = {k: p[k] for k in ("code", "family", "group", "t", "u")} | {
            "at": base.now().isoformat(),
            "parent_hash": base.digest(p),
            "input_hash": p["input_hash"],
            "model_hash": manifest["model_sha256"],
            "plan_hash": manifest["plan_hash"],
            "answers": {n: actual[n] for n in model.CANDIDATES},
            "status": "MODEL_NOT_RELEASED",
            "contract": core.CONTRACT,
            "r92_forecast_hash": base.digest(extra),
            "futures_history_hash": base.digest(history),
        }
        if base.now() >= core.data.deadline(p["u"]):
            continue
        base.save(path, value)
        checked, at = base.read(path), base.now()
        receipt = {
            "readback_at": at.isoformat(),
            "forecast_hash": base.digest(checked),
            "status": "VERIFIED" if checked == value and at < core.data.deadline(p["u"]) else "LATE_OR_INVALID",
        }
        base.save(model.root() / "receipts" / p["u"] / path.name, receipt)
        validate(value, receipt, p, source, manifest, bundle, extra, history)
    return report((manifest, bundle, ctx, parents, extras, history))


def report(loaded=None):
    manifest, bundle, ctx, parents, extras, history = loaded or context()
    paired, cache, sources, valid, revised = defaultdict(list), {}, {}, [], set()
    at = base.now()
    for path in sorted((model.root() / "forward").glob("*/*.json")):
        value = base.read(path)
        model.require(path.stem == value["code"] and path.parent.name == value["u"], "FORWARD_PATH_CHANGED")
        p = parents[value["code"], value["u"]]
        if p["u"] not in sources:
            sources[p["u"]] = core.load_input(p["u"], ctx)
        receipt = base.read(model.root() / "receipts" / p["u"] / path.name)
        validate(value, receipt, p, sources[p["u"]], manifest, bundle, extras[p["code"], p["u"]], history)
        valid.append(value)
        outcome = core.root() / "outcomes" / p["u"] / path.name
        if not outcome.exists():
            continue
        first = base.read(outcome)
        pair = core.evidence.validated_outcome(p, first, cache, at)
        for revision in (core.root() / "revisions" / p["u"] / p["code"]).glob("*.json"):
            changed = base.read(revision)
            core.evidence.validated_outcome(p, changed, cache, at)
            model.require(changed["first_outcome_hash"] == base.digest(first), "OUTCOME_REVISION_CHANGED")
            revised.add((p["code"], p["u"]))
        for name, answer in (p["answers"] | extras[p["code"], p["u"]]["answers"] | value["answers"]).items():
            paired[name].append(p | pair | {"prediction": answer["prediction"]})
    closed = [
        str(d)
        for d in base.calendar()[0]
        if str(d) >= model.FIRST_TARGET and core.data.deadline(str(d)) <= min(at, core.evidence.sprint_end())
    ]
    count = sum(f["u"] in closed for f in valid)
    matched = len(next(iter(paired.values()), []))
    value = {
        "at": at.isoformat(),
        "verified_forecasts": len(valid),
        "matched": matched,
        "pending": len(valid) - matched,
        "new_branches": 2,
        "same_question_branches": 12,
        "matched_forward_metrics": {n: base.metrics(v) for n, v in paired.items()},
        "missing_due_predictions": len(closed) * 30 - count,
        "closed_targets": closed,
        "eligible_coverage": count / (len(closed) * 30) if closed else None,
        "whole_watchlist_coverage": count / (len(closed) * 43) if closed else None,
        "revised_questions": len(revised),
        "new_source_requests": 0,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(model.root() / "report.json", value, replace=True)
    return value


def preflight():
    manifest, bundle = model.models()
    ctx = core.context()
    checked = core.day16(ctx)
    source = base.read(base.ROOT / "core-market-independent-forward-v1/input.json")
    history = model.data.history()
    checks = 0
    for f in checked:
        actual = model.answers(
            model.data.extend(source["market"], f["t"], f["u"], history["snapshot"]["rows"]), f["group"], bundle
        )
        for name in model.CANDIDATES:
            model.require(actual[name]["prediction"] in (0, 1), "PREFLIGHT_DIRECTION_INVALID")
            checks += 1
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_ONLY_NOT_ACTUAL_FORWARD",
        "plan_hash": manifest["plan_hash"],
        "checks": checks,
        "new_source_requests": 0,
        "new_fits": 0,
    }
    base.save(model.root() / "preflight.json", value, replace=True)
    return value
