"""第101轮仅组合真实提前落盘的R94/R96答案，保留全部既有分支和原始收据。"""

from collections import defaultdict
from datetime import datetime

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_market_futures_equal as model
from app.services import direction_1d_sprint_option_futures_forward as prior_forward

prior_model = model.previous
FALLBACK = "FULL_INTERVAL_FXI_SIGN3_LR504"


def record_receipt(number, parent, record, created):
    """父答案必须是同题、同输入、对应冻结模型且已完成回读，不能事后补录为提前预测。"""
    path = base.ROOT / f"round-{number}"
    receipt = base.read(path / "receipts" / parent["u"] / f"{parent['code']}.json")
    manifest = base.read(path / "result.json")
    model.require(
        all(record[k] == parent[k] for k in ("code", "family", "group", "t", "u", "input_hash")),
        "SOURCE_QUESTION_CHANGED",
    )
    model.require(
        record["model_hash"] == manifest["model_sha256"] and record["plan_hash"] == manifest["plan_hash"],
        "SOURCE_MODEL_CHANGED",
    )
    model.require(
        record["status"] == "MODEL_NOT_RELEASED" and record["contract"] == core.CONTRACT, "SOURCE_CONTRACT_CHANGED"
    )
    model.require(
        receipt["status"] == "VERIFIED" and receipt["forecast_hash"] == base.digest(record), "SOURCE_RECEIPT_CHANGED"
    )
    model.require(
        datetime.fromisoformat(manifest["at"])
        <= datetime.fromisoformat(record["at"])
        <= datetime.fromisoformat(receipt["readback_at"])
        <= created,
        "SOURCE_READBACK_LATE",
    )


def answer(parent, parts):
    return model.combine(
        parts[94]["answers"][model.MEMBERS[0]], parts[96]["answers"][model.MEMBERS[1]], parent["answers"][FALLBACK]
    )


def validate(value, receipt, parent, source, manifest, bundle, extra, parts):
    parent_receipt = base.read(core.root() / "receipts" / parent["u"] / f"{parent['code']}.json")
    created = datetime.fromisoformat(value["at"])
    readback = datetime.fromisoformat(receipt["readback_at"])
    model.require(
        parent_receipt["forecast_hash"] == base.digest(parent)
        and datetime.fromisoformat(parent_receipt["readback_at"]) <= created,
        "PARENT_RECEIPT_CHANGED_OR_LATE",
    )
    model.require(all(value[k] == parent[k] for k in ("code", "family", "group", "t", "u")), "FORWARD_ID_CHANGED")
    model.require(
        value["u"] >= model.FIRST_TARGET
        and datetime.fromisoformat(parent["at"]) <= created <= readback < core.data.deadline(value["u"]),
        "FORWARD_LATE",
    )
    model.require(
        datetime.fromisoformat(manifest["at"]) <= created < core.evidence.sprint_end(), "MODEL_OR_SPRINT_LATE"
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
        receipt["status"] == "VERIFIED" and receipt["forecast_hash"] == base.digest(value), "FORWARD_RECEIPT_CHANGED"
    )
    model.require(
        value["status"] == "MODEL_NOT_RELEASED" and value["contract"] == core.CONTRACT, "FORWARD_CONTRACT_CHANGED"
    )
    model.require(value["r100_forecast_hash"] == base.digest(extra), "R100_PARENT_CHANGED")
    record_receipt(100, parent, extra, created)
    for number in (94, 96):
        record_receipt(number, parent, parts[number], created)
        model.require(
            value["member_forecast_hashes"][str(number)] == base.digest(parts[number]), "MEMBER_FORECAST_CHANGED"
        )
    model.require(value["answers"] == {model.CANDIDATES[0]: answer(parent, parts)}, "FORWARD_ANSWERS_CHANGED")
    return value


def context():
    manifest, bundle = model.models()
    prior_manifest, prior_bundle, ctx, candidates, prior_extras, prior_sources = prior_forward.context()
    parents, extras, parts, sources = {}, {}, {}, {}
    for key, p in candidates.items():
        path = prior_model.root() / "forward" / p["u"] / f"{p['code']}.json"
        if not path.exists():
            continue
        if p["u"] not in sources:
            sources[p["u"]] = core.load_input(p["u"], ctx)
        extra = base.read(path)
        receipt = base.read(prior_model.root() / "receipts" / p["u"] / path.name)
        prior_forward.validate(
            extra, receipt, p, sources[p["u"]], prior_manifest, prior_bundle, prior_extras[key], prior_sources[p["u"]]
        )
        parts[key] = {n: base.read(base.ROOT / f"round-{n}" / "forward" / p["u"] / path.name) for n in (94, 96)}
        parents[key], extras[key] = p, extra
    return manifest, bundle, ctx, parents, extras, parts


def run():
    # 上游完成真实采集、各原模型预测及校验后，只读这些记录组合，不新增HTTP或拟合。
    prior_forward.run()
    manifest, bundle, ctx, parents, extras, parts = context()
    sources = {}
    for key, p in parents.items():
        path = model.root() / "forward" / p["u"] / f"{p['code']}.json"
        if path.exists() or base.now() >= min(core.data.deadline(p["u"]), core.evidence.sprint_end()):
            continue
        if p["u"] not in sources:
            sources[p["u"]] = core.load_input(p["u"], ctx)
        created = base.now()
        for number, record in [(100, extras[key]), *parts[key].items()]:
            record_receipt(number, p, record, created)
        value = {k: p[k] for k in ("code", "family", "group", "t", "u")} | {
            "at": created.isoformat(),
            "parent_hash": base.digest(p),
            "input_hash": p["input_hash"],
            "model_hash": manifest["model_sha256"],
            "plan_hash": manifest["plan_hash"],
            "answers": {model.CANDIDATES[0]: answer(p, parts[key])},
            "status": "MODEL_NOT_RELEASED",
            "contract": core.CONTRACT,
            "r100_forecast_hash": base.digest(extras[key]),
            "member_forecast_hashes": {str(n): base.digest(v) for n, v in parts[key].items()},
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
        validate(value, receipt, p, sources[p["u"]], manifest, bundle, extras[key], parts[key])
    return report((manifest, bundle, ctx, parents, extras, parts))


def combined_answers(parent, extra, value):
    """明确保留CORE八分支及所有增量模型，共十九个同题答案。"""
    combined = dict(parent["answers"])
    for number in (92, 94, 95, 96, 97, 98, 99):
        record = base.read(base.ROOT / f"round-{number}" / "forward" / parent["u"] / f"{parent['code']}.json")
        combined.update(record["answers"])
    combined.update(extra["answers"])
    combined.update(value["answers"])
    model.require(len(combined) == 19, "FORWARD_BRANCH_COUNT_CHANGED")
    return combined


def report(loaded=None):
    manifest, bundle, ctx, parents, extras, parts = loaded or context()
    paired, cache, sources, valid, revised = defaultdict(list), {}, {}, [], set()
    at = base.now()
    for path in sorted((model.root() / "forward").glob("*/*.json")):
        value = base.read(path)
        model.require(path.stem == value["code"] and path.parent.name == value["u"], "FORWARD_PATH_CHANGED")
        key = value["code"], value["u"]
        p = parents[key]
        if p["u"] not in sources:
            sources[p["u"]] = core.load_input(p["u"], ctx)
        receipt = base.read(model.root() / "receipts" / p["u"] / path.name)
        validate(value, receipt, p, sources[p["u"]], manifest, bundle, extras[key], parts[key])
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
            revised.add(key)
        for name, choice in combined_answers(p, extras[key], value).items():
            paired[name].append(p | pair | {"prediction": choice["prediction"]})
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
        "new_branches": 1,
        "same_question_branches": 19,
        "new_feature_fallback_forecasts": sum(
            v["answers"][model.CANDIDATES[0]]["route"] != model.CANDIDATES[0] for v in valid
        ),
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
    manifest, bundles = model.models()
    ctx = core.context()
    checked = core.day16(ctx)
    source = base.read(base.ROOT / "core-market-independent-forward-v1/input.json")
    basis_history, intraday_history = model.basis.data.history(), model.intraday.data.history()
    checks = 0
    for f in checked:
        a = model.basis.answers(
            model.basis.data.extend(source["market"], f["t"], f["u"], basis_history["snapshot"]["rows"]),
            f["group"],
            bundles[0],
        )
        b = model.intraday.answers(
            model.intraday.data.extend(source["market"], f["t"], f["u"], intraday_history["snapshot"]["rows"]),
            f["group"],
            bundles[1],
        )
        result = model.combine(a[model.MEMBERS[0]], b[model.MEMBERS[1]], f["answers"][FALLBACK])
        model.require(result["prediction"] in (0, 1), "PREFLIGHT_DIRECTION_INVALID")
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
