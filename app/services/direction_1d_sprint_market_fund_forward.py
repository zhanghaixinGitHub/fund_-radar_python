"""基金差异模型共享第70轮独立输入，但按每只基金代码计算与复验答案，拒绝错组和换码。"""

from collections import defaultdict
from datetime import datetime

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_etf_runtime_v2 as runtime
from app.services import direction_1d_sprint_market_only as parent
from app.services import direction_1d_sprint_market_only_data as data
from app.services import direction_1d_sprint_market_only_forward as evidence


def parent_context():
    manifest, bundle = parent.models()
    forecasts, invalid = evidence.verified_forecasts(manifest, bundle)
    if invalid:
        raise ValueError("MARKET_CHILD_PARENT_INVALID")
    return {(f["code"], f["u"]): f for f in forecasts}


def validate(service, value, receipt, p, source, market, manifest, bundle):
    created, readback = datetime.fromisoformat(value["at"]), datetime.fromisoformat(receipt["readback_at"])
    if (
        value["parent_hash"] != base.digest(p)
        or any(value[k] != p[k] for k in ("code", "u", "t", "family", "group"))
        or value["u"] < service.FIRST_TARGET
        or not datetime.fromisoformat(p["at"]) <= created <= readback < data.deadline(value["u"])
        or created >= evidence.sprint_end()
        or datetime.fromisoformat(manifest["at"]) > created
        or receipt["status"] != "VERIFIED"
        or receipt["forecast_hash"] != base.digest(value)
        or value["model_hash"] != manifest["model_sha256"]
        or value["plan_hash"] != manifest["plan_hash"]
        or value["input_hash"] != base.digest(source)
        or value["input_hash"] != p["input_hash"]
        or value["status"] != "MODEL_NOT_RELEASED"
        or value["market"] != market
        or not runtime.answers_match(value["answers"], service.live_answers(market, p["group"], bundle, p["code"]))
    ):
        raise ValueError("MARKET_CHILD_FORECAST_CHANGED")
    return value


def tick(service):
    manifest, bundle = service.models()
    if base.now() >= evidence.sprint_end():
        return report(service, manifest, bundle)
    parents = parent_context()
    inputs, markets = {}, {}
    for p in parents.values():
        target = p["u"]
        if target < service.FIRST_TARGET or base.now() >= data.deadline(target):
            continue
        path = service.root() / "forward" / target / f"{p['code']}.json"
        if path.exists():
            continue
        if target not in inputs:
            inputs[target] = data.load(target)
            markets[target] = service.live_market(inputs[target])
        source, market = inputs[target], markets[target]
        value = {k: p[k] for k in ("code", "family", "group", "u", "t")} | {
            "at": base.now().isoformat(),
            "parent_hash": base.digest(p),
            "input_hash": base.digest(source),
            "model_hash": manifest["model_sha256"],
            "plan_hash": manifest["plan_hash"],
            "market": market,
            "answers": service.live_answers(market, p["group"], bundle, p["code"]),
            "status": "MODEL_NOT_RELEASED",
        }
        base.save(path, value)
        loaded, at = base.read(path), base.now()
        receipt = {
            "readback_at": at.isoformat(),
            "forecast_hash": base.digest(value),
            "status": "VERIFIED" if loaded == value and at < data.deadline(target) else "LATE_OR_INVALID",
        }
        base.save(service.root() / "receipts" / target / path.name, receipt)
        validate(service, value, receipt, p, source, market, manifest, bundle)
    return report(service, manifest, bundle, parents)


def report(service, manifest=None, bundle=None, parents=None):
    if not (service.root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    if manifest is None:
        manifest, bundle = service.models()
    if parents is None:
        parents = parent_context()
    inputs, markets, observations, paired = {}, {}, {}, defaultdict(list)
    valid, invalid, revised = [], [], set()
    at = base.now()
    for path in sorted((service.root() / "forward").glob("*/*.json")):
        try:
            value = base.read(path)
            p = parents[value["code"], value["u"]]
            if path.stem != value["code"] or path.parent.name != value["u"]:
                raise ValueError("MARKET_CHILD_PATH_CHANGED")
            if value["u"] not in inputs:
                inputs[value["u"]] = data.load(value["u"])
                markets[value["u"]] = service.live_market(inputs[value["u"]])
            receipt = base.read(service.root() / "receipts" / value["u"] / path.name)
            validate(service, value, receipt, p, inputs[value["u"]], markets[value["u"]], manifest, bundle)
            valid.append(value)
        except Exception as exc:
            invalid.append({"path": path.relative_to(service.root()).as_posix(), "error": base.error_code(exc)})
            continue
        outcome_path = parent.root() / "outcomes" / value["u"] / path.name
        if not outcome_path.exists():
            continue
        outcome = base.read(outcome_path)
        pair = evidence.validated_outcome(p, outcome, observations, at)
        for revision_path in (parent.root() / "revisions" / p["u"] / p["code"]).glob("*.json"):
            revision = base.read(revision_path)
            evidence.validated_outcome(p, revision, observations, at)
            if revision["first_outcome_hash"] != base.digest(outcome):
                raise ValueError("MARKET_CHILD_REVISION_PARENT_CHANGED")
            revised.add((p["code"], p["u"]))
        for name, choice in (p["answers"] | value["answers"]).items():
            paired[name].append(p | pair | {"prediction": choice["prediction"]})
    closed = [
        str(d)
        for d in base.calendar()[0]
        if str(d) >= service.FIRST_TARGET and data.deadline(str(d)) <= min(at, evidence.sprint_end())
    ]
    count = sum(v["u"] in closed for v in valid)
    matched = len(next(iter(paired.values()), []))
    value = {
        "at": at.isoformat(),
        "verified_forecasts": len(valid),
        "invalid_forecasts": invalid,
        "pending": len(valid) - matched,
        "matched": matched,
        "matched_forward_metrics": {n: base.metrics(v) for n, v in paired.items()},
        "revised_questions": len(revised),
        "closed_targets": closed,
        "missing_due_predictions": len(closed) * len(data.scope()) - count,
        "eligible_coverage": count / (len(closed) * len(data.scope())) if closed else None,
        "whole_watchlist_coverage": count / (len(closed) * 43) if closed else None,
        "outcome_policy": "R70_FIRST_OBSERVED_RAW_NAV_PAIR_WITH_REVISIONS_RETAINED",
        "new_source_requests": 0,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(service.root() / "report.json", value, replace=True)
    return value
