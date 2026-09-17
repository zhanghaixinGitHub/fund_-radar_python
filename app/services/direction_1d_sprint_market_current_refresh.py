"""第79轮：只用实际收到的新成熟标签刷新当前模型，旧历史成绩不重新评分。"""

import hashlib
import shutil
from datetime import datetime
from decimal import Decimal

import joblib

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_odd_features as previous
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_market_only_data as data

CANDIDATES = ("CURRENT_0916_MARKET_LR3_504", "CURRENT_0916_MARKET_SIGN_LR3_504")
CONTROLS = ("FROZEN_R70_MARKET_LR3", "FROZEN_R70_MARKET_SIGN_LR3", "MARKET_MAJORITY3", "SPX_SIGN", "ALWAYS_UP")
BRANCHES = CANDIDATES + CONTROLS
FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "b9d76837c377fc49537fe216b15ca1ac71b891bef10e5bd1cb6564acefaf4b5a"
active = baseline.active


def root():
    return base.ROOT / "round-79"


def fresh_root():
    return base.ROOT / "fresh-current-market-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_REFRESH_" + reason)


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_current_refresh.py",
        "scripts/direction_1d_sprint_market_current_refresh.py",
        "tests/test_direction_1d_sprint_market_current_refresh.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def validate_spec(proposal, scope_size):
    require(
        proposal["candidates"] == list(CANDIDATES)
        and proposal["controls"] == list(CONTROLS)
        and proposal["branch_count"] == len(BRANCHES) == 7
        and len(set(BRANCHES)) == 7
        and proposal["preflight_branch_checks"] == len(BRANCHES) * scope_size
        and proposal["max_development_fits"] == 0
        and proposal["max_current_fits"] == 6
        and proposal["reproductions"] == 1
        and proposal["current_cutoff"] == "2026-09-16"
        and proposal["old_current_cutoff"] == "2026-09-15",
        "DECLARED_BRANCHES_OR_BUDGET_CHANGED",
    )


def verify_inputs(p):
    for path, digest in p["input_hashes"].items():
        require(base.digest(base.read(base.ROOT / path)) == digest, "FROZEN_INPUT_CHANGED")
    require(base.digest(data.scope()) == p["scope_hash"], "SCOPE_CHANGED")


def plan():
    proposal = base.read(root() / "proposal-before-implementation.json")
    validate_spec(proposal, len(data.scope()))
    require(base.digest(proposal) == PROPOSAL_HASH, "PROPOSAL_CHANGED")
    require(
        hashlib.sha256((root() / "design-before-implementation.md").read_bytes()).hexdigest()
        == proposal["design_sha256"],
        "DESIGN_CHANGED",
    )
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        require(p["fingerprint"] == fingerprint() and p["calendar_hash"] == base.calendar()[1], "CODE_CHANGED")
        verify_inputs(p)
        return p
    active()
    require(base.digest(previous.models()[0]) == proposal["r78_result_hash"], "PARENT_CHANGED")
    require(base.digest(baseline.models()[0]) == proposal["r70_result_hash"], "BASELINE_CHANGED")
    for name in ("plan", "rows", "result"):
        require(
            base.digest(base.read(fresh_root() / f"{name}.json")) == proposal[f"fresh_{name}_hash"], "FRESH_CHANGED"
        )
    fresh_plan = base.read(fresh_root() / "plan.json")
    paths = [
        "history.json",
        "round-79/proposal-before-implementation.json",
        "round-78/result.json",
        "round-70/result.json",
        "fresh-current-market-feasibility-v1/plan.json",
        "fresh-current-market-feasibility-v1/result.json",
        "fresh-current-market-feasibility-v1/rows.json",
        "fresh-current-market-feasibility-v1/nav-snapshot.json",
        fresh_plan["nav_source_ref"],
    ]
    groups = sorted({f["group"] for f in data.scope()})
    for name in CONTROLS[:2]:
        for group in groups:
            control = previous.control_path(name, group, "current")[0].with_suffix(".json")
            paths.append(control.relative_to(base.ROOT).as_posix())
    p = {
        "at": base.now().isoformat(),
        "round": 79,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(data.scope()),
        "input_hashes": {name: base.digest(base.read(base.ROOT / name)) for name in paths},
        "proposal_hash": PROPOSAL_HASH,
        "candidates": list(CANDIDATES),
        "controls": list(CONTROLS),
        "current_fit_cutoff": proposal["current_cutoff"],
        "old_current_cutoff": proposal["old_current_cutoff"],
        "first_forward_target": FIRST_TARGET,
        "development_fits": 0,
        "current_fits": 6,
        "reproductions": 1,
        "preflight_branch_checks": 210,
        "training": "Unchanged R70 raw3/sign3 LR recipe; 504 mature full-market dates",
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return p


def rebuild_added(snapshot, old, frozen, successor, markets):
    """从冻结真实响应重建标签。日期截止约束成熟日，实际收到时间另与真实冻结时刻比较。

    快照at是请求批次开始时间，逐基金received_at可能稍晚；不得误判为未来响应。
    所有新增标签仍是相邻中国交易日的原始单位净值变化，净值不作为模型输入。
    """
    freeze_at = datetime.fromisoformat(frozen["at"])
    require(freeze_at.tzinfo is not None and not snapshot["errors"], "SNAPSHOT_INVALID")
    require(
        datetime.fromisoformat(snapshot["at"]) < freeze_at < datetime.fromisoformat(snapshot["expires_at"]),
        "SNAPSHOT_TIME_INVALID",
    )
    previous_day = {v: k for k, v in successor.items()}
    added, excluded = [], []
    max_old = max(r["u"] for r in old)
    for fund in sorted(snapshot["funds"], key=lambda f: f["fund_code"]):
        code = fund["fund_code"]
        points = {r["date"]: r for r in fund["rows"]}
        require(len(points) == len(fund["rows"]), "DUPLICATE_NAV_DATE")
        for u in frozen["candidate_target_dates"]:
            t = previous_day[u]
            a, z = points.get(t), points.get(u)
            reason = (
                "MISSING_NAV_PAIR"
                if not a or not z
                else "MISSING_ANN_DATE"
                if not a.get("ann_date") or not z.get("ann_date")
                else "NO_ACTUAL_RECEIPT_TIME"
                if not a.get("received_at") or not z.get("received_at")
                else None
            )
            if reason:
                excluded.append({"code": code, "u": u, "reason": reason})
                continue
            for point in (a, z):
                received, nav = datetime.fromisoformat(point["received_at"]), Decimal(point["nav"])
                require(received.tzinfo is not None and received < freeze_at, "RECEIPT_TIME_INVALID")
                require(nav.is_finite() and nav > 0, "NAV_INVALID")
                require(
                    point["date"] <= str(received.date()) and point["ann_date"] <= str(received.date()), "FUTURE_NAV"
                )
            maturity = max(successor[u], a["ann_date"], z["ann_date"])
            if maturity >= frozen["new_cutoff"]:
                excluded.append({"code": code, "u": u, "reason": "LABEL_NOT_MATURE_BEFORE_CUTOFF", "mature": maturity})
                continue
            require(u > max_old and successor[t] == u, "TARGET_NOT_ADJACENT_OR_NEW")
            truth = base.label(a["nav"], z["nav"])
            added.append(
                {
                    "code": code,
                    "family": fund["family"],
                    "group": fund["group"],
                    "t": t,
                    "u": u,
                    "mature": maturity,
                    "y": truth["y"],
                    "actual_direction": truth["actual_direction"],
                    "market": markets[t, u],
                    "old_question": False,
                    "fund_nav_used_as_predictor": False,
                    "label_t_hash": base.digest(a),
                    "label_u_hash": base.digest(z),
                    "label_source_kind": "ACTUAL_RESPONSE_OBSERVATION",
                    "label_observation_ref": frozen["nav_source_ref"],
                    "label_observation_hash": frozen["nav_snapshot_hash"],
                }
            )
    return added, excluded


def dataset():
    """重验原38949行和新30行的独立原始来源；不访问供应商、不更新快照、不计算成绩。"""
    p = plan()
    frozen, stored = base.read(fresh_root() / "plan.json"), base.read(fresh_root() / "rows.json")
    snapshot = base.read(fresh_root() / "nav-snapshot.json")
    require(
        hashlib.sha256((fresh_root() / "check.py").read_bytes()).hexdigest() == frozen["script_sha256"],
        "SOURCE_SCRIPT_CHANGED",
    )
    require(base.digest(frozen) == stored["plan_hash"], "SOURCE_PLAN_CHANGED")
    require(
        base.digest(snapshot)
        == frozen["nav_snapshot_hash"]
        == base.digest(base.read(base.ROOT / frozen["nav_source_ref"])),
        "NAV_SNAPSHOT_CHANGED",
    )
    old, proof = data.dataset()
    require(proof["rows_hash"] == frozen["parent_rows_hash"] == stored["parent_rows_hash"], "OLD_ROWS_CHANGED")
    expected = {f["fund_code"]: f for f in base.read(base.ROOT / "history.json")["funds"]}
    found = {f["fund_code"]: f for f in snapshot["funds"]}
    require(set(expected) == set(found) and len(found) == len(snapshot["funds"]) == 30, "NAV_SCOPE_CHANGED")
    for code, fund in found.items():
        require(
            all(fund[k] == expected[code][k] for k in ("family", "group", "source_fund_code")), "FUND_IDENTITY_CHANGED"
        )
    spx = base.read(data.overnight.root() / "spx.json")
    etfs, cnya = data.etfs.history(), data.cnya.history()
    require(datetime.fromisoformat(frozen["at"]) < datetime.fromisoformat(spx["expires_at"]), "SPX_EXPIRED_AT_FREEZE")
    sources = {"spx": base.digest(spx), "etfs": base.digest(etfs), "cnya": base.digest(cnya)}
    require(sources == stored["source_market_hashes"], "MARKET_SOURCE_CHANGED")
    days = list(map(str, base.calendar()[0]))
    successor = dict(zip(days[:-1], days[1:], strict=True))
    prior = {v: k for k, v in successor.items()}
    markets = {
        (prior[u], u): data.feature_values(prior[u], u, spx["rows"], etfs, cnya)
        for u in frozen["candidate_target_dates"]
        if successor[u] < frozen["new_cutoff"]
    }
    added, excluded = rebuild_added(snapshot, old, frozen, successor, markets)
    require(added == stored["added_rows"] and excluded == stored["excluded"], "REBUILT_LABELS_CHANGED")
    require(len(added) == 30 and {r["u"] for r in added} == {"2026-09-14"}, "NEW_ROWS_OR_DATES_CHANGED")
    merged = sorted(old + added, key=lambda r: (r["u"], r["family"], r["code"]))
    require(merged == stored["merged_rows"] and len(merged) == 38979, "MERGED_ROWS_CHANGED")
    require(len({(r["code"], r["u"]) for r in merged}) == len(merged), "DUPLICATE_QUESTION")
    require(frozen["new_cutoff"] == p["current_fit_cutoff"], "CUTOFF_CHANGED")
    return merged, {
        "at": base.now().isoformat(),
        "fresh_rows_hash": base.digest(stored),
        "old_rows": len(old),
        "added_rows": len(added),
        "added_dates": 1,
        "merged_rows": len(merged),
        "no_new_2026_scores": True,
        "new_requests": 0,
    }


def fit(rows, name, cutoff):
    require(name in CANDIDATES, "CANDIDATE_INVALID")
    # 只换成熟训练数据，原配方、两种变换和权重函数直接复用，避免暗中改变优化条件。
    return baseline.fit(rows, baseline.CANDIDATES[CANDIDATES.index(name)], cutoff)


def fit_checkpoint(rows, name, cutoff, label, plan_hash):
    path = root() / "checkpoints" / f"{label}-{name}.joblib"
    receipt, attempt = path.with_suffix(".json"), path.with_suffix(".attempt.json")
    if receipt.exists():
        meta = base.read(receipt)
        require(
            meta["plan_hash"] == plan_hash
            and meta["cutoff"] == cutoff
            and meta["name"] == name
            and meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(),
            "CHECKPOINT_CHANGED",
        )
        return joblib.load(path)
    require(not attempt.exists(), "FIT_ALREADY_ATTEMPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "plan_hash": plan_hash})
    head = fit(rows, name, cutoff)
    joblib.dump(head, path)
    base.save(
        receipt,
        {
            "at": base.now().isoformat(),
            "name": name,
            "cutoff": cutoff,
            "plan_hash": plan_hash,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    base.save(root() / "training" / f"{label}-{name}.json", {k: v for k, v in head.items() if k != "model"})
    return head


def batch_answers(values, name, head=None):
    require(name in BRANCHES, "BRANCH_INVALID")
    if name in CANDIDATES:
        original = baseline.CANDIDATES[CANDIDATES.index(name)]
    elif name in CONTROLS[:2]:
        original = baseline.CANDIDATES[CONTROLS.index(name)]
    else:
        original = name
    return baseline.batch_answers(values, original, head)


def answers(market, group, bundle):
    return {n: batch_answers([market], n, bundle.get(n, {}).get(group))[0] for n in BRANCHES}


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    rows, proof = dataset()
    if not (root() / "training-question-proof.json").exists():
        base.save(root() / "training-question-proof.json", proof)
    bundle, ph = {}, base.digest(p)
    for name in CANDIDATES + CONTROLS[:2]:
        bundle[name] = {}
        for group in sorted({r["group"] for r in rows}):
            active()
            bundle[name][group] = (
                fit_checkpoint(
                    [r for r in rows if r["group"] == group], name, p["current_fit_cutoff"], f"current-{group}", ph
                )
                if name in CANDIDATES
                else previous.frozen_head(name, group, "current", p["old_current_cutoff"])
            )
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    result = {
        "at": base.now().isoformat(),
        "plan_hash": ph,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fingerprint": fingerprint(),
        "calendar_hash": p["calendar_hash"],
        "development_fits": 0,
        "current_fits": 6,
        "new_rows": 30,
        "new_dates": 1,
        "current_fit_cutoff": p["current_fit_cutoff"],
        "historical_recipe_reference": "round-70/result.json",
        "historical_reference_is_new_result": False,
        "kind": "CURRENT_REFIT_ONLY_NOT_ACCURACY_IMPROVEMENT",
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(root() / "result.json", result)
    return result


def models():
    p, result = plan(), base.read(root() / "result.json")
    path = root() / "models.joblib"
    require(
        result["plan_hash"] == base.digest(p)
        and result["fingerprint"] == fingerprint()
        and result["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(),
        "MODEL_OR_PLAN_CHANGED",
    )
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    rows, _ = dataset()
    market = next(r["market"] for r in reversed(rows) if r["market"]["available"])
    for fund in data.scope():
        require(set(answers(market, fund["group"], bundle)) == set(BRANCHES), "INCOMPLETE_ANSWERS")
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_NOT_FORWARD",
        "branch_checks": len(data.scope()) * len(BRANCHES),
    }
    base.save(root() / "preflight.json", value)
    return value


def tick():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_current_refresh as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_current_refresh as service

    return forward.report(service)


live_market = previous.live_market
live_answers = answers
