"""补齐第74轮预声明的五输入对照，复用原模型和原答案，新增拟合为零。"""

import hashlib
import shutil
from collections import defaultdict

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_child_forward_v2 as forward
from app.services import direction_1d_sprint_market_gap_ablation as original
from app.services import direction_1d_sprint_market_gap_delta as five

CANDIDATES = original.CANDIDATES
CONTROLS = (
    "FROZEN_R70_MARKET_LR3",
    "FROZEN_R70_MARKET_SIGN_LR3",
    "FROZEN_R73_MARKET_LR5",
    "MARKET_MAJORITY3",
    "SPX_SIGN",
    "ALWAYS_UP",
)
BRANCHES = CANDIDATES + CONTROLS
FIRST_TARGET = original.FIRST_TARGET
REPAIR_HASH = "d2cfab8168775ab76d8ec4ad9cffcbd05e60a80e0f895486ab3f774ee80a9a67"
live_market = original.live_market


def root():
    return base.ROOT / "round-74-v2"


def fingerprint():
    value = original.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_gap_ablation_v2.py",
        "scripts/direction_1d_sprint_market_gap_ablation_v2.py",
        "tests/test_direction_1d_sprint_market_gap_ablation_v2.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def plan():
    repair = base.read(root() / "repair-plan-before-implementation.json")
    if base.digest(repair) != REPAIR_HASH:
        raise ValueError("ABLATION_REPAIR_PLAN_CHANGED")
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ABLATION_REPAIR_CODE_OR_CALENDAR_CHANGED")
        for file, digest in value["input_hashes"].items():
            if base.digest(base.read(base.ROOT / file)) != digest:
                raise ValueError("ABLATION_REPAIR_INPUT_CHANGED")
        return value
    original.active()
    old_result, _ = original.models()
    five_result, _ = five.models()
    if base.digest(old_result) != repair["parent_result_hash"] or base.digest(five_result) != repair["r73_result_hash"]:
        raise ValueError("ABLATION_REPAIR_PARENT_RESULT_CHANGED")
    paths = [original.root() / "result.json", five.root() / "result.json"]
    paths += list((original.root() / "folds").glob("*.json"))
    paths += list((five.root() / "folds").glob(f"*-{five.CANDIDATES[0]}.json"))
    value = {
        "at": base.now().isoformat(),
        "repair_hash": REPAIR_HASH,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "candidates": list(CANDIDATES),
        "controls": list(CONTROLS),
        "input_hashes": {p.relative_to(base.ROOT).as_posix(): base.digest(base.read(p)) for p in paths},
        "new_fit_count": 0,
        "source_requests": 0,
        "new_cost_cny": 0,
        "new_2026_scores": False,
        "source_timing": original.plan()["source_timing"],
        "first_forward_target": FIRST_TARGET,
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        if name.endswith("market_gap_ablation_v2.py"):
            dest = root() / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base.PROJECT / name, dest)
    return value


def batch_answers(values, name, head=None):
    if name == CONTROLS[2]:
        return five.batch_answers(values, five.CANDIDATES[0], head)
    if name not in BRANCHES:
        raise ValueError("ABLATION_REPAIR_BRANCH_INVALID")
    return original.batch_answers(values, name, head)


def answers(market, group, bundle):
    return {n: batch_answers([market], n, bundle.get(n, {}).get(group))[0] for n in BRANCHES}


live_answers = answers


def composed():
    old_result, old_bundle = original.models()
    five_result, five_bundle = five.models()
    # 丢弃原包中被放到规则名称下的闲置头；规则自身仍按固定三票算。
    bundle = {n: old_bundle[n] for n in CANDIDATES + CONTROLS[:2]}
    bundle[CONTROLS[2]] = five_bundle[five.CANDIDATES[0]]
    return old_result, five_result, bundle


def train():
    """仅复制/核对冻结开发答案并登记模型组合，不调用任何fit或供应商接口。"""
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    original.active()
    old_result, five_result, _ = composed()
    output = defaultdict(list)
    for name in BRANCHES:
        source_root, source_name = (five.root(), five.CANDIDATES[0]) if name == CONTROLS[2] else (original.root(), name)
        for path in sorted((source_root / "folds").glob(f"*-{source_name}.json")):
            rows = base.read(path)
            prefix = path.name[: -len(source_name) - len(".json")]
            dest = root() / "folds" / f"{prefix}{name}.json"
            if dest.exists():
                if base.read(dest) != rows:
                    raise ValueError("ABLATION_REPAIR_SAVED_FOLD_CHANGED")
            else:
                base.save(dest, rows)
            output[name].extend(rows)
    expected = sorted((r["code"], r["u"], r["y"]) for r in output[CANDIDATES[0]])
    if len(expected) != 7290 or any(
        sorted((r["code"], r["u"], r["y"]) for r in rows) != expected for rows in output.values()
    ):
        raise ValueError("ABLATION_REPAIR_COMMON_QUESTIONS_CHANGED")
    metrics = {
        subset: {
            n: base.metrics([r for r in rows if subset == "all" or r["old_question"] == (subset == "old")])
            for n, rows in output.items()
        }
        for subset in ("all", "old", "added")
    }
    for subset in metrics:
        for name in BRANCHES:
            source, source_name = (five_result, five.CANDIDATES[0]) if name == CONTROLS[2] else (old_result, name)
            if metrics[subset][name] != source["metrics"][subset][source_name]:
                raise ValueError("ABLATION_REPAIR_ORIGINAL_METRICS_CHANGED")
    manifest = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "original_r74_model_sha256": old_result["model_sha256"],
        "r73_model_sha256": five_result["model_sha256"],
        "bindings": {
            n: ("R73:" + five.CANDIDATES[0] if n == CONTROLS[2] else "R74:" + n) for n in CANDIDATES + CONTROLS[:3]
        },
        "new_fit_count": 0,
    }
    base.save(root() / "models-manifest.json", manifest)
    result = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "fingerprint": fingerprint(),
        "model_sha256": hashlib.sha256((root() / "models-manifest.json").read_bytes()).hexdigest(),
        "model_format": "VERIFIED_COMPOSITION_OF_ORIGINAL_LOCAL_MODELS",
        "metrics": metrics,
        "new_development_fits": 0,
        "new_current_fits": 0,
        "original_r74_development_fits": 24,
        "original_r74_current_fits": 6,
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
        "kind": "PREDECLARED_CONTROL_RESTORED_WITHOUT_RETRAINING",
    }
    base.save(root() / "result.json", result)
    return result


def models():
    p, value = plan(), base.read(root() / "result.json")
    path = root() / "models-manifest.json"
    manifest = base.read(path)
    old, five_result, bundle = composed()
    if (
        value["plan_hash"] != base.digest(p)
        or manifest["plan_hash"] != base.digest(p)
        or value["fingerprint"] != fingerprint()
        or value["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        or manifest["original_r74_model_sha256"] != old["model_sha256"]
        or manifest["r73_model_sha256"] != five_result["model_sha256"]
    ):
        raise ValueError("ABLATION_REPAIR_MODEL_BINDING_CHANGED")
    return value, bundle


def preflight():
    _, bundle = models()
    rows, _ = original.data.dataset()
    market = next(r["market"] for r in reversed(rows) if r["market"]["available"])
    for fund in original.data.scope():
        answers(market, fund["group"], bundle)
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_NOT_FORWARD",
        "branch_checks": len(original.data.scope()) * len(BRANCHES),
    }
    base.save(root() / "preflight.json", value)
    return value


def tick():
    from app.services import direction_1d_sprint_market_gap_ablation_v2 as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_gap_ablation_v2 as service

    return forward.report(service)
