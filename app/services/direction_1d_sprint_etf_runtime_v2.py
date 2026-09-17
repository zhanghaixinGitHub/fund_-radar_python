"""ETF两轮的独立预测验证版本：方向严格一致，浮点分数只允许1e-12舍入误差。

并行树求和顺序会使同一模型重复推理相差约1e-16；这不应被误报为模型或答案篡改。
原模型和第48/49轮代码不改，新增版本与每份未来答案绑定；身份、方向、原始数据及时间仍严格核验。
"""

import hashlib
import math
import shutil
from collections import defaultdict
from datetime import date, datetime, time

import numpy as np
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_us_etf as r48
from app.services import direction_1d_sprint_us_etf_data as us_etf_data
from app.services import direction_1d_sprint_us_etf_direct as r49

VERSION = "ETF_RUNTIME_V2_FLOAT_TOLERANCE"
MODULES = (r48, r49)
SCORE_ATOL = 1e-12


def folder():
    return base.ROOT / "runtime-etf-v2"


def fingerprint():
    value = r49.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_etf_runtime_v2.py",
        "scripts/direction_1d_sprint_etf_runtime_v2.py",
        "tests/test_direction_1d_sprint_etf_runtime_v2.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def freeze():
    """冻结独立执行版本；模型身份沿原结果核验，既有答案如存在只登记哈希，不覆盖。"""
    path = folder() / "plan.json"
    if path.exists():
        return verify(r49)
    r49.active()
    models = {
        s.root().name: {"result_hash": base.digest(s.models()[0]), "plan_hash": base.digest(s.plan())} for s in MODULES
    }
    legacy = {
        str(p.relative_to(base.ROOT)): base.digest(base.read(p))
        for s in MODULES
        for p in (s.root() / "forward").glob("*/*.json")
    }
    value = {
        "at": base.now().isoformat(),
        "version": VERSION,
        "fingerprint": fingerprint(),
        "models": models,
        "legacy_forecasts": legacy,
        "score_atol": SCORE_ATOL,
        "new_fits": 0,
        "new_provider_requests": 0,
        "change": "Only finite research_score accepts absolute1e-12 roundoff; all other fields exact",
        "reason": "Same frozen HK tree re-inference differed approximately1e-16 in R49 fallback audit",
        "calendar_hash": base.calendar()[1],
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = folder() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def verify(s):
    """执行及回读均验证模型、日历和226份源文件；新容差不能绕过任何原始数据哈希。"""
    manifest = base.read(folder() / "plan.json")
    if s not in MODULES or manifest["version"] != VERSION or manifest["score_atol"] != SCORE_ATOL:
        raise ValueError("ETF_RUNTIME_SCOPE_CHANGED")
    if manifest["fingerprint"] != fingerprint() or manifest["calendar_hash"] != base.calendar()[1]:
        raise ValueError("ETF_RUNTIME_CODE_OR_CALENDAR_CHANGED")
    expected = manifest["models"][s.root().name]
    if expected["result_hash"] != base.digest(base.read(s.root() / "result.json")) or expected[
        "plan_hash"
    ] != base.digest(s.plan()):
        raise ValueError("ETF_RUNTIME_MODEL_BINDING_CHANGED")
    return manifest


def validate_runtime_binding(s, path, value, manifest):
    """旧答案只能来自冻结时登记的内容；新答案必须写入本执行版本的不可变摘要。"""
    if "runtime_version" not in value:
        expected = manifest["legacy_forecasts"].get(str(path.relative_to(base.ROOT)))
        if expected is None or expected != base.digest(value):
            raise ValueError("ETF_RUNTIME_UNREGISTERED_LEGACY_ANSWER")
    elif value["runtime_version"] != VERSION or value.get("runtime_manifest_hash") != base.digest(manifest):
        raise ValueError("ETF_RUNTIME_FORECAST_BINDING_CHANGED")


def answers_match(saved, expected):
    """容差只适用于有限的研究分数；预测方向、基线、翻转标记和分数类型逐项严格相同。"""
    if saved.keys() != expected.keys():
        return False
    for name, actual in saved.items():
        wanted = expected[name]
        if actual.keys() != wanted.keys():
            return False
        for key, value in actual.items():
            if key != "research_score":
                if value != wanted[key]:
                    return False
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isinstance(wanted[key], (int, float))
                or not math.isfinite(value)
                or not math.isfinite(wanted[key])
                or not 0 <= value <= 1
                or not 0 <= wanted[key] <= 1
                or abs(value - wanted[key]) > SCORE_ATOL
            ):
                return False
    return True


def preflight():
    """在真实任务环境验证两个冻结模型各30只基金，共120条推理路径，不创建未来答案。"""
    rows, _ = r49.dataset()
    selected = {}
    for row in reversed(rows):
        selected.setdefault(row["code"], row)
    count = 0
    for s in MODULES:
        verify(s)
        _, bundle = s.models()
        for row in selected.values():
            for name in s.CANDIDATES:
                model = bundle[name][row["group"]]
                first, second = s.answer(row["z"], name, model), s.answer(row["z"], name, model)
                if not answers_match({name: first}, {name: second}):
                    raise ValueError("ETF_RUNTIME_PREFLIGHT_ANSWER_CHANGED")
                count += 1
    value = {"at": base.now().isoformat(), "branch_checks": count, "kind": "DRY_RUN_NOT_FORWARD"}
    base.save(folder() / "preflight.json", value)
    return value


def tick(s):
    manifest = verify(s)
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    w = base.window(at)
    if at >= end or w["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        return report(s)
    target = w["target_nav_date"]
    if target < s.FIRST_TARGET:
        return report(s)
    paths = [
        p
        for p in (previous.root() / "forward" / target).glob("*.json")
        if not (s.root() / "forward" / target / p.name).exists()
    ]
    if not paths:
        return report(s)
    overnight.source()
    model_manifest, bundle = s.models()
    hk_input = hk_live.capture(at)
    us_etf_input = us_etf_data.capture(base.now())
    if hk_input is None or us_etf_input is None:
        return report(s)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = s.live_vector(source, original, hk_input["rows"], us_etf_input["rows"])
            choices = {n: s.answer(z, n, bundle[n][original["group"]]) for n in s.CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
                "us_etf_input_hash": base.digest(us_etf_input),
                "original_hash": base.digest(original),
                "model_hash": model_manifest["model_sha256"],
                "answers": choices,
                "z": z,
                "status": "MODEL_NOT_RELEASED",
                "runtime_version": VERSION,
                "runtime_manifest_hash": base.digest(manifest),
            }
            saved = s.root() / "forward" / target / path.name
            base.save(saved, value)
            readback = base.now()
            verified = base.read(saved) == value and readback < deadline
            base.save(
                s.root() / "receipts" / target / path.name,
                {
                    "readback_at": readback.isoformat(),
                    "forecast_hash": base.digest(value),
                    "status": "VERIFIED" if verified else "LATE_OR_INVALID",
                },
            )
    return report(s)


def report(s):
    manifest = verify(s)
    if not (s.root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    original_report = base.report()
    result = base.read(s.root() / "result.json")
    bundle = s.models()[1] if any((s.root() / "forward").glob("*/*.json")) else None
    paired, good, late, pending, closed = defaultdict(list), 0, 0, 0, 0
    for path in (s.root() / "forward").glob("*/*.json"):
        value = base.read(path)
        validate_runtime_binding(s, path, value, manifest)
        receipt_path = s.root() / "receipts" / value["u"] / path.name
        receipt = base.read(receipt_path) if receipt_path.exists() else {}
        deadline = datetime.combine(date.fromisoformat(value["u"]), time(8, 30), base.ZONE)
        if (
            value["u"] < s.FIRST_TARGET
            or receipt.get("status") != "VERIFIED"
            or receipt.get("forecast_hash") != base.digest(value)
            or datetime.fromisoformat(receipt["readback_at"]) >= deadline
            or datetime.fromisoformat(value["at"]) >= deadline
        ):
            late += 1
            continue
        p5, p4, source, original = dual.read_parent(previous.root() / "forward" / value["u"] / path.name)
        if (
            any(
                value[k] != base.digest(v)
                for k, v in (("parent_hash", p5), ("source_hash", source), ("original_hash", original))
            )
            or value["model_hash"] != result["model_sha256"]
        ):
            raise ValueError("ETF_RUNTIME_V2_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        us_etf_input = us_etf_data.load(value["u"])
        if value["us_etf_input_hash"] != base.digest(us_etf_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"], s.live_vector(source, original, hk_input["rows"], us_etf_input["rows"]), rtol=0, atol=1e-12
        ):
            raise ValueError("ETF_RUNTIME_V2_VECTOR_CHANGED")
        expected_answers = {n: s.answer(value["z"], n, bundle[n][original["group"]]) for n in s.CANDIDATES}
        if not answers_match(value["answers"], expected_answers):
            raise ValueError("ETF_RUNTIME_V2_SAVED_ANSWER_CHANGED")
        good += 1
        closed += value["u"] in original_report["closed_targets"]
        outcome_path = base.ROOT / "outcomes" / value["u"] / path.name
        if not outcome_path.exists():
            pending += 1
            continue
        outcome = base.read(outcome_path)
        if outcome["forecast_hash"] != base.digest(original):
            raise ValueError("OUTCOME_INPUT_CHANGED")
        for name, choice in (value["answers"] | p5["answers"] | p4["answers"] | original["answers"]).items():
            paired[name].append(original | outcome | {"prediction": choice["prediction"]})
        paired["ALWAYS_UP"].append(original | outcome | {"prediction": 1})
    closed_targets = [d for d in original_report["closed_targets"] if d >= s.FIRST_TARGET]
    due = len(closed_targets) * original_report["eligible_funds"]
    whole = len(closed_targets) * original_report["watchlist_funds"]
    value = {
        "at": base.now().isoformat(),
        "primary_candidate": result["winner"],
        "verified_forecasts": good,
        "invalid_or_late": late,
        "pending": pending,
        "eligible_coverage": closed / due if due else None,
        "whole_watchlist_coverage": closed / whole if whole else None,
        "missing_due_predictions": due - closed,
        "matched_forward_metrics": {n: base.metrics(v) for n, v in paired.items()},
        "new_cost_cny": 0,
        "model_released": False,
    }
    base.save(s.root() / "report.json", value, replace=True)
    return value
