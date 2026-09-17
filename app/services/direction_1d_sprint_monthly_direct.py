"""第60轮：仅改变三输入一日模型的历史更新频率，比较月初与季初重训。

所有原题及缺数据时的季度HK回退保持不变；当前统一重训时点的参数与第50轮完全相同，
直接绑定已有未来预测分支，避免复制同一模型后将重复答案当作新的独立证据。
"""

import hashlib
import shutil

import joblib
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_etf_joint as original
from app.services import direction_1d_sprint_stacking_monotonic as previous

CANDIDATE = "MONTHLY_REFRESH_DIRECT_ETF3_LR504"


def root():
    return base.ROOT / "round-60"


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_monthly_direct.py",
        "scripts/direction_1d_sprint_monthly_direct.py",
        "tests/test_direction_1d_sprint_monthly_direct.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def selected(rows, cutoff):
    """保持原模型的504个成熟目标日及数据可用条件，不能混入月内尚未发生的标签。"""
    return original.adaptive.training_rows(
        [r for r in rows if r["u"] < cutoff and r["mature"] < cutoff and original.available(r["z"])],
        cutoff,
        "MONTHLY_BAL504",
    )


def month_window(month):
    if month not in range(1, 13):
        raise ValueError("MONTHLY_DEVELOPMENT_MONTH_INVALID")
    return f"2025-{month:02d}-01", "2026-01-01" if month == 12 else f"2025-{month + 1:02d}-01"


def quarter_head(month, group):
    """季度对照只从原冻结产物读取，校验其时间及内容；不重新训练或替换原模型。"""
    q = (month - 1) // 3 + 1
    path = original.root() / f"checkpoints/{q}-{group}-{original.LEARNED[0]}.joblib"
    receipt = base.read(path.with_suffix(".json"))
    if receipt["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("MONTHLY_QUARTER_MODEL_CHANGED")
    trained = joblib.load(path)
    if trained["us_etf"]["cutoff"] != month_window(3 * q - 2)[0]:
        raise ValueError("MONTHLY_QUARTER_CUTOFF_CHANGED")
    return trained, receipt


def plan():
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        if p["fingerprint"] != fingerprint() or p["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND60_CODE_OR_CALENDAR_CHANGED")
        return p
    original.active()
    result, _ = original.models()
    proposal = base.read(root() / "proposal-before-implementation.json")
    prior = original.plan()
    expected = {
        "new_development_fits": 24,
        "reused_quarter_fits": 12,
        "new_current_fits": 0,
        "reused_current_models": 3,
        "max_reproduction_count": 1,
        "new_provider_requests": 0,
        "new_cost_cny": 0,
    }
    if (
        proposal["budget"] != expected
        or proposal["candidate"] != CANDIDATE
        or proposal["prior_result_hash"] != base.digest(result)
        or proposal["current_cutoff"] != prior["current_fit_cutoff"]
    ):
        raise ValueError("MONTHLY_PROPOSAL_CHANGED")
    p = {
        "at": base.now().isoformat(),
        "candidate": CANDIDATE,
        "proposal": proposal,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "current_fit_cutoff": prior["current_fit_cutoff"],
        "input_hashes": prior["input_hashes"] | {"round-60/proposal-before-implementation.json": base.digest(proposal)},
        "target": "Next adjacentCN rawunitNAV UP/NON_UP,flatseparately;not monthlyreturn",
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        destination = root() / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, destination)
    return p


def checkpoint(rows, month, group):
    """每个非季度起始月只拟合一次；中断保留attempt，禁止自动重复拟合。"""
    start, _ = month_window(month)
    old, old_receipt = quarter_head(month, group)
    chosen = selected(rows, start)
    if month in (1, 4, 7, 10):
        if old["us_etf"]["fit_hash"] != base.digest(chosen):
            raise ValueError("MONTHLY_REUSED_TRAINING_CHANGED")
        return old, {"reused": True, "source": old_receipt}
    path = root() / f"checkpoints/{month:02d}-{group}.joblib"
    receipt_path, attempt = path.with_suffix(".json"), path.with_suffix(".attempt.json")
    if receipt_path.exists():
        receipt = base.read(receipt_path)
        if receipt["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest() or receipt[
            "training_hash"
        ] != base.digest(chosen):
            raise ValueError("MONTHLY_NEW_CHECKPOINT_CHANGED")
        return joblib.load(path), receipt
    if attempt.exists():
        raise ValueError("MONTHLY_INTERRUPTED_NO_AUTORETRY")
    base.save(attempt, {"at": base.now().isoformat(), "cutoff": start, "group": group, "new_fit": True})
    head = original.fit(rows, original.LEARNED[0], start)
    if head["fit_hash"] != base.digest(chosen) or head["max_mature_date"] >= start or head["fit_end"] >= start:
        raise ValueError("MONTHLY_FIT_TEMPORAL_BOUNDARY_CHANGED")
    trained = old | {"us_etf": head, "new_fit_count": 1}
    joblib.dump(trained, path)
    receipt = {
        "at": base.now().isoformat(),
        "reused": False,
        "cutoff": start,
        "group": group,
        "training_hash": head["fit_hash"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "quarter_control_sha256": old_receipt["sha256"],
    }
    base.save(receipt_path, receipt)
    return trained, receipt


def current_alias(rows):
    """统一在当前时点拟合时与原模型数学上相同，校验训练行和实际输出后复用。"""
    result, models = original.models()
    p = original.plan()
    references = {}
    with threadpool_limits(limits=2):
        for group, trained in models[original.LEARNED[0]].items():
            group_rows = [r for r in rows if r["group"] == group]
            chosen = selected(group_rows, p["current_fit_cutoff"])
            head = trained["us_etf"]
            if head["fit_hash"] != base.digest(chosen):
                raise ValueError("MONTHLY_CURRENT_ALIAS_ROWS_CHANGED")
            if head["cutoff"] != p["current_fit_cutoff"] or head["model"].C != 0.1 or head["model"].fit_intercept:
                raise ValueError("MONTHLY_CURRENT_ALIAS_RECIPE_CHANGED")
            # 在每只基金的最新输入上核对实际推断，不增加任何供应商调用或当前拟合。
            latest = {}
            for row in group_rows:
                if row["code"] not in latest or row["u"] > latest[row["code"]]["u"]:
                    latest[row["code"]] = row
            answers = original.batch_answers([r["z"] for r in latest.values()], original.LEARNED[0], trained)
            references[group] = {
                "training_hash": head["fit_hash"],
                "cutoff": head["cutoff"],
                "inference_checks": len(answers),
                "answer_hash": base.digest(answers),
            }
    return {
        "at": base.now().isoformat(),
        "candidate": CANDIDATE,
        "source_candidate": original.LEARNED[0],
        "source_model_sha256": result["model_sha256"],
        "references": references,
        "source_forward_directory": "round-50/forward",
        "source_receipt_directory": "round-50/receipts",
        "kind": "SAME_CURRENT_MODEL_NOT_NEW_INDEPENDENT_BRANCH",
        "new_current_fits": 0,
        "new_forward_accuracy_claim": False,
        "new_2026_audit_metrics": False,
    }


def train():
    p = plan()
    if (root() / "result.json").exists():
        return base.read(root() / "result.json")
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("MONTHLY_INPUT_CHANGED")
    rows, proofs = original.dataset()
    if not (root() / "training-question-proof.json").exists():
        base.save(root() / "training-question-proof.json", proofs)
    output, windows = [], []
    with threadpool_limits(limits=2):
        for month in range(1, 13):
            original.active()
            start, end = month_window(month)
            for group in sorted({r["group"] for r in rows}):
                group_rows = [r for r in rows if r["group"] == group]
                trained, receipt = checkpoint(group_rows, month, group)
                exam = [r for r in group_rows if start <= r["u"] < end]
                answers = original.batch_answers([r["z"] for r in exam], original.LEARNED[0], trained)
                scored = [
                    {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")} | a
                    for r, a in zip(exam, answers, strict=True)
                ]
                path = root() / f"folds/{month:02d}-{group}-{CANDIDATE}.json"
                if path.exists():
                    previous_answers = base.read(path)
                    if len(previous_answers) != len(scored) or any(
                        any(a[k] != c[k] for k in ("code", "family", "group", "u", "y", "actual_direction"))
                        for a, c in zip(previous_answers, scored, strict=True)
                    ):
                        raise ValueError("MONTHLY_SAVED_QUESTION_IDENTITY_CHANGED")
                    if not original.runtime.answers_match(
                        {
                            str(i): {
                                k: r[k]
                                for k in ("prediction", "research_score", "kind", "baseline_prediction", "flipped")
                            }
                            for i, r in enumerate(previous_answers)
                        },
                        {
                            str(i): {
                                k: r[k]
                                for k in ("prediction", "research_score", "kind", "baseline_prediction", "flipped")
                            }
                            for i, r in enumerate(scored)
                        },
                    ):
                        raise ValueError("MONTHLY_SAVED_ANSWERS_CHANGED")
                else:
                    base.save(path, scored)
                output.extend(scored)
                windows.append({"month": month, "group": group, "receipt": receipt, "questions": len(scored)})
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "month": month}, replace=True)
    expected = []
    for path in sorted((original.root() / "folds").glob(f"*-{original.LEARNED[0]}.json")):
        expected.extend(base.read(path))

    def keys(values):
        return sorted((r["code"], r["u"], r["y"]) for r in values)

    if len(output) != 5670 or keys(output) != keys(expected) or len({r["u"] for r in output}) != 199:
        raise ValueError("MONTHLY_COMMON_QUESTIONS_CHANGED")
    alias = current_alias(rows)
    base.save(root() / "current-model-alias.json", alias)
    value = {
        "at": base.now().isoformat(),
        "candidate": CANDIDATE,
        "plan_hash": base.digest(p),
        "metrics": base.metrics(output),
        "windows": windows,
        "new_development_fits": 24,
        "reused_quarter_fits": 12,
        "new_current_fits": 0,
        "current_alias_hash": base.digest(alias),
        "new_provider_requests": 0,
        "new_cost_cny": 0,
        "new_2026_audit_metrics": False,
        "kind": "DEVELOPMENT_CADENCE_DIAGNOSTIC_NOT_NEW_CURRENT_MODEL",
    }
    base.save(root() / "result.json", value)
    return value
