"""训练器只接收当前窗口的FIT/CAL及EXAM输入，不接收来源或考试答案路径。"""

from calendar import monthrange
from collections import Counter
from datetime import date
from decimal import Decimal
from uuid import UUID

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.direction_training_artifacts import digest
from app.services.direction_training_protocol import TREE, VERSIONS
from app.services.historical_nav_calibration import fit_calibrator, predict_calibrated_scores
from app.services.historical_nav_evaluation import PreparedRow, fixed_momentum_score
from app.services.historical_nav_training import (
    fit_logistic_artifact,
    predict_artifact_scores,
    restore_logistic_artifact,
    training_rows_hash,
)


def recent_lower(end: date):
    offset = end.year * 12 + end.month - 1 - 18
    year, month = offset // 12, offset % 12 + 1
    return date(year, month, min(end.day, monthrange(year, month)[1]))


def prepared(rows, lower, upper):
    output, keys = [], set()
    if not rows or len(rows) > 10000:
        raise ValueError("TRAINING_ROW_BUDGET")
    for row in rows:
        if set(row) != {"input", "answer"}:
            raise ValueError("UNEXPECTED_TRAINING_FIELDS")
        item = DirectionInput.model_validate(row["input"])
        answer = DirectionAnswer.model_validate(row["answer"])
        if item.key != answer.key or item.key in keys or not lower < item.cutoff <= answer.available_at <= upper:
            raise ValueError("TRAINING_STAGE_BOUNDARY")
        keys.add(item.key)
        output.append(
            PreparedRow(
                UUID(int=0),
                item.fund,
                item.cutoff,
                item.cutoff,
                answer.available_at,
                tuple(Decimal(str(x)) for x in item.x),
                answer.y,
                digest(row),
            )
        )
    if set(Counter(r.fund_code for r in output)) != {"001632", "006730", "008888"}:
        raise ValueError("TRAINING_COHORT_MISMATCH")
    return tuple(output)


def baseline_scores(history, exam):
    counts = Counter(r.fund_code for r in history)
    rates = {f: Decimal(sum(r.y for r in history if r.fund_code == f)) / n for f, n in counts.items()}
    return {
        "ALWAYS_UP": [1.0 for _ in exam],
        "TRAIN_UP_FREQUENCY": [float(rates[i.fund]) for i in exam],
        "MOMENTUM_20D": [float(i.x[1] > 0) for i in exam],
        "FIXED_MOMENTUM_SCORE": [float(fixed_momentum_score(Decimal(str(i.x[1])))) for i in exam],
    }


def execute_job(payload):
    if set(payload) != {"candidate", "window", "fit", "cal", "exam", "base"}:
        raise ValueError("UNEXPECTED_JOB_FIELDS")
    branch, window = payload["candidate"], payload["window"]
    fit_end, cal_end, exam_end = [date.fromisoformat(window[k]) for k in ("fit_end", "cal_end", "exam_end")]
    if not date(2021, 1, 1) <= fit_end < cal_end < exam_end <= date(2024, 12, 31):
        raise ValueError("JOB_DATES_PROTECTED")
    fit = prepared(payload["fit"], date(2020, 12, 31), fit_end)
    # A/B/C不接收CAL答案，校准与频率对照只在其各自作业中接收。
    cal = prepared(payload["cal"], fit_end, cal_end) if payload["cal"] else ()
    exam = [DirectionInput.model_validate(r) for r in payload["exam"]]
    if not exam or len(exam) > 10000 or len({i.key for i in exam}) != len(exam):
        raise ValueError("EXAM_BUDGET_OR_DUPLICATE")
    if any(not cal_end < i.cutoff <= exam_end for i in exam):
        raise ValueError("EXAM_TIME_BOUNDARY")
    x = tuple(i.x for i in exam)
    if branch == "BASELINES":
        if not cal:
            raise ValueError("BASELINE_HISTORY_MISSING")
        return {"status": "PREDICTED", "scores": baseline_scores(fit + cal, exam), "artifact": None}
    if branch not in ("A", "A_CAL", "B", "C"):
        raise ValueError("UNKNOWN_CANDIDATE")
    if branch in ("A", "B", "C") and (cal or payload["base"] is not None):
        raise ValueError("UNEXPECTED_CAL_OR_BASE")
    start = date(2021, 1, 1)
    if branch == "B":
        lower = recent_lower(fit_end)
        recent = tuple(r for r in fit if r.available_at > lower)
        if recent == fit:
            return {"status": "NOT_DISTINCT", "scores": {}, "artifact": None}
        if any(sum(r.fund_code == f for r in recent) < 252 for f in ("001632", "006730", "008888")):
            return {"status": "INSUFFICIENT_DATA", "scores": {}, "artifact": None}
        fit, start = recent, date.fromordinal(lower.toordinal() + 1)
    if any(sum(r.fund_code == f for r in fit) < 252 for f in ("001632", "006730", "008888")):
        raise ValueError("FIT_INSUFFICIENT")
    status = "PREDICTED"
    if branch in ("A", "B"):
        model = fit_logistic_artifact(fit, start_date=start, end_date=fit_end, versions=VERSIONS)
        scores, artifact = predict_artifact_scores(model, x), model.model_dump(mode="json")
    elif branch == "A_CAL":
        base = restore_logistic_artifact(payload["base"])
        if base.train_end_date != fit_end or base.train_hash != training_rows_hash(fit):
            raise ValueError("CAL_BASE_MISMATCH")
        if any(sum(r.fund_code == f for r in cal) < 60 for f in ("001632", "006730", "008888")):
            raise ValueError("CAL_INSUFFICIENT")
        model = fit_calibrator(base, cal, end_date=cal_end)
        slope = model.calibrator.slope
        status = "REVERSED_PENDING_VALIDATION" if slope < 0 else "CONSTANT_PENDING_VALIDATION" if slope == 0 else status
        scores, artifact = predict_calibrated_scores(model, x), model.model_dump(mode="json")
    else:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits

        counts = Counter(r.fund_code for r in fit)
        with threadpool_limits(limits=1):
            model = HistGradientBoostingClassifier(**TREE).fit(
                np.asarray([r.x for r in fit], dtype=float),
                [r.y for r in fit],
                sample_weight=[len(fit) / (len(counts) * counts[r.fund_code]) for r in fit],
            )
            if model.classes_.tolist() != [0, 1]:
                raise ValueError("TREE_SINGLE_CLASS")
            scores = model.predict_proba(np.asarray(x, dtype=float))[:, 1].tolist()
        artifact = {
            "kind": "TREE_REPLAY_RECIPE",
            "parameters": TREE,
            "train_rows_hash": digest(payload["fit"]),
            "fit_count": len(fit),
            "per_fund": dict(counts),
        }
    return {"status": status, "scores": {branch: list(scores)}, "artifact": artifact}
