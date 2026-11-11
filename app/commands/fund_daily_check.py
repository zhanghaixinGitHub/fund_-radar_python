"""每日只读核对资料、持仓评分及四周期正式归档，漏跑和缺回执一律不算成功。"""

import argparse
import json
from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_daily_runtime import PROJECT, LocalDailyBackend
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save


def assess(state, followthrough, scope, ratings_readback, checked_at, live_window=None):
    issues = []
    if state.get("owner_user_id") != followthrough.get("owner_user_id"):
        issues.append("OWNER_MISMATCH")
    if set(state.get("scope", [])) != set(scope):
        issues.append("HOLDINGS_CHANGED")
    if state.get("scope_hash") != digest(scope):
        issues.append("SCOPE_RECEIPT_MISMATCH")
    if state.get("checked_at", "")[:10] != checked_at[:10]:
        issues.append("RUN_NOT_CHECKED_TODAY")
    if state.get("window") != followthrough.get("window"):
        issues.append("PREDICTION_WINDOW_CHANGED")
    if live_window and state.get("window", {}).get("target_nav_date") != live_window["target_nav_date"]:
        issues.append("LIVE_PREDICTION_WINDOW_CHANGED")
    if followthrough.get("status") != "CLOSED" or not followthrough.get("all_predictions_generated"):
        issues.append("PREDICTIONS_INCOMPLETE")
    materials = state.get("materials_terminal") or {}
    if not materials.get("terminal"):
        issues.append("MATERIALS_NOT_FINISHED")
    elif any(
        c.get("status") not in {"SUCCEEDED", "PARTIAL_SUCCESS"}
        for c in materials.get("children", [])
        if c.get("job_type") not in {"FUND_RATINGS", "DIRECTION_1D_PREDICTIONS"}
    ):
        issues.append("MATERIAL_STAGE_FAILED")
    outcome = state.get("prediction_outcomes") or {}
    summary = outcome.get("summary") or {}
    counts = {"total", "checked", "completed", "created", "pending", "failed", "remaining"}
    if not counts <= summary.keys():
        issues.append("OUTCOME_RECEIPT_MISSING")
    elif any(summary[k] for k in ("pending", "failed", "remaining")):
        issues.append("OLD_PREDICTION_OUTCOMES_INCOMPLETE")
    rating = state.get("ratings") or {}
    if rating.get("status") != "SUCCEEDED" or set(rating.get("scope", [])) != set(scope):
        issues.append("RATINGS_INCOMPLETE")
    if len(ratings_readback) != len(scope):
        issues.append("RATING_READBACK_INCOMPLETE")
    elif {r.get("fundCode") for r in ratings_readback} != set(scope):
        issues.append("RATING_READBACK_SCOPE_MISMATCH")
    elif any(r.get("status") not in {"RATED", "SCORED"} or r.get("score") is None for r in ratings_readback):
        issues.append("RATING_RESULTS_NOT_COMPLETE")
    return {
        "status": "SUCCEEDED" if not issues else "INCOMPLETE",
        "checked_at": checked_at,
        "scope_kind": "SIMULATED_HOLDINGS",
        "scope": scope,
        "fund_count": len(scope),
        "issues": issues,
        "predictions": followthrough,
        "ratings_readback": ratings_readback,
        "prediction_outcomes": outcome,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    at = now().isoformat()
    directory = args.run_dir or PROJECT / ".local-runs" / ("daily-prediction-" + now().strftime("%Y%m%d"))
    path = directory / "run.json"
    if not path.exists():
        result = {"status": "NOT_STARTED", "checked_at": at, "issues": ["TODAYS_RUN_MISSING"]}
    else:
        backend = LocalDailyBackend(directory)
        try:
            _, _, scope, live_window = backend.context()
            # 重新交叉核对两端原预测、当前资料摘要和本人归档，不只读昨天的成功标志。
            followthrough = backend.guard(directory / "check-followthrough.json", None)
            ratings = [backend.get(f"http://127.0.0.1:8000/internal/v1/funds/{c}/rating") for c in scope]
            result = assess(read(path), followthrough, scope, ratings, at, live_window)
        except Exception as error:
            result = {
                "status": "FAILED",
                "checked_at": at,
                "issues": [type(error).__name__],
                "reason": str(error)[:300],
            }
        finally:
            backend.close()
    output = PROJECT / ".local-runs" / "daily-checks" / (now().strftime("%Y%m%d") + ".json")
    versioned_save(output, result)
    print(
        json.dumps({"status": result["status"], "issues": result["issues"], "output": str(output)}, ensure_ascii=False)
    )
    raise SystemExit(0 if result["status"] == "SUCCEEDED" else 2)


if __name__ == "__main__":
    main()
