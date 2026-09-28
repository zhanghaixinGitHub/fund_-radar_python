"""修复协议独立计数：使用标准库逐日重算，不调用模型训练或比较函数。"""

import argparse
import json
import re
from pathlib import Path

from scripts.fund_002112_round3_audit import audit, read


def recount_gate(comparison, *, historical):
    """根据已经逐日独立核验过的计数重算预定门槛，不能信任 passed 字段。"""
    models = comparison["models"]
    tree = models["T20"]
    checks = {
        "total_strictly_above_controls": all(tree["correct"] > models[v]["correct"] for v in ("N7", "L20")),
        "total_strictly_above_constants": all(
            tree["correct"] > value["correct"] for value in comparison["constants"].values()
        ),
    }
    for kind in ("DOWN", "FLAT", "UP"):
        checks["class_not_worse_" + kind] = all(
            tree["class_correct"][kind] >= models[v]["class_correct"][kind] for v in ("N7", "L20")
        )
    if historical:
        checks["at_least_two_quarters_not_worse_L20"] = (
            sum(quarter["T20"]["correct"] >= quarter["L20"]["correct"] for quarter in comparison["quarters"].values())
            >= 2
        )
    assert checks == comparison["checks"]
    assert all(checks.values()) == comparison["numerical_passed"]
    return checks


def verify(path):
    evidence, daily = audit(path)
    comparison = read(path / "comparison.json")
    historical = comparison["historical"]
    assert historical is not None and len(daily["historical"]) == 161
    historical_checks = recount_gate(historical, historical=True)
    decision = read(path / "historical-decision.json")
    assert decision["passed"] == all(historical_checks.values())
    development = comparison["development"]
    if development is not None:
        assert decision["passed"] and len(daily["development"]) == 230
        development_checks = recount_gate(development, historical=False)
        assert read(path / "development-decision.json")["passed"] == all(development_checks.values())
    else:
        assert not decision["passed"]
        development_checks = None
    attempts = list((path / "attempts").glob("*.json"))
    calls = list((path / "classifier-started").glob("*.json"))
    completed = list((path / "classifier-completed").glob("*.json"))
    assert len(attempts) == len(calls) == len(completed) == (19 if development is not None else 13)
    evidence.update(
        historical_checks=historical_checks,
        development_checks=development_checks,
        new_real_fits=len(calls),
        inherited_fits=5,
        lifetime_real_fits=39 + len(calls),
        new_fits_during_audit=0,
        passed=True,
    )
    return evidence, daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"002112-r3r-[0-9a-f]{24}", args.run_id):
        parser.error("新协议运行编号不合法")
    path = Path(__file__).resolve().parents[1] / ".local-runs/fund-exposure-002112/round3-repair-runs" / args.run_id
    evidence, daily = verify(path)
    # 写文件使用既有的不可变 JSON 封装；计数逻辑本身仅使用上面的标准库。
    from app.services.fund_002112_round3_data import write_once

    write_once(path / "independent-audit.json", evidence)
    write_once(path / "daily-comparison.json", daily)
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
