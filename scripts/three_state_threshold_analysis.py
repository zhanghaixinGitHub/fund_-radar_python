"""只读分析冻结训练段的持平带；不读取选优段成绩、不拟合或修改任何模型。"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.auto_model_contract import thaw_value  # noqa: E402
from app.services.prediction_contract import PredictionFailure, reinvested_series, target_dates  # noqa: E402
from app.services.prediction_features import ZONE, cash_events  # noqa: E402
from app.services.prediction_research import label_values  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402


def analyze():
    """使用已有自动周期的不可变基金输入；收益标签与训练器复用同一公开时间边界。"""
    engine = create_engine(
        get_settings().ai_database_url,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c statement_timeout=10000",
        },
    )
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        cycle = (
            c.execute(
                text("""SELECT cycle_id,spec FROM prediction_auto_cycle p
          WHERE EXISTS(SELECT 1 FROM prediction_auto_input i WHERE i.cycle_id=p.cycle_id
            AND input_key LIKE 'fund:%') ORDER BY created_at DESC LIMIT 1""")
            )
            .mappings()
            .one()
        )
        inputs = list(
            c.execute(
                text("""SELECT payload FROM prediction_auto_input
          WHERE cycle_id=:id AND input_key LIKE 'fund:%' ORDER BY input_key LIMIT 501"""),
                {"id": cycle["cycle_id"]},
            ).scalars()
        )
        assert len(inputs) <= 500
        read_at = str(c.execute(text("SELECT current_timestamp")).scalar())
    spec = cycle["spec"]
    boundary = datetime.fromisoformat(spec["trainEnd"] + "T23:59:00+08:00")
    bands = {
        "T5_V1": ("0.002", "0.003", "0.005"),
        "T20_V1": ("0.005", "0.01", "0.015"),
        "M6_V1": ("0.02", "0.03", "0.05"),
    }
    counts, failures = defaultdict(Counter), Counter()
    for frozen in inputs:
        data = thaw_value(frozen)
        if "frozenFailure" in data:
            failures[data["frozenFailure"]["code"]] += 1
            continue
        sessions = data["calendar"].sessions
        starts = [d for d in sessions if spec["trainStart"] <= str(d) <= spec["trainEnd"]][::5]
        for horizon, choices in bands.items():
            for day in starts:
                target = target_dates(data["calendar"], datetime.combine(day, time(10), ZONE), horizon)
                end = target["endDate"]
                if not end or end >= spec["trainEnd"]:
                    continue
                dates = [d for d in sessions if str(day) <= str(d) <= end]
                try:
                    nav, _ = label_values(data, set(dates), boundary)
                    values = reinvested_series(dates, nav, cash_events(data, dates, boundary, replay=True))
                    value = values[-1] / values[0] - 1
                    group = data["fund"]["fund_type"]
                    for band in choices:
                        direction = "UP" if value > Decimal(band) else "DOWN" if value < -Decimal(band) else "FLAT"
                        counts[(horizon, band, group)][direction] += 1
                except PredictionFailure as error:
                    failures[error.payload["code"]] += 1
    result = {
        "readAt": read_at,
        "cycleId": str(cycle["cycle_id"]),
        "trainStart": spec["trainStart"],
        "trainEnd": spec["trainEnd"],
        "inputCount": len(inputs),
        "stride": 5,
        "evidenceLevel": "DEVELOPMENT_ONLY",
        "failures": dict(failures),
        "groups": [
            {"horizon": h, "band": b, "fundType": g, "counts": dict(v)} for (h, b, g), v in sorted(counts.items())
        ],
    }
    destination = Path(".local-runs/three-state/threshold-analysis.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    analyze()
