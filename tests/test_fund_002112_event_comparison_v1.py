"""离线研究边界验证：时间、未来数据隔离、文本实际参与与失败拟合不重复。"""

from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_protocol as production
from scripts import fund_002112_event_comparison_data_v1 as data
from scripts import fund_002112_event_comparison_v1 as experiment


@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2026-09-29T14:59:59+08:00", ("2026-09-28", "2026-09-29")),
        ("2026-09-29T15:00:00+08:00", ("2026-09-29", "2026-09-30")),
        ("2026-09-29T23:00:00+08:00", ("2026-09-29", "2026-09-30")),
        ("2026-09-30T00:00:00+08:00", ("2026-09-29", "2026-09-30")),
        ("2026-10-01T08:30:00+08:00", ("2026-09-30", "2026-10-08")),
        ("2026-10-04T23:00:00+08:00", ("2026-09-30", "2026-10-08")),
        ("2026-09-29T07:00:00+00:00", ("2026-09-29", "2026-09-30")),
    ],
)
def test_clock_matches_existing_production(stamp, expected):
    sessions = [str(day) for day in production.calendar()[0]]
    assert data.prediction_target(sessions, stamp) == expected
    actual = production.window(data.moment(stamp))
    assert (actual["base_nav_date"], actual["target_nav_date"]) == expected


def test_strict_nav_waits_for_base_publication():
    sessions = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(62)]
    source = {
        "sessions": sessions,
        "nav": {
            day: {"unit_nav": str(1 + i / 100), "available_at": day + "T21:00:00+08:00"}
            for i, day in enumerate(sessions)
        },
    }
    base = sessions[-2]
    assert data.strict_nav(source, base, base + "T20:59:59+08:00")[0] is None
    vector, proof = data.strict_nav(source, base, base + "T21:00:00+08:00")
    assert np.allclose(vector, experiment.independent_nav(proof["values"]), rtol=0, atol=1e-12)
    del source["nav"][sessions[12]]
    assert data.strict_nav(source, base, base + "T23:00:00+08:00")[1]["reason"] == "NAV_GAP"


def event_source():
    return {
        "sessions": ["2026-09-28", "2026-09-29", "2026-09-30"],
        "profiles": {},
        "company_events": [],
        "reports": [
            {
                "fund_code": "002112",
                "available_at": "2026-09-01T00:00:00+08:00",
                "report_end": "2026-06-30",
                "raw": {"sha256": "report"},
                "holdings": [{"stock_code": "300001.SZ", "nav_weight_pct": 10, "stock_name": "示例公司"}],
            }
        ],
        "documents": [
            {
                "id": "event",
                "event_type": "ANNOUNCEMENT",
                "source_kind": "company",
                "available_at": "2026-09-30T08:00:00+08:00",
                "title": "示例公司风险假设",
                "facts": ["这是尚未证实的风险推演"],
                "stage": "IMPLEMENTATION",
                "uncertain_claim": True,
                "direction": "PRESSURE",
                "links": [{"code": "300001.SZ", "at_event_weight": 0.1}],
            }
        ],
    }


def test_new_morning_event_and_unconfirmed_stage():
    source = event_source()
    assert not data.related_events(source, "2026-09-29", "2026-09-29T23:00:00+08:00")
    vector, text, proof = data.event_features(source, "2026-09-29", "2026-09-30T08:30:00+08:00")
    assert proof["text_event_ids"] == ["event"]
    assert "风险推演" in text
    assert proof["events"][0]["stage"] == "UNCONFIRMED_DISCUSSION"
    assert vector[data.EVENT_NAMES.index("ANNOUNCEMENT_5_pressure_weight")] == 0
    assert vector[data.EVENT_NAMES.index("ANNOUNCEMENT_5_proposal_weight")] == 0.1
    source["reports"][0]["available_at"] = "2026-10-01T00:00:00+08:00"
    assert not data.related_events(source, "2026-09-29", "2026-09-30T08:30:00+08:00")


def synthetic_rows():
    rows = []
    for i in range(36):
        groups = {g: [float(np.sin(i + j)) for j in range(len(data.NAMES[g]))] for g in data.GROUPS}
        groups["C_EVENTS"][7] = None
        rows.append(
            {
                "phase": "MORNING_0830",
                "target": f"2024-02-{i + 1:02d}",
                "mature_at": "2024-03-10T08:00:00+08:00",
                "groups": groups,
                "label": "UP" if i % 3 else "DOWN",
                "text": "已有公告公司业绩" if i % 3 else "已有新闻风险",
            }
        )
    return rows


def test_text_is_train_only_and_reload_probabilities_are_independent():
    training = synthetic_rows()
    bundle = experiment.fit_bundle(training, "C_EVENTS", "2024-03-11T08:30:00+08:00")
    vocabulary = bundle["vectorizer"].vocabulary_.copy()
    means = bundle["scaler"].mean_.copy()
    unseen = {**training[0], "text": "未来独有消息"}
    matrix = experiment.transform(bundle, [unseen])
    assert bundle["vectorizer"].vocabulary_ == vocabulary
    assert "未来" not in vocabulary
    assert np.array_equal(bundle["scaler"].mean_, means)
    assert np.allclose(
        experiment.probabilities(bundle, matrix), experiment.probabilities(bundle, matrix, True), atol=1e-12
    )
    assert experiment.probabilities(bundle, matrix)[0, 1] == 0  # 缺少FLAT样本不伪造持平概率。
    assert any(name.startswith("text:") for name in bundle["feature_names"])
    with pytest.raises(AssertionError):
        experiment.fit_bundle(training, "A_NAV", "2024-03-01T08:30:00+08:00")


def test_quarter_split_keeps_future_labels_out():
    rows = synthetic_rows()[:3]
    rows[0].update(target="2024-12-31", mature_at="2025-01-02T08:00:00+08:00")
    rows[1].update(target="2024-12-30", mature_at="2024-12-31T08:00:00+08:00")
    rows[2].update(target="2025-01-02", mature_at="2025-01-03T08:00:00+08:00")
    plans = [p for p in experiment.planned(rows) if p[0].startswith("MORNING_0830-2025Q1")]
    assert len(plans) == 3
    assert all([r["target"] for r in p[2]] == ["2024-12-30"] for p in plans)
    assert all([r["target"] for r in p[3]] == ["2025-01-02"] for p in plans)


def test_failed_fit_counts_and_never_automatically_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "prepare", lambda: {})
    data.save_lines(tmp_path / "dataset.jsonl", [])
    monkeypatch.setattr(experiment, "planned", lambda _: iter([("one", "A_NAV", [], [], "now", "2025Q1")]))
    calls = []

    def failure(*args):
        calls.append(True)
        raise ValueError("EXPECTED_FAILURE")

    monkeypatch.setattr(experiment, "fit_bundle", failure)
    with pytest.raises(ValueError, match="EXPECTED_FAILURE"):
        experiment.train()
    with pytest.raises(RuntimeError, match="NO_AUTOMATIC_RETRY"):
        experiment.train()
    assert len(calls) == 1
    assert [r["status"] for r in data.jsonl(tmp_path / "fit-ledger.jsonl")] == ["RESERVED", "FAILED"]
