"""未来对照关键证据边界；全部使用合成输入和临时目录，零真实模型训练。"""

from copy import deepcopy
from datetime import timedelta

import pytest
from app.services import direction_1d_activity_study as activity
from app.services import direction_1d_independent as p
from app.services import direction_1d_independent_audit as audit
from app.services import direction_1d_independent_data as data
from app.services.direction_1d_protocol import FEATURE_VERSION, FEATURES, PROTOCOL, TARGET, calendar


@pytest.fixture
def case():
    sessions = [str(d) for d in calendar()[0]]
    target = "2026-09-14"
    base, _, _ = p.bounds(sessions, target)
    i = sessions.index(base)
    at = p.instant("2026-09-13T14:00:00+08:00")
    leaf = {"leaf": True, "value": 0.0, "threshold": 0.0, "depth": 0, "count": 10}
    candidate = {
        "candidate": activity.CANDIDATE,
        "features": activity.FEATURES,
        "recipe": activity.tree.TREE_RECIPE,
        "feature_recipe": activity.FEATURE_RECIPE,
        "target_definition": TARGET,
        "model_released": False,
        "mean": [0.0] * 12,
        "scale": [1.0] * 12,
        "baseline": 0.0,
        "trees": [[deepcopy(leaf)] for _ in range(100)],
    }
    fixed = {
        "protocol": PROTOCOL,
        "horizon": 1,
        "target_definition": TARGET,
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "coef": [1.0] * 7,
        "mean": [0.0] * 7,
        "scale": [1.0] * 7,
        "intercept": 0.0,
        "majority": 0,
        "group_id": "CN_EQUITY",
        "train_as_of": "2026-09-11T23:00:00+08:00",
    }
    member = {
        "fund_code": "001632",
        "family": "family1",
        "group": "CN_EQUITY",
        "source_id": "source",
        "index_code": "930653.CSI",
        "mapping_hash": "mapping",
        "fixed_model_id": "fixed",
        "eligible": True,
    }
    expires = "2027-09-01T00:00:00+08:00"
    snapshot = {
        "fund_code": "001632",
        "family": "family1",
        "group": "CN_EQUITY",
        "base": base,
        "target": target,
        "observed_at": at.isoformat(),
        "source": {"source_id": "source", "enabled": True, "expires_at": expires},
        "events_status": "UNKNOWN",
        "events": [],
        "nav": [
            {
                "date": d,
                "unit_nav": str(1 + n / 1000),
                "content_hash": f"nav{n}",
                "updated_at": "2026-09-11T22:00:00+08:00",
                "ann_date": "2026-09-12",
            }
            for n, d in enumerate(sessions[i - 60 : i + 1])
        ],
        "market": {
            "mapping_hash": "mapping",
            "amount_unit": "THOUSAND_CNY",
            "received_at": at.isoformat(),
            "persisted_at": at.isoformat(),
            "expires_at": expires,
            "series": {
                "000300.SH": [
                    {"date": d, "close": str(100 + n), "amount": "200" if n == 20 else "100"}
                    for n, d in enumerate(sessions[i - 20 : i + 1])
                ],
                "930653.CSI": [
                    {"date": d, "close": str(200 + n), "amount": None} for n, d in enumerate(sessions[i - 5 : i + 1])
                ],
            },
        },
    }
    contract = {
        "members": {"001632": member},
        "sessions": sessions,
        "schedule": {"targets": [target]},
        "candidate_expires_at": expires,
        "fixed_expires_at": {"fixed": expires},
    }
    return {
        "snapshot": snapshot,
        "contract": contract,
        "models": {"candidate": candidate, "fixed": {"fixed": fixed}},
        "at": at,
        "member": member,
    }


def answer(case):
    return p.answers(case["snapshot"], case["contract"]["sessions"], case["member"], case["models"], case["at"])


def archived(tmp_path, case, clock=None):
    return p.archive(tmp_path, case["contract"], case["snapshot"], case["models"], clock=clock or (lambda: case["at"]))


def folder(tmp_path):
    return tmp_path / "questions/2026-09-14/001632"


def test_calendar_requires_120_and_10_and_preserves_missing_dates():
    sessions = [str(d) for d in calendar()[0]]
    actual = p.schedule(sessions, "2026-09-14")
    assert len(actual["targets"]) == 73
    assert actual["missing_target_days"] == 47
    assert actual["missing_grace_days"] == 10
    assert not actual["complete"]
    complete = p.schedule(sessions, "2026-01-05")
    assert complete["complete"] and len(complete["grace_targets"]) == 10


def test_frozen_policy_no_fit_no_silent_change_of_threshold():
    assert p.POLICY["new_training_budget"] == 0
    assert p.POLICY["bootstrap_repetitions"] == 10000
    assert p.POLICY["bootstrap_seed"] == 20260913
    assert p.POLICY["checkpoints"] == [20, 60, 120]


def test_same_snapshot_feature_values_and_half_boundary(case):
    x, full = p.input_vector(case["snapshot"], case["contract"]["sessions"], case["member"], case["at"])
    assert full[:7] == x
    assert full[-1] == 1.0  # T不进入前20日均值。
    assert full[7] == pytest.approx(120 / 119 - 1)
    assert full[9] == pytest.approx(205 / 204 - 120 / 119)
    result = answer(case)
    assert result["scores"]["ACTIVITY12"] == 0.5
    assert result["directions"]["ACTIVITY12"] == 0
    assert set(result["directions"]) == {"FIXED7", "ACTIVITY12", *p.POLICY["baselines"]}


@pytest.mark.parametrize(
    "key,value",
    [
        ("amount_unit", "CNY"),
        ("mapping_hash", "changed"),
        ("expires_at", "2026-09-12T00:00:00+08:00"),
        ("persisted_at", "2026-09-14T08:31:00+08:00"),
    ],
)
def test_market_unit_mapping_expiry_and_late_receipt_rejected(case, key, value):
    case["snapshot"]["market"][key] = value
    with pytest.raises(ValueError):
        answer(case)


@pytest.mark.parametrize("series", ["nav", "000300.SH", "930653.CSI"])
def test_missing_intermediate_date_never_filled_or_dropped(case, series):
    rows = case["snapshot"]["nav"] if series == "nav" else case["snapshot"]["market"]["series"][series]
    del rows[3]
    with pytest.raises(ValueError, match="INCOMPLETE"):
        answer(case)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
def test_invalid_amount_cannot_become_a_score(case, value):
    case["snapshot"]["market"]["series"]["000300.SH"][-1]["amount"] = value
    with pytest.raises(ValueError, match="AMOUNT_INVALID"):
        answer(case)


@pytest.mark.parametrize(
    "field,value", [("family", "other"), ("fund_code", "other"), ("group", "CN_BOND"), ("base", "2026-09-10")]
)
def test_input_owner_scope_group_and_date_are_bound(case, field, value):
    case["snapshot"][field] = value
    with pytest.raises(ValueError):
        answer(case)


def test_missing_candidate_keeps_fixed_and_baselines(case, tmp_path):
    case["snapshot"]["market"] = None
    case["snapshot"]["candidate_missing_reason"] = "INDEX_HISTORY_INCOMPLETE"
    result = archived(tmp_path, case)
    assert result["status"] == "VERIFIED"
    assert result["prediction"]["scores"]["ACTIVITY12"] is None
    assert result["prediction"]["scores"]["FIXED7"] is not None


def test_disabled_source_or_future_nav_revision_rejected(case):
    case["snapshot"]["source"]["enabled"] = False
    with pytest.raises(ValueError, match="SOURCE_EXPIRED_OR_DISABLED"):
        answer(case)
    case["snapshot"]["source"]["enabled"] = True
    case["snapshot"]["nav"][-1]["updated_at"] = "2026-09-14T12:00:00+08:00"
    with pytest.raises(ValueError, match="VERSION_OR_TIME"):
        answer(case)


def test_naive_time_rejected(case):
    case["snapshot"]["observed_at"] = "2026-09-13T14:00:00"
    with pytest.raises(ValueError, match="TIMEZONE_REQUIRED"):
        answer(case)


@pytest.mark.parametrize("change", ["missing_tree", "negative_scale", "future_train", "wrong_group"])
def test_model_integrity_and_applicability(case, change):
    if change == "missing_tree":
        case["models"]["candidate"]["trees"].pop()
    elif change == "negative_scale":
        case["models"]["candidate"]["scale"][11] = -1
    elif change == "future_train":
        case["models"]["fixed"]["fixed"]["train_as_of"] = "2026-09-15T00:00:00+08:00"
    else:
        case["models"]["fixed"]["fixed"]["group_id"] = "CN_BOND"
    with pytest.raises(ValueError):
        answer(case)


def test_first_snapshot_cannot_be_replaced_by_retry(case, tmp_path):
    first = archived(tmp_path, case)
    original_hashes = {f.name: f.read_bytes() for f in folder(tmp_path).iterdir()}
    case["snapshot"]["nav"][-1]["unit_nav"] = "2.0"
    assert archived(tmp_path, case) == first
    assert {f.name: f.read_bytes() for f in folder(tmp_path).iterdir()} == original_hashes


def test_atomic_write_interruption_never_publishes_partial_file(tmp_path, monkeypatch):
    def stopped(*args):
        raise OSError("simulated interruption before publication")

    monkeypatch.setattr(p.os, "link", stopped)
    with pytest.raises(OSError):
        p.seal(tmp_path / "a.json", {"value": 1})
    assert not list(tmp_path.iterdir())


def test_atomic_existing_file_cannot_be_overwritten(tmp_path):
    p.seal(tmp_path / "a.json", {"value": 1})
    with pytest.raises(FileExistsError):
        p.seal(tmp_path / "a.json", {"value": 2})
    assert p.unseal(tmp_path / "a.json") == {"value": 1}


def test_restore_after_prediction_written_before_deadline(case, tmp_path):
    first = archived(tmp_path, case)
    for name in ("receipt.json", "confirmation.json", "readback.json"):
        (folder(tmp_path) / name).unlink()
    assert archived(tmp_path, case) == first


def test_at_deadline_is_missed_and_cannot_backfill(case, tmp_path):
    result = archived(tmp_path, case, lambda: p.instant("2026-09-14T08:30:00+08:00"))
    assert result["status"] == "MISSED_DEADLINE"
    assert not folder(tmp_path).exists()


def test_restore_crosses_deadline_is_not_a_valid_answer(case, tmp_path):
    times = iter(
        [
            case["at"],
            case["at"],
            case["at"],
            p.instant("2026-09-14T08:30:00+08:00"),
            p.instant("2026-09-14T08:30:01+08:00"),
            p.instant("2026-09-14T08:30:02+08:00"),
        ]
    )
    result = archived(tmp_path, case, lambda: next(times))
    assert result["status"] == "LATE_ARCHIVE"


def test_changed_model_and_resealed_answer_detected(case, tmp_path):
    archived(tmp_path, case)
    changed = deepcopy(case["models"])
    changed["fixed"]["fixed"]["intercept"] += 1
    with pytest.raises(ValueError, match="CONTRACT_CHANGED"):
        p.verify_question(folder(tmp_path), case["contract"], changed)
    path = folder(tmp_path) / "prediction.json"
    raw = p.read(path)
    raw["payload"]["scores"]["ACTIVITY12"] = 0.9
    path.write_text(p.canonical(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        p.verify_question(folder(tmp_path), case["contract"], case["models"])


def outcome(case, value="1.06000000000000001"):
    return {
        "fund_code": "001632",
        "date": "2026-09-14",
        "unit_nav": value,
        "content_hash": p.digest(value),
        "source_id": "source",
        "observed_at": "2026-09-14T22:00:00+08:00",
        "updated_at": "2026-09-14T21:00:00+08:00",
        "ann_date": "2026-09-14",
        "expires_at": "2027-09-01T00:00:00+08:00",
    }


def test_future_outcome_stays_pending_and_raw_precision_is_used(case, tmp_path):
    archived(tmp_path, case)
    assert p.record_outcome(
        folder(tmp_path), case["contract"], case["models"], outcome(case), clock=lambda: case["at"]
    ) == {"status": "PENDING_TARGET"}
    result = p.record_outcome(
        folder(tmp_path),
        case["contract"],
        case["models"],
        outcome(case),
        clock=lambda: p.instant("2026-09-14T22:01:00+08:00"),
    )
    assert result["first_answer"]["y"] == 1
    assert result["first_answer"]["nav_return"] == "0E-12"


def test_flat_is_non_up_and_revisions_do_not_change_first_score(case, tmp_path):
    archived(tmp_path, case)

    def clock():
        return p.instant("2026-09-14T22:01:00+08:00")

    first = p.record_outcome(folder(tmp_path), case["contract"], case["models"], outcome(case, "1.06"), clock=clock)
    revised = p.record_outcome(folder(tmp_path), case["contract"], case["models"], outcome(case, "1.08"), clock=clock)
    assert first["first_answer"]["actual_direction"] == "FLAT"
    assert revised["status"] == "REVISION_RECORDED"
    assert revised["first_answer"]["y"] == 0 and revised["current_answer"]["y"] == 1


def test_unannounced_or_future_observed_outcome_not_scored(case, tmp_path):
    archived(tmp_path, case)
    observation = outcome(case)
    observation["ann_date"] = "2026-09-15"

    def clock():
        return p.instant("2026-09-14T22:01:00+08:00")

    assert (
        p.record_outcome(folder(tmp_path), case["contract"], case["models"], observation, clock=clock)["status"]
        == "PENDING_ANNOUNCEMENT"
    )
    observation["observed_at"] = "2026-09-15T22:00:00+08:00"
    with pytest.raises(ValueError, match="TIME_INVALID"):
        p.record_outcome(folder(tmp_path), case["contract"], case["models"], observation, clock=clock)


def test_start_gate_preserves_missing_calendar(monkeypatch, tmp_path):
    draft = {"blockers": ["OFFICIAL_CALENDAR_120_PLUS_10_INCOMPLETE"], "schedule": {"complete": False}}
    monkeypatch.setattr(audit, "verify", lambda root: {"draft": draft})
    with pytest.raises(ValueError, match="START_BLOCKED:OFFICIAL_CALENDAR"):
        audit.require_startable(tmp_path)


def test_provider_response_duplicate_future_missing_and_preclose():
    rows = [["000300.SH", "20260910", 100, 99, 1], ["000300.SH", "20260911", 101, 100, 2]]

    def raw(r):
        return p.canonical({"code": 0, "data": {"fields": data.FIELDS, "items": r}}).encode()

    wanted = ["2026-09-10", "2026-09-11"]
    assert len(data.parse_index(raw(rows), "000300.SH", wanted)) == 2
    for invalid in (rows + [rows[0]], rows[:1], [["000300.SH", "20260914", 100, 99, 1]]):
        with pytest.raises(ValueError):
            data.parse_index(raw(invalid), "000300.SH", wanted)
    rows[1][3] = 98
    with pytest.raises(ValueError, match="PRE_CLOSE_DISCONTINUITY"):
        data.parse_index(raw(rows), "000300.SH", wanted)


def test_fund_coverage_keeps_excluded_owner_funds_and_training_groups():
    context = {"profiles": [], "codes": ["001632", "999999"], "source": {"source_id": "source"}}
    result = audit.coverage(context, {"fund_codes": []}, {}, {})
    assert [r["fund_code"] for r in result] == context["codes"]
    assert all(not r["eligible"] and "NOT_IN_CANDIDATE_TRAINING_SCOPE" in r["reasons"] for r in result)


def test_models_expire_without_extending_retention(case, tmp_path):
    case["contract"]["candidate_expires_at"] = (case["at"] - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="MODEL_SOURCE_EXPIRED"):
        archived(tmp_path, case)


def test_completed_answer_can_be_read_after_deadline_without_new_writes(case, tmp_path):
    first = archived(tmp_path, case)
    before = {f.name: f.read_bytes() for f in folder(tmp_path).iterdir()}
    assert archived(tmp_path, case, lambda: p.instant("2026-09-14T09:00:00+08:00")) == first
    assert {f.name: f.read_bytes() for f in folder(tmp_path).iterdir()} == before


def test_existing_nav_version_expiry_cannot_be_reset_by_new_read(case):
    row = {"nav_date": "2026-09-11", "unit_nav": "1.06", "ann_date": "2026-09-12", "content_hash": "upstream"}
    h = p.digest(
        {
            "nav_date": row["nav_date"],
            "unit_nav": row["unit_nav"],
            "ann_date": row["ann_date"],
            "upstream_hash": "upstream",
            "source_id": "source",
        }
    )
    old = {
        "content_hash": h,
        "received_at": case["at"] - timedelta(days=5),
        "expires_at": case["at"] + timedelta(days=3),
    }
    expires, evidence = data.nav_retention([row], [old], "source", 365, case["at"])
    assert expires == old["expires_at"]
    assert evidence[0]["basis"] == "EXISTING_VERSION_NO_RENEWAL"
    old["expires_at"] = case["at"] - timedelta(seconds=1)
    with pytest.raises(ValueError, match="NAV_PREVIOUS_VERSION_EXPIRED"):
        data.nav_retention([row], [old], "source", 365, case["at"])


@pytest.mark.parametrize("market_failed", [False, True])
def test_runtime_forecast_to_maturity_uses_same_immutable_answer(case, tmp_path, monkeypatch, market_failed):
    """模拟时钟穿过提前留档、重复触发、目标成熟三个节点；禁用所有外部读取。"""
    from app.services import direction_1d_independent_runtime as runtime

    contract = case["contract"]
    contract["coverage"] = [case["member"]]
    i = contract["sessions"].index("2026-09-14")
    contract["schedule"]["grace_targets"] = contract["sessions"][i + 1 : i + 11]
    monkeypatch.setattr(runtime, "load", lambda root: (contract, case["models"]))
    monkeypatch.setattr(runtime, "fresh_scope", lambda value: {"codes": ["001632"]})
    stamp = [p.instant("2026-09-14T08:20:00+08:00")]
    monkeypatch.setattr(p, "now", lambda: stamp[0])
    # archive/record_outcome的默认参数在导入时已绑定原时钟；测试显式注入同一个模拟时钟。
    original_archive, original_outcome = p.archive, p.record_outcome
    monkeypatch.setattr(p, "archive", lambda *args: original_archive(*args, clock=lambda: stamp[0]))
    monkeypatch.setattr(p, "record_outcome", lambda *args: original_outcome(*args, clock=lambda: stamp[0]))

    def get_market(*args):
        if market_failed:
            raise ValueError("INDEX_HISTORY_INCOMPLETE")
        return deepcopy(case["snapshot"]["market"])

    def get_nav(*args):
        return {**deepcopy(case["snapshot"]), "nav_source_versions": []}

    monkeypatch.setattr(data, "collect_market", get_market)
    monkeypatch.setattr(data, "collect_nav", get_nav)
    first = runtime.tick(tmp_path)
    assert first["forecast"]["results"]["001632"] == "VERIFIED"
    original = (folder(tmp_path) / "prediction.json").read_bytes()
    repeated = runtime.tick(tmp_path)
    assert repeated["forecast"]["status"] == "SLOT_ALREADY_ATTEMPTED"
    stamp[0] = p.instant("2026-09-14T22:30:00+08:00")
    monkeypatch.setattr(runtime, "read_outcome", lambda *args: outcome(case, "1.07"))
    completed = runtime.tick(tmp_path)
    assert completed["status"] == "STAGE_FINISHED"
    assert (folder(tmp_path) / "prediction.json").read_bytes() == original
    result = p.unseal(tmp_path / "final-report.json")
    assert result["common_count"] == (0 if market_failed else 1)
    assert result["conservative_all_mature"]["FIXED7"] == 1
    assert result["coverage"]["candidate"] == (0 if market_failed else 1)


def test_base_revision_only_adds_diagnostic_and_keeps_first_target(case, tmp_path):
    archived(tmp_path, case)
    obs = outcome(case, "1.07")

    def clock():
        return p.instant("2026-09-14T22:01:00+08:00")

    first = p.record_outcome(folder(tmp_path), case["contract"], case["models"], obs, clock=clock)
    obs["base_revision"] = {**obs, "date": case["snapshot"]["base"], "unit_nav": "1.08", "content_hash": "base-revised"}
    revision = p.record_outcome(folder(tmp_path), case["contract"], case["models"], obs, clock=clock)
    assert revision["status"] == "REVISION_RECORDED"
    assert revision["first_answer"] == first["first_answer"]
    assert revision["first_answer"]["y"] == 1
    versions = [p.unseal(f) for f in (folder(tmp_path) / "outcomes").glob("*.json")]
    assert any(
        v["base_revision_answer_diagnostic_only"] and v["base_revision_answer_diagnostic_only"]["y"] == 0
        for v in versions
    )
