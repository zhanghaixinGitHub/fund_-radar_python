"""货币聚合组件的边界验证；全部使用临时原件，不连接数据库或网络。"""

import hashlib
import tempfile
import unittest
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.services import fund_rating_money_aggregate_components as money


class MoneyAggregateComponentsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.inputs = {
            "fund_code": "000009",
            "legal_name": money.LEGAL_NAME,
            "share_class": "A",
            "as_of_date": money.AS_OF,
            "primary_ref": "a" * 64,
            "sources": {},
        }
        dimensions = {
            key: {"lower": "75", "upper": "75", "candidate": "75"}
            for key in ("risk", "execution", "management", "liquidity", "fee")
        }
        dimensions.update(
            {key: {"lower": "0", "upper": "100", "candidate": None} for key in ("return", "stability", "holdings")}
        )
        self.baseline = {
            "fund_code": "000009",
            "as_of_date": money.AS_OF,
            "identity": {"legal_name": money.LEGAL_NAME, "manager_name": "同一管理人"},
            "share_class": "A",
            "verified_at": "2026-10-23T10:09:28+00:00",
            "original_report_archive": {"sha256": "a" * 64},
            "money_inputs": {"method_version": "M3", "holdings": "原区间"},
            "reviewed_document": {"dimensions": dimensions, "outer_weights_percent": {"return": 15}},
        }
        self.actual = {
            "dimensions": {
                "return": {"lower": "62.5", "upper": "62.5", "candidate": "62.5"},
                "stability": {"lower": "72.425", "upper": "73.3333", "candidate": None},
            },
            "complete_score": None,
            "statistics_assumption_used": False,
        }

    def prepared(self):
        request = {
            "fund_code": "000009",
            "expected_current_ref": "current",
            "input_archive": money.archive_json(self.root, self.inputs),
        }
        with patch.object(money, "calculate", return_value=deepcopy(self.actual)):
            bundle, actual = money.prepare(
                request,
                current_bundle=self.baseline,
                current_ref="current",
                root=self.root,
                frozen=datetime(2026, 10, 30, tzinfo=UTC),
            )
        return bundle, actual

    def test_prepare_keeps_original_five_points_holdings_identity_and_validity(self):
        bundle, _ = self.prepared()
        self.assertEqual(bundle["verified_at"], self.baseline["verified_at"])
        self.assertEqual(bundle["identity"], self.baseline["identity"])
        for key in money.UNCHANGED:
            self.assertEqual(
                bundle["reviewed_document"]["dimensions"][key], self.baseline["reviewed_document"]["dimensions"][key]
            )
        self.assertIsNone(bundle["reviewed_document"]["dimensions"]["stability"]["candidate"])

    def test_snapshot_binding_accepts_added_sixth_point(self):
        bundle, _ = self.prepared()
        with patch.object(money, "calculate", return_value=self.actual):
            money.check_binding(bundle, root=self.root)

    def test_validity_refresh_is_rejected_in_snapshot(self):
        bundle, _ = self.prepared()
        bundle["verified_at"] = "2026-10-30T00:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "UNCHANGED_IDENTITY_OR_VALIDITY"):
            money.check_binding(bundle, root=self.root)

    def test_unchanged_dimension_edit_is_rejected(self):
        bundle, _ = self.prepared()
        bundle["reviewed_document"]["dimensions"]["holdings"]["lower"] = "75"
        with self.assertRaisesRegex(ValueError, "UNRELATED_DIMENSION_CHANGED"):
            money.check_binding(bundle, root=self.root)

    def test_old_five_point_precondition_only_applies_to_new_prepare(self):
        bundle, _ = self.prepared()
        request = {
            "fund_code": "000009",
            "expected_current_ref": "next",
            "input_archive": bundle[money.KEY]["input_archive"],
        }
        with self.assertRaisesRegex(ValueError, "PREPARE_EXISTING_FIVE_POINTS"):
            money.prepare(request, current_bundle=bundle, current_ref="next", root=self.root, frozen=datetime.now(UTC))

    def test_snapshot_interval_cannot_be_promoted_to_midpoint(self):
        bundle, _ = self.prepared()
        bundle["reviewed_document"]["dimensions"]["stability"]["candidate"] = "72.87915"
        with patch.object(money, "calculate", return_value=self.actual):
            with self.assertRaisesRegex(ValueError, "SNAPSHOT_COMPONENT_POINT_CHANGED"):
                money.replay_snapshot(bundle, root=self.root)

    def test_typed_input_must_match_archived_input(self):
        bundle, _ = self.prepared()
        changed = deepcopy(self.inputs)
        changed["share_class"] = "B"
        with self.assertRaisesRegex(ValueError, "SNAPSHOT_TYPED_INPUT_CHANGED"):
            money.check_binding(bundle, changed, root=self.root)

    def test_source_bytes_checked_even_after_calculation_cache_hit(self):
        content = b'{"actual":"original"}'
        digest = hashlib.sha256(content).hexdigest()
        relative = f"sources/{digest}.json"
        path = self.root / relative
        path.parent.mkdir()
        path.write_bytes(content)
        inputs = {"sources": {digest: {"file": relative, "sha256": digest, "kind": "json"}}}
        with patch.object(money, "_calculate_cached", return_value={"point": [62.5]}) as cached:
            first = money.calculate(inputs, root=self.root)
            first["point"].append(99)
            self.assertEqual(money.calculate(inputs, root=self.root), {"point": [62.5]})
            path.write_bytes(content + b"tampered")
            with self.assertRaisesRegex(ValueError, "SOURCE_HASH"):
                money.calculate(inputs, root=self.root)
            self.assertEqual(cached.call_count, 2)

    def test_future_acquisition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_FUTURE_RESEARCH_DATE"):
            money._source_dates({"acquired_at": "2026-10-31T00:00:00+08:00"})

    def test_future_fundamental_publication_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_FUTURE_FUNDAMENTAL_PUBLICATION"):
            money._source_dates({"role": "STATUTORY_AGGREGATE_RETURN", "published_on": "2026-10-08"})

    def test_actual_late_historical_income_availability_is_retained(self):
        source = {
            "role": "PEER_ORIGINAL_DAILY_INCOME",
            "published_on": "2026-10-08",
            "acquired_at": "2026-10-29T10:01:55.669Z",
        }
        before = deepcopy(source)
        money._source_dates(source)
        self.assertEqual(source, before)
        money._source_dates({"acquired_at": None, "published_on": None})

    def test_strict_common_domain_existence_does_not_impute_daily_observations(self):
        start = date(2023, 9, 23)
        income = {(start + timedelta(days=i)).isoformat(): money.D(0) for i in range(8)}
        result = money._confirm_nonempty_domain(income, {"2023-09-30": money.D(0)}, [])
        self.assertTrue(result["full_common_baseline_domain_nonempty"])
        self.assertGreater(money.D(result["minimum_original_interval_slack"]), 0)
        self.assertFalse(result["daily_observation_imputation_used"])

    def test_contradictory_precision_constraints_cannot_make_unique_empty_set(self):
        start = date(2023, 9, 23)
        income = {(start + timedelta(days=i)).isoformat(): money.D(0) for i in range(8)}
        with self.assertRaisesRegex(ValueError, "FULL_COMMON_BASELINE_DOMAIN_INFEASIBLE"):
            money._confirm_nonempty_domain(income, {"2023-09-30": money.D(100)}, [])


if __name__ == "__main__":
    unittest.main()
