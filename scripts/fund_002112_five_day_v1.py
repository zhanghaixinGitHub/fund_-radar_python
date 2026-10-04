"""002112 五交易日研究：复用已核验输入，独立冻结目标、预算和历史评价。

不修改一期脚本或产物，不读取个人持仓，不写业务数据库和模型绑定。
输入截点仍为 D 日 08:00；标签为 NAV(D+5) 相对 NAV(D)，不是五个自然日。
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

from scripts import fund_002112_signal_common_v1 as c
from scripts import fund_002112_signal_features_v1 as features
from scripts import fund_002112_signal_forward_v1 as forward
from scripts import fund_002112_signal_train_v1 as train

ROOT = c.io.RESEARCH / "five-day-optimization/20261001-v1"
PRIOR = c.DEFAULT_ROOT
HORIZON = 5
END = "2026-09-30"
GROUPS = ("B0", "B1", "B2", "B3")


def make_rows(original, sessions, nav, end=END):
    """只改答案区间，不移动输入时点；尾部未到期行单列，不能伪造历史预测。"""
    positions = {d: i for i, d in enumerate(sessions)}
    rows, pending = [], []
    for old in original:
        index = positions[old["base"]]
        if index + HORIZON >= len(sessions):
            raise ValueError("CALENDAR_END_UNKNOWN")
        row = copy.deepcopy(old)
        row.update(
            target=sessions[index + HORIZON],
            one_day_target=old["target"],
            base_session_index=index,
            horizon_sessions=HORIZON,
            source_row_sha256=c.io.digest(old),
        )
        if row["target"] > end:
            pending.append({**row, "label_mature_at": None, "state": "OUTSIDE_FROZEN_HISTORY_NOT_A_PREDICTION"})
            continue
        for day in (row["base"], row["target"]):
            if day not in nav or Decimal(nav[day]["unit_nav"]) <= 0:
                raise ValueError("LABEL_NAV_MISSING_OR_INVALID:" + day)
        row["label_mature_at"] = max((nav[d]["available_at"] for d in (row["base"], row["target"])), key=c.moment)
        rows.append(row)
    return rows, pending


def dividend_audit(root, nav):
    """核对研究期间分红与累计/单位净值差，累计净值不替代再投收益。

    本轮仅在整个历史区间没有分红、两来源核验一致时使用单位净值方向。
    如存在分红或差值变化，停止本轮准备，不能临时忽略除息日期以提高成绩。
    """
    evidence = c.io.read(root / "dividend-local-evidence.json")
    if evidence["read_only"] != [{"transaction_read_only": "on"}]:
        raise ValueError("READONLY_EVIDENCE_REQUIRED")
    if not any(x["status"] == "SUCCEEDED" and x["dividends_verified_at"][:10] >= END for x in evidence["watermark"]):
        raise ValueError("DIVIDEND_WATERMARK_INCOMPLETE")
    start = min(nav)
    events = [x for x in evidence["dividends"] if x["ex_date"] is None or start <= x["ex_date"] <= END]
    if events:
        raise ValueError("DIVIDEND_ADJUSTMENT_REQUIRES_SEPARATE_VERIFIED_LABELS")
    provenance_path = c.io.RESEARCH / "recent-input-completion/20260930-v1/input-provenance.json"
    provenance = c.io.read(provenance_path)
    totals, paths = {}, [provenance_path, root / "dividend-local-evidence.json"]
    for expected_hash, source in provenance["nav"]["official_sources"].items():
        path = Path(source["path"])
        if c.io.sha(path) != expected_hash:
            raise ValueError("OFFICIAL_NAV_HASH_CHANGED")
        paths.append(path)
        for r in c.io.payload(c.io.read(path))["dataList"]:
            if r["date"] in nav:
                value = (Decimal(r["netvalue"]), Decimal(r["totalnetvalue"]))
                if r["date"] in totals and totals[r["date"]] != value:
                    raise ValueError("OFFICIAL_NAV_VERSION_CONFLICT")
                totals[r["date"]] = value
    for r in evidence["nav_0930"]:
        value = (Decimal(r["unit_nav"]), Decimal(r["accumulated_nav"]))
        if r["nav_date"] in totals and totals[r["nav_date"]] != value:
            raise ValueError("LOCAL_OFFICIAL_NAV_CONFLICT")
        totals[r["nav_date"]] = value
    if set(nav) - set(totals):
        raise ValueError("ACCUMULATED_NAV_CROSSCHECK_MISSING")
    for d, r in nav.items():
        if totals[d][0] != Decimal(r["unit_nav"]):
            raise ValueError("LABEL_VERSION_CONFLICT:" + d)
    differences = {str(t - u) for u, t in totals.values()}
    if len({Decimal(x) for x in differences}) != 1:
        raise ValueError("CASH_OR_SPLIT_BASIS_UNRESOLVED")
    result = {
        "passed": True,
        "range": [start, END],
        "verified_nav_dates": len(nav),
        "dividend_events_in_range": 0,
        "historical_dividend_records": len(evidence["dividends"]),
        "constant_accumulated_minus_unit": str(next(iter(totals.values()))[1] - next(iter(totals.values()))[0]),
        "label_basis": "UNIT_NAV_DIRECTION; no cash distributions in this audited interval",
        "not_a_total_return_backtest": True,
        "evidence_limit": "本机已核验分红台账与官网累计/单位净值交叉检查；不声称取得历史逐时版本存档。",
    }
    c.io.save(root / "dividend-audit.json", result)
    return paths


def novelty_audit(root, rows):
    """不打开预测成绩，只统计已核验新披露/可比进展及逐日真实覆盖。

    首次披露不等于超出市场预期；没有分析师预期和市场吸收证据时明确未知。
    """
    directory = PRIOR / "features-r4"
    events = c.lines(directory / "events.jsonl")
    by_id = {e["id"]: e for e in events}
    lineage = {r["target"]: r for r in c.lines(directory / "event-lineage.jsonl")}
    qualified = [e for e in events if e["status"] == "QUALIFIED"]
    actual, links = set(), 0
    yearly = {}
    for year in ("2024", "2025", "2026"):
        subset = [r for r in rows if r["target"].startswith(year)]
        ids = set()
        for row in subset:
            proof = lineage[row["one_day_target"]]
            assert proof["as_of"] == row["as_of"]
            for e in proof["B1_B2"]["events"]:
                if e["status"] != "QUALIFIED":
                    continue
                assert c.moment(e["version_available_at"]) <= c.moment(row["as_of"])
                assert c.moment(e["prediction_report"]["available_at"]) <= c.moment(row["as_of"])
                assert c.moment(e["event_report"]["report_available_at"]) <= c.moment(e["version_available_at"])
                assert e["original_weight"] > 0
                ids.add(e["id"])
                links += 1
        actual.update(ids)
        yearly[year] = {
            "rows": len(subset),
            "direct_event_ids": len(ids),
            "event_days": sum((r["E"][12] or 0) > 0 for r in subset),
            "trigger_days": sum(r["trigger"] for r in subset),
            "fields_nonzero_or_known": {
                name: sum(r["E"][j] is not None and (j in (2, 3, 4, 6) or r["E"][j] != 0) for r in subset)
                for j, name in enumerate(features.FIELDS)
            },
        }
    for e in qualified:
        for previous_id in e["previous_event_ids"]:
            previous = by_id[previous_id]
            assert c.moment(previous["version_available_at"]) <= c.moment(e["version_available_at"])
            assert previous["code"] == e["code"]
    supplement = {e["id"] for e in events if not e.get("cache_reused", False)}
    result = {
        "at": c.io.now(),
        "passed": True,
        "does_not_read_scores": True,
        "source_documents": len(events),
        "qualified_events": len(qualified),
        "qualified_flag_counts": dict(Counter(f for e in qualified for f in e["flags"])),
        "qualified_with_previous_event": sum(bool(e["previous_event_ids"]) for e in qualified),
        "actual_direct_events": len(actual),
        "verified_row_event_links": links,
        "actual_events_from_phase1_supplement": len(actual & supplement),
        "by_target_year": yearly,
        "unproven": ["是否超出市场预期", "市场是否已经消化事件", "五日方向是否受益"],
        "missing": ["同期间可比指引修订", "订单与已公开同口径营收分母", "配对公司风险变化"],
        "decision": "REUSE_FROZEN_FEATURES_TO_TEST_HORIZON_ONLY",
        "new_body_requests": 0,
        "public_requests": 0,
        "llm_requests": 0,
    }
    c.io.save(root / "event-increment-audit.json", result)
    c.io.save_lines(root / "actual-event-provenance.jsonl", [by_id[i] for i in sorted(actual)])
    return result


def prepare(root):
    if not c.io.read(root / "protection-summary.json")["complete"]:
        raise ValueError("PROTECTION_BASELINE_REQUIRED")
    c.verify_files(c.io.read(PRIOR / "training-freeze.json"))
    if not c.io.read(PRIOR / "independent-verification-strict.json")["passed"]:
        raise ValueError("PRIOR_VERIFICATION_REQUIRED")
    original = c.lines(PRIOR / "features-r4/inputs.jsonl")
    nav, sessions = train.labels_nav(), c.bundle()["sessions"]
    dividend_paths = dividend_audit(root, nav)
    rows, pending = make_rows(original, sessions, nav)
    calendars = {y: train.calendar(rows, y) for y in ("2025", "2026")}
    # 按五日终点划分年份，训练答案必须在首个预测 D 的 08:00 前成熟。
    # 保证训练区间终点早于评估区间起点，不能只沿用一日成熟字段。
    for updates in calendars.values():
        for update in updates:
            first = rows[update["evaluate"][0]]
            assert all(rows[i]["target"] < first["base"] for i in update["training"])
    novelty_audit(root, rows)
    c.io.save_lines(root / "inputs.jsonl", rows)
    c.io.save_lines(root / "outside-history.jsonl", pending)
    c.io.save(root / "nav.json", nav)
    c.io.save(root / "sessions.json", sessions)
    c.io.save(root / "training-calendar.json", calendars)
    labels = train.reference.labels(rows, nav)
    c.io.save_lines(
        root / "label-lineage.jsonl",
        [
            {
                "base": r["base"],
                "target": r["target"],
                "as_of": r["as_of"],
                "label_mature_at": r["label_mature_at"],
                "label": label,
                "source_values": {d: nav[d] for d in (r["base"], r["target"])},
            }
            for r, label in zip(rows, labels, strict=True)
        ],
    )
    paths = [
        PRIOR / "training-freeze.json",
        PRIOR / "features-r4/inputs.jsonl",
        PRIOR / "features-r4/events.jsonl",
        PRIOR / "features-r4/event-lineage.jsonl",
        PRIOR / "features-r4/feature-protocol.json",
        PRIOR / "features-r4/source-review.json",
        PRIOR / "independent-verification-strict.json",
        c.bodies.BUNDLE,
        c.OLD / "reserved-labels.json",
        *dividend_paths,
    ]
    c.register_sources(root, paths)
    c.io.save(
        root / "prepared.json",
        {
            "at": c.io.now(),
            "rows": len(rows),
            "outside_history_rows": len(pending),
            "by_target_year": dict(Counter(r["target"][:4] for r in rows)),
            "first_training_rows": len(calendars["2025"][0]["training"]),
            "input_content_unchanged": True,
            "new_features": 0,
            "planned_fits": sum(len(u) for u in calendars.values()) * 3,
        },
    )
    c.stage(
        root,
        "数据与信息增量核验",
        "COMPLETE",
        ["prepared.json", "event-increment-audit.json", "dividend-audit.json"],
        ["六个季末净值历史首次公开时点仍缺证", "事件超预期与价格吸收未证明"],
        "先测试并冻结，再执行固定拟合",
    )


def freeze(root):
    if (root / "training-freeze.json").exists():
        c.verify_files(c.io.read(root / "training-freeze.json"))
        return
    if not c.io.read(root / "test-results.json")["passed"]:
        raise ValueError("TESTS_REQUIRED")
    data = c.io.read(root / "prepared.json")
    c.io.save(
        root / "training-protocol.json",
        {
            "at": c.io.now(),
            "horizon_sessions": 5,
            "prediction_time": "D 08:00 Asia/Shanghai",
            "label": "sign(NAV(D+5)-NAV(D)); D close is unknown at prediction time",
            "year_partition": "five-day target end date",
            "history_end": END,
            "development": "2025; previously observed NAV history; exploratory selection only",
            "diagnostic": "2026; previously exposed NAV values; not an unseen test",
            "recipe": train.RECIPE,
            "cadence_sessions": 20,
            "planned_fits": data["planned_fits"],
            "maximum_fits": 80,
            "replay_fits_max": 2,
            "failure_and_retry_consume_budget": True,
            "budget_identity": str(ROOT),
            "separate_from_phase1_budget": True,
            "candidates": {
                "B0": "NAV15",
                "B1": "NAV15+EVENT20",
                "B2": "75% B0 + 25% B1 when triggered",
                "B3": "75% B0 + 25% decay event model when triggered",
            },
            "trigger_and_decay": c.io.read(PRIOR / "features-r4/feature-protocol.json"),
            "selection": {
                "extra_correct": 5,
                "quarters_not_worse": 3,
                "max_down_recall_drop": 0.05,
                "brier_not_worse": True,
                "nonoverlap_phases_not_worse": 4,
                "paired_block_ci_lower_gt_zero": True,
                "beat_always_up": True,
                "tie_break": ["correct_desc", "brier_asc", "B1", "B2", "B3"],
            },
            "overlap_handling": {
                "train": (
                    "expanding chronological training; both endpoint NAVs mature strictly before update; "
                    "purge overlapping label intervals"
                ),
                "diagnostics": "all 5 fixed base-session modulo phases reported; no phase selected",
                "uncertainty": (
                    "paired moving blocks length20, 2000 resamples, seed0, 95%; "
                    "not independent daily Bernoulli trials"
                ),
            },
            "selective_output": {
                "max_probability_min": 0.55,
                "top_two_margin_min": 0.10,
                "role": "secondary diagnostic, not candidate selection; retain every abstained date",
            },
            "new_public_requests": 0,
            "new_llm_requests": 0,
            "new_bodies": 0,
            "probability_atol": 1e-12,
            "adoption": False,
            "future_observation_auto_start": False,
        },
    )
    files = [
        root / n
        for n in (
            "inputs.jsonl",
            "nav.json",
            "sessions.json",
            "training-calendar.json",
            "training-protocol.json",
            "source-manifest.json",
            "event-increment-audit.json",
            "dividend-audit.json",
            "test-results.json",
            "prepared.json",
            "outside-history.jsonl",
            "label-lineage.jsonl",
            "actual-event-provenance.jsonl",
        )
    ]
    files += [Path(p) for p in c.io.read(root / "source-manifest.json")["files"]]
    files += list(c.io.PY.glob("scripts/*fund_002112_five_day*_v1.py"))
    files += [c.io.PY / "scripts/fund_002112_signal_verify_v1.py"]
    files += [Path(m.__file__) for m in (c, features, forward, train, train.reference, c.io, c.bodies, c.nav_inputs)]
    for path in files.copy():
        if path.suffix == ".py":
            dest = root / "code-snapshot" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                with dest.open("xb") as stream:
                    stream.write(path.read_bytes())
            assert c.io.sha(path) == c.io.sha(dest)
            files.append(dest)
    c.io.save(
        root / "training-freeze.json",
        {
            "at": c.io.now(),
            "python": sys.version,
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "joblib")},
            "files": {str(p.resolve()): c.io.sha(p) for p in set(files)},
        },
    )
    c.stage(
        root, "五日协议冻结", "COMPLETE", ["training-freeze.json", "training-protocol.json"], [], "39次2025开发拟合"
    )


def selective_metrics(preds, truth):
    """固定阈值只做附加诊断；弃权样本仍在全体分母，不能挑阈值换漂亮成绩。"""
    selected = []
    for p in preds:
        values = sorted(p["probabilities"], reverse=True)
        if values[0] >= 0.55 and values[0] - values[1] >= 0.10:
            selected.append(p)
    correct = sum(p["predicted"] == truth[p["target"]] for p in selected)
    down = sum(v == "DOWN" for d, v in truth.items() if d in {p["target"] for p in preds})
    caught = sum(truth[p["target"]] == p["predicted"] == "DOWN" for p in selected)
    return {
        "all_dates": len(preds),
        "judged_dates": len(selected),
        "abstained_dates": len(preds) - len(selected),
        "coverage": len(selected) / len(preds),
        "judged_correct": correct,
        "judged_accuracy": correct / len(selected) if selected else None,
        "correct_over_all_dates": correct / len(preds),
        "down_days": down,
        "down_caught": caught,
        "down_not_caught_including_abstention": down - caught,
        "abstained_targets": [p["target"] for p in preds if p not in selected],
    }


def score(predictions, rows, nav):
    result = train.score(predictions, rows, nav)
    truth = dict(zip([r["target"] for r in rows], train.reference.labels(rows, nav), strict=True))
    by_target = {r["target"]: r for r in rows}
    base = predictions["B0"]
    for group in GROUPS:
        ps = predictions[group]
        differences = [
            int(p["predicted"] == truth[p["target"]]) - int(b["predicted"] == truth[b["target"]])
            for p, b in zip(ps, base, strict=True)
        ]
        indices = [by_target[p["target"]]["base_session_index"] for p in ps]
        assert all(b - a == 1 for a, b in zip(indices, indices[1:], strict=False))
        result[group]["paired_block_ci95"] = forward.bootstrap_difference(differences)
        result[group]["accuracy_gain"] = float(np.mean(differences))
        phases = {}
        for phase in range(5):
            subset = [p for p in ps if by_target[p["target"]]["base_session_index"] % 5 == phase]
            bases = [p for p in base if by_target[p["target"]]["base_session_index"] % 5 == phase]
            phases[str(phase)] = {
                **train.metrics(subset, truth),
                "baseline_correct": train.metrics(bases, truth)["correct"],
                "extra_correct": sum(p["predicted"] == truth[p["target"]] for p in subset)
                - sum(p["predicted"] == truth[p["target"]] for p in bases),
            }
        result[group]["all_nonoverlap_phases"] = phases
        result[group]["selective_diagnostic"] = selective_metrics(ps, truth)
    return result


def select(scores):
    base = scores["B0"]
    decisions = {}
    for group in ("B1", "B2", "B3"):
        s = scores[group]
        gates = {
            "full_243_dates": s["days"] == base["days"] == 243,
            "five_more_correct": s["correct"] >= base["correct"] + 5,
            "three_quarters": sum(s["quarters"][q]["correct"] >= b["correct"] for q, b in base["quarters"].items())
            >= 3,
            "down_recall": s["recall"]["DOWN"] >= base["recall"]["DOWN"] - 0.05,
            "brier": s["brier"] <= base["brier"],
            "four_nonoverlap_phases": sum(v["extra_correct"] >= 0 for v in s["all_nonoverlap_phases"].values()) >= 4,
            "positive_block_interval": s["paired_block_ci95"][0] > 0,
            "beat_always_up": s["correct"] > scores["always_up"]["correct"],
        }
        decisions[group] = {"passed": all(gates.values()), "gates": gates}
    passed = [g for g, v in decisions.items() if v["passed"]]
    selected = min(passed, key=lambda g: (-scores[g]["correct"], scores[g]["brier"], g)) if passed else "B0"
    return {
        "at": c.io.now(),
        "selected": selected,
        "candidates": decisions,
        "future_observation_eligible": bool(passed),
        "selection_data": "2025_ONLY",
        "adopted": False,
        "future_samples": 0,
    }


def execute(root):
    """唯一拟合入口；版本目录不能自行获得新额度，失败预留也计入80上限。"""
    if root.resolve() != ROOT.resolve():
        raise ValueError("FIXED_EXPERIMENT_ROOT_REQUIRED")
    freeze(root)
    rows, nav = c.lines(root / "inputs.jsonl"), c.io.read(root / "nav.json")
    for year in ("2025", "2026"):
        c.verify_files(c.io.read(root / "training-freeze.json"))
        if year == "2026" and not (root / "diagnostic-entry.json").exists():
            c.io.save(
                root / "diagnostic-entry.json",
                {
                    "at": c.io.now(),
                    "selection_sha256": c.io.sha(root / "selection.json"),
                    "not_an_unseen_test": True,
                    "previously_exposed_nav_history": True,
                },
            )
        for update in train.calendar(rows, year):
            for group in train.GROUPS:
                train.fit_one(root, group, update, rows, nav)
        predictions = train.predictions(root, rows, year)
        for group, ps in predictions.items():
            c.io.save_lines(root / "predictions" / f"{year}-{group}.jsonl", ps)
        scores = score(predictions, rows, nav)
        c.io.save(root / f"scores-{year}.json", scores)
        if year == "2025" and not (root / "selection.json").exists():
            c.io.save(root / "selection.json", select(scores))
        c.stage(root, year + "历史比较", "COMPLETE", [f"scores-{year}.json"], [], "按冻结方案继续验收")
        print(c.io.canonical({"year": year, "correct": {g: v["correct"] for g, v in scores.items()}}), flush=True)
    selected = c.io.read(root / "selection.json")["selected"]
    groups = ["B0"] + (["B1"] if selected in ("B1", "B2") else ["B3_MODEL"] if selected == "B3" else [])
    for group in groups:
        update = train.calendar(rows, "2025")[0]
        train.fit_one(root, group, update, rows, nav, replay=True)
        identifier = group + "__" + update["id"]
        old = c.lines(root / "runs" / identifier / "predictions.jsonl")
        replay = c.lines(root / "runs" / (identifier + "__replay") / "predictions.jsonl")
        np.testing.assert_allclose(
            [p["probabilities"] for p in old], [p["probabilities"] for p in replay], rtol=0, atol=1e-12
        )
    c.io.save(root / "model-verification.json", train.verify_models(root, rows, nav))
    c.verify_files(c.io.read(root / "training-freeze.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "train"))
    args = parser.parse_args()
    with c.io.writer_lock(ROOT), c.io.offline_guard():
        {"prepare": prepare, "freeze": freeze, "train": execute}[args.action](ROOT)


if __name__ == "__main__":
    main()
