"""剩余资料审查的独立读回复核：不导入特征生成器、不计算方向标签、不拟合。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".local-runs/fund-exposure-002112/remaining-admission-review/20261001-v1"


def read(path):
    """只读JSON；来源路径由本阶段冻结清单给定。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """实际文件字节摘要，不把包内业务摘要当成文件摘要。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    """重建全部元数据窗口及15字段，逐条比对正文引句，并验证旧产物未变。"""
    freeze = read(STAGE / "source-freeze.json")
    plan = read(STAGE / "protocol.json")
    assert sha(STAGE / "protocol.json") == freeze["protocol_sha256"]
    assert all(sha(p) == h for p, h in freeze["files"].items())
    ledger = read(STAGE / "actual-read-ledger.json")
    assert all(freeze["files"].get(r["path"]) == r["sha256"] for r in ledger)
    result = read(STAGE / "admission-results.json")
    assert sha(ROOT / "scripts/fund_002112_remaining_admission_v1.py") == result["script_sha256"]
    assert sha(STAGE / "code-snapshot/fund_002112_remaining_admission_v1.py") == result["script_sha256"]
    assert (STAGE / "fit-ledger.jsonl").stat().st_size == 0
    facts = None
    closed = []
    for p in freeze["files"]:
        if Path(p).suffix != ".bin":
            continue
        value = read(p)
        payload = value.get("payload", value)
        if "funds" in payload:
            facts = payload
        if "years" in value:
            closed.extend((a, b) for y in value["years"] for a, b in y["closed_ranges"])
    assert facts is not None
    sessions = []
    day = date(2015, 1, 1)
    while day <= date(2025, 12, 31):
        if day.weekday() < 5 and not any(a <= day.isoformat() <= b for a, b in closed):
            sessions.append(day.isoformat())
        day += timedelta(days=1)
    positions = {d: i for i, d in enumerate(sessions)}
    actual = defaultdict(dict)
    for line in (STAGE / "peer-metadata-feature-audit.jsonl").read_text("utf-8").splitlines():
        row = json.loads(line)
        assert row["origin"] not in actual[row["fund_code"]]
        actual[row["fund_code"]][row["origin"]] = row
    summaries = {r["fund_code"]: r for r in read(STAGE / "peer-admission.json")["funds"]}
    checked = 0
    max_error = 0.0
    report_count = 0
    for code in plan["fund_codes"]:
        package = facts["funds"][code]
        rows = package["nav"]["rows"]
        nav = {r["date"]: float(r["nav"]) for r in rows}
        avail = {
            r["date"]: (date.fromisoformat(r["ann_date"]) + timedelta(days=1)).isoformat() + "T08:00:00+08:00"
            for r in rows
        }
        assert len(nav) == len(rows) and all(np.isfinite(x) and x > 0 for x in nav.values())
        assert max(nav) <= "2024-12-31"
        assert package["nav"]["fund_code"] == code
        assert all(r["fund_code"] == code for r in package["reports"])
        report_count += len(package["reports"])
        skip = Counter()
        origins = []
        mature = 0
        for origin in sorted(nav):
            if origin not in positions:
                skip["NOT_TRADING_SESSION"] += 1
                continue
            i = positions[origin]
            if i + 1 == len(sessions) or sessions[i + 1] not in nav:
                skip["NEXT_SESSION_NAV_ABSENT"] += 1
                continue
            window = None
            for lag in range(21):
                end = i - 1 - lag
                if end < 60:
                    continue
                ds = sessions[end - 60 : end + 1]
                if any(d not in nav or avail[d] > origin + "T08:00:00+08:00" for d in ds):
                    continue
                values = np.array([nav[d] for d in ds])
                if np.ptp(values[1:]) == 0:
                    continue
                window = ds
                break
            if window is None:
                skip["NO_VALID61_CONTIGUOUS_KNOWN_NAV_WINDOW_WITHIN20SESSION_LAG"] += 1
                continue
            saved = actual[code][origin]
            target = sessions[i + 1]
            label_time = max(avail[origin], avail[target])
            assert saved["target"] == target and saved["nav_window_dates"] == window
            assert saved["latest_feature_nav_date"] == window[-1] and saved["lag_sessions"] == lag
            assert saved["as_of"] == origin + "T08:00:00+08:00"
            assert saved["max_input_metadata_available_at"] == max(avail[d] for d in window)
            assert saved["label_mature_metadata_at"] == label_time
            assert saved["strict_training_admitted"] is False
            assert window[-1] < origin < target
            mature += label_time < "2024-12-31T08:00:00+08:00"
            returns = values[1:] / values[:-1] - 1
            trailing = 0
            for value in reversed(returns):
                if value >= 0:
                    break
                trailing += 1
            recent = values[1:]
            numbers = [values[-1] / values[-1 - n] - 1 for n in (5, 20, 60)]
            numbers += [
                np.std(returns[-20:]),
                np.min(recent / np.maximum.accumulate(recent) - 1),
                (values[-1] - recent.min()) / np.ptp(recent),
                float(trailing),
                float(lag),
            ]
            numbers += [values[-1] / values[-1 - n] - 1 for n in (1, 2, 10, 40)]
            numbers += [np.std(returns[-5:]), np.std(returns), np.sqrt(np.mean(np.minimum(returns[-20:], 0) ** 2))]
            error = float(np.max(np.abs(np.asarray(numbers) - saved["feature_values_15"])))
            assert error <= 1e-13
            max_error = max(max_error, error)
            origins.append(origin)
            checked += 1
        summary = summaries[code]
        assert set(origins) == set(actual[code])
        assert len(origins) == summary["N_NE"]["metadata_constructible_rows"]
        assert mature == summary["N_NE"]["mature_before_first_2025_origin"]
        assert dict(skip) == summary["N_NE"]["unconstructible_reasons"]
        assert summary["N_NE"]["strictly_admitted_rows"] == 0
        assert summary["nav"]["rows"] == len(rows)
        assert summary["report_version"]["raw_first_seen_verified"] == 0
    assert checked == result["metadata_constructible_rows"] == 10658
    # 引句回到同一冻结PDF及同一页，核验原文内容，不推定早年公开时间。
    quotes_checked = 0
    extracts = read(STAGE / "static-original-evidence.json")
    for entry in extracts:
        pdf = PdfReader(entry["source"]["raw_path"])
        pages = {}
        assert entry["pages_read"] <= 16 and entry["known_version_bound"].startswith("2026")
        for quote in entry["quotes"]:
            number = quote["page"]
            assert 1 <= number <= entry["pages_read"]
            if number not in pages:
                pages[number] = pdf.pages[number - 1].extract_text() or ""
            assert quote["quote"] in pages[number]
            quotes_checked += 1
    notices = read(STAGE / "manager-notice-admission.json")
    for entry in notices:
        pdf = PdfReader(entry["source"]["path"])
        text = "\n".join(p.extract_text() or "" for p in pdf.pages[:6])
        assert entry["fund_code_literal_found"] == ("002112" in text or "001412" in text)
        assert entry["fund_name_literal_found"] == ("鑫星" in text)
        assert entry["available_at"] > "2025-12-31T23:59:59+08:00" and not entry["eligible_for2025"]
    attrs = read(STAGE / "attribute-admission.json")
    assert [sum(any(q["field"] == field for q in e["quotes"]) for e in extracts) for field in
            ("contract_date", "C_share_code", "C_date_16", "C_subscription_date_18")] == [15, 15, 12, 3]
    input_path = next(
        p for p in freeze["files"] if p.endswith("update-frequency-development\\20261001-v1\\inputs.jsonl")
    )
    input_rows = [json.loads(line) for line in Path(input_path).read_text("utf-8").splitlines()]
    ages = [
        (date.fromisoformat(r["base"]) - date(2015, 6, 19)).days
        for r in input_rows if r["target"].startswith("2025")
    ]
    assert [min(ages), max(ages), len(set(ages))] == [3483, 3847, 243]
    assert attrs["static"][2]["distinct_values"] == 243
    failure = read(STAGE / "validation-failures.json")["failures"][0]
    assert sha(failure["code_copy"]) == failure["code_sha256"]
    assert all(sha(STAGE / p) == h for p, h in failure["partial_artifacts"].items())
    failure2 = read(STAGE / "validation-failures-02.json")
    assert sha(failure2["code_copy"]) == failure2["code_sha256"]
    prior_file = next(
        p for p in freeze["files"]
        if p.endswith("event-time-combination-review\\20261001-v1\\final-verification.json")
    )
    prior = read(prior_file)
    assert all(sha(Path(prior_file).parent / p) == h for p, h in prior["artifacts"].items())
    assert all(sha(p) == h for p, h in prior["source_code"].items())
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    assert now < plan["deadline_at"]
    output = {
        "at": now, "status": "PASS", "source_files_verified": len(freeze["files"]),
        "analysis_read_entries_verified": len(ledger), "funds_independently_recomputed": len(summaries),
        "metadata_rows_independently_recomputed": checked, "feature_values_compared": checked * 15,
        "maximum_feature_absolute_error": max_error, "embedded_reports": report_count,
        "target_pdfs_quote_readback": len(extracts), "quotes_readback": quotes_checked,
        "manager_pdfs_readback": len(notices),
        "manager_code_literal_matches": sum(e["fund_code_literal_found"] for e in notices),
        "prior_stage_artifact_hashes_verified": len(prior["artifacts"]),
        "prior_stage_code_hashes_verified": len(prior["source_code"]), "failure_records_preserved": 2,
        "supervised_fits": 0, "strictly_admitted_new_rows": 0, "verification_script_sha256": sha(__file__),
        "new_2026_predictions_or_scores_computed": False,
    }
    target = STAGE / "independent-verification-final.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    return output


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
