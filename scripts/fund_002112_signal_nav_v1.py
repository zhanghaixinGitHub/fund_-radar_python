"""逐日复原61个连续交易日净值窗口，保留未知公开时间，不向前挪动截点。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import fund_002112_price_sources_v1 as price
from scripts import fund_002112_signal_common_v1 as c


def audit_row(row, sessions, nav, unresolved):
    cutoff = row["as_of"]
    # base 是08:00的判断日D，target 是下一交易日U；不能用U替代D选窗口。
    end, lag, values = c.nav_inputs.choose_nav_window(sessions, nav, row["base"])
    expected_index = sessions.index(row["base"]) - 1
    attempts = []
    for offset in range((lag if lag is not None else 20) + 1):
        idx = expected_index - offset
        days = sessions[idx - 60 : idx + 1]
        blockers = []
        for day in days:
            source = nav.get(day)
            if source is None:
                blockers.append({"date": day, "reason": "NAV_RECORD_MISSING"})
            elif c.moment(source["available_at"]) > c.moment(cutoff):
                known = unresolved.get(day)
                blockers.append(
                    {
                        "date": day,
                        "reason": "AVAILABLE_AFTER_CUTOFF",
                        "record": source,
                        "classification": "CONTIGUOUS_WINDOW_BLOCKED_PUBLICATION_VERSION_UNRESOLVED",
                        "evidence": known,
                        "missing_evidence": (
                            "缺少在该预测截点前公开、含相同基金与相同净值的历史原件；公告字段与季报日期相同不足以提前。"
                        ),
                    }
                )
        attempts.append({"end": sessions[idx], "lag": offset, "blockers": blockers})
    if end:
        idx = sessions.index(end)
        window = sessions[idx - 60 : idx + 1]
        ne = price.nav_extra([float(nav[d]["unit_nav"]) for d in window])
        assert values == row["groups"]["N"] and ne == row["groups"]["NE"], "OLD_BASELINE_NOT_REPRODUCED"
        assert all(c.moment(nav[d]["available_at"]) <= c.moment(cutoff) for d in window)
    else:
        window, ne = [], None
    return {
        "base": row["base"],
        "target": row["target"],
        "as_of": cutoff,
        "expected_nav_date": sessions[expected_index],
        "window_end": end,
        "window_dates": window,
        "old_lag": row["groups"]["N"][7],
        "new_lag": lag,
        "attempts": attempts,
        "first_blocker": next((a["blockers"][0] for a in attempts if a["blockers"]), None),
        "window_source_keys": {d: nav[d]["raw_sha256"] for d in window},
        "value_changed": False,
        "time_changed": False,
        "baseline_reproduced": bool(end),
        "decision": "RETAIN_CONSERVATIVE_OLD_WINDOW" if lag else "LATEST_CONTIGUOUS_WINDOW_AVAILABLE",
    }


def run(root):
    root = Path(root)
    b = c.bundle()
    previous = c.io.RESEARCH / "recent-input-completion/20260930-v1"
    evidence = c.io.read(previous / "nav-unresolved-evidence.json")
    unresolved = {r["nav_date"]: r for r in evidence["items"]}
    audits = [audit_row(r, b["sessions"], b["nav"], unresolved) for r in c.original_rows()]
    focused = [r for r in audits if r["target"][:4] in ("2025", "2026") and r["old_lag"] > 1]
    assert len(focused) == 71
    # 复核来源原值；原始官网响应与输入快照的基金代码、日期、数值相符才通过。
    provenance = c.io.read(previous / "input-provenance.json")
    originals = {}
    paths = [
        c.bodies.BUNDLE,
        c.OLD / "feature-revision-r3/inputs.jsonl",
        previous / "nav-unresolved-evidence.json",
        previous / "input-provenance.json",
        previous / "nav-provider-date-recheck.json",
    ]
    for key, source in provenance["nav"]["official_sources"].items():
        p = Path(source["path"])
        assert c.io.sha(p) == key
        payload = c.io.payload(c.io.read(p))
        originals[key] = {v["date"]: v for v in payload["dataList"]}
        paths.append(p)
    from decimal import Decimal

    for day, record in b["nav"].items():
        raw = originals[record["raw_sha256"]][day]
        assert raw["fundcode"] == "002112" and Decimal(str(raw["netvalue"])) == Decimal(record["unit_nav"])
    # 原件只证明当前版本数值；历史首次公开时点仍按已有较晚约束，不由当前读取反推。
    for item in evidence["items"]:
        for report in item["matching_report_send_date"]:
            p = Path(report["raw_path"])
            assert c.io.sha(p) == report["raw_sha256"]
            paths.append(p)
    c.io.save(root / "nav-sources.json", b["nav"])
    c.io.save_lines(root / "nav-audit.jsonl", audits)
    c.io.save_lines(root / "nav-focus-71.jsonl", focused)
    c.io.save(
        root / "nav-repairs.json",
        {"repairs": [], "reason": "没有足以提前历史净值版本时点的新增证据，原值及时间约束均保留。"},
    )
    summary = {
        "all_rows": len(audits),
        "focused_rows": len(focused),
        "reproduced": sum(r["baseline_reproduced"] for r in audits),
        "blocking_nav_dates": dict(Counter(r["first_blocker"]["date"] for r in focused)),
        "by_year": {
            year: {
                "rows": sum(r["target"].startswith(year) for r in audits),
                "old_distribution": dict(Counter(r["old_lag"] for r in audits if r["target"].startswith(year))),
                "new_distribution": dict(Counter(r["new_lag"] for r in audits if r["target"].startswith(year))),
            }
            for year in ("2025", "2026")
        },
        "source_values_verified": len(b["nav"]),
        "training_row_change": 0,
        "confirmed_mechanism": "季末单条净值的较晚可用约束阻断61日连续窗口；未跳过任何交易日。",
        "unresolved_root": "不能区分首次晚公布、供应商用季报日覆盖日净值日期或后续版本修订；保留限制。",
    }
    c.io.save(root / "nav-summary.json", summary)
    c.register_sources(root, paths, "nav-source-manifest.json")
    c.stage(
        root,
        "净值时效",
        "COMPLETE_WITH_CONSERVATIVE_UNRESOLVED",
        ["nav-audit.jsonl", "nav-focus-71.jsonl", "nav-summary.json"],
        list(summary["blocking_nav_dates"]),
        "固定最小事件输入",
    )
    print(c.io.canonical(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)
