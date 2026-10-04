"""封存旧评估日期和新增测试输入；准备阶段不读取新增日期的净值答案。"""

from __future__ import annotations

import json

from scripts import fund_002112_semantic_upgrade_v1 as core

io, ROOT = core.io, core.ROOT


def prepare():
    if (ROOT / "exposure-audit.json").exists():
        return io.read(ROOT / "exposure-audit.json")
    sources = []
    exposed = set()
    # 查全部旧预测文件，只输出目标日期；本轮目录排除，避免把自身诊断混成历史暴露。
    for path in io.RESEARCH.rglob("*.jsonl"):
        if ROOT in path.parents or "predict" not in str(path.relative_to(io.RESEARCH)).lower():
            continue
        days = set()
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                obj = json.loads(line)
                target = obj.get("target") or obj.get("target_date") or obj.get("date")
                if isinstance(target, str) and target.startswith("2026-"):
                    days.add(target[:10])
        if days:
            exposed |= days
            sources.append({"path": str(path), "sha256": io.sha(path), "dates": sorted(days)})
    bundle = io.read(core.previous.BUNDLE)
    rows = core.previous.read_lines(ROOT / "numeric-inputs.jsonl")
    extra_day = "2026-09-30"
    independent = extra_day not in exposed
    if independent:
        sessions = bundle["sessions"]
        pos = sessions.index(extra_day)
        base = sessions[pos - 1]
        cutoff = base + "T08:00:00+08:00"
        end, _, n = core.previous.holdings.choose_nav_window(sessions, bundle["nav"], base)
        ni = sessions.index(end)
        ne = core.price.nav_extra([float(bundle["nav"][d]["unit_nav"]) for d in sessions[ni - 60 : ni + 1]])
        reports = io.read(ROOT / "reports.json")
        report = core.previous.choose_report(reports, cutoff)
        h, proof = core.partial_holdings(
            report, bundle["stocks"], sessions, cutoff, io.read(ROOT / "suspension-evidence.json")["events"]
        )
        row = {
            "as_of": cutoff,
            "base": base,
            "target": extra_day,
            "session_index": rows[-1]["session_index"] + 1,
            "label_mature_at": "2026-10-01T08:00:00+08:00",
            "groups": {"N": n, "NE": ne},
            "H": h,
        }
        assert rows[-1]["target"] == base and extra_day not in {r["target"] for r in rows}
        rows.append(row)
        io.save(
            ROOT / "reserved-independent-input.json",
            {
                "at": io.now(),
                "input": row,
                "holdings_lineage": proof,
                "target_value_not_read": True,
                "limitations": "新增仅一天；此前2026已用于研究。未获取历史首次抓取证明。",
            },
        )
    io.save_lines(ROOT / "numeric-expanded.jsonl", rows)
    result = {
        "at": io.now(),
        "old_prediction_files": sources,
        "exposed_dates": sorted(exposed),
        "independent_dates": [extra_day] if independent else [],
        "rule": "2025确定候选；2026旧日期仅诊断；新增一天先保存预测再读答案，不据此重新挑选",
    }
    io.save(ROOT / "exposure-audit.json", result)
    return {
        "old_files": len(sources),
        "exposed_count": len(exposed),
        "exposed_max": max(exposed),
        "independent_dates": result["independent_dates"],
    }


if __name__ == "__main__":
    print(io.canonical(prepare()), flush=True)
