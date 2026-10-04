"""真实下一交易日输入对齐及训练标签成熟的跨年反例。"""

from scripts import fund_002112_next_day_search_v1 as n


def test_next_day_uses_origin_features_and_origin_training_cutoff(tmp_path, monkeypatch):
    morning, target = tmp_path / "morning", tmp_path / "next"
    n.io.save(target / "plan.json", {"registered_at": "2026-09-30T23:00:00+08:00"})
    n.io.save(target / "dependency-manifest.json", {})
    nav_file = tmp_path / "nav.json"
    n.io.save(nav_file, {})
    data = [
        ("2024-12-26", "2024-12-25", "2024-12-27", 26),
        ("2024-12-27", "2024-12-26", "2024-12-30", 27),
        ("2024-12-30", "2024-12-27", "2024-12-31", 30),
        ("2024-12-31", "2024-12-30", "2025-01-02", 31),
        ("2025-01-02", "2024-12-31", "2025-01-03", 102),
        ("2025-12-31", "2025-12-30", "2026-01-05", 1231),
        ("2026-01-05", "2025-12-31", "2026-01-06", 105),
    ]
    rows = [
        {
            "target": day,
            "base": base,
            "as_of": day + "T08:00:00+08:00",
            "label_mature_at": mature + "T08:00:00+08:00",
            "session_index": i,
            "groups": {"N": [value]},
        }
        for i, (day, base, mature, value) in enumerate(data)
    ]
    n.io.save_lines(morning / "inputs.jsonl", rows)
    monkeypatch.setattr(n.kernel, "ROOT", morning)
    monkeypatch.setattr(n, "sources_file", lambda: nav_file)
    n.build(target)
    frozen = {r["target"]: r for r in n.io.lines(target / "inputs.jsonl")}
    # 目标日的新信息102不能流入12月31日发出的预测，只能用预测日自身输入31。
    assert frozen["2025-01-02"]["groups"]["N"] == [31]
    assert frozen["2025-01-02"]["as_of"] == "2024-12-31T08:00:00+08:00"
    split = n.io.read(target / "splits.json")["V1"]
    assert split["cutoff_exclusive"] == "2024-12-31T08:00:00+08:00"
    # 12月30日的答案直到预测时点08:00才成熟，按排他边界不能用于提前训练。
    assert split["train_dates"] == ["2024-12-27"]
    assert frozen["2026-01-05"]["groups"]["N"] == [1231]
