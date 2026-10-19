"""数据库等前置条件失败必须留下本轮失败，不能读回旧的成功回执。"""

import json
import sys

from app.commands import check_market_holdings as command


def test_failed_invocation_replaces_previous_success_receipt(tmp_path, monkeypatch):
    path = tmp_path / "latest-result.json"
    path.write_text('{"status":"SUCCEEDED","run_id":"old"}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_market_holdings", "--output", str(path)])
    monkeypatch.setattr(
        command.MarketHoldingsMaintenance, "check", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError())
    )
    assert command.main() == 1
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["status"] == "FAILED" and value["failure"] == "ConnectionError"
    assert "run_id" not in value and value["finished_at"]
