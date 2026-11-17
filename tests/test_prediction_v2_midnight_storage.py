"""午夜准备前已经发生的磁盘错误必须阻止业务写入。"""

from datetime import datetime

import pytest
from app.commands import fund_prediction_v2 as command


def test_midnight_storage_error_stops_before_prepare(monkeypatch, tmp_path):
    instant = datetime.fromisoformat("2026-11-17T00:30:00+08:00")
    error_at = datetime.fromisoformat("2026-11-17T00:10:00+08:00")
    prepared = []
    monkeypatch.setattr(command, "now", lambda: instant)
    monkeypatch.setattr("sys.argv", ["fund_prediction_v2", "prepare", "--run-dir", str(tmp_path),
                                    "--selection", str(tmp_path / "selection.json")])

    def check(since):
        if datetime.fromisoformat(since) <= error_at <= instant:
            raise OSError("V2_NEW_STORAGE_ERROR_STOP_WRITES")
        return {"count": 0}

    monkeypatch.setattr(command, "storage_check", check)
    monkeypatch.setattr(command, "prepare_trial", lambda *args: prepared.append(args) or {"selection": []})
    monkeypatch.setattr(command, "save_once", lambda *args: None)
    with pytest.raises(OSError, match="NEW_STORAGE_ERROR_STOP_WRITES"):
        command.main()
    assert prepared == []
