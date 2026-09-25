"""002112 持仓补齐试点命令；每个阶段可以恢复，不自动购买数据或覆盖原预测。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.integrations.dbfund_reports import acquire  # noqa: E402
from app.services.fund_exposure_audit import audit  # noqa: E402
from app.services.fund_exposure_common import initialize  # noqa: E402
from app.services.fund_exposure_features import build_dataset, study  # noqa: E402
from app.services.fund_exposure_quotes import acquire_quotes, safe_error  # noqa: E402
from app.services.fund_exposure_runtime import capture, disable, enable, execution_lock, tick  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="002112 零新增采购数据补齐")
    commands = {
        "init": initialize,
        "reports": acquire,
        "quotes": acquire_quotes,
        "features": build_dataset,
        "study": study,
        "capture": capture,
        "enable": enable,
        "disable": disable,
        "tick": tick,
        "audit": audit,
    }
    parser.add_argument("command", choices=list(commands))
    command = parser.parse_args().command
    with execution_lock() as locked:
        if not locked:
            raise SystemExit("EXPOSURE_BUSY")
        result = commands[command]()
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "reason": safe_error(exc)}, ensure_ascii=False))
        raise SystemExit(1) from None
