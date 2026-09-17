"""执行有界风格指数历史采集；异常仅打印固定错误码，详细请求响应保存在研究目录。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_style_data as data  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "acquire"))
    action = parser.parse_args().action
    try:
        print(json.dumps(getattr(data, action)()))
    except Exception as exc:
        print(json.dumps({"error": base.error_code(exc)}))
        raise SystemExit(1) from None
