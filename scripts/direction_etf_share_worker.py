"""进程资源限制生效后才加载ETF份额对照的数值库。"""

import json
import sys


def main():
    data = sys.stdin.buffer.read(8 * 1024**2 + 1)
    if len(data) > 8 * 1024**2:
        raise ValueError("ETF_SHARE_WORKER_INPUT_BUDGET")
    from app.services.direction_etf_share_models import execute_job

    try:
        result = execute_job(json.loads(data))
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)[:180]}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
