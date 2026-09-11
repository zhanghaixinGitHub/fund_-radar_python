"""资源限制生效后才读取滚动作业及加载数值库。"""

import json
import sys


def main():
    content = sys.stdin.buffer.read(8 * 1024**2 + 1)
    if len(content) > 8 * 1024**2:
        raise ValueError("ROLLING_WORKER_INPUT_BUDGET")
    from app.services.direction_rolling_models import execute_job

    try:
        result = execute_job(json.loads(content))
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)[:180]}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
