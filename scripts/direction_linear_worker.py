"""资源限制生效后仅消费当前窗口的线性训练作业，不连接来源。"""

import json
import sys


def main():
    content = sys.stdin.buffer.read(8 * 1024**2 + 1)
    if len(content) > 8 * 1024**2:
        raise ValueError("WORKER_INPUT_BUDGET")
    from app.services.direction_linear_models import execute_job

    try:
        result = execute_job(json.loads(content))
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)[:240]}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
