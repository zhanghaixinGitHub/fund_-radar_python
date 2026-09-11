"""在资源限制内消费单个四组对照作业，不访问来源或考试答案。"""

import json
import sys


def main():
    content = sys.stdin.buffer.read(8 * 1024**2 + 1)
    if len(content) > 8 * 1024**2:
        raise ValueError("WORKER_INPUT_BUDGET")
    from app.services.direction_algorithm_models import execute_job

    try:
        result = execute_job(json.loads(content))
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)[:240]}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
