"""由本地runner启动的内部训练进程；无路径参数，无数据库调用。"""

import json
import sys


def main():
    content = sys.stdin.buffer.read(8 * 1024**2 + 1)
    if len(content) > 8 * 1024**2:
        raise ValueError("JOB_INPUT_TOO_LARGE")
    from app.services.direction_training_models import execute_job

    try:
        result = execute_job(json.loads(content))
    except Exception as error:
        result = {"status": "FAILED", "reason": getattr(error, "code", type(error).__name__)}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
