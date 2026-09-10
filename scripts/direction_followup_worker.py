"""固定研究作业的受限子进程；不打开来源和答案文件。"""

import json
import sys


def main():
    content = sys.stdin.buffer.read(8 * 1024**2 + 1)
    if len(content) > 8 * 1024**2:
        raise ValueError("WORKER_INPUT_BUDGET")
    from app.services.direction_followup_models import execute_job

    try:
        result = execute_job(json.loads(content))
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
