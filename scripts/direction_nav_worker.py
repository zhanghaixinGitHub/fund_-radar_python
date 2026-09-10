"""在父进程安装内存/时间限制并交付输入后才导入模型。"""

import json
import sys


def main():
    content = sys.stdin.buffer.read(8 * 1024**2 + 1)
    if len(content) > 8 * 1024**2:
        raise ValueError("WORKER_INPUT_BUDGET")
    from app.services.direction_nav_models import execute_job

    try:
        result = execute_job(json.loads(content))
    except Exception as error:
        result = {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)[:240]}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
