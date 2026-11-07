"""008888已核验文本的单次清理；默认只形成精确计划，--execute才消费计划。"""

import argparse
import json
from pathlib import Path

from app.commands.material_text_pilot import DEFAULT_RUN
from app.services.material_pilot_cleanup import execute, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from app.services.material_file_access import legacy_workflow_error

    legacy_workflow_error()
    result = execute(args.run.resolve()) if args.execute else prepare(args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
