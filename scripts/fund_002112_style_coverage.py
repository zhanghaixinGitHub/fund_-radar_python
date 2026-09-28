"""固定资料内的方向覆盖核查；extract 后人工复核冻结，再执行 count。"""

import argparse
import json

from app.services.fund_002112_style_coverage import count, extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "count"))
    args = parser.parse_args()
    print(json.dumps(extract() if args.command == "extract" else count(), ensure_ascii=False, indent=2))
