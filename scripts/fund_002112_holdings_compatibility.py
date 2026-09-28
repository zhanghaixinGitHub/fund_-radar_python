"""002112 冻结持仓核查；不训练、不联网，结果相同时复用，禁止覆盖不同结果。"""

import argparse
import json

from app.services.fund_002112_holdings_compatibility import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
