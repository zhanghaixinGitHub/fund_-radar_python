"""执行两只参考基金的本地资料审计；协议须已单独冻结，无采集或训练入口。"""

import json

from app.services.fund_002112_peer_coverage_audit import run

if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
