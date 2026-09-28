"""运行固定的零拟合参考基金关联与输入证据检查。"""

import json

from app.services.fund_002112_transfer_evidence import run

if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
