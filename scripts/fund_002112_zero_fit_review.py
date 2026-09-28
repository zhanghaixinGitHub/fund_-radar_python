"""固定入口：只复盘已有预测，不调用任何模型训练或业务服务。"""

import json

from app.services.fund_002112_zero_fit_review import run

if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
