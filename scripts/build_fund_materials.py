"""显式发布 002112 已采集公共资料快照；没有外部网络调用。"""

import json

from app.services.fund_materials_build import build

if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
