"""本地 002112 样例研究入库；默认仅准备，--apply 才提交完整事务。"""

import argparse
import json

from app.services.fund_002112_input_audit import OUTPUT
from app.services.fund_002112_news_import import prepare_existing_bundle
from app.services.fund_002112_zero_fit_review import save_once
from app.services.news_research_storage import read_research_bundle, save_research_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    bundle = prepare_existing_bundle()
    payload = bundle.model_dump(mode="json")
    save_once(OUTPUT / "news-storage-prepared.json", payload)
    if args.apply:
        result = save_research_bundle(bundle)
        if read_research_bundle(result["bundle_hash"]) != payload:
            raise ValueError("NEWS_DURABLE_READBACK_FAILED")
        # 首次提交与幂等重试分开留档；重试不覆盖首次创建事实。
        name = "news-storage-created.json" if result["created"] else "news-storage-retry.json"
        save_once(OUTPUT / name, {**result, "durable_readback_equal": True})
    else:
        result = {"prepared_cards": len(bundle.records), "database_written": False}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
