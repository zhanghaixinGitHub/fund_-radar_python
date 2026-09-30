"""为当前消息快照补充正文要点；默认预览，--apply 原子发布且保留旧快照。

只处理固定基金当前最多 50 条已保存消息，不联网、不更新核对日期、不触发提醒。
与原同步任务使用同一把数据库锁，避免覆盖正在发布的消息快照。
"""

import argparse
import json

from app.db.session import get_engine
from app.services.announcement_key_points import RULE, with_reading_summary
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read, save
from app.services.fund_news_sync import directory
from sqlalchemy import text


def refresh(*, apply=False):
    with get_engine().connect() as connection:
        if not connection.execute(text("SELECT pg_try_advisory_lock(721130,2112)")).scalar_one():
            raise ValueError("NEWS_SYNC_BUSY")
        try:
            pointer = directory() / "current.json"
            reference = read(pointer)
            previous = read(directory() / "snapshots" / (reference["hash"] + ".json"))
            if digest(previous) != reference["hash"] or previous["fund_code"] != "002112":
                raise ValueError("NEWS_SNAPSHOT_INVALID")
            if len(previous["items"]) > 50:
                raise ValueError("NEWS_SNAPSHOT_SCOPE_EXCEEDED")
            items = [with_reading_summary(item) for item in previous["items"]]
            updated = {**previous, "items": items}
            key = digest(updated)
            if apply and key != reference["hash"]:
                path = directory() / "snapshots" / (key + ".json")
                if not path.exists():
                    save(path, updated)
                elif read(path) != updated:
                    raise ValueError("NEWS_SUMMARY_SNAPSHOT_CONFLICT")
                if read(pointer) != reference:
                    raise ValueError("NEWS_SUMMARY_SOURCE_CHANGED")
                save(pointer, {"hash": key}, replace=True)
            return {"applied": apply, "rule": RULE, "previous": reference["hash"], "current": key,
                    "items": len(items), "withPoints": sum(bool(i["summary_evidence"]) for i in items),
                    "checkedAt": previous["checked_at"]}
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(721130,2112)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    print(json.dumps(refresh(apply=parser.parse_args().apply), ensure_ascii=False))
