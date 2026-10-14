"""002112 当前资料的9B下一交易日诊断；数据库只读、首次请求一次、不发布。

显式下一交易日可在本日未收盘时生成，比较基准仍是本日，不能拿最新已公布
净值替换基准。早期预测与正式已收盘窗口分开；全部边界保存在独立实验目录。
"""

import argparse
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from datetime import time as day_time
from pathlib import Path

from app.db.session import get_nav_preview_engine
from app.repositories import direction_1d as repository
from app.services import fund_information_snapshot as snapshot_service
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.fund_exposure_common import ROOT as MATERIAL_ROOT
from app.services.fund_exposure_common import read as read_material
from app.services.fund_materials_store import source_path
from sqlalchemy import event, text

MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    """首次输入、调用和失败均独占保存，不覆盖旧实验。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def next_window(now):
    """下一交易日相对其前一交易日的净值变化；即使基准尚未收盘也不改日期。"""
    days, version = calendar()
    today = now.astimezone(ZONE).date()
    target = next(day for day in days if day > today)
    baseline = days[days.index(target) - 1]
    return {
        "generation_policy": "LOCAL_NEXT_SESSION_EARLY_DIAGNOSTIC",
        "base_nav_date": str(baseline),
        "target_nav_date": str(target),
        "calendar_version": version,
        "status": "DIAGNOSTIC_ONLY",
        "window_open_at": datetime.combine(baseline, day_time(15), ZONE).isoformat(),
        "deadline_at": datetime.combine(target, day_time(15), ZONE).isoformat(),
    }


def forward_time(data):
    """本地提前预测时刻契约；不用正式服务的‘今日未收盘则预测今日’策略。"""
    cutoff = datetime.fromisoformat(data["as_of"]).astimezone(ZONE)
    baseline, target = data["window"]["base_nav_date"], data["window"]["target_nav_date"]
    base_close = datetime.combine(datetime.fromisoformat(baseline).date(), day_time(15), ZONE)
    latest = max((str(r["nav_date"]) for r in data["nav"]), default=None)
    if latest != data["latest_nav_date"] or (latest and latest > baseline):
        raise ValueError("NEXTDAY_NAV_DATE_INVALID")
    pending = cutoff < base_close
    stocks = {c["code"]: {"date": (c.get("quote") or {}).get("date")} for c in data["companies"]}
    markets = {code: {"date": q["date"]} for code, q in data["market"].items()}
    for value in [*stocks.values(), *markets.values()]:
        day = value["date"]
        if day and day > str(cutoff.date()):
            raise ValueError("FUTURE_QUOTE_FORBIDDEN")
        value["status"] = "MISSING" if not day else "READY" if day == baseline else "LAST_AVAILABLE_CLOSE"
    return {
        "cutoff": cutoff.isoformat(),
        "target_date": target,
        "baseline_date": baseline,
        "latest_nav_date": latest,
        "calendar_hash": data["window"]["calendar_version"],
        "baseline_close_status": "NOT_YET_OCCURRED" if pending else "CLOSED",
        "nav_status": "READY" if latest == baseline else "NOT_YET_PUBLISHED" if pending else "STALE_OR_MISSING",
        "stocks": stocks,
        "markets": markets,
        "baseline_complete": latest == baseline,
        "target_close_status": "NOT_YET_OCCURRED",
        "statements": [
            f"预测目标{target}，与{baseline}单位净值比较；资料截止北京时间{cutoff:%Y-%m-%d %H:%M:%S}。",
            f"最新已保存净值截至{latest}，不能把它的日期替换为比较基准日。",
            "本次是基准尚未收盘的提前预测，未取得基准净值不能当作已知数值。"
            if pending
            else "本次各项行情和净值按实际资料日期分别保留。",
            "次日涨跌准确性只能等目标日净值公布后核验，本轮不读取未来答案。",
        ],
    }


def read_only_observe(connection, code, source, rows, now):
    """原observe会写来源版本表；此诊断只在本地记录实际读取回执，禁止入库。"""
    result = []
    for row in rows:
        if row["ann_date"] and row["ann_date"] > now.date():
            raise ValueError("FUTURE_NAV_PUBLICATION")
        result.append(
            {
                "nav_date": str(row["nav_date"]),
                "unit_nav": str(row["unit_nav"]),
                "ann_date": str(row["ann_date"]) if row["ann_date"] else None,
                "upstream_hash": row["content_hash"],
                "source_id": str(source["source_id"]),
                "received_at": now.isoformat(),
                "receipt_scope": "LOCAL_READ_ONLY_OBSERVATION",
            }
        )
    return result


def guard_sql(statement):
    """只读事务之外再拒绝写SQL；事务级只读仍是最终数据库保障。"""
    if statement.strip().split(maxsplit=1)[0].upper() not in {"SELECT", "SHOW", "SET"}:
        raise ValueError("LOCAL_DATABASE_WRITE_FORBIDDEN")


class ReadOnlyEngine:
    def __init__(self, engine):
        self.engine = engine

    @contextmanager
    def begin(self):
        with self.engine.connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION READ ONLY"))
            yield connection


def restore_documents(data):
    """恢复准备器长度裁剪的同源全文；原本只保存摘录的仍明确为摘录。"""
    receipts = []
    for document in data["documents"]:
        if not document["body_truncated"]:
            continue
        identity = document["id"].removeprefix("LIVE_CNINFO:")
        candidate = source_path(MATERIAL_ROOT, "supplement/company-documents/" + digest(identity) + ".json")
        if not candidate.exists():
            raise ValueError("TRUNCATED_SOURCE_NOT_RESTORABLE:" + document["id"])
        original = read_material(candidate)
        if original["receipt"]["sha256"] != document["source_hash"]:
            raise ValueError("RESTORED_SOURCE_HASH_MISMATCH")
        body = "\n".join(original["pages"])
        if not body or "【中间部分未纳入】" in body:
            raise ValueError("RESTORED_BODY_INVALID")
        receipts.append(
            {
                "id": document["id"],
                "previous_characters": len(document["body"]),
                "full_characters": len(body),
                "source_path": str(candidate),
                "source_sha256": sha(candidate),
            }
        )
        document.update(body=body, body_scope="FULL_TEXT", body_truncated=False)
    return receipts


def prepare(root):
    """从当前已保存材料和数据库重新读取，不使用旧预测答案或旧输入快照。"""
    if root.exists():
        raise ValueError("NEW_RUN_DIRECTORY_REQUIRED")
    root.mkdir(parents=True)
    engine = get_nav_preview_engine()

    @event.listens_for(engine, "before_cursor_execute")
    def before_sql(connection, cursor, statement, parameters, context, executemany):
        guard_sql(statement)

    with engine.connect() as connection, connection.begin():
        connection.execute(text("SET TRANSACTION READ ONLY"))
        now = connection.execute(text("SELECT clock_timestamp()")).scalar_one().astimezone(ZONE)
    if str(now.date()) != "2026-10-13":
        raise ValueError("CLIENT_DATE_MISMATCH")
    # 以下替换仅存在于这个独立命令进程；不修改项目服务源码或运行中的后端。
    snapshot_service.get_engine = lambda: ReadOnlyEngine(engine)
    snapshot_service.window = next_window
    snapshot_service.time_context = forward_time
    repository.observe = read_only_observe
    data = snapshot_service.prepare(now)
    restored = restore_documents(data)
    data["time_context"] = forward_time(data)
    for document in data["documents"]:
        if datetime.fromisoformat(document["available_at"]) > now or document["date"] > str(now.date()):
            raise ValueError("SOURCE_AFTER_CUTOFF")
    save(root / "snapshot.json", data)
    save(root / "source-restoration.json", restored)
    save(root / "time-context.json", data["time_context"])
    save(
        root / "input-receipt.json",
        {
            "as_of": data["as_of"],
            "target": data["window"]["target_nav_date"],
            "baseline": data["window"]["base_nav_date"],
            "latest_nav_date": data["latest_nav_date"],
            "snapshot_hash": digest(data),
            "documents": len(data["documents"]),
            "source_characters": sum(len(d["body"]) for d in data["documents"]),
            "excerpts": sum(d["body_scope"] != "FULL_TEXT" for d in data["documents"]),
            "truncated": sum(d["body_truncated"] for d in data["documents"]),
            "restored": len(restored),
            "source_selection": "现有服务的近两周资料窗口和来源核验；未按模型输出挑材料",
            "database_read_only": True,
            "database_writes": 0,
            "paid_calls": 0,
            "old_prediction_or_old_snapshot_loaded": False,
            "production_switched": False,
        },
    )
    print(json.dumps(read(root / "input-receipt.json"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
