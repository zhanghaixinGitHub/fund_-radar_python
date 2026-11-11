"""补建指定基金当前正式批次的评价；业务数据库强制只读，不重发评分。"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.models.fund_rating import RatingBatch, RatingCurrent, RatingResult
from app.services.fund_rating import read
from app.services.fund_rating_narrative_store import prepare
from app.services.fund_rating_reviewed_snapshot import archive_root


def generate(codes, *, force=False, progress=None):
    if not 1 <= len(codes) <= 100 or any(len(c) != 6 or not c.isascii() or not c.isdigit() for c in codes):
        raise ValueError("NARRATIVE_CODES_INVALID")
    engine = get_nav_preview_engine()
    items = []
    for code in dict.fromkeys(codes):
        # 摘要经过同样的正式批次门禁，不读取待修复的评价文件。
        detail = read([code], engine=engine)["items"][0]
        if detail["status"] not in {"SCORED", "RATED", "PARTIAL"}:
            items.append({"fund_code": code, "state": "WITHHELD", "rating_status": detail["status"]})
            continue
        with Session(engine) as session:
            session.execute(text("SET TRANSACTION READ ONLY"))
            session.execute(text("SET LOCAL statement_timeout='5s'"))
            batch = session.get(RatingBatch, detail["ratingRef"])
            result = session.scalar(
                select(RatingResult).where(RatingResult.batch_id == batch.batch_id, RatingResult.fund_code == code)
            )
            try:
                items.append(prepare(result, batch.input_snapshot, root=archive_root(), force=force))
            except (ValueError, KeyError, FileNotFoundError) as error:
                items.append({"fund_code": code, "state": "FAILED", "reason": str(error)[:160]})
        if progress:
            progress(items[-1])
    return {"items": items}


def generate_all(*, force=False, progress=None):
    """全部指当前已有正式评分；无评分和失效评分不凭空生成结论。"""
    with Session(get_nav_preview_engine()) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        codes = sorted(
            session.scalars(
                select(RatingResult.fund_code)
                .join(RatingCurrent, RatingCurrent.batch_id == RatingResult.batch_id)
                .distinct()
            ).all()
        )
    items = []
    for start in range(0, len(codes), 100):
        items.extend(generate(codes[start : start + 100], force=force, progress=progress)["items"])
    return {"items": items, "counts": dict(Counter(row["state"] for row in items)), "scope_count": len(codes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codes", nargs="*")
    parser.add_argument("--all", action="store_true", help="处理所有当前正式评分，不扩展每日评分任务范围")
    parser.add_argument("--report", type=Path, help="将本轮处理结果保存到D盘")
    parser.add_argument("--force", action="store_true", help="重新核对原件并生成当前同批次评价")
    args = parser.parse_args()
    if bool(args.codes) == args.all:
        parser.error("提供基金代码或--all，二者选一")
    result = generate_all(force=args.force) if args.all else generate(args.codes, force=args.force)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        target = args.report.resolve()
        if target.drive.upper() != "D:":
            raise ValueError("NARRATIVE_REPORT_MUST_USE_D_DRIVE")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
