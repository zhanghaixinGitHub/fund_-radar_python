"""盘点小范围评分缺项，或按资料核验回执维护补件记录。"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_engine, get_nav_preview_engine
from app.models.fund_rating import RatingBatch
from app.repositories.fund_rating import lock_category
from app.services.fund_exposure_common import PROJECT
from app.services.fund_market_preparation import validate_codes
from app.services.fund_market_review import rating_review_required
from app.services.fund_rating import read
from app.services.fund_rating_evidence_preview import preview
from app.services.fund_rating_pipeline import current_inputs
from app.services.fund_rating_reviewed_snapshot import archive_root
from app.services.fund_rating_supplements import record, report

DATABASE = PROJECT / ".local-runs" / "rating-evidence" / "supplements.sqlite3"


def audit(codes: tuple[str, ...]) -> dict:
    """现有评分缺项只作为查证线索。已有分也不能证明新规则所需的原件齐全。"""
    engine = get_nav_preview_engine()
    items = []
    missing = Counter()
    for code in codes:
        detail = read([code], detail=True, engine=engine)
        withheld = detail["status"] in {"STALE", "WITHDRAWN"}
        dimensions = []
        for dim in detail["dimensions"]:
            state = "POINT" if dim.get("score") is not None else "INTERVAL" if dim.get("scoreRange") else "MISSING"
            if withheld:
                state = "WITHHELD"
            elif state != "POINT":
                missing[dim["key"]] += 1
            dimensions.append({"key": dim["key"], "name": dim["name"], "state": state, "reason": dim["explanation"]})
        items.append(
            {
                "fund_code": code,
                "as_of_date": detail.get("asOfDate"),
                "rating_ref": detail.get("ratingRef"),
                "rating_status": detail["status"],
                "rating_message": detail.get("message"),
                "dimensions": dimensions,
            }
        )
    return {"items": items, "dimensions_without_points": dict(missing)}


def compare(codes: tuple[str, ...]) -> dict:
    engine = get_nav_preview_engine()
    displayed = {item["fundCode"]: item for item in read(list(codes), engine=engine)["items"]}
    with Session(engine) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        current = current_inputs(session, list(codes))
    root = archive_root()
    items = []
    for code in codes:
        old = displayed[code]
        row = current.get(code)
        if row is None:
            items.append({"fund_code": code, "status": "NO_CURRENT_INPUT"})
            continue
        if row["batch_id"] != old.get("ratingRef"):
            raise ValueError("EVIDENCE_PREVIEW_CURRENT_CHANGED")
        bundle = next(member["bundle"] for member in row["input_snapshot"]["members"] if member["fund_code"] == code)
        if bundle["fund_code"] != code or bundle["as_of_date"] != str(row["as_of_date"]):
            raise ValueError("EVIDENCE_PREVIEW_BATCH_BINDING")
        result = preview(bundle, root, review_required=rating_review_required(code))
        items.append(
            {
                "fund_code": code,
                "as_of_date": bundle["as_of_date"],
                "rating_ref": row["batch_id"],
                "old_status": old["status"],
                "old_score": old.get("score"),
                **result,
            }
        )
    return {"items": items}


def activate(codes: tuple[str, ...], *, dry_run: bool = False) -> dict:
    from app.services.fund_rating_evidence_publication import dossier_for
    from app.services.fund_rating_reviewed_snapshot import publish_reviewed

    root = archive_root()
    with Session(get_engine()) as session, session.begin():
        if not session.scalar(text("SELECT pg_try_advisory_xact_lock(hashtextextended('fund-rating-task',0))")):
            raise ValueError("RATING_TASK_BUSY")
        current = current_inputs(session, list(codes))
        if set(current) != set(codes):
            raise ValueError("RATING_EVIDENCE_V1_NO_CURRENT_INPUT")
        categories = {session.get(RatingBatch, row["batch_id"]).category_code for row in current.values()}
        for category in sorted(categories):
            lock_category(session, category)
        if {code: row["batch_id"] for code, row in current_inputs(session, list(codes)).items()} != {
            code: row["batch_id"] for code, row in current.items()
        }:
            raise ValueError("EVIDENCE_PREVIEW_CURRENT_CHANGED")
        bundles = [
            next(m["bundle"] for m in current[code]["input_snapshot"]["members"] if m["fund_code"] == code)
            for code in codes
        ]
        pending = [bundle for bundle in bundles if not bundle.get("evidence_v1_calculation")]
        if not pending:
            return {"already_active": list(codes)}
        result = publish_reviewed(session, dossier_for(pending, root), "启用已验证的近一年指数基金评分规则", root=root)
        if dry_run:
            session.rollback()
        return {**result, "dry_run": dry_run}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    audit_command = actions.add_parser("audit", help="只读盘点指定基金的八维现状")
    audit_command.add_argument("codes", nargs="+")
    preview_command = actions.add_parser("preview", help="读取封存原件，对照一年候选分与正式分")
    preview_command.add_argument("codes", nargs="+")
    activate_command = actions.add_parser("activate", help="启用已核准份额的近一年正式评分")
    activate_command.add_argument("codes", nargs="+")
    activate_command.add_argument("--dry-run", action="store_true", help="执行发布核验后回滚数据库事务")
    record_command = actions.add_parser("record", help="导入已核的补件回执")
    record_command.add_argument("receipt", type=Path)
    actions.add_parser("report", help="查看补件待办和逾期统计")
    args = parser.parse_args()

    if args.action in {"audit", "preview", "activate"}:
        codes = validate_codes(args.codes)
        if len(codes) > 5:
            parser.error("一次最多盘点五只基金")
        if args.action == "activate":
            result = activate(codes, dry_run=args.dry_run)
        else:
            result = audit(codes) if args.action == "audit" else compare(codes)
    elif args.action == "record":
        if args.receipt.stat().st_size > 65_536:
            parser.error("回执大小超过限制")
        receipt = json.loads(args.receipt.read_text(encoding="utf-8-sig"))
        result = {"recorded": record(DATABASE, receipt, sources_root=archive_root() / "sources")}
    else:
        result = report(DATABASE)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
