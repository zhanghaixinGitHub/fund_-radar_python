"""python -m app.commands.fund_ratings：本地核验、导入、试算和受控恢复，不新增定时任务。"""

import argparse
import json
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.fund_rating import RatingBatch
from app.repositories.fund_rating import publish, withdraw
from app.services.fund_rating import update_ratings
from app.services.fund_rating_admin import activate, ingest, trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "update", "import", "activate", "withdraw", "rollback"))
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fund-code")
    parser.add_argument("--family")
    parser.add_argument("--batch")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.action == "update":
        result = update_ratings(args.fund_code, as_of=args.as_of)
    else:
        with Session(get_engine()) as session, session.begin():
            if args.action == "audit":
                if not args.as_of:
                    parser.error("audit 必须明确 --as-of；仅输出盘点，不采集、不发布")
                result = trial(session, args.as_of)
            elif args.action == "import":
                if not args.file:
                    parser.error("import 需要经过核验的规范化原件 --file")
                result = {"evidence_hash": ingest(session, json.loads(args.file.read_text(encoding="utf-8")))}
            elif args.action == "activate":
                if not args.file or not args.family or not args.reason:
                    parser.error("activate 需要真实试算 --file、--family 和审阅记录 --reason")
                activate(session, args.family, json.loads(args.file.read_text(encoding="utf-8")), args.reason)
                result = {"activated": args.family}
            elif args.action in {"withdraw", "rollback"}:
                if not args.batch or not args.reason:
                    parser.error("撤回/回退必须明确 --batch 和 --reason")
                if args.action == "withdraw":
                    withdraw(session, args.batch, args.reason)
                else:
                    batch = session.get(RatingBatch, args.batch)
                    if not batch:
                        parser.error("所选批次不存在")
                    publish(session, batch, rollback_reason=args.reason)
                result = {"action": args.action, "batch": args.batch}
    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
