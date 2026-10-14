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
    parser.add_argument(
        "action",
        choices=(
            "audit",
            "update",
            "import",
            "activate",
            "withdraw",
            "rollback",
            "collect-case",
            "prepare-public",
            "activate-public",
        ),
    )
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fund-code")
    parser.add_argument("--family")
    parser.add_argument("--batch")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.action == "collect-case":
        if args.fund_code != "002112" or not args.as_of:
            parser.error("collect-case 必须明确 --fund-code 002112 和 --as-of")
        from app.core.config import get_settings
        from app.integrations.fund_rating_public import collect_risk_free
        from app.services.direction_1d_protocol import calendar

        sessions, _ = calendar()
        start_month = args.as_of.replace(year=args.as_of.year - 3, day=1)
        start = max(d for d in sessions if (d.year, d.month) == (start_month.year, start_month.month))
        value = collect_risk_free(
            start, args.as_of, Path(get_settings().fund_insight_directory) / "ratings" / "risk-free"
        )
        result = {
            "source": "财政部-中国国债收益率曲线",
            "rows": len(value["rows"]),
            "start": value["start"],
            "end": value["end"],
            "usage": value["usage"],
        }
    elif args.action == "update":
        result = update_ratings(args.fund_code, as_of=args.as_of)
    else:
        with Session(get_engine()) as session, session.begin():
            if args.action == "prepare-public":
                if args.fund_code != "002112" or not args.as_of or not args.output:
                    parser.error("prepare-public 必须明确 --fund-code 002112、--as-of 和 --output；只准备原件，不出级")
                from datetime import UTC, datetime

                from app.services.fund_rating_public_inputs import prepare_public_bundle

                result = prepare_public_bundle(session, args.as_of, datetime.now(UTC))
            elif args.action == "activate-public":
                if not args.file or not args.reason:
                    parser.error("activate-public 需要真实 --file 和 --reason，不能省略验证")
                from app.services.fund_rating_public_validation import activate_public

                result = {
                    "methodology_id": activate_public(
                        session, json.loads(args.file.read_text(encoding="utf-8")), args.reason
                    )
                }
            elif args.action == "audit":
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
