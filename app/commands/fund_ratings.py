"""python -m app.commands.fund_ratings：本地核验、导入、试算和受控恢复，不新增定时任务。"""

import argparse
import json
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.fund_rating import RatingBatch
from app.repositories.fund_rating import publish, rollback_switch, withdraw
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
            "publish-reviewed",
        ),
    )
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fund-code")
    parser.add_argument("--event-review", action="store_true", help="已核验重大事项的单基金复核，必须明确截止日")
    parser.add_argument("--family")
    parser.add_argument(
        "--basis",
        choices=(
            "PUBLIC_DISCLOSURE_DOMESTIC_EQUITY_V1",
            "PUBLIC_DISCLOSURE_DOMESTIC_INDEX_V1",
            "PUBLIC_DISCLOSURE_DOMESTIC_INDEX_RECONCILED_V1",
            "PUBLIC_DISCLOSURE_GLOBAL_EQUITY_CNY_V1",
            "PUBLIC_DISCLOSURE_ETF_LINK_V1",
            "PUBLIC_REVIEWED_ACTIVE_QUALITY_V2",
            "PUBLIC_REVIEWED_PASSIVE_QUALITY_V2",
        ),
        help="市场公开资料必须显式指定已验证口径，不由基金代码猜测",
    )
    parser.add_argument("--batch")
    parser.add_argument("--from-batch", help="跨口径回退时明确指定撤销的当前批次，防止误改其他类别")
    parser.add_argument("--reason")
    parser.add_argument("--source-root", type=Path, help="已经取得并核验的官方原件目录；准备命令不联网、不出级")
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
        if args.event_review and (not args.fund_code or not args.as_of):
            parser.error("重大事项复核必须指定 --fund-code 和 --as-of，并先补齐本期八维原件")
        result = update_ratings(args.fund_code, as_of=args.as_of, event_review=args.event_review)
    else:
        with Session(get_engine()) as session, session.begin():
            if args.action == "publish-reviewed":
                if not args.file or not args.reason:
                    parser.error("publish-reviewed 需要已归档研究清单 --file 和明确 --reason")
                from app.services.fund_rating_reviewed_snapshot import publish_reviewed

                result = publish_reviewed(session, json.loads(args.file.read_text(encoding="utf-8")), args.reason)
            elif args.action == "prepare-public":
                if args.fund_code not in {"002112", "016008"} or not args.as_of or not args.output:
                    parser.error("prepare-public 必须明确已支持基金代码、--as-of 和 --output；只准备原件，不出级")
                from datetime import UTC, datetime

                from app.services.fund_rating_public_inputs import prepare_public_bundle

                if args.fund_code == "016008":
                    if not args.source_root:
                        parser.error("016008 必须提供 --source-root 官方原件目录")
                    from app.services.fund_rating_public_etf import prepare_etf_bundle

                    result = prepare_etf_bundle(args.source_root, args.as_of, datetime.now(UTC))
                else:
                    result = prepare_public_bundle(session, args.as_of, datetime.now(UTC))
            elif args.action == "activate-public":
                if not args.file or not args.reason:
                    parser.error("activate-public 需要真实 --file 和 --reason，不能省略验证")
                from app.services.fund_rating_public_policy import (
                    BASIS,
                    ETF_BASIS,
                    INDEX_BASES,
                    MARKET_BASES,
                    REVIEWED_BASES,
                )
                from app.services.fund_rating_public_validation import activate_public

                if args.fund_code not in {None, "002112", "016008"} and args.basis not in {
                    ETF_BASIS,
                    *MARKET_BASES,
                    *INDEX_BASES,
                    *REVIEWED_BASES,
                }:
                    parser.error("市场资料需显式 --basis，并通过真实八维原件验证")

                result = {
                    "methodology_id": activate_public(
                        session,
                        json.loads(args.file.read_text(encoding="utf-8")),
                        args.reason,
                        basis=args.basis or (ETF_BASIS if args.fund_code == "016008" else BASIS),
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
                elif args.from_batch:
                    rollback_switch(session, args.from_batch, args.batch, args.reason)
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
