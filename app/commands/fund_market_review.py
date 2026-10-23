"""本机资料审阅命令；不会联网、预测、修改权重或发布评级。"""

import argparse
import json
from pathlib import Path

from app.services.fund_market_review import approve_comprehensive, record_event_review, record_prediction_event_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("event", "prediction-event", "comprehensive", "inspect"))
    parser.add_argument("--fund-code", required=True)
    parser.add_argument("--note-file", type=Path)
    parser.add_argument("--materials", type=Path, help="本基金补充资料的真实路径；不接受其他基金资料")
    parser.add_argument("--resolutions", type=Path, help="逐项缺口处置及原文证据，仅用于预测，不更改评级准入")
    parser.add_argument("--article")
    parser.add_argument("--source-hash")
    parser.add_argument("--decision", choices=("NO_RATING_IMPACT", "REQUIRES_RATING_REVIEW", "RATING_REVIEW_COMPLETED"))
    parser.add_argument("--rating-batch")
    parser.add_argument("--evidence-file", type=Path, help="本次预测事项审阅引用的连续原文片段JSON数组")
    args = parser.parse_args()
    if args.action != "inspect" and args.note_file is None:
        parser.error("公告或综合审阅必须提供 --note-file")
    note = args.note_file.read_text(encoding="utf-8") if args.note_file else ""
    resolutions = json.loads(args.resolutions.read_text(encoding="utf-8")) if args.resolutions else None
    if args.action == "event":
        if not all((args.article, args.source_hash, args.decision)):
            parser.error("公告复核必须指定公告编号、原文摘要和结论")
        result = record_event_review(
            args.fund_code, args.article, args.source_hash, args.decision, note, rating_batch_id=args.rating_batch
        )
    elif args.action == "prediction-event":
        if not all((args.article, args.source_hash, args.evidence_file)):
            parser.error("预测事项审阅必须指定公告编号、原文摘要和 --evidence-file")
        quotes = json.loads(args.evidence_file.read_text(encoding="utf-8"))
        result = record_prediction_event_review(args.fund_code, args.article, args.source_hash, note, quotes)
    elif args.action == "inspect":
        from app.services.fund_workflow_materials import inspect_materials

        if args.materials is None:
            parser.error("资料交接检查必须指定 --materials")
        result = inspect_materials(args.fund_code, args.materials, resolutions=resolutions)
    else:
        result = approve_comprehensive(args.fund_code, note, materials_path=args.materials, resolutions=resolutions)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
