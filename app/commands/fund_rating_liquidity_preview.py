"""从现有正式评分包读取少量基金，输出候选对照 JSON。"""

import argparse
import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.services.fund_market_preparation import validate_codes
from app.services.fund_market_review import rating_review_required
from app.services.fund_rating_liquidity_preview import preview
from app.services.fund_rating_pipeline import current_inputs
from app.services.fund_rating_reviewed_snapshot import archive_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codes", nargs="+", help="要核对的六位基金代码")
    parser.add_argument("--mask-payment-check", action="store_true", help="用已有证据附做付款项遮蔽对照")
    args = parser.parse_args()
    try:
        codes = validate_codes(args.codes)
    except ValueError:
        parser.error("基金代码须为六位数字")
    if len(codes) > 5:
        parser.error("一次试算最多五只基金")
    with Session(get_nav_preview_engine()) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        current = current_inputs(session, list(codes))

    root = archive_root()
    results = []
    for code in codes:
        row = current.get(code)
        if row is None:
            results.append({"fund_code": code, "status": "NO_CURRENT_INPUT"})
            continue
        bundle = next(member["bundle"] for member in row["input_snapshot"]["members"] if member["fund_code"] == code)
        if bundle.get("fund_code") != code or bundle.get("as_of_date") != str(row["as_of_date"]):
            raise ValueError("RATING_LIQUIDITY_PREVIEW_CURRENT_BINDING")
        result = preview(
            bundle, root=root, event_review=rating_review_required(code), mask_payment_check=args.mask_payment_check
        )
        results.append({"fund_code": code, "rating_ref": row["batch_id"], **result})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
