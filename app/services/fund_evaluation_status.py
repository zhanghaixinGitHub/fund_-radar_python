"""历史综合评价的资料状态；规则与资料未通过时只公开缺口，绝不拼凑总分。"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.services.direction_1d_protocol import ZONE
from app.services.fund_exposure_common import read


@lru_cache(maxsize=2)
def _coverage(path: str, modified_ns: int):
    """按不可变审计文件缓存并校验摘要，页面批量读取不会重算或发起来源请求。"""
    value = read(Path(path))
    checked = datetime.fromisoformat(value["at"])
    if checked.tzinfo is None or checked > datetime.now(ZONE) or len(value["funds"]) > 5000:
        raise ValueError("EVALUATION_COVERAGE_INVALID")
    codes = [r["fund_code"] for r in value["funds"]]
    if len(set(codes)) != len(codes):
        raise ValueError("EVALUATION_COVERAGE_DUPLICATE")
    return value


def batch(codes: list[str], *, root: Path | None = None) -> dict:
    """一页最多 100 只，一次读取同一审计批次；不发布未经定稿规则计算的分数或排名。"""
    if not 1 <= len(codes) <= 100 or any(len(c) != 6 or not c.isascii() or not c.isdigit() for c in codes):
        raise ValueError("EVALUATION_SCOPE_INVALID")
    folder = (root or Path(get_settings().fund_insight_directory)) / "coverage"
    files = sorted(folder.glob("????????-??????.json"))
    evidence = _coverage(str(files[-1].resolve()), files[-1].stat().st_mtime_ns) if files else None
    rows = {} if evidence is None else {r["fund_code"]: r for r in evidence["funds"]}
    items = []
    for code in dict.fromkeys(codes):
        row = rows.get(code)
        reasons = ["该基金的历史评价资料尚未核对完整。"]
        if row:
            reasons = ["历史收益中的分红处理尚未核对完整。", "经理经验、资产规模及同类划分的资料仍不充分。"]
            category = next(
                (
                    r
                    for r in evidence["source_categories"]
                    if r["type"] == row["fund_type"] and r["invest_type"] == row["invest_type"]
                ),
                None,
            )
            if category and category["share_count"] < 30:
                reasons.insert(0, "已核对范围内的同类资料不足，暂不计算同类排名。")
        items.append(
            {
                "fundCode": code,
                "score": None,
                "rank": None,
                "scoreDate": None,
                "coverageCheckedAt": evidence["at"] if row else None,
                "navAsOfDate": row.get("last_nav_date") if row else None,
                "message": "资料暂不完整，暂不评分。",
                "reasons": reasons,
                "note": "历史评价用于同类比较，不代表未来收益；暂不评分不等于低分。",
            }
        )
    return {"items": items}
