"""人工缺证结论单独留存，绑定当前评级批次，不改动正式评分。"""

from datetime import date
from hashlib import file_digest
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_file_access import physical_path

logger = get_logger(__name__)
INCOMPLETE_STATUSES = {"PARTIAL", "NOT_RATED"}


def _path(rating: dict) -> Path:
    return (
        Path(get_settings().fund_insight_directory) / "ratings" / "completion-reviews"
        / rating["fundCode"] / f"{rating.get('ratingRef') or 'unrated'}.json"
    )


def _identity(rating: dict) -> dict:
    return {key: rating.get(key) for key in ("fundCode", "ratingRef", "asOfDate")}


def incomplete_reviewed_on(rating: dict) -> str | None:
    if rating["status"] not in INCOMPLETE_STATUSES:
        return None
    try:
        path = _path(rating)
        if not physical_path(path).is_file():
            return None
        review = read(path)
        reviewed_on = date.fromisoformat(review["reviewedOn"])
        if review["rating"] == _identity(rating) and reviewed_on <= now().date():
            return reviewed_on.isoformat()
    except FileNotFoundError:
        pass  # 已有评级兼容无人工回执的情况。
    except (OSError, KeyError, TypeError, ValueError):
        logger.warning("fund_rating_completion_review >>> unreadable review, fund_code=%s", rating["fundCode"])
    return None


def record_incomplete_review(code: str, batch: str | None, receipt: Path, reason: str) -> dict:
    """只从本机人工命令调用。已有部分分或未评分才可登记，并留存核查回执身份。"""
    from app.services.fund_rating import read as read_rating

    rating = read_rating([code])["items"][0]
    if rating["status"] not in INCOMPLETE_STATUSES or rating["ratingRef"] != batch:
        raise ValueError("当前评级状态或批次与核查结论不匹配")
    reason = reason.strip()
    receipt = receipt.resolve(strict=True)
    if not reason or not receipt.is_file() or receipt.stat().st_size == 0:
        raise ValueError("必须提供非空的核查结论和回执")
    with receipt.open("rb") as source:
        sha256 = file_digest(source, "sha256").hexdigest()
    review = {
        "rating": _identity(rating),
        "reviewedOn": now().date().isoformat(),
        "reason": reason,
        "receipt": {"path": str(receipt), "sha256": sha256},
    }
    versioned_save(_path(rating), review)
    return review
