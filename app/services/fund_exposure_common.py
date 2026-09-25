"""002112 持仓数据试点的固定范围、文件留存与证据校验。"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.services.direction_1d_protocol import ZONE, canonical, digest

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / ".local-runs" / "fund-exposure-002112"
FUND = "002112"
MASTER = "001412"


def now() -> datetime:
    """运行时间使用真实北京时间，历史数据不得改写为过去已采集。"""
    return datetime.now(ZONE)


def save(path: Path, value, *, replace: bool = False) -> str:
    """JSON 同时保存内容哈希；默认只创建，运行状态才允许原子替换。"""
    raw = canonical({"hash": digest(value), "payload": value}).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, raw, replace=replace)
    return digest(value)


def _publish(path: Path, raw: bytes, *, replace=False):
    """先写完临时文件再公开名称；并行验收不会读到半份 JSON，首次创建仍禁止覆盖。"""
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
        if replace:
            os.replace(temporary, path)
        else:
            # NTFS/POSIX 硬链接都要求目标不存在；不会覆盖已有不可变证据。
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path: Path):
    """读取前核对哈希，拒绝损坏或被静默改写的证据。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"hash", "payload"} or digest(value["payload"]) != value["hash"]:
        raise ValueError("EXPOSURE_EVIDENCE_HASH_MISMATCH")
    return value["payload"]


def blob(raw: bytes, suffix: str) -> tuple[str, str]:
    """原始公开资料按内容寻址保存，名称不包含网页输入或凭据。"""
    if suffix not in {"pdf", "json", "html"}:
        raise ValueError("EXPOSURE_SUFFIX_INVALID")
    key = hashlib.sha256(raw).hexdigest()
    path = ROOT / "raw" / f"{key}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != key:
            raise ValueError("EXPOSURE_RAW_HASH_MISMATCH")
    else:
        try:
            _publish(path, raw)
        except FileExistsError:
            # 不同公告可能引用相同原文，并发写入内容相同的文件可以直接复用。
            if hashlib.sha256(path.read_bytes()).hexdigest() != key:
                raise ValueError("EXPOSURE_RAW_HASH_MISMATCH") from None
    return key, path.relative_to(ROOT).as_posix()


def initialize() -> dict:
    """采集和评分之前冻结预算及三类样本门槛，不自动购权、不读 2025 标签。"""
    path = ROOT / "plan.json"
    if path.exists():
        return read(path)
    plan = {
        "version": "FUND_EXPOSURE_002112_V1",
        "fund_code": FUND,
        "fund_master_code": MASTER,
        "created_at": now().isoformat(),
        "report_year_start": 2020,
        "historical_feature_start": "2021-01-01",
        "development_end": "2024-12-31",
        "fit_end": "2023-12-31",
        "check_start": "2024-01-01",
        "protected_label_years": [2025],
        "reconstructed_2026_labels_allowed": False,
        "target": "UNIT_NAV_DIRECTION_THREE_STATE_V2",
        "minimum_fit_dates": 252,
        "minimum_class_dates": 30,
        "candidates": ["NAV7", "NAV7_HOLDINGS", "NAV7_HOLDINGS_MARKET"],
        "maximum_fits": 6,
        "new_purchase_cny": 0,
        "maximum_report_pages": 8,
        "maximum_reports": 80,
        "maximum_provider_requests_per_command": 300,
        "maximum_retries_per_request": 1,
        "historical_report_availability": "report_declared_publication_next_calendar_day_0800_Asia_Shanghai",
        "historical_quote_availability": "next_trading_day_0800_Asia_Shanghai_reconstruction",
        "holdings_selection": "latest_report_end_then_latest_available_disclosure_never_renormalize_partial",
        "max_report_age_days": 210,
        "minimum_quote_weight_coverage": 1.0,
        "automatic_model_adoption": False,
    }
    save(path, plan)
    return plan
