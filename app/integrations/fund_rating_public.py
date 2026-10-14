"""评级试算的公开国债曲线采集；仅显式维护命令调用，不购权、不自动登记生产授权。"""

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx

from app.services.fund_exposure_common import read, save

URL = "https://yield.chinabond.com.cn/cbweb-czb-web/czb/historyQuery"


def collect_risk_free(start: date, end: date, root: Path, *, client=None) -> dict:
    """使用官网历史查询，不超过其单次一年范围；保存真实回执，失败保留旧完整指针。

    3月、1年国债到期收益率是两种待验证的无风险代理，不能冒称真实隔夜现金收益。
    网站公开可读不代表取得了再发布或指数开发授权；评级正式使用另经来源门禁。
    """
    if not start <= end or (end - start).days > 1200 or end > datetime.now(UTC).date():
        raise ValueError("RATING_RISK_FREE_SCOPE")
    if client is None:
        with httpx.Client(timeout=httpx.Timeout(15, connect=5), follow_redirects=False) as owned:
            return collect_risk_free(start, end, root, client=owned)
    receipts, rows = [], {}
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=364))
        params = {"startDate": str(cursor), "endDate": str(stop), "gjqx": "0.25,1", "locale": "cn_ZH", "qxmc": "1"}
        with client.stream("POST", URL, params=params) as response:
            response.raise_for_status()
            raw = bytearray()
            for part in response.iter_bytes():
                raw.extend(part)
                if len(raw) > 2_000_000:
                    raise ValueError("RATING_RISK_FREE_RESPONSE_LIMIT")
        raw = bytes(raw)
        key = hashlib.sha256(raw).hexdigest()
        path = root / "raw" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != key:
                raise ValueError("RATING_RISK_FREE_HASH_MISMATCH")
        else:
            from app.services.fund_exposure_common import _publish

            _publish(path, raw)
        receipt = {
            "url": URL,
            "params": params,
            "sha256": key,
            "file": path.relative_to(root).as_posix(),
            "received_at": datetime.now(UTC).isoformat(),
            "source_code": "CHINABOND_PUBLIC_RESEARCH",
        }
        save(
            root / "receipts" / f"{hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()}.json",
            receipt,
        )
        value = json.loads(raw)
        if value.get("flag") != "0" or not value.get("heList"):
            raise ValueError("RATING_RISK_FREE_SOURCE_EMPTY")
        for item in value["heList"]:
            day = date.fromisoformat(item["workTime"])
            if not cursor <= day <= stop or str(day) in rows:
                raise ValueError("RATING_RISK_FREE_SOURCE_DATES")
            yields = {k: Decimal(item[k]) for k in ("threeMonth", "oneYear")}
            if any(not v.is_finite() or not -10 <= v <= 100 for v in yields.values()):
                raise ValueError("RATING_RISK_FREE_SOURCE_RANGE")
            rows[str(day)] = {k: str(v) for k, v in yields.items()}
        receipts.append(receipt)
        cursor = stop + timedelta(days=1)
    result = {
        "start": str(start),
        "end": str(end),
        "currency": "CNY",
        "unit": "PERCENT_PER_YEAR",
        "usage": "RESEARCH_ONLY_PENDING_SOURCE_REVIEW",
        "rows": dict(sorted(rows.items())),
        "receipts": receipts,
    }
    from app.services.fund_rating_inputs import digest

    archive = root / "versions" / f"{digest(result)}.json"
    if not archive.exists():
        save(archive, result)
    save(root / "risk-free.json", result, replace=True)
    return result


def risk_free_intervals(snapshot: dict, days: list[date], *, term: str) -> list[Decimal]:
    """候选口径：使用区间起点的国债年收益率按实际天数/365计息，绝不回填未来利率。

    只允许日历原值逐日匹配。节假日由前一估值日起息，缺失交易日不无限向前填补。
    """
    if term not in {"threeMonth", "oneYear"} or snapshot["currency"] != "CNY" or snapshot["unit"] != "PERCENT_PER_YEAR":
        raise ValueError("RATING_RISK_FREE_BASIS")
    if days != sorted(set(days)) or len(days) < 2:
        raise ValueError("RATING_RISK_FREE_DATES")
    result = []
    for a, b in zip(days, days[1:], strict=False):
        row = snapshot["rows"].get(str(a))
        if row is None:
            raise ValueError("RATING_RISK_FREE_DATE_MISSING:" + str(a))
        yield_rate = Decimal(row[term]) / 100
        if not yield_rate.is_finite() or yield_rate <= -1:
            raise ValueError("RATING_RISK_FREE_SOURCE_RANGE")
        result.append((1 + yield_rate) ** (Decimal((b - a).days) / 365) - 1)
    return result


def read_risk_free(root: Path, frozen: datetime) -> dict | None:
    path = root / "risk-free.json"
    if not path.exists():
        return None
    result = read(path)
    from app.services.fund_rating_materials import _raw

    for receipt in result["receipts"]:
        _raw(root, receipt, frozen)
    return result
