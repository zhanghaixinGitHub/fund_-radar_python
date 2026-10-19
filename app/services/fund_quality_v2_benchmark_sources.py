"""复用发行人逐日合同基准HTTP原响应；不采集、不发布、不用成功标志证明曲线齐全。"""

import hashlib
import json
import re
from datetime import datetime
from decimal import InvalidOperation
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app.services.fund_exposure_common import _publish
from app.services.fund_quality_v2_paired_history import benchmark_records
from app.services.fund_rating_rules import decimal


def reuse_benchmark_originals(code: str, source_root: Path, rating_root: Path, frozen: datetime) -> list[dict]:
    """核原始收据、实际请求URL、字节和产品响应后复制到隔离档案。

    清单中的fund_code只用于路由。只接纳富国/万家已有公开GET日基准端点，
    实际URL查询参数必须绑定本份额；原件位于来源目录raw内，取得时间不能
    晚于冻结时刻。返回原始引用，不补published_on、不声称历史首次公布；
    本份额净值、同期合同、完整窗口和舍入差异由成对历史链路另行重建。
    """
    source_root = source_root.resolve()
    manifest = source_root / "quality-benchmark-originals.json"
    if not manifest.exists():
        return []
    if not re.fullmatch(r"[0-9]{6}", code) or manifest.stat().st_size > 1_000_000:
        raise ValueError("QUALITY_BENCHMARK_MANIFEST_CODE_OR_SIZE")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data = data.get("payload", data)
    if (data.get("schema") != "QUALITY_BENCHMARK_SOURCE_RECEIPTS_V1"
            or not isinstance(data.get("items"), list) or len(data["items"]) > 200
            or any(not isinstance(item, dict) for item in data["items"])):
        raise ValueError("QUALITY_BENCHMARK_MANIFEST_SCHEMA")
    result = []
    for item in data["items"]:
        if item.get("fund_code") != code:
            continue
        receipt = item["receipt"]
        if not isinstance(receipt, dict) or not isinstance(receipt.get("params"), dict):
            raise ValueError("QUALITY_BENCHMARK_RECEIPT_SCHEMA")
        address = urlsplit(receipt["url"])
        if (item.get("source_kind") != "ISSUER_PRODUCT_DAILY_BENCHMARK"
                or receipt.get("method") != "GET" or receipt.get("status") != 200
                or address.scheme != "https" or address.port not in {None, 443}
                or address.username or address.password or address.fragment):
            raise ValueError("QUALITY_BENCHMARK_RECEIPT_OR_HTTPS")
        query = parse_qs(address.query, keep_blank_values=True)
        if (address.hostname, address.path) == ("www.fullgoal.com.cn", "/ws-business-server/fund/getIncomeList"):
            if query.get("productCode") != [code]:
                raise ValueError("QUALITY_BENCHMARK_REQUEST_IDENTITY")
        elif (address.hostname, address.path) == ("www.wjasset.com", "/common-web/ets40/yieldListQueryAPI"):
            if query.get("fundCode") != [code] or query.get("incomeType") != ["8"]:
                raise ValueError("QUALITY_BENCHMARK_REQUEST_IDENTITY")
        else:
            raise ValueError("QUALITY_BENCHMARK_SOURCE_ENDPOINT_UNSUPPORTED")
        acquired = datetime.fromisoformat(receipt["received_at"])
        path = (source_root / receipt["file"]).resolve()
        if (acquired.tzinfo is None or acquired > frozen or not path.is_relative_to(source_root / "raw")
                or not path.is_file() or path.stat().st_size > 2_000_000):
            raise ValueError("QUALITY_BENCHMARK_TIME_PATH_OR_SIZE")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != receipt["sha256"]:
            raise ValueError("QUALITY_BENCHMARK_ORIGINAL_HASH")
        ref = {"label": "benchmark", "url": receipt["url"], "method": "GET",
               "params": receipt["params"], "sha256": sha,
               **({"format": item["format"]} if item.get("format") else {})}
        try:
            records, adapter = benchmark_records(raw, ref, fund_code=code)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("QUALITY_BENCHMARK_RESPONSE_SCHEMA") from exc
        # 原响应有空单元格不补值，后续逐窗口对齐会拒收缺失日；已披露的
        # 数字须有限且累计因子为正，不能把HTML、NaN或占位文字当真实收益。
        populated = 0
        for row in records.values():
            for key in ("fund", "benchmark"):
                if row[key] not in (None, ""):
                    try:
                        value = decimal(row[key])
                    except InvalidOperation as exc:
                        raise ValueError("QUALITY_BENCHMARK_NUMBER_INVALID") from exc
                    if value <= -100:
                        raise ValueError("QUALITY_BENCHMARK_NONPOSITIVE_FACTOR")
            populated += all(row[key] not in (None, "") for key in ("fund", "benchmark"))
        if not populated:
            raise ValueError("QUALITY_BENCHMARK_NO_OBSERVED_PAIR")
        target = rating_root / "sources" / (sha + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("QUALITY_BENCHMARK_DESTINATION_CONFLICT")
        else:
            _publish(target, raw)
        result.append({**ref, "file": target.relative_to(rating_root).as_posix(),
            "acquired_at": receipt["received_at"], "title": item["title"],
            "source_kind": "ISSUER_DAILY_BENCHMARK_HTTP_ORIGINAL", "adapter": adapter,
            "origin_receipt_file": str(manifest),
            "time_scope": "ACTUAL_RECEIPT_NOT_HISTORICAL_FIRST_PUBLICATION_OR_REVISION_PROOF"})
    return result
