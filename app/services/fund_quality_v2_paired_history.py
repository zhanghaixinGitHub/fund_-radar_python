"""独立于V1八维准入，重建同份额净值与发行人合同基准的成对历史。

只接纳已有适配的两家发行人产品日曲线，不以通用指数、净值涨幅或
API成功标志代替合同基准。调用者负责来源字节、授权与合同原文绑定。
"""

import json
from datetime import date
from decimal import Decimal, localcontext
from urllib.parse import parse_qs, urlsplit

from app.services.fund_rating_rules import decimal


def benchmark_records(raw: bytes, ref: dict, *, fund_code: str) -> tuple[dict, str]:
    """重读两家既有发行人响应的请求身份、业务码、日期及原始收益字段。

    返回按原始自然日索引的记录和适配器名称。此步骤仅证明响应结构与
    产品请求对应；没有净值、合同和完整估值日对齐时，不生成质量指标。
    """
    address = urlsplit(ref["url"])
    if address.scheme != "https":
        raise ValueError("QUALITY_PAIRED_BENCHMARK_HTTPS_REQUIRED")
    data = json.loads(raw, parse_float=Decimal)
    params = ref.get("params", {})
    if (address.hostname, address.path) == ("www.wjasset.com", "/common-web/ets40/yieldListQueryAPI"):
        if (
            ref.get("format") != "ISSUER_DAILY_BENCHMARK_WJ"
            or params.get("fundCode") != fund_code
            or params.get("incomeType") != "8"
            or data.get("code") != "ETS-5BP0000"
            or not isinstance(data.get("results"), list)
        ):
            raise ValueError("QUALITY_PAIRED_WJ_PRODUCT_REQUEST_OR_RESPONSE")
        records = [
            {"day": row.get("tradeDate"), "benchmark": row.get("benchmarkRatio"), "fund": row.get("returnRatio")}
            for row in data["results"]
        ]
        adapter = "WJ_ISSUER_PRODUCT_DAILY_BENCHMARK"
    elif (address.hostname, address.path) == ("www.fullgoal.com.cn", "/ws-business-server/fund/getIncomeList"):
        query = parse_qs(address.query, keep_blank_values=True)
        if (
            params.get("productCode") != fund_code
            or query.get("productCode", [fund_code]) != [fund_code]
            or data.get("code") != 0
            or not isinstance(data.get("data"), list)
            or any(row.get("productCode") != fund_code for row in data["data"])
        ):
            raise ValueError("QUALITY_PAIRED_FULLGOAL_PRODUCT_REQUEST_OR_RESPONSE")
        records = [
            {"day": row.get("navDate"), "benchmark": row.get("benchmarkRate"), "fund": row.get("syNavRate")}
            for row in data["data"]
        ]
        adapter = "FULLGOAL_ISSUER_PRODUCT_DAILY_BENCHMARK"
    else:
        raise ValueError("QUALITY_PAIRED_BENCHMARK_ADAPTER_UNSUPPORTED")
    if not records or len(records) > 10000:
        raise ValueError("QUALITY_PAIRED_BENCHMARK_SIZE")
    by_day = {}
    for record in records:
        day = str(date.fromisoformat(record["day"]))
        if day in by_day:
            raise ValueError("QUALITY_PAIRED_DUPLICATE_BENCHMARK_DAY")
        by_day[day] = record
    return by_day, adapter


def paired_history(curve: dict, raw: bytes, ref: dict, *, fund_code: str) -> dict:
    """逐日匹配37个月末净值与合同基准，并用发行人基金曲线交叉核验。

    输入curve须来自已核验的本份额净值覆盖，不允许缩窗或补值。返回曲线
    与舍入差异证据；缺日、空基准、身份冲突或两份基金曲线不符均拒绝。
    """
    by_day, adapter = benchmark_records(raw, ref, fund_code=fund_code)
    days, ends = curve["dates"], curve["month_ends"]
    if days != sorted(set(days)) or len(ends) != 37 or ends[0] != days[0] or ends[-1] != days[-1]:
        raise ValueError("QUALITY_PAIRED_OWN_CURVE_WINDOW")
    missing = [day for day in days if day not in by_day or by_day[day]["benchmark"] in (None, "")]
    if missing:
        raise ValueError("QUALITY_PAIRED_BENCHMARK_DATES_MISSING:" + ",".join(missing))
    if any(by_day[day]["fund"] in (None, "") for day in days):
        raise ValueError("QUALITY_PAIRED_PUBLISHER_FUND_CURVE_MISSING")
    with localcontext() as context:
        context.prec = 40
        benchmark = [1 + decimal(by_day[day]["benchmark"]) / 100 for day in days]
        publisher = [1 + decimal(by_day[day]["fund"]) / 100 for day in days]
        nav = list(map(decimal, curve["adjusted_nav"]))
        if len(nav) != len(days) or any(value <= 0 for value in benchmark + publisher + nav):
            raise ValueError("QUALITY_PAIRED_CURVE_LENGTH_OR_POSITIVITY")
        # 沿用已验证发行人适配的公开舍入上界，不因历史结果不符而放宽。
        bound = Decimal(".0001") / min(nav) + Decimal(".000001") / min(publisher)
        difference = max(abs(a / nav[0] - b / publisher[0]) for a, b in zip(nav, publisher, strict=True))
        if difference > bound:
            raise ValueError("QUALITY_PAIRED_PROVIDER_PUBLISHER_NAV_CONFLICT")
        return {
            "dates": list(days), "month_ends": list(ends), "adjusted_nav": list(curve["adjusted_nav"]),
            "benchmark_index": list(map(str, benchmark)),
            "publisher_cross_check": {
                "adapter": adapter, "maximum_normalized_difference": str(difference),
                "rounding_bound": str(bound), "source_hashes": [ref["sha256"]],
                "scope": "ACTUAL_ISSUER_CONTRACT_BENCHMARK_NOT_GENERIC_INDEX",
            },
        }
