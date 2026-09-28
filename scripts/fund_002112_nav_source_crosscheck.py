"""只读核对已冻结的四个历史区间；不写数据库、不修订原实验、不训练。"""

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[1] / ".local-runs/fund-exposure-002112"
OUT = ROOT / "peer-mechanism-review/20260928-v1"
SNAPSHOT = ROOT / "training-ready/sources/98fb1c51f2582687a09ac0c404d514c2ef74520e4cb63a9a9f89a300ef150611.json"
WINDOWS = [
    ["2023-09-18", "2023-09-28"],
    ["2023-12-14", "2023-12-22"],
    ["2024-02-05", "2024-02-08"],
    ["2024-05-20", "2024-05-23"],
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    """独占创建，已占用的请求号不能再发请求或覆盖收据。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def parse_rows(provider, data, start, end):
    """验证页完整性、基金及日期范围；不接受缺页、重复日或非正净值。"""
    if provider == "dbfund":
        raw = data["dataList"]
        if int(data["totalCount"]) != len(raw) or int(data["totalPage"]) != 1:
            raise ValueError("INCOMPLETE_OFFICIAL_PAGE")
        rows = []
        for item in raw:
            if item["fundcode"] != "002112":
                raise ValueError("WRONG_FUND")
            rows.append({"date": item["date"], "nav": str(item["netvalue"])})
    else:
        raw = data["Data"]["LSJZList"]
        if int(data["TotalCount"]) != len(raw) or int(data["PageIndex"]) != 1:
            raise ValueError("INCOMPLETE_SECONDARY_PAGE")
        rows = [{"date": item["FSRQ"], "nav": str(item["DWJZ"])} for item in raw]
    if not rows or len({r["date"] for r in rows}) != len(rows):
        raise ValueError("EMPTY_OR_DUPLICATE_DATES")
    for row in rows:
        if not start <= row["date"] <= end or not row["date"] < "2025-01-01":
            raise ValueError("OUTSIDE_FROZEN_WINDOW")
        value = Decimal(row["nav"])
        if not value.is_finite() or value <= 0:
            raise ValueError("INVALID_NAV")
    return sorted(rows, key=lambda row: row["date"])


def run():
    protocol = read(OUT / "source-check-protocol.json")
    if protocol["windows"] != WINDOWS or protocol["fund"] != "002112" or protocol["maximum_public_requests"] != 8:
        raise ValueError("SOURCE_SCOPE_CHANGED")
    sources = OUT / "public-sources"
    sources.mkdir(exist_ok=True)
    save(
        sources / "code-freeze.json",
        {
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256((OUT / "source-check-protocol.json").read_bytes()).hexdigest(),
        },
    )
    receipts = []
    # 无重试、无跟随跳转；先登记再请求，失败同样占用一次读取额度。
    with httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False) as client:
        for provider in ("dbfund", "eastmoney"):
            for start, end in WINDOWS:
                number = len(receipts) + 1
                if provider == "dbfund":
                    url = "https://www.dbfund.com.cn/common-web/chart/fundnettable/getFundNetTableJson"
                    params = {"fundcode": "002112", "from": start, "to": end, "pages": "1-100"}
                    referer = "https://www.dbfund.com.cn/products/hunhe/002112/index.html"
                else:
                    url = "https://api.fund.eastmoney.com/f10/lsjz"
                    params = {"fundCode": "002112", "pageIndex": 1, "pageSize": 20, "startDate": start, "endDate": end}
                    referer = "https://fundf10.eastmoney.com/"
                receipt = {
                    "number": number,
                    "provider": provider,
                    "window": [start, end],
                    "url": url,
                    "params": params,
                    "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                }
                save(sources / f"{number:02d}-request.json", receipt)
                try:
                    response = client.get(url, params=params, headers={"User-Agent": "Mozilla/5.0", "Referer": referer})
                    content = response.content
                    receipt.update(
                        status_code=response.status_code,
                        bytes=len(content),
                        response_sha256=hashlib.sha256(content).hexdigest(),
                    )
                    if len(content) > 5_000_000:
                        raise ValueError("RESPONSE_TOO_LARGE")
                    with (sources / f"{number:02d}-response.json").open("xb") as stream:
                        stream.write(content)
                    response.raise_for_status()
                    receipt["rows"] = parse_rows(provider, response.json(), start, end)
                    receipt["valid"] = True
                except Exception as error:
                    receipt.update(valid=False, error_type=type(error).__name__, error=str(error))
                save(sources / f"{number:02d}-receipt.json", receipt)
                receipts.append(receipt)
                print(json.dumps({k: receipt.get(k) for k in ("number", "provider", "valid", "error")}), flush=True)
    snapshot = read(SNAPSHOT)["payload"]
    frozen = {r["date"]: r for r in snapshot["funds"]["002112"]["nav"]["rows"]}
    comparisons = []
    for receipt in receipts:
        for row in receipt.get("rows", []):
            old = frozen.get(row["date"])
            comparisons.append(
                {
                    "provider": receipt["provider"],
                    "date": row["date"],
                    "current_public_nav": row["nav"],
                    "frozen_nav": old["nav"] if old else None,
                    "equal": Decimal(row["nav"]) == Decimal(old["nav"]) if old else None,
                    "receipt_number": receipt["number"],
                }
            )
    result = {
        "requests": len(receipts),
        "valid_requests": sum(r["valid"] for r in receipts),
        "comparisons": comparisons,
        "disagreements": [r for r in comparisons if r["equal"] is False],
        "interpretation": (
            "Current public values only; no proof of first publication or permission to rewrite frozen values"
        ),
        "actual_new_fits": 0,
    }
    save(OUT / "source-crosscheck.json", result)
    print(
        json.dumps(
            {
                "requests": result["requests"],
                "valid_requests": result["valid_requests"],
                "compared_rows": len(comparisons),
                "disagreements": len(result["disagreements"]),
            }
        )
    )


if __name__ == "__main__":
    run()
