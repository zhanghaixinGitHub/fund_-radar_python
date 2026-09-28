"""独立准备的有界公开补证：仅两只参考基金历史净值和报告目录，不调用付费接口。"""

from datetime import datetime
from urllib.parse import urlparse

import httpx
from app.services.fund_002112_zero_fit_review import ROOT, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-training-ready/20260928-v1"


def fetch(name, url, *, params=None, form=None):
    """保存开始记录、原始响应与完成回执；失败占额度且不自动重试、不覆盖。

    日期在发请求前限定，避免拉取全历史后再过滤封存年份。最多 80 次请求，
    每次 5 秒连接/25 秒读取、5 MB 上限，无登录、Cookie 注入和跳转绕过。
    """
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
        raise ValueError("INVALID_REQUEST_NAME")
    if urlparse(url).scheme != "https":
        raise ValueError("HTTPS_REQUIRED")
    if url == "https://api.fund.eastmoney.com/f10/lsjz":
        if form or params["fundCode"] not in {"017493", "160323"}:
            raise ValueError("NAV_CODE_SCOPE")
        if not "2021-01-01" <= params["startDate"] <= params["endDate"] <= "2023-12-31":
            raise ValueError("NAV_DATE_SCOPE")
        if not 1 <= params["pageSize"] <= 1000 or not 1 <= params["pageIndex"] <= 40:
            raise ValueError("NAV_PAGE_SCOPE")
    elif url == "https://www.cninfo.com.cn/new/hisAnnouncement/query":
        if params or form["searchkey"] not in {"华夏磐泰", "东方红新动力"}:
            raise ValueError("REPORT_FUND_SCOPE")
        if form["seDate"] != "2020-01-01~2024-12-31" or form["pageSize"] != 30:
            raise ValueError("REPORT_DATE_SCOPE")
        if not 1 <= form["pageNum"] <= 20:
            raise ValueError("REPORT_PAGE_SCOPE")
    elif url == "https://www.dfham.com/common-web/cms/content/getContents":
        if form or params["categoryId"] not in {"924f6d20a7d747df8941d0684f7c86fc", "f76f1a8e80e049e09e6b3d643e11535d"}:
            raise ValueError("ISSUER_CATEGORY_SCOPE")
        if params["pageSize"] not in {30, 100} or not 1 <= params["pageNumber"] <= 10:
            raise ValueError("ISSUER_PAGE_SCOPE")
    else:
        raise ValueError("PUBLIC_ENDPOINT_NOT_ALLOWED")
    folder = OUTPUT / "public-sources"
    folder.mkdir(exist_ok=True)
    reserve, finish = folder / (name + "-start.json"), folder / (name + "-receipt.json")
    spec = {"url": url, "params": params, "form": form}
    if reserve.exists():
        old = read_json(reserve)
        if any(old[k] != spec[k] for k in spec):
            raise ValueError("REQUEST_SPEC_CHANGED")
        result = read_json(finish) if finish.exists() else old
        if result.get("path") and file_hash(result["path"]) != result["sha256"]:
            raise ValueError("RAW_SOURCE_CHANGED")
        return result
    if len(list(folder.glob("*-start.json"))) >= 80:
        raise ValueError("PUBLIC_REQUEST_LIMIT")
    result = {**spec, "started_at": datetime.now().astimezone().isoformat(), "status": "RESERVED"}
    save_once(reserve, result)
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"} if params else {}
        if urlparse(url).hostname == "www.dfham.com":
            headers = {"Referer": "https://www.dfham.com/product/jijin/hunhe/piangu/000480/notice/index.html"}
        with httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False, headers=headers) as client:
            with client.stream("POST" if form else "GET", url, params=params, data=form) as response:
                result["http_status"] = response.status_code
                response.raise_for_status()
                raw = bytearray()
                for piece in response.iter_bytes():
                    raw.extend(piece)
                    if len(raw) > 5_000_000:
                        raise ValueError("RESPONSE_TOO_LARGE")
        path = folder / (name + "-raw.json")
        with path.open("xb") as stream:
            stream.write(raw)
        result.update(status="RECEIVED", path=str(path), sha256=file_hash(path), bytes=len(raw))
    except (httpx.HTTPError, ValueError) as exc:
        result.update(status="UNRESOLVED", reason=type(exc).__name__)
    result["finished_at"] = datetime.now().astimezone().isoformat()
    save_once(finish, result)
    return result


def nav(code):
    """按冻结净值依赖日期范围做免费交叉核对，不能用新来源覆盖原净值和公告日。"""
    return fetch(
        "nav-" + code,
        "https://api.fund.eastmoney.com/f10/lsjz",
        params={
            "fundCode": code,
            "pageIndex": 1,
            "pageSize": 1000,
            "startDate": "2022-12-05" if code == "017493" else "2021-01-18",
            "endDate": "2023-12-29",
        },
    )


def catalog(code, page):
    """只补公告目录用于报告身份/修订核查，不下载或扩展新闻训练资料。"""
    return fetch(
        f"catalog-{code}-{page}",
        "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        form={
            "column": "fund",
            "tabName": "fulltext",
            "searchkey": "东方红新动力" if code == "017493" else "华夏磐泰",
            "pageNum": page,
            "pageSize": 30,
            "seDate": "2020-01-01~2024-12-31",
            "isHLtitle": "true",
            "category": "",
            "stock": "",
            "secid": "",
            "plate": "",
            "trade": "",
            "sortName": "",
            "sortType": "",
        },
    )
