"""001021本A官方单位净值和分红再投重放；旧四/五项版本原样委托。"""

import copy
import re
from datetime import datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from bs4 import BeautifulSoup

from app.services import fund_rating_bond_index_holdings_partial as previous
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_metrics import total_return

VERSION = "BOND_INDEX_FIVE_PARTIAL_OFFICIAL_NAV_REINVEST_20261111_V1"
PROOF_VERSION = "001021_A_OFFICIAL_UNIT_NAV_AND_REINVEST_NOTICE_V1"
NAV_URL = "https://www.chinaamc.com/product/fundLishijingzhi.do"
HISTORY_URL = "https://www.chinaamc.com/product/fundLishifenhong.do?fundcode=001021"
IDENTITY_URL = "https://www.chinaamc.com/fund/001021/lishifenhong.shtml?source=click"
NOTICE_URL = "https://www.chinaamc.com/upload/resources/file/2025/12/17/7f4430ff29714e90a111803bc5af48a4.pdf"
NOTICE_HASH = "72436c26fb479d95f84de500150019a123e93eb1810b871be400ab1e10bea6e6"
HISTORY_HASH = "9701848c232b356db03f8c4b7dbd367ebacdb16a2eacc554bb86e4d1602a0eda"
IDENTITY_HASH = "02ccc495d3940fa575d1d1b9ace34f132a8c5b4aafd3517c152148776a10cb5f"
NAV_HASHES = (
    "76982cfff3c5a6047c85ef55e890eb73d125dc15e0bc00733a83751b1ff729fd",
    "9da96ab2d1aae45909cf9d7813bb2166ea6365a5e834abeaca529ffc14b69156",
    "b5094823db0ce966b219e240ec75fe635f70a112f74fe4b698124334dccad4fb",
    "04e6e1caba1cc9c8a5a51640b64a932f85c3c1335dd107c43a3b9c10c8e2c3fd",
    "bc7081c04f43ad622f668b7b01d276aae92d4e3ed894c5509409a88a6cfb071a",
    "9c01184448c8c61e887f5f4cfdb4feb93dab4e0851945056b8add1cb39385976",
    "d77bfe20206bc7ed1bfa67c3e96fd9a81d6f9ef235aca867c6163ebe941c2a19",
    "81f6fb5cf4df720d81953dfd404e1b9ec1cf6bc4820a8dc25b7302a036ca77c6",
    "db289bd3fd9a8fc029e60dad83b94252caa942ec054febad3973509b56f4b66e",
    "3b86252f3607f5d6e0b5b1b4e7c7551d003b5387edadb7538e9badb56677957f",
    "698b1db2190b07574be8860243bb6166ef641cc80ec8d1820a9661723832bebe",
    "bb61d6bfa49b633fde9858219ffd3f468afc541d95ec8e877b4e80f625ea5433",
    "160b585409bf7c9e2c4abb45088c0ba91358b3ea9730220e788e3d84860176e1",
)
EVENT = {"ex_date": "2025-12-19", "reinvestment_nav_date": "2025-12-19", "cash_per_share": "0.020"}


def require(ok, code):
    if not ok:
        raise ValueError("001021_OFFICIAL_NAV_" + code)


def _nav_originals(inputs, root):
    """测试可将新增原件映射到隔离根；正式重放仍只读统一来源目录。"""
    return Originals(
        {"sources": inputs["nav_reconstruction"]["sources"], "as_of_date": inputs["as_of_date"]}, root, lazy=True
    )


def _ref(source, sha, *, url, role, kind, published=None):
    ref = source.sources[sha]
    require(ref["sha256"] == sha and ref["url"] == url and ref["kind"] == kind, "SOURCE_IDENTITY")
    require(ref.get("source_date_role") == role and ref.get("published_on") == published, "SOURCE_DATE_ROLE")
    return ref


def _html(source, sha):
    return BeautifulSoup(source.load(sha), "html.parser")


def _rows(soup):
    result = []
    for tr in soup.select("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
        if cells and re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]):
            result.append(cells)
    return result


def rebuild(inputs, *, root):
    proof = inputs["nav_reconstruction"]
    require(proof["version"] == PROOF_VERSION and proof["event"] == EVENT, "PROOF_VERSION_OR_EVENT")
    require(proof["fund_code"] == "001021" and proof["share_class"] == "A" and proof["currency"] == "CNY", "SHARE")
    require(proof["window"] == ["2025-09-30", "2026-09-30"], "WINDOW")
    require(proof["nav_pages"] == [{"page": i, "source_hash": h} for i, h in enumerate(NAV_HASHES, 1)], "PAGE_MANIFEST")
    require(set(proof["sources"]) == {*NAV_HASHES, NOTICE_HASH, HISTORY_HASH, IDENTITY_HASH}, "SOURCE_MANIFEST")
    checked = datetime.fromisoformat(proof["sources_checked_at"])
    require(checked.tzinfo is not None, "SOURCE_REVIEW_TIME")
    for ref in proof["sources"].values():
        acquired = datetime.fromisoformat(ref["acquired_at"])
        require(acquired.tzinfo is not None and acquired <= checked, "SOURCE_ACQUISITION_TIME")
    source = _nav_originals(inputs, Path(root))
    _ref(source, IDENTITY_HASH, url=IDENTITY_URL, role="OFFICIAL_A_DIVIDEND_PAGE_IDENTITY", kind="html")
    identity = _html(source, IDENTITY_HASH)
    require("华夏亚债中国指数A" in identity.title.get_text() and "001021" in identity.get_text(), "OFFICIAL_A_IDENTITY")
    require(
        any(f.get("src") == "/product/fundLishifenhong.do?fundcode=001021" for f in identity.select("iframe")),
        "HISTORY_IFRAME",
    )
    _ref(source, NOTICE_HASH, url=NOTICE_URL, role="ORIGINAL_DIVIDEND_NOTICE", kind="pdf", published="2025-12-17")
    notice = normalized_text(source.plain(NOTICE_HASH))
    require(
        all(
            t in notice
            for t in [
                "亚债中国债券指数基金",
                "华夏基金管理有限公司",
                "001021001023",
                "元/10份基金份额",
                "0.2000.238",
                "除息日2025年12月19日",
                "确定日为2025年12月19日",
                "2025年12月22日直接计入",
                "公告送出日期:2025年12月17日",
            ]
        ),
        "ORIGINAL_DIVIDEND_TERMS",
    )
    _ref(source, HISTORY_HASH, url=HISTORY_URL, role="OBSERVED_OFFICIAL_A_EVENT_HISTORY", kind="html")
    history = _html(source, HISTORY_HASH)
    require(
        [t.get_text(" ", strip=True) for t in history.select("th")]
        == ["年份", "权益登记日", "每份分红（元）", "分红发放日"],
        "EVENT_HEADERS",
    )
    event_rows = [r for r in _rows(history)]  # 此表首列为年份，另按真实第二列日期解析。
    require(not event_rows, "EVENT_TABLE_SCHEMA")
    event_rows = [[td.get_text(" ", strip=True) for td in tr.select("td")] for tr in history.select("tr")]
    event_rows = [r for r in event_rows if len(r) == 4 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r[1])]
    require(
        len(event_rows) == 8 and event_rows[0] == ["2025", "2025-12-19", "0.0200", "2025-12-22"],
        "EVENT_HISTORY_COVERAGE",
    )
    window_events = {r[1]: D(r[2]) for r in event_rows if inputs["as_of_date"] >= r[1] > "2025-09-30"}
    require(window_events == {EVENT["ex_date"]: D(EVENT["cash_per_share"])}, "EVENT_HISTORY_MISMATCH")

    official, order = {}, []
    for page, sha in enumerate(NAV_HASHES, 1):
        ref = _ref(source, sha, url=NAV_URL, role="HISTORICAL_DAILY_NAV_BUSINESS_DATE_PER_ROW", kind="html")
        # 官网首次查询true；翻页必须留空，否则服务器会重置到第一页。
        form = {
            "fundcode": "001021",
            "beginDate": "2025-09-30",
            "endDate": "2026-09-30",
            "isQuery": "true" if page == 1 else "",
            "pageIndex": str(page),
        }
        require(ref.get("method") == "POST_READ_ONLY_HISTORY_QUERY" and ref.get("form") == form, "PAGE_REQUEST")
        soup = _html(source, sha)
        for key in ["fundcode", "beginDate", "endDate"]:
            field = soup.select_one(f"input[name={key}]")
            require(field is not None and field.get("value") == form[key], "PAGE_QUERY_IDENTITY")
        current = soup.select("a.cur")
        require(len(current) == 1 and current[0].get_text(strip=True) == str(page), "PAGE_SELECTED")
        page_numbers = [
            int(m[1]) for a in soup.select("a[onclick]") if (m := re.fullmatch(r"thisPage\((\d+)\)", a["onclick"]))
        ]
        require(max(page_numbers) == 13, "PAGE_TOTAL")
        require(
            [h.get_text(" ", strip=True) for h in soup.select("th")]
            == ["净值日期", "单位净值", "累计净值", "资产净值"],
            "NAV_HEADERS",
        )
        rows = _rows(soup)
        require(len(rows) == (20 if page < 13 else 2), "PAGE_ROW_COUNT")
        for row in rows:
            require(len(row) == 4 and all(re.fullmatch(r"\d+\.\d{4}", v) for v in row[1:3]), "NAV_ROW")
            day = row[0]
            require(day not in official, "DUPLICATE_DATE")
            official[day] = {"unit_nav": D(row[1]), "accumulated_nav": D(row[2]), "source_hash": sha}
            order.append(day)
    old_source = Originals(inputs, Path(root), lazy=True)
    base = previous.previous.base
    days, ends, old_values = base._nav(inputs, old_source)
    require(len(days) == 242 and order == list(reversed(days)), "COMPLETE_REAL_CALENDAR")
    provider = old_source.json(inputs["refs"]["provider"])
    rows = {r["nav_date"]: r for r in provider["history"]["rows"]}
    for day in days:
        require(
            all(official[day][k] == D(rows[day][k]) for k in ["unit_nav", "accumulated_nav"]),
            "PROVIDER_UNIT_CROSS_CHECK",
        )
        # 历史净值按业务估值日取窗；ann_date可能是后续公开/更新日，原样保留而不伪造或抹早。
        require(day <= inputs["as_of_date"], "NAV_AFTER_VALUATION_CUTOFF")
    events = [r for r in provider["dividend_records"] if days[0] < r["ex_date"] <= days[-1]]
    require(
        len(events) == 1
        and events[0]["ex_date"] == EVENT["ex_date"]
        and D(events[0]["cash_dividend"]) == D(".020")
        and events[0]["process_status"] == "实施",
        "PROVIDER_EVENT_CROSS_CHECK",
    )
    half = normalized_text("\n".join(old_source.pages(inputs["primary_ref"])))
    require("本基金本报告期未进行利润分配" in half, "HALF_YEAR_NO_DISTRIBUTION")
    for a, b in zip(days[:-1], days[1:], strict=True):
        increment = (official[b]["accumulated_nav"] - official[b]["unit_nav"]) - (
            official[a]["accumulated_nav"] - official[a]["unit_nav"]
        )
        require(increment == window_events.get(b, D(0)), "CUMULATIVE_CASH_CONSERVATION")
    units = [official[d]["unit_nav"] for d in days]
    require(all(v > 0 for v in units), "NONPOSITIVE_NAV")
    cash = [window_events.get(d, D(0)) for d in days]
    values = total_return(units, cash, [D(1)] * len(days))
    return (
        days,
        ends,
        values,
        {
            "method": "OFFICIAL_UNIT_NAV_ORIGINAL_NOTICE_CASH_REINVESTMENT_EX_DATE",
            "official_unit_and_accumulated_rows_checked": 242,
            "nav_page_count": 13,
            "cash_events": {d: str(v) for d, v in window_events.items()},
            "reinvestment_nav_date": "2025-12-19",
            "reinvestment_unit_nav": str(official["2025-12-19"]["unit_nav"]),
            "arrival_date_not_reinvestment_date": "2025-12-22",
            "original_provider_source_hash": inputs["refs"]["provider"],
            "original_provider_adjusted_nav_used_or_rewritten": False,
            "source_reviewed_at": proof["sources_checked_at"],
            "nav_date_role": "HISTORICAL_DAILY_NAV_BUSINESS_DATE_PER_ROW",
            "availability_note": "按业务估值日取窗，保留真实采集和复核时间；原ann_date不冒充估值日或盘前已知日。",
            "source_hashes": sorted(source.raw),
            "late_q3_report_used": False,
            "old_adjusted_annual_return_for_audit_only": str(old_values[-1] / old_values[0] - 1),
            "official_reinvest_annual_return": str(values[-1] / values[0] - 1),
        },
    )


def calculate(inputs, *, root: Path):
    if inputs.get("version") in {previous.VERSION, previous.OLD_VERSION}:
        return previous.calculate(inputs, root=root)
    require(inputs.get("version") == VERSION, "VERSION")
    old = copy.deepcopy(inputs)
    old["version"] = previous.VERSION
    result = previous.calculate(old, root=root)
    result["dimensions"]["risk"] = None  # 新证据失败不能回退到已知不等价的旧复权点。
    result["facts"].pop("risk", None)
    with localcontext() as ctx:
        ctx.prec = 65
        try:
            days, ends, values, reconstruction = rebuild(inputs, root=root)
            peak, drawdown = values[0], D(0)
            peak_day, pair = days[0], None
            for day, value in zip(days, values, strict=True):
                if value > peak:
                    peak, peak_day = value, day
                loss = 1 - value / peak
                if loss > drawdown:
                    drawdown, pair = loss, {"peak_date": peak_day, "trough_date": day}
            result["dimensions"]["risk"] = str(100 * (1 - drawdown))
            result["facts"]["risk"] = {
                "maximum_drawdown": str(drawdown),
                "first_nav": str(values[0]),
                "last_nav": str(values[-1]),
                "fund_total_return": reconstruction["official_reinvest_annual_return"],
                "source_kind": reconstruction["method"],
                "peak_trough_dates": pair,
                "scope": "本A官方单位净值及公告再投重建的一年已实现最大回撤；不代表全部债券风险已核。",
            }
            result["facts"]["window"] = {
                "from": ends[0],
                "through": ends[-1],
                "month_ends": ends,
                "valuation_points": len(days),
                "nav_basis": reconstruction["method"],
            }
            result["facts"]["nav_reconstruction"] = reconstruction
            result["facts"]["source_sha256"] = sorted(
                set(result["facts"]["source_sha256"]) | set(reconstruction["source_hashes"])
            )
        except (ValueError, KeyError, TypeError) as exc:
            result["facts"]["evidence_errors"]["risk"] = str(exc)
    result["facts"]["unscored_dimensions"].update(
        {
            "return": "准确同口径首尾基准端点不足，连续收益得分不唯一",
            "efficiency": "缺准确基准日线，跟踪误差未知",
            "stability": "12个月同口径完整比较未闭合",
        }
    )
    result["facts"]["compiler_version"] = VERSION
    result["facts"]["complete_type_validated"] = False
    return result
