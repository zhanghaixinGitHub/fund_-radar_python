"""境内直接复制及ETF联接指数C份额的R5原件编译器。

复用完整股票、真实份额净值、同指数参照和逐日成交原件。股票复制按已冻结
R5的股票内部口径；全资产余额用于支付、受限范围，不能把旧全资产行业距离
误当股票复制的前置条件。本模块不联网、不写库、不接收维度分或总分。
"""

import json
import re
import unicodedata
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_accounting import accounting_observations
from app.services.fund_quality_v2_asset_structure import capco_publication, gross_asset_book
from app.services.fund_quality_v2_balance_proofs import identify_nonnegative_parts, strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_stock_book import periodic_financial_view
from app.services.fund_rating_direct_index_math import capacity, performance, replication, share_flow
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_management_inputs import _source_text
from app.services.fund_rating_management_inputs import calculate as management_calculate
from app.services.fund_rating_market_review import fee_profile, normalized_text
from app.services.fund_rating_passive_portfolio import portfolio as complete_portfolio
from app.services.fund_rating_public_market import provider_history
from app.services.fund_rating_public_stock_tables import parse
from app.services.fund_rating_r5 import number
from app.services.material_content import content_digest, source_bytes, source_exists
from app.services.material_content import document_reader as PdfReader

VERSION = "R5_DIRECT_INDEX_ORIGINALS_20261028_1"
FEEDER_VERSION = "R5_FEEDER_INDEX_ORIGINALS_20261028_1"
LIABILITIES = (
    "短期借款",
    "交易性金融负债",
    "衍生金融负债",
    "卖出回购金融资产款",
    "应付清算款",
    "应付赎回款",
    "应付管理人报酬",
    "应付托管费",
    "应付销售服务费",
    "应付投资顾问费",
    "应交税费",
    "应付利润",
    "递延所得税负债",
    "其他负债",
)


def physical_page_prefix(pages: list[str]) -> list[str]:
    """核实整份原PDF的物理页序后去页首页码，不删证券数字或任意独行数。

    富国完整PDF的封面无页码，其余每页抽取文本均先输出物理页号；
    多页股票表连接时它会混入上一只股票。显式版式标记只在至少三页
    正文且每个正文页号均等于物理位置时可用，任何一页不符就拒绝。
    """
    if len(pages) < 4:
        raise ValueError("R5_ORIGINAL_PDF_PHYSICAL_PAGE_PREFIX_UNVERIFIED")
    result = [pages[0]]
    for index, page in enumerate(pages[1:], 2):
        match = re.match(r"^\s*(\d{1,4})[ \t]*\r?\n", page)
        if not match or int(match[1]) != index:
            raise ValueError("R5_ORIGINAL_PDF_PHYSICAL_PAGE_PREFIX_UNVERIFIED")
        result.append(page[match.end() :])
    return result


class Originals:
    """每次复算重新核原字节；路径限正式来源目录，摘要不可当成外部事实。"""

    def __init__(self, inputs: dict, root: Path, *, lazy: bool = False):
        self.root = root.resolve()
        self.sources = inputs["sources"]
        self.raw = {}
        self.pages_cache = {}
        self.plain_cache = {}
        self.as_of_date = inputs["as_of_date"]
        if not lazy:
            for digest in self.sources:
                self.load(digest)

    def load(self, digest: str) -> bytes:
        if digest not in self.raw:
            ref = self.sources[digest]
            path = (self.root / ref["file"]).resolve()
            if not path.is_relative_to(self.root / "sources") or not source_exists(path):
                raise ValueError("R5_DIRECT_INDEX_SOURCE_SCOPE")
            raw = source_bytes(path)
            if len(raw) > 50_000_000 or content_digest(raw) != digest or ref["sha256"] != digest:
                raise ValueError("R5_DIRECT_INDEX_SOURCE_HASH")
            if ref.get("published_on") and ref["published_on"] > self.as_of_date:
                raise ValueError("R5_DIRECT_INDEX_FUTURE_DOCUMENT")
            self.raw[digest] = raw
        return self.raw[digest]

    def json(self, digest: str):
        return json.loads(self.load(digest))

    def pages(self, digest: str) -> list[str]:
        if digest not in self.pages_cache:
            ref = self.sources[digest]
            if ref["file"].lower().endswith(".pdf"):
                self.pages_cache[digest] = [p.extract_text() or "" for p in PdfReader(self.load(digest)).pages]
                if ref.get("folio_prefix") == "PHYSICAL_PAGE_ONE_BASED":
                    self.pages_cache[digest] = physical_page_prefix(self.pages_cache[digest])
            else:
                body = self.json(digest)["data"]["notice_content"]
                self.pages_cache[digest] = [BeautifulSoup(body, "html.parser").get_text("\n", strip=True)]
        return self.pages_cache[digest]

    def plain(self, digest: str) -> str:
        if digest not in self.plain_cache:
            self.load(digest)
            self.plain_cache[digest] = _source_text(self.root, self.sources[digest])
        return self.plain_cache[digest]

    def claim(self, claim: dict) -> None:
        quote = normalized_text(claim["quote"])
        if not quote or quote not in self.plain(claim["source_hash"]):
            raise ValueError("R5_DIRECT_INDEX_QUOTE_MISMATCH")


def _table(raw: dict) -> list[dict]:
    if raw.get("code") != 0 or not isinstance(raw.get("data", {}).get("items"), list):
        raise ValueError("R5_DIRECT_INDEX_PROVIDER_RESPONSE")
    fields = raw["data"]["fields"]
    return [dict(zip(fields, row, strict=True)) for row in raw["data"]["items"]]


def _contract(source: Originals, digest: str, code: str, index_name: str) -> str:
    """核每期实际合同描述；现金利率措辞原样留存，共同曲线只作同参照评价。"""
    plain = normalized_text("\n".join(source.pages(digest)))
    if code not in plain[:15000] or index_name + "指数" not in plain:
        raise ValueError("R5_DIRECT_INDEX_CONTRACT_IDENTITY:" + code)
    matches = []
    for match in re.finditer("业绩比较基准", plain):
        block = plain[match.end() : match.end() + 200]
        if all(term in block for term in [index_name + "指数", "95%", "5%", "活期存款", "税后"]):
            matches.append(block.split("风险收益特征")[0])
    if not matches:
        raise ValueError("R5_DIRECT_INDEX_CONTRACT_95_5_NOT_VERIFIED:" + code)
    return matches[0]


def contract_history_refs(
    source, annuals: dict, alternatives: dict, start: str, asof: str, member: str, frozen: dict
) -> dict:
    """同产品原合同须覆盖整个三年窗口；首年年报失败才准用已冻结的完整季度。

    季度替代只核同一年度本窗口全部日期的实际合同，不缩短净值/持仓窗口，
    不借目标ETF年报。失败响应、所属基金公告目录和连续覆盖证明均须原件核准。
    """
    from app.integrations.public_fund_reports import report_period as catalog_period

    first = date.fromisoformat(start).year
    needed = {str(y) for y in range(first, date.fromisoformat(asof).year)}
    if (
        set(annuals) & set(alternatives)
        or set(annuals) | set(alternatives) != needed
        or len(set(annuals.values())) != len(annuals)
    ):
        raise ValueError("R5_DIRECT_INDEX_CONTRACT_HISTORY_REQUIRED:" + member)
    result = dict(annuals)
    for year, digest in annuals.items():
        period = report_period("\n".join(source.pages(digest)))
        needed_start = max(year + "-01-01", start)
        if str(period[1]) != year + "-12-31" or period[0].year != int(year) or str(period[0]) > needed_start:
            raise ValueError("R5_DIRECT_INDEX_CONTRACT_ANNUAL_PERIOD:" + member)
    for year, alternative in alternatives.items():
        if year != str(first) or not frozen.get("first_window_annual_alternative"):
            raise ValueError("R5_CONTRACT_QUARTER_FIRST_YEAR_FROZEN_RULE_REQUIRED")
        failure, catalog = alternative["failure_source"], alternative["catalog_source"]
        ref = source.sources[failure]
        directory = source.sources[catalog]
        raw = source.json(catalog)
        entries = [r for r in raw.get("Data", []) if r["ID"] == ref.get("params", {}).get("art_code")]
        if (
            ref.get("url") != "https://np-cnotice-fund.eastmoney.com/api/content/ann"
            or directory.get("url") != "https://api.fund.eastmoney.com/f10/JJGG"
            or directory.get("params", {}).get("fundcode") != member
            or raw.get("ErrCode") != 0
            or len(entries) != 1
            or catalog_period(entries[0]["TITLE"]) != (year + "-12-31", "ANNUAL")
            or source.json(failure).get("data", {}).get("notice_content")
        ):
            raise ValueError("R5_CONTRACT_ANNUAL_FAILURE_ORIGINAL_REQUIRED")
        quarters = alternative["quarter_refs"]
        if not 1 <= len(quarters) <= 4 or len(set(quarters)) != len(quarters):
            raise ValueError("R5_CONTRACT_QUARTER_REFS_INVALID")
        periods = []
        for digest in quarters:
            if source.sources[digest].get("archive_kind") == "DERIVED":
                raise ValueError("R5_CONTRACT_QUARTER_ORIGINAL_REQUIRED")
            period = report_period("\n".join(source.pages(digest)))
            quarter = (period[0].month - 1) // 3 + 1
            ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            if (
                period[0].year != first
                or period[0].day != 1
                or period[0].month not in {1, 4, 7, 10}
                or str(period[1]) != year + "-" + ends[quarter]
            ):
                raise ValueError("R5_CONTRACT_QUARTER_PERIOD_INVALID")
            periods.append((*period, digest))
        cursor, end = date.fromisoformat(max(start, year + "-01-01")), date.fromisoformat(year + "-12-31")
        for begin, stop, digest in sorted(periods):
            if begin > cursor or stop < cursor:
                raise ValueError("R5_CONTRACT_QUARTERS_DO_NOT_COVER_WINDOW")
            cursor = stop + timedelta(days=1)
            result[str(stop)] = digest
        if cursor <= end:
            raise ValueError("R5_CONTRACT_QUARTERS_DO_NOT_COVER_WINDOW")
    return result


def th_own_nav_check(rows: list[dict], observations: dict, adjusted: dict, units: dict, common: list[str]) -> dict:
    """本份额单位净值仍逐值相同；复权净值只按原官网明示四位小数核舍入。

    有历史分红的份额，本地八位复权值可能与官网四位原值不同。不能要求
    舍入文本逐字相等，也不能用任意容差或缩放掩盖另一条净值曲线。
    """
    published = {str(date.fromisoformat(r["fundDate"])): r for r in rows}
    largest, rounded = Decimal(0), 0
    for day in common:
        row = observations[day]
        if row["unit_nav"] != units[day]:
            raise ValueError("R5_DIRECT_INDEX_TH_OWN_NAV_CONFLICT")
        difference = abs(row["publisher_nav"] - adjusted[day])
        largest = max(largest, difference)
        if difference:
            literal = published[day].get("fundAdjustedNav")
            if (
                not isinstance(literal, str)
                or not re.fullmatch(r"\d+\.\d{4}", literal)
                or difference > Decimal(".00005")
            ):
                raise ValueError("R5_DIRECT_INDEX_TH_OWN_NAV_CONFLICT")
            rounded += 1
    return {
        "own_share_observations_checked": len(common),
        "four_decimal_adjusted_rows_checked": rounded,
        "maximum_adjusted_nav_difference": str(largest),
        "publisher_adjusted_nav_half_quantum": ".00005",
    }


def th_publisher_adjusted_nav(observations: dict, adjusted: dict, units: dict, common: list[str]) -> tuple[dict, dict]:
    """官网与供应商复权口径不同时，只以单位净值同值确认份额后采用官网复权值。"""
    values = dict(adjusted)
    for day in common:
        row = observations[day]
        if row["unit_nav"] != units[day] or row["publisher_nav"] <= 0:
            raise ValueError("R5_DIRECT_INDEX_TH_OWN_NAV_CONFLICT")
        values[day] = row["publisher_nav"]
    return values, {
        "mode": "ISSUER_PUBLISHED_ADJUSTED_NAV",
        "own_share_observations_checked": len(common),
        "provider_unit_nav_rows_checked": len(common),
        "provider_adjusted_nav_not_substituted_for_issuer_series": True,
    }


def _history(source: Originals, inputs: dict) -> dict:
    refs = inputs["target_refs"]
    provider = source.json(refs["provider"])
    if (
        provider.get("fund_code") != inputs["fund_code"]
        or provider.get("source", {}).get("enabled") is not True
        or not provider.get("source", {}).get("authorization_verified_at")
    ):
        raise ValueError("R5_DIRECT_INDEX_OWN_AUTHORIZED_HISTORY")
    _, history = provider_history(provider, date.fromisoformat(inputs["as_of_date"]))
    if source.sources[refs["benchmark"]].get("format") == "ISSUER_DAILY_BENCHMARK_SOUTHERN":
        from app.services.fund_rating_southern_benchmark import compile_history

        return compile_history(source, inputs, history)
    if source.sources[refs["benchmark"]].get("format") == "ISSUER_DAILY_BENCHMARK_GT":
        from app.services.fund_rating_gt_benchmark import compile_history

        return compile_history(source, inputs, history)
    if source.sources[refs["benchmark"]].get("format") == "ISSUER_DAILY_BENCHMARK_GF":
        from app.services.fund_rating_gf_benchmark import compile_history

        return compile_history(source, inputs, history)
    if source.sources[refs["benchmark"]].get("format") == "ISSUER_DAILY_BENCHMARK_CMF":
        from app.services.fund_rating_cmf_benchmark import compile_history

        return compile_history(source, inputs, history)
    if source.sources[refs["benchmark"]].get("format") == "ISSUER_DAILY_BENCHMARK_JS":
        from app.services.fund_rating_js_benchmark import compile_history

        return compile_history(source, inputs, history)
    raw = source.json(refs["benchmark"])
    if source.sources[refs["benchmark"]].get("format") == "ISSUER_DAILY_BENCHMARK_TH":
        from app.services.fund_rating_benchmark_bridge import reconstruct
        from app.services.fund_rating_public_etf import _benchmark_expression, _benchmark_series

        expression = _benchmark_expression(
            normalized_text("\n".join(source.pages(refs["summary"]) + source.pages(refs["prospectus"])))
        )
        # 原响应可能还带截止日以后的新净值，而当日基准尚未公布。只核实际
        # 评分窗，保留原字节不改写；窗内缺基准仍由适配器拒绝，不能用未来行。
        raw = deepcopy(raw)
        for key in ["fundHistoryRecord", "indexHistoryRecord"]:
            rows = raw["payload"][key]["fundDayRecordList"]
            raw["payload"][key]["fundDayRecordList"] = [
                r for r in rows if date.fromisoformat(r["fundDate"]).isoformat() in set(history["dates"])
            ]
        series = _benchmark_series(raw, source.sources[refs["benchmark"]], inputs["fund_code"], expression)
        observations = {r["navDate"]: r for r in series["rows"]}
        common = sorted(set(history["dates"]) & observations.keys())
        adjusted = dict(zip(history["dates"], map(number, history["adjusted_nav"]), strict=True))
        units = {r["nav_date"]: number(r["unit_nav"]) for r in history["rows"]}
        if inputs.get("use_publisher_adjusted_nav"):
            adjusted, own_check = th_publisher_adjusted_nav(observations, adjusted, units, common)
            history["adjusted_nav"] = [str(adjusted[day]) for day in history["dates"]]
        else:
            own_check = th_own_nav_check(
                raw["payload"]["fundHistoryRecord"]["fundDayRecordList"], observations, adjusted, units, common
            )
        if inputs.get("use_official_benchmark_levels"):
            history["benchmark_index"] = [
                str(observations[day]["benchmark_level"]) if day in observations else None for day in history["dates"]
            ]
            history["publisher_cross_check"] = {
                **own_check,
                "mode": "COMPLETE_ISSUER_PUBLISHED_BENCHMARK_LEVELS",
                "missing_publisher_dates": sorted(set(history["dates"]) - observations.keys()),
            }
            return history
        bridge = inputs["benchmark_reconstruction"]
        frozen = source.json(bridge["protocol_hash"])
        rate = bridge["annual_cash_rate"]
        if frozen["candidate_cash_rate"] != rate or frozen["frozen_at"] > inputs["verified_at"]:
            raise ValueError("R5_BRIDGE_PROTOCOL_CHANGED")
        rate_text = source.plain(bridge["rate_source"])
        if rate != ".0035" or "活期存款0.350.35" not in rate_text:
            raise ValueError("R5_BRIDGE_CASH_RATE_ORIGINAL_NOT_VERIFIED")
        prices = {}
        for row in _table(source.json(bridge["index_daily"])):
            day = date.fromisoformat(row["trade_date"]).isoformat()
            if row["ts_code"] != inputs["index"]["code"] or day in prices:
                raise ValueError("R5_BRIDGE_INDEX_IDENTITY_OR_DUPLICATE")
            if day in adjusted:
                prices[day] = row["close"]
        checked = reconstruct(
            history["dates"],
            prices,
            {d: observations[d]["benchmark_level"] for d in common},
            annual_cash_rate=rate,
            expected_missing_dates=bridge["expected_missing_dates"],
        )
        history["benchmark_index"] = checked.pop("levels")
        history["publisher_cross_check"] = {**checked, **own_check}
        return history
    rows = raw.get("data")
    if raw.get("code") != 0 or not isinstance(rows, list) or len({r["navDate"] for r in rows}) != len(rows):
        raise ValueError("R5_DIRECT_INDEX_ISSUER_BENCHMARK_RESPONSE")
    if any(r["productCode"] != inputs["fund_code"] for r in rows):
        raise ValueError("R5_DIRECT_INDEX_ISSUER_BENCHMARK_SHARE")
    by = {r["navDate"]: r for r in rows}
    missing = sorted(set(history["dates"]) - by.keys())
    allowed_missing = inputs.get("allow_missing_benchmark_before")
    if missing and (not allowed_missing or any(day >= allowed_missing for day in missing)):
        raise ValueError("R5_DIRECT_INDEX_BENCHMARK_MISSING_DATE")
    available = [day for day in history["dates"] if day in by]
    with localcontext() as context:
        context.prec = 70
        benchmark = [1 + number(by[d]["benchmarkRate"]) / 100 if d in by else None for d in history["dates"]]
        publisher = [1 + number(by[d]["syNavRate"]) / 100 for d in available]
        own = [number(v) for v in history["adjusted_nav"]]
        if min([v for v in benchmark if v is not None] + publisher + own) <= 0:
            raise ValueError("R5_DIRECT_INDEX_NONPOSITIVE_HISTORY")
        bound = Decimal(".0001") / min(own) + Decimal(".000001") / min(publisher)
        difference = max(
            abs(own[history["dates"].index(day)] / own[0] - (1 + number(by[day]["syNavRate"]) / 100) / publisher[0])
            for day in available
        )
        if difference > bound:
            raise ValueError("R5_DIRECT_INDEX_ISSUER_NAV_CONFLICT")
    history["benchmark_index"] = [str(value) if value is not None else None for value in benchmark]
    history["publisher_cross_check"] = {
        "maximum_difference": str(difference),
        "rounding_bound": str(bound),
        "missing_publisher_dates": missing,
    }
    return history


def _peer_nav(source: Originals, peer: dict, code: str, days: list[str], asof: str) -> list[str]:
    # 本适配器只接纳明确无分红、单位/累计/复权值一致的原响应；非零事件另走逐笔再投资适配。
    dividends = _table(source.json(peer["dividends"]))
    if dividends:
        raise ValueError("R5_DIRECT_INDEX_PEER_DIVIDEND_ADAPTER_REQUIRED:" + code)
    rows = _table(source.json(peer["nav"]))
    if not rows or len({r["nav_date"] for r in rows}) != len(rows):
        raise ValueError("R5_DIRECT_INDEX_PEER_DUPLICATE_NAV:" + code)
    mapped = {}
    for row in rows:
        day, announced = str(date.fromisoformat(row["nav_date"])), str(date.fromisoformat(row["ann_date"]))
        if row["ts_code"] != code + ".OF" or announced > asof:
            raise ValueError("R5_DIRECT_INDEX_PEER_IDENTITY_OR_FUTURE_NAV:" + code)
        values = [number(row[k]) for k in ["unit_nav", "accum_nav", "adj_nav"]]
        if min(values) <= 0 or len(set(values)) != 1:
            raise ValueError("R5_DIRECT_INDEX_PEER_ADJUSTMENT_NOT_CLOSED:" + code)
        mapped[day] = str(values[0])
    if not set(days) <= mapped.keys():
        raise ValueError("R5_DIRECT_INDEX_PEER_COMMON_DATES_MISSING:" + code)
    return [mapped[d] for d in days]


def _liabilities(pages: list[str], total: str) -> dict:
    """完整负债分项对账；应付清算款横杠仅在全表非负守恒唯一确定时识别。"""
    body = "\n".join(pages)
    start, stop = body.find("短期借款"), body.find("负债合计")
    if start < 0 or stop <= start:
        raise ValueError("R5_DIRECT_INDEX_LIABILITY_BODY")
    body = body[start:stop]
    parts, original = {}, {}
    slot = r"(?:[\d,]+\.\d{2}|[-－—])"
    for name in LIABILITIES:
        matches = re.findall(rf"(?m)^\s*{name}\s+(?:[567]\.4\.\d+(?:\.\d+)*\s+)?({slot})\s+({slot})\s*$", body)
        if len(matches) != 1:
            raise ValueError("R5_DIRECT_INDEX_LIABILITY_ROW:" + name)
        cell = matches[0][0]
        parts[name] = None if cell in "-－—" else cell.replace(",", "")
        original[name] = cell
    proof = identify_nonnegative_parts(total, parts)
    if proof["state"] != "POINT_IDENTIFIED":
        raise ValueError("R5_DIRECT_INDEX_LIABILITY_AMBIGUOUS")
    return {**proof, "original_cells": original}


def _flows(
    pages: list[str],
    *,
    allow_reversed: bool = False,
    allow_interleaved: bool = False,
    additional_shares: tuple[str, ...] = (),
    allow_period_variant: bool = False,
) -> tuple[list[dict], list[dict]]:
    additional_shares = tuple(additional_shares)
    raw_lines = "\n".join(pages).splitlines()
    if not additional_shares:
        # 个别原表A标签独占下一行，而C仍在第一行第二列。先用保留的
        # 水平位置把A接回第一列，不能把消空白后的prefixprefixCA误认成C/A反序。
        starts = [i for i, line in enumerate(raw_lines) if "开放式基金份额变动" in normalized_text(line)]
        if starts:
            for index in range(starts[-1] + 1, min(starts[-1] + 25, len(raw_lines) - 1)):
                line, following = raw_lines[index].expandtabs(), raw_lines[index + 1].expandtabs()
                header = re.fullmatch(r"\s*项目\s+(?P<prefix>\S(?:.*?\S)?)\s{2,}(?P=prefix)\s+C\s*", line)
                detached = re.fullmatch(r"(?P<indent> *)A *", following)
                if not header or not detached:
                    continue
                second_start = line.find(header["prefix"], header.end("prefix"))

                # 中文字宽与英文/空格不同。PDF保留的是显示列位置，不能拿
                # Python字符下标比较，否则正确居中的A会被判到第二列。
                def display_width(text):
                    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)

                first_column = display_width(line[: header.start("prefix")])
                second_column = display_width(line[:second_start])
                if not first_column <= len(detached["indent"]) < second_column:
                    raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_WRAPPED_CLASS_ALIGNMENT")
                raw_lines[index] = line[: header.end("prefix")] + "A" + line[header.end("prefix") :]
                raw_lines[index + 1] = ""
    if additional_shares:
        # 个别PDF把多列金额的最后一位小数放到下一行；只在完整份额行和
        # 纯数字续行之间按原顺序接回，后续仍须通过各份额的份额守恒。
        labels = (
            "本报告期期初基金份额总额",
            "本报告期基金总申购份额",
            "减:本报告期基金总赎回份额",
            "减：本报告期基金总赎回份额",
            "本报告期期末基金份额总额",
        )
        wrapped_labels = (
            ("本报告期期初", "基金份额总额"),
            ("本报告期基金", "总申购份额"),
            ("减:本报告期基", "金总赎回份额"),
            ("减：本报告期基", "金总赎回份额"),
            ("本报告期期末", "基金份额总额"),
        )
        for index, line in enumerate(raw_lines[:-1]):
            continuation = raw_lines[index + 1]
            standard = any(re.match(r"^\s*" + label + r"\s+", line) for label in labels)
            wrapped_suffix = next(
                (
                    suffix
                    for prefix, suffix in wrapped_labels
                    if re.match(r"^\s*" + re.escape(prefix) + r"\s+", line) and suffix in continuation
                ),
                None,
            )
            if not standard and wrapped_suffix is None:
                continue
            cells = re.split(r"\s{2,}", line.strip())
            digits = re.findall(r"(?<!\S)\d(?!\S)", continuation)
            missing = [
                column
                for column, value in enumerate(cells[1:], 1)
                if re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)\.\d", value)
            ]
            expected_cells = 3 + len(additional_shares)
            if (
                len(cells) == expected_cells
                and len(missing) == len(digits)
                and all(re.fullmatch(r"\d", x) for x in digits)
            ):
                for column, digit in zip(missing, digits, strict=True):
                    cells[column] += digit
                if wrapped_suffix is not None:
                    cells[0] += wrapped_suffix
                raw_lines[index] = "  ".join(cells)
                raw_lines[index + 1] = ""
    text = normalized_text("\n".join(raw_lines))
    start = text.rfind("开放式基金份额变动")
    end = text.find("重大事件揭示", start)
    if end < 0:
        alternatives = [text.find(term, start) for term in ["基金管理人运用固有资金", "备查文件目录"]]
        end = min((position for position in alternatives if position >= 0), default=-1)
    if start < 0 or end < 0:
        raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_BODY")
    section = text[start:end]
    names = {
        "opening": "本报告期期初基金份额总额",
        "subscribed": "本报告期基金总申购份额",
        "redeemed": "本报告期基金总赎回份额",
        "split": "本报告期基金拆分变动份额",
        "closing": "本报告期期末基金份额总额",
    }
    period_names = {
        "opening": "报告期期初基金份额总额",
        "subscribed": "报告期期间基金总申购份额",
        "redeemed": "减:报告期期间基金总赎回份额",
        "split": "报告期期间基金拆分变动份额",
        "closing": "报告期期末基金份额总额",
    }
    # 原表可能使用“天弘中证光伏A/C”等正式简称，不必带“指数”两字。
    # 必须先核同一简称的A、C列及次序，不能只按第二个金额猜C类。
    heading = (
        section.split("基金合同生效日", 1)[0]
        if "基金合同生效日" in section
        else section.split("报告期期初基金份额总额", 1)[0]
    )
    heading = re.sub(r"^开放式基金份额变动.*?单位[:：]份(?:项目)?", "", heading)
    if additional_shares:
        order = ["A", "C", *additional_shares]
        columns = None
        if additional_shares == ("E",):
            columns = re.fullmatch(r"(?P<prefix>.+?)A(?P=prefix)C(?P=prefix)E", heading)
        elif additional_shares == ("D",):
            # 三年持有产品的A/C/D原表：同名、列序和逐列份额对账都必须成立。
            columns = re.fullmatch(r"(?P<prefix>.+?)A(?P=prefix)C(?P=prefix)D", heading)
        elif additional_shares == ("I",):
            # 南方A/C/I原表同名前缀、后缀和列序逐项核准，仍逐列核申赎守恒。
            columns = re.fullmatch(r"(?P<prefix>.+?)A(?P=prefix)C(?P=prefix)I", heading) or re.fullmatch(
                r"(?P<prefix>.+?)(?P=prefix)(?P=prefix)(?P<suffix>.+?)A(?P=suffix)C(?P=suffix)I", heading
            )
        elif additional_shares == ("Y",):
            columns = re.fullmatch(r"(?P<prefix>.+?)A(?P=prefix)C(?P=prefix)Y", heading) or re.fullmatch(
                r"(?P<prefix>.+?)(?P=prefix)(?P=prefix)(?P<suffix>.*?)A(?P=suffix)C(?P=suffix)Y", heading
            )
            if not columns:
                split = re.fullmatch(
                    r"(?P<prefix>.+?)(?P=prefix)(?P<other>.+?)(?P<suffix>.*?)A(?P=suffix)C(?P<other_suffix>.*?)Y",
                    heading,
                )
                if split and split["prefix"] + split["suffix"] == split["other"] + split["other_suffix"]:
                    columns = split
        elif additional_shares == ("Y", "I"):
            columns = re.fullmatch(r"(?P<prefix>.+?)A(?P=prefix)C(?P=prefix)Y(?P=prefix)I", heading)
            if not columns:
                split = re.fullmatch(
                    r"(?P<prefix>.+?)(?P=prefix)(?P=prefix)(?P=prefix)(?:项目)?"
                    r"(?P<suffix>.+?)(?P=suffix)(?P=suffix)(?P=suffix)Y"
                    r"(?P<lof>\(LOF\)|（LOF）)A(?P=lof)C(?P=lof)I",
                    heading,
                )
                if split:
                    columns = split
        if not columns:
            raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_COLUMNS")
    else:
        order = ["A", "C"]
        columns = re.fullmatch(r"(.+?)A(.+?)C", heading)
        if not columns or columns[1] != columns[2]:
            # 广发原表把两列同名基金的前半行并排输出，后半行才分别带A/C：
            # “...ETF发 ...ETF发 / 项目 / 起联接A 起联接C”。仅认前缀两次
            # 完全相同、后缀两次完全相同且次序为A后C的完整表头，不依金额猜列。
            wrapped = re.fullmatch(r"(?P<prefix>.+?)(?P=prefix)(?:项目)?(?P<suffix>.+?)A(?P=suffix)C", heading)
            if not wrapped:
                reversed_columns = allow_reversed and re.fullmatch(r"(.+?)C(.*?)A", heading)
                adjacent_reversed = allow_reversed and re.fullmatch(r"(?P<prefix>.+?)(?P=prefix)CA", heading)
                if not adjacent_reversed and (not reversed_columns or reversed_columns[1] != reversed_columns[2]):
                    raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_COLUMNS")
                order = ["C", "A"]
    parsed = {share: {"share": share} for share in ["A", "C", *additional_shares]}
    slot = r"(?:\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2}|[-－—])"
    interleaved = {
        "opening": [
            ("本报告期期初基金", "份额总额"),
            ("本报告期期初基金份", "额总额"),
            ("本报告期期初", "基金份额总额"),
        ],
        "subscribed": [("本报告期基金总申", "购份额"), ("本报告期基金总申购", "份额"), ("本报告期基金", "总申购份额")],
        "redeemed": [
            ("减:本报告期基金", "总赎回份额"),
            ("减:本报告期基金总", "赎回份额"),
            ("减:本报告期基", "金总赎回份额"),
        ],
        "split": [("本报告期基金拆分", "变动份额"), ("本报告期基金拆分变", "动份额"), ("本报告期基金", "拆分变动份额")],
        "closing": [
            ("本报告期期末基金", "份额总额"),
            ("本报告期期末基金份", "额总额"),
            ("本报告期期末", "基金份额总额"),
        ],
    }
    value_pattern = rf"({slot})" * len(order)
    for field, label in names.items():
        labels = [label, period_names[field]] if allow_period_variant else [label]
        match = None
        for candidate in labels:
            if allow_period_variant and field == "split":
                match = re.search(candidate + r"\(份额" + value_pattern + "减少以", section)
                if not match:
                    match = re.search(candidate + r"\(份额减少以.{0,12}填列\)" + value_pattern, section)
            if not match:
                match = re.search(candidate + value_pattern, section)
            if match:
                break
        if not match and allow_interleaved:
            for prefix, suffix in interleaved[field]:
                match = re.search(re.escape(prefix) + value_pattern + re.escape(suffix), section)
                if match:
                    break
        if not match:
            raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_ROW:" + field)
        for share, value in zip(order, match.groups(), strict=True):
            parsed[share][field] = None if value in "-－—" else value.replace(",", "")
    rows = list(parsed.values())
    return rows, share_flow(rows)


def _mappings(source: Originals, inputs: dict, securities: set[str]) -> tuple[dict, dict]:
    registry = source.json(inputs["issuer_registry"])["stockList"]
    issuers = {r["code"]: r["orgId"] for r in registry}
    from app.services.fund_rating_security_disclosures import verified_securities

    identities = verified_securities(source, inputs)
    issuers.update({short: value["issuer"] for short, value in identities.items()})
    classification = inputs["classification"]
    article = capco_publication(
        source.raw[classification["article"]], attachment_url=source.sources[classification["pdf"]]["url"]
    )
    if article["period_end"] != inputs["holding_date"] or article["published_on"] > inputs["as_of_date"]:
        raise ValueError("R5_DIRECT_INDEX_CLASSIFICATION_PERIOD")
    # 只提取需要的真实行。保留细行业C39/J66，不以门类C/J替代；派生登记行再核原PDF。
    pdf_plain = normalized_text("\n".join(source.pages(classification["pdf"])))
    industries = {}
    for sec in sorted(securities):
        short = sec[:6]
        if short not in issuers or not issuers[short]:
            raise ValueError("R5_DIRECT_INDEX_ISSUER_UNMAPPED:" + sec)
        row = classification["rows"].get(short)
        if short in identities:
            # 退市证券不在当期上市公司目录时，必须有同期资料明示的相同
            # 两位细行业码；不能沿用旧上市行业或退回行业门类。
            identity = identities[short]
            # 分类原表把门类K、门类名称及大类70分列，正文并不存在连写K70。
            # 必须找到同一当期原行同时核准代码与名称，不能用字面拼接命中代替。
            taxonomy_rows = []
            for candidate in classification["rows"].values():
                raw_row = normalized_text(candidate["raw_row"])
                if (
                    candidate["industry_code"] != identity["industry_code"]
                    or candidate.get("industry_name") != identity["industry_name"]
                    or raw_row not in pdf_plain
                    or not raw_row.startswith(candidate["security"])
                ):
                    continue
                after = raw_row.split(candidate["section_code"], 1)[1]
                first_number = re.search(r"\d{2}", after)
                if first_number and candidate["section_code"] + first_number[0] == identity["industry_code"]:
                    taxonomy_rows.append(raw_row)
            if not taxonomy_rows:
                raise ValueError("R5_SECURITY_INDUSTRY_TAXONOMY_CONFLICT:" + sec)
            industries[sec] = identity["industry_code"]
            continue
        if row is not None:
            raw_row = normalized_text(row["raw_row"])
            section, industry = row["section_code"], row["industry_code"]
            if raw_row not in pdf_plain or not raw_row.startswith(short) or not re.fullmatch(r"[A-S]\d{2}", industry):
                raise ValueError("R5_DIRECT_INDEX_CLASSIFICATION_ROW:" + sec)
            after = raw_row.split(section, 1)[1]
            first_number = re.search(r"\d{2}", after)
            if not first_number or section + first_number[0] != industry:
                raise ValueError("R5_DIRECT_INDEX_CLASSIFICATION_FINE_CODE:" + sec)
        else:
            supplement = classification["ipo_supplements"].get(short)
            if not supplement:
                raise ValueError("R5_DIRECT_INDEX_INDUSTRY_UNMAPPED:" + sec)
            source.claim(supplement)
            quote, industry = normalized_text(supplement["quote"]), supplement["industry_code"]
            if short not in source.plain(supplement["source_hash"]) or not re.search(
                r"(?:行业分类代码|行业代码).*?" + re.escape(industry), quote
            ):
                raise ValueError("R5_DIRECT_INDEX_IPO_CLASSIFICATION:" + sec)
        industries[sec] = industry
    return {sec: issuers[sec[:6]] for sec in securities}, industries


def holder_execution_evidence(source: Originals, review: dict, primary_hash: str) -> None:
    """核持有人履职原文；不同托管报告措辞须有同等级实质依据。

    默认模板保持原“监督/复核”要求。显式的另一版式必须同时核本基金
    托管人对实际申赎计价未发现损害、完整报告核查，以及管理人明确的
    当期实际申赎运作和无风险事件声明；不能仅由一般合规承诺推定正常。
    本函数不评价现金可用范围，也不解除未闭合的全资产受限检查。
    """
    execution = normalized_text(review["execution"]["quote"])
    source.claim(review["execution"])
    profile = review.get("execution_profile", "SUPERVISION_AND_REVIEW")
    if profile == "SUPERVISION_AND_REVIEW":
        if not all(x in execution for x in ["报告期", "监督", "复核"]):
            raise ValueError("R5_DIRECT_INDEX_ACTUAL_HOLDER_EXECUTION_REQUIRED")
        return
    if profile == "TRUSTEE_SUPERVISION_AND_RECONCILIATION":
        if review["execution"]["source_hash"] != primary_hash or not all(
            term in execution for term in ["报告期", "监督", "对账"]
        ):
            raise ValueError("R5_DIRECT_INDEX_ACTUAL_HOLDER_EXECUTION_REQUIRED")
        return
    if profile == "TRUSTEE_QUARTER_REPORT_REVIEW":
        if review["execution"]["source_hash"] != primary_hash or not all(
            term in execution for term in ["复核", "投资组合报告", "保证复核内容不存在虚假记载"]
        ):
            raise ValueError("R5_DIRECT_INDEX_ACTUAL_HOLDER_EXECUTION_REQUIRED")
        return
    if profile not in {
        "TRUSTEE_NAV_REVIEW_AND_ACTUAL_REDEMPTION_OPERATION",
        "TRUSTEE_REDEMPTION_PRICING_AND_REPORT_REVIEW",
    }:
        raise ValueError("R5_HOLDER_EXECUTION_PROFILE_UNSUPPORTED")
    operation = review.get("actual_operation")
    if (
        not operation
        or review["execution"]["source_hash"] != primary_hash
        or operation.get("source_hash") != primary_hash
    ):
        raise ValueError("R5_HOLDER_EXECUTION_PRIMARY_REPORT_REQUIRED")
    source.claim(operation)
    operation_text = normalized_text(operation["quote"])
    required_trustee = [
        ("5.1报告期内本基金托管人遵规守信情况声明",),
        ("5.2托管人对报告期内本基金投资运作",),
        ("托管人未发现损害基金份额持有人利益的行为", "托管人未发现损害基金持有人利益的行为"),
        ("5.3托管人对本中期报告", "5.3托管人对中期报告"),
        ("进行了核查",),
        ("以上内容真实、准确和完整",),
    ]
    pricing_reviewed = "基金份额申购赎回价格计算" in execution or "基金份额申购赎回价格的计算" in execution
    if profile == "TRUSTEE_NAV_REVIEW_AND_ACTUAL_REDEMPTION_OPERATION":
        operation_valid = all(
            term in operation_text
            for term in [
                "报告期内",
                "本基金严格遵守合同要求",
                "认真对待投资者日常申购",
                "赎回及成份股调整",
                "未发生任何风险事件",
            ]
        )
    else:
        no_damage_terms = (
            "没有损害基金持有人利益的行为",
            "无损害基金持有人利益的行为",
            "无损害基金份额持有人利益的行为",
            "无重大违法、违规行为",
        )
        operation_valid = (
            "4.2管理人对报告期内本基金运作遵规守信情况的说明" in operation_text
            and re.search(r"(?:严格)?遵守.{0,100}基金合同", operation_text) is not None
            and any(term in operation_text for term in no_damage_terms)
        )
    if (
        not pricing_reviewed
        or not all(any(term in execution for term in alternatives) for alternatives in required_trustee)
        or not operation_valid
    ):
        raise ValueError("R5_HOLDER_EXECUTION_SUBSTANTIVE_EVIDENCE_REQUIRED")


def compile_observations(inputs: dict, *, root: Path) -> dict:
    """原件复读后编译所有原始子项，并返回金额/参照证明；不计算或导入总分。"""
    # 本版式仍重放原七维；最后一维要求完整基础证据及全部结构条件同分。
    # 未知实际集中程度保持为空，不能由缺失的单一投资者表猜成分散。
    if inputs.get("holder_structure_condition_protocol"):
        from app.services.fund_rating_cmf_liquidity import PROTOCOL
        from app.services.fund_rating_cmf_liquidity import compile_observations as compile_cmf

        if inputs["holder_structure_condition_protocol"] != PROTOCOL or inputs.get("component_scope"):
            raise ValueError("R5_CMF_COMPLETE_COMPILATION_PROTOCOL")
        return compile_cmf(inputs, root=root)
    # 查询和发布会多次校验同一不可变包。每次仍核全部原件摘要及截止日，
    # 只复用相同内容的解析/Decimal算术，避免每次详情重读六套完整报告。
    # 缓存不改verified_at、过期时间或源文件，损坏原件即使已有缓存也拒绝。
    Originals(inputs, root)
    return deepcopy(_compile_cached(canonical(inputs), str(root.resolve())))


@lru_cache(maxsize=16)
def _compile_cached(serialized: str, root_name: str) -> dict:
    inputs, root = json.loads(serialized), Path(root_name)
    if inputs.get("version") not in {VERSION, FEEDER_VERSION} or inputs.get("share_class") != "C":
        raise ValueError("R5_DIRECT_INDEX_COMPILER_SCOPE")
    feeder = inputs["version"] == FEEDER_VERSION
    source = Originals(inputs, root)
    code, peers, refs = inputs["fund_code"], inputs["peers"], inputs["target_refs"]
    expected = inputs["cohort"]["peers"]
    frozen = source.json(inputs["cohort"]["source_hash"])
    if (
        frozen["groups"][code]["peers"] != expected
        or frozen["frozen_at"] != inputs["cohort"]["frozen_at"]
        or frozen["selection"] != inputs["cohort"]["selection"]
    ):
        raise ValueError("R5_DIRECT_INDEX_COHORT_ARCHIVE_CONFLICT")
    if not expected or sorted(peers) != sorted(expected) or code in peers:
        raise ValueError("R5_DIRECT_INDEX_FROZEN_PEER_CLOSURE")
    if inputs["cohort"]["frozen_at"] > inputs["verified_at"]:
        raise ValueError("R5_DIRECT_INDEX_COHORT_FREEZE_ORDER")
    if feeder and (
        frozen["groups"][code].get("route") != "FEEDER" or set(inputs.get("etf_refs", {})) != {code, *expected}
    ):
        raise ValueError("R5_FEEDER_FROZEN_ROUTE_OR_TARGETS")
    own_pages = source.pages(refs["half"])
    body = normalized_text("\n".join(own_pages))
    if code not in body[:15000] or normalized_text(inputs["legal_name"]) not in body[:15000]:
        raise ValueError("R5_DIRECT_INDEX_OWN_REPORT_IDENTITY")
    history = _history(source, inputs)
    if history["dates"][-1] != inputs["as_of_date"]:
        raise ValueError("R5_DIRECT_INDEX_HISTORY_CUTOFF")
    contracts = {}
    all_reports = {code: refs["half"], **{c: p["report"] for c, p in peers.items()}}
    portfolios, values = {}, {}
    for member, digest in all_reports.items():
        if source.sources[digest].get("archive_kind") == "DERIVED":
            raise ValueError("R5_PASSIVE_FINANCIAL_ORIGINAL_REQUIRED")
        pages = source.pages(digest)
        view, _ = periodic_financial_view(pages)
        book = strict_balance_values(view, observed_on=inputs["holding_date"])["point_book"]
        if book is None:
            raise ValueError("R5_DIRECT_INDEX_PORTFOLIO_BALANCE:" + member)
        if feeder:
            from app.services.fund_rating_passive_holdings import _fund_holding, _identity
            from app.services.fund_rating_passive_portfolio import lookthrough

            own_portfolio = complete_portfolio(pages, holding_date=inputs["holding_date"])
            etf = inputs["etf_refs"][member]
            etf_pages = _identity(source, etf["source_hash"], etf["fund_code"], etf["legal_name"])
            etf_portfolio = complete_portfolio(etf_pages, holding_date=inputs["holding_date"])
            relation = _fund_holding(source, digest, etf, book["fund_amount"])
            budget = lookthrough(own_portfolio, etf_portfolio, fund_holding_cny=relation["amount_cny"])
            portfolios[member] = {**budget, "balance": book, "own": own_portfolio, "etf": etf_portfolio, "etf_ref": etf}
        else:
            stocks = parse(view, book["stock_amount"], book["net_assets"], require_industry=True)
            portfolios[member] = {"positions": stocks["holdings"], "balance": book, "stock_total": book["stock_amount"]}
        annuals = inputs["target_annuals"] if member == code else peers[member]["annuals"]
        alternatives = (
            inputs.get("target_contract_alternatives", {})
            if member == code
            else peers[member].get("contract_alternatives", {})
        )
        historical_refs = contract_history_refs(
            source, annuals, alternatives, history["dates"][0], inputs["as_of_date"], member, frozen
        )
        if feeder:
            for historical in {**historical_refs, "current": digest}.values():
                original = normalized_text("\n".join(source.pages(historical)))
                # 个别原转载丢失2.2数字而保留“基金产品说明”完整标题；
                # 身份区起点、法律全名和实际目标代码仍须同时原文核准。
                relation = re.search(
                    r"2\.1\.1目标基金基本情况(.{1,2500}?)(?:(?:2\.2)?基金产品说明|"
                    r"2\.1\.2目标基金产品说明)",
                    original,
                )
                if (
                    not relation
                    or etf["fund_code"] not in relation[1]
                    or normalized_text(etf["legal_name"]) not in relation[1]
                ):
                    raise ValueError("R5_FEEDER_HISTORICAL_TARGET_RELATIONSHIP:" + member)
        contracts[member] = {
            year: _contract(source, ref, member, inputs["index"].get("contract_name", inputs["index"]["name"]))
            for year, ref in {**historical_refs, "current": digest}.items()
        }
        nav = (
            history["adjusted_nav"]
            if member == code
            else _peer_nav(source, peers[member], member, history["dates"], inputs["as_of_date"])
        )
        values[member] = performance(history["dates"], history["month_ends"], nav, history["benchmark_index"])
    index_rows = _table(source.json(inputs["index"]["weights"]))
    # 原目录可能按市场一次返回全量指数；按原代码唯一定位，不要求
    # 调用者裁剪原响应或把派生的一行摘要冒充HTTP原件。
    metadata = [
        row for row in _table(source.json(inputs["index"]["metadata"])) if row["ts_code"] == inputs["index"]["code"]
    ]
    if (
        len(metadata) != 1
        or metadata[0]["ts_code"] != inputs["index"]["code"]
        or metadata[0]["name"] != inputs["index"]["name"]
        or metadata[0]["publisher"]
        not in {"中证指数有限公司", "中证指数有限公司、上海证券交易所", "深圳证券信息有限公司"}
    ):
        raise ValueError("R5_DIRECT_INDEX_PUBLISHER_IDENTITY")
    if inputs["index"].get("contract_name", inputs["index"]["name"]) != inputs["index"]["name"]:
        claim = inputs["index"].get("identity_claim")
        if not claim:
            raise ValueError("R5_PASSIVE_INDEX_ALIAS_ORIGINAL_REQUIRED")
        source.claim(claim)
        if not all(
            value in normalized_text(claim["quote"])
            for value in (
                inputs["index"]["contract_name"] + "指数",
                inputs["index"]["code"].split(".")[0],
                inputs["index"]["name"],
            )
        ):
            raise ValueError("R5_PASSIVE_INDEX_ALIAS_ORIGINAL_CONFLICT")
    weights = {}
    for row in index_rows:
        if (
            row["index_code"] != inputs["index"]["code"]
            or date.fromisoformat(row["trade_date"]).isoformat() != inputs["holding_date"]
        ):
            raise ValueError("R5_DIRECT_INDEX_WEIGHT_IDENTITY_OR_DATE")
        if row["con_code"] in weights:
            raise ValueError("R5_DIRECT_INDEX_WEIGHT_DUPLICATE")
        weights[row["con_code"]] = str(number(row["weight"]) / 100)
    if abs(sum(map(number, weights.values())) - 1) > Decimal(".000005") * len(weights):
        raise ValueError("R5_DIRECT_INDEX_WEIGHT_TOTAL_NOT_ROUNDING")
    securities = set(weights) | {p["stock_code"] for v in portfolios.values() for p in v["positions"]}
    issuers, industries = _mappings(source, inputs, securities)
    market = {sec: {} for sec in securities}
    days = inputs["market"]["dates"]
    if days[-1] != inputs["as_of_date"] or set(days) != inputs["market"]["receipts"].keys():
        raise ValueError("R5_DIRECT_INDEX_CAPACITY_CUTOFF")
    for day, digest in inputs["market"]["receipts"].items():
        for row in _table(source.json(digest)):
            sec = row["ts_code"]
            if sec in market:
                if date.fromisoformat(row["trade_date"]).isoformat() != day or day in market[sec]:
                    raise ValueError("R5_DIRECT_INDEX_MARKET_DATE_OR_DUPLICATE")
                market[sec][day] = row["amount"]
    for member, portfolio in portfolios.items():
        values[member].update(
            replication(portfolio["positions"], portfolio["stock_total"], weights, issuers, industries)
        )
        values[member]["market_capacity"] = capacity(portfolio["positions"], portfolio["stock_total"], days, market)[
            "K"
        ]
    # 正式JSON归档会排序字典键；引用数组也必须有固定顺序，才能跨进程复算。
    nodes, all_refs = {}, sorted(source.sources)
    for name in [
        "return_1y",
        "return_3y",
        "extra_drawdown",
        "extra_downside",
        "tracking_error",
        "issuer_replication",
        "industry_replication",
        "market_capacity",
    ]:
        nodes[name] = {
            "kind": "rank",
            "target": values[code][name],
            "direction": "higher" if name.startswith("return_") else "lower",
            "expected_products": ["CN_PUBLIC_FUND:" + c for c in expected],
            "peers": [{"product_id": "CN_PUBLIC_FUND:" + c, "value": values[c][name]} for c in expected],
            "source_refs": all_refs,
        }
    with localcontext() as context:
        context.prec = 70
        for name, count in [("monthly_success", 36), ("annual_success", 25)]:
            rows = [
                {**row, "reference": str(sum(number(values[c][name][i]["target"]) for c in expected) / len(expected))}
                for i, row in enumerate(values[code][name])
            ]
            nodes[name] = {"kind": "windows", "rows": rows, "expected_windows": count, "source_refs": all_refs}
    # 完整流动性缺项不能阻塞另外七维；此模式显式只编译非流动性原始观察，
    # 完整评分入口仍要求全八维，不能拿本模式生成完整总分或等级。
    component_only = inputs.get("component_scope") == "NON_LIQUIDITY_SEVEN_DIMENSIONS"
    if inputs.get("component_scope") and not component_only:
        raise ValueError("R5_PASSIVE_COMPONENT_SCOPE")
    liabilities, rows, flow_proof = None, None, None
    if not component_only:
        own = complete_portfolio(own_pages, holding_date=inputs["holding_date"], include_restrictions=True)
        view, _ = periodic_financial_view(own_pages)
        # 全资产范围只用于完整受限/现金审查；直接股票R5复制不依赖全资产行业距离。
        if not feeder:
            gross_asset_book(view)
        accounting = accounting_observations(view, date.fromisoformat(inputs["holding_date"]))["observations"]
        if number(accounting["money_funds"]["value"]) != number(accounting["demand_deposits_total"]["value"]):
            raise ValueError("R5_DIRECT_INDEX_NON_DEMAND_CASH_REQUIRES_PURPOSE_ADAPTER")
        liabilities = _liabilities(view, own["balance"]["liabilities"])
        nodes["known_payment"] = {
            "kind": "purpose_covered_payment",
            "as_of": inputs["holding_date"],
            "currency": "CNY",
            "bank_principal": accounting["demand_deposits_principal"]["value"],
            "broker_settlement_principal": "0",
            "settlement_liabilities": liabilities["point_values"]["应付清算款"],
            "all_book_liabilities": own["balance"]["liabilities"],
            "scope_closed": True,
            "restriction_amount_pending": False,
            "disclosed_restrictions": [],
            "source_refs": [refs["half"]],
            "scope_note": "已核货币资金全部为活期存款，取本金；备付金、保证金、应收及国债不作通用付款本金。"
            "证券专户可计本金在本已闭合货币资金范围为零，不将其他账户未明用途视为零余额。",
        }
        nodes["restricted"] = {
            "kind": "restricted",
            "restricted_amount": own["restricted_assets"]["restricted_amount"],
            "net_assets": own["balance"]["net_assets"],
            "scope_closed": True,
            "source_refs": [refs["half"]],
        }
        if feeder:
            # 联接自身的完整受限金额与真实持有ETF对应的底层受限风险预算分别
            # 取得，不能仅看联接证券表为空就把底层限售当0。普通支付仍只取
            # 联接本人的已核本金，不把ETF内部现金当成联接可以直接支取的钱。
            etf = inputs["etf_refs"][code]
            underlying = complete_portfolio(
                source.pages(etf["source_hash"]), holding_date=inputs["holding_date"], include_restrictions=True
            )
            with localcontext() as context:
                context.prec = 70
                budget = number(portfolios[code]["etf_scale"]) * number(
                    underlying["restricted_assets"]["restricted_amount"]
                )
                combined = number(own["restricted_assets"]["restricted_amount"]) + budget
            nodes["restricted"].update(
                restricted_amount=str(combined), source_refs=sorted({refs["half"], etf["source_hash"]})
            )
        rows, flow_proof = _flows(own_pages)
        review = inputs["holder_review"]
        for name in ["structure", "mechanism", "execution", "cash_restriction_scope"]:
            source.claim(review[name])
        structure = normalized_text(review["structure"]["quote"])
        dispersed = re.search(r"(?:未出现|无单一投资者).*(?:达到|超过)(?:本基金总份额)?20%", structure)
        if not dispersed:
            dispersed = re.search(r"本基金本报告期无需要披露的单一投资者持有基金份额比例达到或超过20%的情况", structure)
        structure_type = "DISPERSED" if dispersed else "CONCENTRATED_OPEN_END"
        if structure_type == "CONCENTRATED_OPEN_END" and not all(x in structure for x in ["20%", "采取措施", "赎回"]):
            raise ValueError("R5_DIRECT_INDEX_CONCENTRATED_EXECUTION_REVIEW_REQUIRED")
        mechanism = normalized_text(review["mechanism"]["quote"])
        execution = normalized_text(review["execution"]["quote"])
        if not all(x in mechanism for x in ["赎回", "巨额"]):
            raise ValueError("R5_DIRECT_INDEX_ACTUAL_HOLDER_EXECUTION_REQUIRED")
        holder_execution_evidence(source, review, refs["half"])
        if "未发现" not in execution or not review.get("actual_effect_review_complete"):
            raise ValueError("R5_DIRECT_INDEX_HOLDER_ADVERSE_EFFECT_SCOPE")
        nodes["holders"] = {
            "kind": "holder_execution_r5",
            "structure_type": structure_type,
            "structure_closed": True,
            "share_flow_closed": True,
            "redemption_mechanism_verified": True,
            "actual_execution_verified": True,
            "actual_redemption_shares": str(sum(number(r["redeemed"]) for r in rows)),
            "confirmed_pressure_effects": review["confirmed_pressure_effects"],
            "source_refs": sorted({review[k]["source_hash"] for k in ["structure", "mechanism", "execution"]}),
        }
    management = management_calculate(
        inputs["management_inputs"],
        root=root,
        manager_name=inputs["manager_name"],
        as_of=inputs["as_of_date"],
        fund_code=code,
        legal_name=inputs["legal_name"],
        primary_report_hash=refs["half"],
    )
    if management["score"] is None:
        raise ValueError("R5_DIRECT_INDEX_MANAGEMENT_NOT_CLOSED")
    raw_management = inputs["management_inputs"]
    for name in ["experience", "governance"]:
        raw = raw_management[name]
        nodes[name] = {
            "kind": "state",
            "state": raw["state"],
            "reason": raw_management["governance"]["review_scope"],
            "source_refs": sorted(raw_management["sources"]),
        }
    nodes["continuity"] = {"kind": "responsibility_timeline", **raw_management["continuity"]}
    fees = fee_profile(
        "\n".join(source.pages(refs["summary"])),
        code=code,
        share_class="C",
        legal_name=inputs["legal_name"],
        supporting_text="\n".join(source.pages(refs["prospectus"])),
        published_on=source.sources[refs["summary"]]["published_on"],
    )
    nodes["cost_365"] = {
        "kind": "standard_cost",
        "fee_inputs": fees["fees"],
        "source_refs": [refs["summary"], refs["prospectus"]],
    }
    tracking, tracking_ref = None, None
    for ref in [refs["summary"], refs["prospectus"]]:
        tracking = re.search(
            r"年(?:化)?跟踪误差(?:不超过|控制在|控制)([\d.]+)%", normalized_text("\n".join(source.pages(ref)))
        )
        if tracking:
            tracking_ref = ref
            break
    if not tracking or number(tracking[1]) <= 0:
        raise ValueError("R5_DIRECT_INDEX_TRACKING_CLAUSE_REQUIRED")
    anchor = {
        **source.sources[tracking_ref],
        "source_verified": True,
        "annual_tracking_error_limit_ratio": str(number(tracking[1]) / 100),
        "actual_clause": tracking[0],
    }
    if component_only:
        nodes.pop("market_capacity")
    return {
        "observations": nodes,
        "anchor": anchor,
        "audit": {
            "contracts": contracts,
            "performance": values,
            "publisher_cross_check": history["publisher_cross_check"],
            "liability_proof": liabilities,
            "share_flow_rows": rows,
            "share_flow_proof": flow_proof,
            "management": management,
            "fee_inputs": fees,
            "holdings_scope": "STOCK_INTERNAL_REPLICATION_R5_NOT_GROSS_ASSET_INDUSTRY_DISTANCE",
            **(
                {
                    "feeder_budgets": {
                        member: {
                            "stock_total": p["stock_total"],
                            "etf_scale": p["etf_scale"],
                            "target_etf": p["etf_ref"]["fund_code"],
                        }
                        for member, p in portfolios.items()
                    },
                    "lookthrough_restricted_cny": nodes.get("restricted", {}).get("restricted_amount"),
                }
                if feeder
                else {}
            ),
            "common_reference": "本基金发行人公布的完整合同基准作为全组共同评价参照；"
            "不宣称各同行现金利率用词相同或逐日履约。",
        },
    }
