"""博时油气QDII人民币C：已冻结收益/稳定性原件编译候选。

仅适用018853与独立法律产品162411的人民币C份额007844。前日净值扣现金
分红统计沿本轮冻结协议；本适配要求双方真实无分红原件，故公式退化为真实
端点净值比。不借不同合同的基准补风险/目标执行，不输出完整分或等级。

本入口只接收已核原件输入，部分评分不能据此晋级完整八维。
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from urllib.parse import unquote, urljoin
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

from app.services.material_content import content_digest, source_bytes, source_exists, source_size
from app.services.material_content import document_reader as PdfReader

VERSION = "R5_BOSERA_QDII_CNY_PERFORMANCE_ORIGINALS_20261029_1"
SCOPE = {"return", "stability"}
LEGAL_NAME = "博时标普石油天然气勘探及生产精选行业指数型发起式证券投资基金(QDII)"
AVAILABILITY_SCOPE = "RETROSPECTIVE_VALUATION_STATISTICS_NOT_ASOF_KNOWLEDGE_OR_TRADE_SIGNAL"
MAX_ORIGINAL_BYTES = 16_000_000
MAX_DOCX_XML_BYTES = 8_000_000


def confirmed_material_events(events: list[dict]) -> list[dict]:
    """原R5第348—354行的事件适用范围，不把条件未知升级为已确认失效。

    返回值仅表示已确认、本基金适用、重大且未解决的事项。三周年净资产
    未公开等条件事实可继续为null；空集合不证明这些条件均未触发。
    """
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("BOSERA_QDII_MATERIAL_EVENT_LIST_REQUIRED")
    return [
        e
        for e in events
        if e.get("applicability") == "FUND_CONFIRMED"
        and e.get("status") == "UNRESOLVED"
        and e.get("severity") in {"MATERIAL", "SEVERE"}
    ]


def compact(value):
    return re.sub(r"\s", "", value).replace("（", "(").replace("）", ")").replace("：", ":")


def docx_text(raw: bytes) -> str:
    """只读有界Word正文；不把ZIP声明大小当作解压时实际读取上限。"""
    from app.services.material_content import SourceContent

    if isinstance(raw, SourceContent) and raw.kind == "DOCX":
        return "".join(raw.extracted["paragraphs"])
    if len(raw) > MAX_ORIGINAL_BYTES:
        raise ValueError("BOSERA_QDII_ZIP_ORIGINAL_SIZE_LIMIT")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        bodies = [info for info in entries if info.filename == "word/document.xml"]
        if len(entries) > 512 or len(bodies) != 1:
            raise ValueError("BOSERA_QDII_ZIP_BODY_NOT_UNIQUE_OR_ENTRY_LIMIT")
        info = bodies[0]
        if info.flag_bits & 1 or not 0 < info.file_size <= MAX_DOCX_XML_BYTES:
            raise ValueError("BOSERA_QDII_ZIP_XML_SIZE_LIMIT")
        with archive.open(info) as stream:
            xml = stream.read(MAX_DOCX_XML_BYTES + 1)
        if len(xml) > MAX_DOCX_XML_BYTES or len(xml) != info.file_size or b"<!DOCTYPE" in xml.upper():
            raise ValueError("BOSERA_QDII_ZIP_XML_SIZE_OR_DECLARATION")
    tree = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return "".join(item.text or "" for item in tree.findall(".//w:t", ns))


def nav_availability(nav_day: str, announced_on: str, contract: dict) -> dict:
    """估值日限制经济观察窗；正常晚公告保留实际可得日，绝不宣称当时已知。"""
    valuation = date.fromisoformat(nav_day)
    available = date.fromisoformat(announced_on)
    cutoff = date.fromisoformat(contract["valuation_as_of"])
    research = date.fromisoformat(contract["research_on"])
    if valuation > cutoff:
        raise ValueError("BOSERA_QDII_FUTURE_NAV_VALUATION")
    if available > research or available < valuation:
        raise ValueError("BOSERA_QDII_FUTURE_OR_INVALID_NAV_ANNOUNCEMENT")
    return {
        "nav_date": str(valuation),
        "available_on": str(available),
        "known_on_valuation_as_of": available <= cutoff,
        "research_on": str(research),
        "availability_scope": AVAILABILITY_SCOPE,
    }


def report_cover_identity(cover: str, expected_period: str) -> dict:
    """报告期只取封面标题后的期末日，不把正文前部的上年比较日当作当期。"""
    text = compact(cover)
    match = re.search(r"(20\d{2})年(年度|中期)报告(20\d{2})年(\d{1,2})月(\d{1,2})日基金管理人", text)
    sent = re.search(r"送出日期:(20\d{2})年(\d{1,2})月(\d{1,2})日", text)
    if not match or not sent:
        raise ValueError("BOSERA_QDII_REPORT_COVER_PERIOD_OR_ISSUE_DATE")
    actual = date(*map(int, match.groups()[2:]))
    required_month_day = (12, 31) if match[2] == "年度" else (6, 30)
    if (
        str(actual) != expected_period
        or actual.year != int(match[1])
        or (actual.month, actual.day) != required_month_day
    ):
        raise ValueError("BOSERA_QDII_HISTORICAL_REPORT_PERIOD_BINDING")
    return {"report_period": str(actual), "document_sent_on": str(date(*map(int, sent.groups())))}


def normalized_source_url(value: str) -> str:
    return re.sub(r"(?<!:)/{2,}", "/", unquote(urljoin("https://www.fsfund.com", value)))


class Originals:
    """显式来源目录、角色及字节摘要；不依赖工作目录或预先计算的分数。"""

    def __init__(self, inputs: dict, root: Path):
        self.inputs, self.root = deepcopy(inputs), root.resolve()
        if {"score", "component_scores", "observations", "result", "packet", "complete_score", "grade"} & inputs.keys():
            raise ValueError("BOSERA_QDII_PRECOMPUTED_SCORE_FORBIDDEN")
        if (
            inputs.get("version") != VERSION
            or inputs.get("fund_code") != "018853"
            or inputs.get("share_class") != "C"
            or inputs.get("currency") != "CNY"
            or inputs.get("as_of_date") != "2026-09-30"
            or inputs.get("holding_date") != "2026-06-30"
            or compact(inputs.get("legal_name", "")) != compact(LEGAL_NAME)
            or inputs.get("manager_name") != "博时基金管理有限公司"
        ):
            raise ValueError("BOSERA_QDII_COMPONENT_SCOPE")
        self.sources, self.roles, self.originals = inputs["sources"], inputs["roles"], {}
        for digest, reference in self.sources.items():
            path = (self.root / reference["file"]).resolve()
            if not path.is_relative_to(self.root / "sources") or not source_exists(path):
                raise ValueError("BOSERA_QDII_CANONICAL_SOURCE_REQUIRED")
            if source_size(path) > MAX_ORIGINAL_BYTES:
                raise ValueError("BOSERA_QDII_ORIGINAL_SIZE_LIMIT")
            raw = source_bytes(path)
            if content_digest(raw) != digest or reference.get("sha256") != digest:
                raise ValueError("BOSERA_QDII_ORIGINAL_HASH_CHANGED")
            if reference.get("published_on") and reference["published_on"] > inputs["as_of_date"]:
                raise ValueError("BOSERA_QDII_FUTURE_DISCLOSURE")
            self.originals[digest] = raw
        for role in self.roles.values():
            if role["source_hash"] not in self.originals:
                raise ValueError("BOSERA_QDII_ROLE_SOURCE_MISSING")
        if self.roles["primary-report"]["source_hash"] != inputs["primary_report_hash"]:
            raise ValueError("BOSERA_QDII_PRIMARY_REPORT_BINDING")
        self.time_contract = self._time_contract()
        self.material_event_pending = self._material_event_review()
        self._identity()

    def _time_contract(self):
        """复用原第2节时间定义；非净值资料仍严格受截止前公开边界约束。"""
        contract = self.inputs.get("time_contract", {})
        if (
            contract.get("version") != "FUND_QUALITY_HISTORICAL_NAV_TIME_CONTRACT_V1"
            or contract.get("valuation_as_of") != self.inputs["as_of_date"]
            or contract.get("public_information_cutoff") != self.inputs["as_of_date"]
            or contract.get("availability_scope") != AVAILABILITY_SCOPE
            or contract.get("historically_known_on_valuation_as_of") is not False
        ):
            raise ValueError("BOSERA_QDII_HISTORICAL_NAV_TIME_CONTRACT_REQUIRED")
        research = date.fromisoformat(contract["research_on"])
        if (
            not date.fromisoformat(self.inputs["as_of_date"])
            <= research
            <= datetime.now(ZoneInfo("Asia/Shanghai")).date()
        ):
            raise ValueError("BOSERA_QDII_RESEARCH_DAY_INVALID")
        for reference in self.sources.values():
            if reference.get("acquired_at"):
                acquired = datetime.fromisoformat(reference["acquired_at"])
                if acquired.tzinfo is None:
                    raise ValueError("BOSERA_QDII_SOURCE_ACQUISITION_TIMEZONE_REQUIRED")
                if acquired.astimezone(ZoneInfo("Asia/Shanghai")).date() > research:
                    raise ValueError("BOSERA_QDII_SOURCE_ACQUIRED_AFTER_RESEARCH_DAY")
        digest = self.roles["historical-time-contract"]["source_hash"]
        text = compact(self.raw(digest).decode("utf-8-sig"))
        if (
            contract.get("source_hash") != digest
            or "名义截止日是月末;净值用各自有效日历对应的实际估值日" not in text.replace("；", ";")
            or "取得日、原公开日、净值日、评分研究日分别保存" not in text
            or "完整持仓取截止日前已经公开的最近年报/中报" not in text
        ):
            raise ValueError("BOSERA_QDII_ORIGINAL_TIME_POLICY_NOT_BOUND")
        return deepcopy(contract)

    def _material_event_review(self):
        """重大未决状态来自有原件的显式研究判断，不能默认为False。"""
        review = self.inputs.get("material_event_review")
        if (
            not isinstance(review, dict)
            or review.get("review_complete") is not True
            or type(review.get("material_event_pending")) is not bool
        ):
            raise ValueError("BOSERA_QDII_MATERIAL_EVENT_REVIEW_REQUIRED")
        if (
            review.get("fund_code") != "018853"
            or review.get("as_of_date") != "2026-09-30"
            or compact(review.get("legal_name", "")) != compact(LEGAL_NAME)
            or review.get("known_manager_matters_reviewed") is not True
            or len(review.get("review_scope", "")) < 40
            or len(review.get("product_impact_conclusion", "")) < 40
        ):
            raise ValueError("BOSERA_QDII_MATERIAL_EVENT_REVIEW_SCOPE")
        claims = review.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("BOSERA_QDII_MATERIAL_EVENT_SOURCE_CLAIMS_REQUIRED")
        from bs4 import BeautifulSoup

        for claim in claims:
            digest, quote = claim.get("source_hash"), claim.get("quote", "")
            if digest not in self.originals or len(quote) < 12:
                raise ValueError("BOSERA_QDII_MATERIAL_EVENT_SOURCE_CLAIM_INVALID")
            # 正常晚公告例外仅用于净值；重大事项仍须截止前已公开。
            published = self.sources[digest].get("published_on")
            if not published or date.fromisoformat(published) > date.fromisoformat(self.inputs["as_of_date"]):
                raise ValueError("BOSERA_QDII_MATERIAL_EVENT_PUBLIC_DATE_REQUIRED")
            raw = self.raw(digest)
            if raw.startswith(b"%PDF"):
                text = "\n".join(page.extract_text() or "" for page in PdfReader(raw).pages)
            elif raw.startswith(b"PK"):
                text = docx_text(raw)
            else:
                payload = json.loads(raw)
                text = payload.get("data", {}).get("notice_content") if isinstance(payload, dict) else None
                text = (
                    BeautifulSoup(text, "html.parser").get_text() if text else json.dumps(payload, ensure_ascii=False)
                )
            if compact(quote) not in compact(text):
                raise ValueError("BOSERA_QDII_MATERIAL_EVENT_QUOTE_NOT_IN_ORIGINAL")
        events = review.get("events")
        confirmed = confirmed_material_events(events)
        if review.get("material_event_pending_meaning") != "CONFIRMED_FUND_SPECIFIC_MATERIAL_UNRESOLVED_EVENTS_ONLY":
            raise ValueError("BOSERA_QDII_MATERIAL_EVENT_GATE_MEANING_REQUIRED")
        if any(
            not event.get("source_refs") or not set(event["source_refs"]) <= self.originals.keys() for event in events
        ):
            raise ValueError("BOSERA_QDII_MATERIAL_EVENT_ORIGINAL_MISSING")
        # 该布尔量由原事件列表计算，未知触发金额及业务限制保留在同一列表中。
        pending = bool(confirmed)
        if review["material_event_pending"] is not pending:
            raise ValueError("BOSERA_QDII_MATERIAL_EVENT_GATE_DERIVATION_MISMATCH")
        self.material_event_anchors = {"material_events": {"018853": deepcopy(events)}}
        self.material_event_scope_audit = {
            "rule": review.get("rule_provenance"),
            "material_event_pending_meaning": review["material_event_pending_meaning"],
            "confirmed_material_unresolved": deepcopy(confirmed),
            "all_reviewed_events": deepcopy(events),
            "product_impact_conclusion": review["product_impact_conclusion"],
            "unconfirmed_continuity_facts_preserved": True,
        }
        return pending

    def raw(self, digest):
        return self.originals[digest]

    def reference(self, label):
        role = self.roles[label]
        digest = role["source_hash"]
        ref = self.sources[digest]
        return {**role, **ref, "sha256": digest, "path": str((self.root / ref["file"]).resolve())}

    def receipt(self, label):
        ref = self.reference(label)
        if ref.get("status", ref.get("http_status")) != 200:
            raise ValueError("BOSERA_QDII_SOURCE_RESPONSE_NOT_SUCCESS:" + label)
        return ref

    def json_role(self, label):
        return json.loads(self.raw(self.roles[label]["source_hash"]), parse_float=D)

    def _identity(self):
        """独立核本份额C人民币原概要与本期完整报告，不沿用旧Y标记。"""
        raw = self.raw(self.roles["primary-report"]["source_hash"])
        text = compact(docx_text(raw))
        if not all(s in text[:18000] for s in [compact(LEGAL_NAME), "018853", "2026年6月30日"]):
            raise ValueError("BOSERA_QDII_PRIMARY_REPORT_IDENTITY")
        role = self.roles["share-identity-summary"]
        summary = compact(
            "\n".join(page.extract_text() or "" for page in PdfReader(self.raw(role["source_hash"])).pages)
        )
        if not all(s in summary for s in ["018853", "C人民币", "交易币种人民币", "博时基金管理有限公司"]):
            raise ValueError("BOSERA_QDII_CNY_C_ORIGINAL_IDENTITY")


def checked_receipt(source: Originals, label: str):
    ref = source.receipt(label)
    return ref, source.raw(ref["sha256"])


def iso_day(value: str) -> str:
    """源日期只接受明确ISO或八位自然日，不从取得时刻猜公告日。"""
    if not isinstance(value, str):
        raise ValueError("BOSERA_QDII_NAV_DATE_OR_ANNOUNCEMENT_MISSING")
    if re.fullmatch(r"\d{8}", value):
        value = value[:4] + "-" + value[4:6] + "-" + value[6:]
    return str(date.fromisoformat(value))


def series_from_bosera(source, label, expected_title):
    """官网年区间正常接口，不插值缺日；每条日期和净值都保留原始披露。"""
    result, receipts = {}, []
    for year in ["2023", "2024", "2025", "2026"]:
        receipt, raw = checked_receipt(source, label + "-" + year)
        response = json.loads(raw, parse_float=D)
        expected_start = "20230901" if year == "2023" else year + "0101"
        expected_end = "20260930" if year == "2026" else year + "1231"
        if (
            receipt["params"]["fundCode"] != "018853"
            or receipt["params"].get("startDate") != expected_start
            or receipt["params"].get("endDate") != expected_end
            or response.get("retCode") != "0"
            or response["data"]["title"] != expected_title
        ):
            raise ValueError("BOSERA_OWN_SHARE_RESPONSE")
        for row in response["data"]["resultList"]:
            if len(row) != 1:
                raise ValueError("BOSERA_DATE_ROW_SHAPE")
            day, value = next(iter(row.items()))
            if not re.fullmatch(r"\d{8}", day):
                raise ValueError("BOSERA_DATE_ROW_SHAPE")
            if day > "20260930":
                raise ValueError("BOSERA_QDII_FUTURE_NAV_VALUATION")
            if not expected_start <= day <= expected_end:
                raise ValueError("BOSERA_QDII_NAV_RESPONSE_OUTSIDE_REQUEST")
            day = date(int(day[:4]), int(day[4:6]), int(day[6:])).isoformat()
            if day in result or D(value) <= 0:
                raise ValueError("BOSERA_DUPLICATE_DATE_OR_VALUE")
            result[day] = D(value)
        receipts.append(receipt)
    return result, receipts


def _compile(source: Originals):
    """每次重读并校验双方原始值、同币种份额、历史合同和37个真实月末。"""
    from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
    from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS

    statistic = source.json_role("return-statistic-protocol")
    if (
        statistic.get("direct_formula")
        != (
            "G(s,t)=NAV(t)/NAV(s)*PRODUCT[NAV(previous_trading_day)/"
            "(NAV(previous_trading_day)-cash_per_share)]；s<ex_date<=t"
        )
        or statistic.get("no_imputation") is not True
    ):
        raise ValueError("BOSERA_QDII_FROZEN_RETURN_STATISTIC_CHANGED")
    frozen = source.json_role("comparison-contract")
    if (
        frozen["peer_set"] != ["CN_PUBLIC_FUND:162411"]
        or frozen["peer_share_code"] != "007844"
        or frozen["scope"] != ["return", "stability"]
    ):
        raise ValueError("FROZEN_COMPARISON_SCOPE")
    reports = frozen.get("source_reports", [])
    if (
        len(reports) != 4
        or len({ref.get("sha256") for ref in reports}) != 4
        or {ref.get("report_period") for ref in reports} != {"2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30"}
    ):
        raise ValueError("BOSERA_QDII_FIXED_HISTORICAL_CONTRACT_SET")
    report_checks = []
    for ref in reports:
        raw = source.raw(ref["sha256"])
        if content_digest(raw) != ref["sha256"] or ref["published_on"] > "2026-09-30":
            raise ValueError("PEER_HISTORICAL_REPORT_HASH_OR_CUTOFF")
        reader = PdfReader(raw)
        cover = report_cover_identity(reader.pages[0].extract_text() or "", ref["report_period"])
        catalog_label = "peer-report-catalog-" + ref["report_period"][:4]
        catalog_ref, catalog_raw = checked_receipt(source, catalog_label)
        catalog = json.loads(catalog_raw)
        if catalog.get("code") != "0000" or catalog_ref.get("params", {}).get("fundCode") != "007844":
            raise ValueError("BOSERA_QDII_REPORT_CATALOG_IDENTITY")
        original_url = source.sources[ref["sha256"]]["url"]
        matches = [
            row
            for row in catalog["data"]
            if row.get("articleFile")
            and normalized_source_url(row["articleFile"]) == normalized_source_url(original_url)
        ]
        if len(matches) != 1:
            raise ValueError("BOSERA_QDII_REPORT_CATALOG_ENTRY_NOT_UNIQUE")
        published = date.fromisoformat(matches[0]["publishDate1"])
        if (
            published < date.fromisoformat(cover["document_sent_on"])
            or published > date.fromisoformat(source.inputs["as_of_date"])
            or source.sources[ref["sha256"]].get("published_on") != str(published)
        ):
            raise ValueError("BOSERA_QDII_REPORT_ACTUAL_PUBLICATION_CONFLICT")
        text = compact("\n".join(page.extract_text() or "" for page in reader.pages))
        if not all(
            value in text[:15000] for value in ["007844", "162411", "华宝标普石油天然气上游股票指数证券投资基金"]
        ):
            raise ValueError("PEER_HISTORICAL_LEGAL_PRODUCT")
        if not all(value in text for value in ["本基金原则上采取完全复制策略", "全收益指数"]):
            raise ValueError("PEER_HISTORICAL_STRATEGY")
        # 逐年核实际策略未改变，而不是只核当前名称相似。
        last = text.rfind("基金投资策略的改变")
        statement = text[last : last + 200].split("。", 1)[0]
        if last < 0 or not re.search(r"投资策略(?:未发生变更|未发生改变|未改变)", statement):
            raise ValueError("PEER_STRATEGY_CHANGE_REQUIRES_REVIEW")
        report_checks.append(
            {
                **ref,
                **cover,
                "workpaper_declared_published_on": ref["published_on"],
                "published_on": str(published),
                "publication_catalog_source": catalog_ref["sha256"],
                "path": str(source.root / source.sources[ref["sha256"]]["file"]),
                "actual_strategy_excerpt": text[last : last + 200],
            }
        )
    summary_ref, raw = checked_receipt(source, "peer-original-summary")
    summary = compact("\n".join(page.extract_text() or "" for page in PdfReader(raw).pages))
    if not all(value in summary for value in ["007844", "人民币", "C类份额", "2025年12月13日"]):
        raise ValueError("PEER_CNY_C_CLASS_REQUIRED")
    own, own_receipts = series_from_bosera(source, "bosera-unit", "单位净值")
    cumulative, cumulative_receipts = series_from_bosera(source, "bosera-nav8", "累计净值")
    if own != cumulative:
        raise ValueError("OWN_DIVIDEND_OR_SPLIT_RECONCILIATION_REQUIRED")
    own = {day: value for day, value in own.items() if "2023-09-28" <= day <= "2026-09-30"}
    provider_ref = source.reference("target-provider")
    provider_path = Path(provider_ref["path"])
    prior = source.json_role("target-provider")
    if prior.get("source", {}).get("enabled") is not True or not prior.get("source", {}).get(
        "authorization_verified_at"
    ):
        raise ValueError("BOSERA_QDII_AUTHORIZED_PROVIDER_REQUIRED")
    if (
        prior["fund_code"] != "018853"
        or prior["dividend_records"] != []
        or prior["event_refresh"]["status"] != "SUCCEEDED"
    ):
        raise ValueError("TARGET_OWN_NAV_EVENTS_REVIEW_REQUIRED")
    target_rows = prior["history"]["rows"]
    target_provider = {iso_day(row["nav_date"]): row for row in target_rows}
    if len(target_provider) != len(target_rows):
        raise ValueError("BOSERA_QDII_DUPLICATE_TARGET_PROVIDER_DATE")
    if any(day > source.inputs["as_of_date"] for day in target_provider):
        raise ValueError("BOSERA_QDII_FUTURE_NAV_VALUATION")
    for day, value in own.items():
        if day not in target_provider or any(
            D(target_provider[day][key]) != value for key in ["unit_nav", "accumulated_nav", "adjusted_nav"]
        ):
            raise ValueError("OWN_OFFICIAL_AND_AUTHORIZED_NAV_CONFLICT:" + day)
    peer_ref, raw = checked_receipt(source, "peer-official-nav")
    peer_response = json.loads(raw, parse_float=D)
    if peer_response["code"] != "0000" or peer_ref["params"]["fundCode"] != "007844":
        raise ValueError("PEER_OFFICIAL_HISTORY_IDENTITY")
    peer, peer_events = {}, []
    for row in peer_response["data"]:
        day = iso_day(row["navDate"])
        if day > source.inputs["as_of_date"]:
            raise ValueError("BOSERA_QDII_FUTURE_NAV_VALUATION")
        if row["fundCode"] != "007844" or day in peer or row.get("publishSign") != "1":
            raise ValueError("PEER_OFFICIAL_DATE_OR_IDENTITY")
        if row["nav"] != row["totalNetValue"]:
            peer_events.append(day)
        peer[day] = D(row["nav"])
    p_ref, raw = checked_receipt(source, "peer-nav")
    p = json.loads(raw, parse_float=D)
    if p["code"] != 0 or p.get("data", {}).get("has_more") is not False:
        raise ValueError("PEER_AUTHORIZED_HISTORY_REFUSED")
    p_rows = [dict(zip(p["data"]["fields"], row, strict=True)) for row in p["data"]["items"]]
    p_by_day = {iso_day(row["nav_date"]): row for row in p_rows}
    if len(p_by_day) != len(p_rows):
        raise ValueError("BOSERA_QDII_DUPLICATE_PEER_PROVIDER_DATE")
    if any(day > source.inputs["as_of_date"] for day in p_by_day):
        raise ValueError("BOSERA_QDII_FUTURE_NAV_VALUATION")
    for day in own:
        if day not in peer or day not in p_by_day:
            raise ValueError("PEER_COMMON_WINDOW_DATE_MISSING:" + day)
        if p_by_day[day]["ts_code"] != "007844.OF" or any(
            D(p_by_day[day][key]) != peer[day] for key in ["unit_nav", "accum_nav", "adj_nav"]
        ):
            raise ValueError("PEER_ORIGINAL_NAV_CONFLICT_OR_EVENT:" + day)
    div_ref, raw = checked_receipt(source, "peer-div")
    dividend = json.loads(raw)
    if (
        div_ref.get("params", {}).get("ts_code") != "007844.OF"
        or dividend["code"] != 0
        or dividend["data"]["items"] != []
        or peer_events
    ):
        raise ValueError("PEER_DISTRIBUTIONS_NEED_RECONCILIATION")
    # 周末的财报专用估值保留在审计，统一使用双方都存在的本基金正常披露日期。
    # 不以境内交易所模板强迫填造未公开日；正常估值日须与授权原序列完整一致。
    dates = sorted(day for day in own if date.fromisoformat(day).weekday() < 5)
    expected_dates = sorted(
        day for day in target_provider if "2023-09-28" <= day <= "2026-09-30" and date.fromisoformat(day).weekday() < 5
    )
    if dates != expected_dates:
        raise ValueError("BOSERA_QDII_NORMAL_NAV_DATE_UNIVERSE_NOT_COMPLETE")
    # 真实公告日晚于估值截止可用于本次事后历史统计；不标记为当时可知。
    availability = {
        "018853": {
            day: nav_availability(day, iso_day(target_provider[day]["ann_date"]), source.time_contract) for day in dates
        },
        "007844": {
            day: nav_availability(day, iso_day(p_by_day[day]["ann_date"]), source.time_contract) for day in dates
        },
    }
    months = prior["history"]["month_ends"]
    # 月末必须由真实披露日历独立生成，不能只接受37个任意日期。
    by_month = {}
    for day in dates:
        by_month[day[:7]] = day
    expected_months = [by_month[key] for key in sorted(by_month)]
    if months != expected_months:
        raise ValueError("BOSERA_QDII_MONTH_ENDS_NOT_ACTUAL_LAST_DISCLOSURE")
    if (
        len(months) != 37
        or months[0] != "2023-09-28"
        or months[-1] != "2026-09-30"
        or not set(months) <= set(dates)
        or dates[-1] != "2026-09-30"
    ):
        raise ValueError("EXACT_THREE_YEAR_37_MONTH_END_WINDOW")
    receipts = own_receipts + cumulative_receipts + [peer_ref, p_ref, div_ref, summary_ref]
    registry = {ref["sha256"]: {"path": ref["path"], "published_on": None} for ref in receipts}
    registry[provider_path.stem] = {"path": str(provider_path), "published_on": None}
    registry.update(
        {ref["sha256"]: {"path": ref["path"], "published_on": ref["published_on"]} for ref in report_checks}
    )
    all_refs = sorted(registry)
    with localcontext() as context:
        context.prec = 70
        span = D((date.fromisoformat(months[-1]) - date.fromisoformat(months[0])).days)
        three = D("365.25") / span
        observations = {}
        for name, first, annualized in [("return_1y", months[-13], False), ("return_3y", months[0], True)]:
            target = own[months[-1]] / own[first]
            reference = peer[months[-1]] / peer[first]
            target, reference = (target**three - 1, reference**three - 1) if annualized else (target - 1, reference - 1)
            observations[name] = {
                "kind": "rank",
                "direction": "higher",
                "target": str(target),
                "peers": [{"product_id": "CN_PUBLIC_FUND:162411", "value": str(reference)}],
                "expected_products": frozen["peer_set"],
                "source_refs": all_refs,
            }
        for name, step in [("monthly_success", 1), ("annual_success", 12)]:
            observations[name] = {
                "kind": "windows",
                "expected_windows": 37 - step,
                "source_refs": all_refs,
                "rows": [
                    {
                        "start": months[i - step],
                        "end": months[i],
                        "target": str(own[months[i]] / own[months[i - step]] - 1),
                        "reference": str(peer[months[i]] / peer[months[i - step]] - 1),
                    }
                    for i in range(step, 37)
                ],
            }
    packet = {
        "fund_code": "018853",
        "as_of_date": "2026-09-30",
        "basis": PASSIVE_BASIS,
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "r4_evidence_registry": registry,
        "quality_metrics": {
            "observations": observations,
            "target_product_aliases": frozen["target_aliases"],
            "material_event_pending": source.material_event_pending,
        },
        "quality_public": {"dataDates": {"nav": "2026-09-30", "holdings": "2026-06-30"}},
        "raw_archive": [{"kind": "ORIGINAL", "sha256": digest} for digest in all_refs],
    }
    result = _calculate_frozen(
        packet, PROTOCOL, source.material_event_anchors, selected_dimensions={"return", "stability"}
    )
    assert result["score"] is None and result["grade"] is None
    return {
        "version": "R5_QDII_CNY_PERFORMANCE_COMPONENTS_20261029_1",
        "packet": packet,
        "anchors": deepcopy(source.material_event_anchors),
        "result": result,
        "audit": {
            "own_original_date_count": len(own),
            "valuation_date_count": len(dates),
            "common_month_ends": months,
            "time_contract": deepcopy(source.time_contract),
            "latest_nav_available_on": max(
                row["available_on"] for rows in availability.values() for row in rows.values()
            ),
            "month_end_nav_availability": {
                code: {day: rows[day] for day in months} for code, rows in availability.items()
            },
            "source_acquisition_dates": {
                digest: {key: ref.get(key) for key in ["acquired_at", "imported_at", "upstream_acquired_at"]}
                for digest, ref in source.sources.items()
            },
            "first": dates[0],
            "last": dates[-1],
            "all_common_values_cross_checked": True,
            "own_report_only_weekends_retained_outside_daily_series": sorted(set(own) - set(dates)),
            "peer_original_row_count": len(peer),
            "peer_dividend_response_rows": 0,
            "historical_report_checks": report_checks,
            "cohort_frozen_at": frozen["frozen_at"],
            "common_benchmark_exactly_cancels_for_relative_return": True,
            "benchmark_not_substituted_for_risk_or_efficiency": True,
            "full_score": None,
            "other_unresolved_dimensions_continue": True,
            "material_event_scope": deepcopy(source.material_event_scope_audit),
        },
    }


def compile_observations(inputs: dict, *, root: Path) -> dict:
    """每次从固定原件重建37月末、两期限收益及36/25真实窗口。"""
    source = Originals(inputs, root)
    compiled = _compile(source)
    # 身份、合同与参照冻结原件也是回放链的一部分，不能仅保存净值应答。
    registry = {
        digest: {**ref, "path": str((source.root / ref["file"]).resolve()), "published_on": ref.get("published_on")}
        for digest, ref in source.sources.items()
    }
    packet = compiled["packet"]
    packet["r4_evidence_registry"] = registry
    # 研究规则、同行选择底稿和公告发现目录不是发行人市场原件。
    # 保留来源用途，不能为了统一注册表把这些材料标成原始披露。
    kinds = {}
    for role, kind in [
        ("comparison-contract", "RESEARCH_WORKPAPER"),
        ("return-statistic-protocol", "RESEARCH_POLICY"),
        ("historical-time-contract", "RESEARCH_POLICY"),
        ("target-provider", "LICENSED_PROVIDER_ARCHIVE"),
    ]:
        kinds[source.roles[role]["source_hash"]] = kind
    packet["raw_archive"] = [
        {
            "kind": kinds.get(
                digest,
                "DISCOVERY"
                if ref.get("source_kind") == "DISCOVERY_CATALOG_NOT_CUTOFF_EVIDENCE"
                else ref.get("archive_kind", "ORIGINAL"),
            ),
            "sha256": digest,
        }
        for digest, ref in sorted(source.sources.items())
    ]
    for node in packet["quality_metrics"]["observations"].values():
        node["source_refs"] = sorted(registry)
    return {
        "observations": packet["quality_metrics"]["observations"],
        "packet": packet,
        "anchors": compiled["anchors"],
        "audit": compiled["audit"],
        "version": VERSION,
    }


def calculate(inputs: dict, *, root: Path) -> dict:
    """仅复算两项已闭合维度；总分、等级必须为空，其他缺项仍由原批次保留。"""
    from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen

    compiled = compile_observations(inputs, root=root)
    result = _calculate_frozen(compiled["packet"], PROTOCOL, compiled["anchors"], selected_dimensions=SCOPE)
    if result["score"] is not None or result["grade"] is not None:
        raise ValueError("BOSERA_QDII_COMPONENT_MUST_REMAIN_PARTIAL")
    return {
        "version": VERSION,
        "component_scores": result["dimensions"],
        "observations": compiled["observations"],
        "audit": compiled["audit"],
        "score": None,
        "grade": None,
    }


def check_binding(bundle: dict, inputs: dict) -> None:
    """绑定纠错后本C人民币份额，避免把旧Y快照或其他法律产品混入复算。"""
    if (
        bundle.get("family") != "INDEX"
        or bundle.get("fund_code") != inputs.get("fund_code")
        or bundle.get("share_class") != inputs.get("share_class")
        or inputs.get("currency") != "CNY"
        or str(bundle.get("as_of_date")) != inputs.get("as_of_date")
        or str(bundle.get("holding_date")) != inputs.get("holding_date")
        or bundle["identity"]["legal_name"] != inputs.get("legal_name")
        or bundle["identity"]["manager_name"] != inputs.get("manager_name")
        or bundle["original_report_archive"]["sha256"] != inputs.get("primary_report_hash")
    ):
        raise ValueError("BOSERA_QDII_CURRENT_BINDING")


def prepare(session, request: dict, *, root: Path) -> dict:
    """从最新快照追加两项历史统计；未知存续条件、其余维度和原有效期不变。"""
    from datetime import UTC

    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    code, raw = request["fund_code"], deepcopy(request["inputs"])
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("BOSERA_QDII_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    check_binding(bundle, raw)
    if bundle.get("basis") != BASIS or any(
        bundle.get(k)
        for k in (
            "r5_calculation",
            "passive_components_calculation",
            "passive_performance_calculation",
            "stability_calculation",
            "gold_components_calculation",
            "enhanced_performance_calculation",
        )
    ):
        raise ValueError("BOSERA_QDII_PARTIAL_SNAPSHOT_REQUIRED")
    result = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    for key in sorted(SCOPE):
        explanation = request["explanations"][key].strip()
        if len(explanation) < 40:
            raise ValueError("BOSERA_QDII_EXPLANATION_REQUIRED")
        point = result["component_scores"][key]
        doc["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
        bundle.setdefault("public_explanations", {})[key] = explanation
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        bosera_performance_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {ref["sha256"] for ref in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for sha, ref in raw["sources"].items() if sha not in known)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None:
        raise ValueError("BOSERA_QDII_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
