"""QDII人民币被动基金收益、稳定性两维原件编译候选。

唯一入口calculate(inputs, root=...)，所有资料只能通过inputs的摘要注册表读取
root/sources。没有本机任务目录、旧候选分数、隐式旁文件或网络依赖；输出只含
两个实际复算分项，完整总分始终为空。公共合入由总指挥执行。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import OrderedDict
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext
from itertools import product as combinations
from pathlib import Path
from statistics import median
from threading import RLock

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy
from app.services.material_content import content_digest, source_bytes, source_exists
from app.services.material_content import document_reader as PdfReader

VERSION = "R5_QDII_RELATIVE_RETURN_STABILITY_20261030_5"
D = Decimal
DIMENSIONS = {"return", "stability"}

# 仅缓存PDF提取文字，库内内容摘要、来源版本、日期和来源角色仍由每次
# Originals构造核验。缓存不持有PDF字节、原件对象、基金分值或来源元数据。
_PDF_TEXT_LIMIT_BYTES = 32 * 1024 * 1024
_PDF_TEXT_MAX_ENTRIES = 128
_PDF_TEXT_CACHE: OrderedDict[tuple[str, int | None], tuple[str, int]] = OrderedDict()
_PDF_TEXT_LOCK = RLock()
_PDF_TEXT_BYTES = 0


def _cached_pdf_text(digest: str, raw: bytes, head_pages: int | None) -> str:
    """对已核PDF按摘要及明确页范围复用纯文本；失败不缓存，内存有界。"""
    global _PDF_TEXT_BYTES
    if not raw.startswith(b"%PDF-") or content_digest(raw) != digest:
        raise ValueError("QDII_ORIGINAL_HASH")
    key = (digest, head_pages)
    # 同摘要同页范围只提取一次。锁外的各Originals已各自重新核原件和日期；
    # 锁只控制确定的纯文本计算，不用先前评分或业务审阅结论替代来源检查。
    with _PDF_TEXT_LOCK:
        cached = _PDF_TEXT_CACHE.get(key)
        if cached is not None:
            _PDF_TEXT_CACHE.move_to_end(key)
            return cached[0]
        from app.services.material_content import document_stream

        with document_stream(raw) as stream:
            pdf = PdfReader(stream)
            pages = pdf.pages if head_pages is None else pdf.pages[:head_pages]
            value = "\n".join(page.extract_text() or "" for page in pages)
        size = len(value.encode("utf-8"))
        if size <= _PDF_TEXT_LIMIT_BYTES:
            _PDF_TEXT_CACHE[key] = (value, size)
            _PDF_TEXT_BYTES += size
            while _PDF_TEXT_BYTES > _PDF_TEXT_LIMIT_BYTES or len(_PDF_TEXT_CACHE) > _PDF_TEXT_MAX_ENTRIES:
                _, (_, removed_size) = _PDF_TEXT_CACHE.popitem(last=False)
                _PDF_TEXT_BYTES -= removed_size
        return value


def compact(value: str) -> str:
    """只统一全半角和排版空白，不删证券数字、日期或否定词。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


class Originals:
    """每次计算核库内提取信息、来源范围和真实取得时间；不要求本地原件。"""

    def __init__(self, inputs: dict, root: Path):
        self.root = root.resolve()
        self.refs = inputs["source_registry"]
        self.raw = {}
        self.text_cache = {}
        research = datetime.fromisoformat(inputs["research_on"])
        if research.tzinfo is None:
            raise ValueError("QDII_RESEARCH_TIMESTAMP_REQUIRED")
        for digest, ref in self.refs.items():
            path = (self.root / ref["file"]).resolve()
            if not path.is_relative_to(self.root / "sources") or not source_exists(path):
                raise ValueError("QDII_ORIGINAL_ROOT_SCOPE")
            raw = source_bytes(path)
            if len(raw) > 50_000_000 or content_digest(raw) != digest or ref["sha256"] != digest:
                raise ValueError("QDII_ORIGINAL_HASH")
            received = datetime.fromisoformat(ref["received_at"])
            if received.tzinfo is None or received > research:
                raise ValueError("QDII_SOURCE_NOT_AVAILABLE_ON_RESEARCH_DATE")
            published = ref.get("available_on")
            if published and published > research.date().isoformat():
                raise ValueError("QDII_FUTURE_PUBLICATION")
            if ref["scope"] == "FUNDAMENTAL" and (not published or published > inputs["as_of_date"]):
                raise ValueError("QDII_FUTURE_FUNDAMENTAL_DOCUMENT")
            self.raw[digest] = raw

    def json(self, digest: str):
        return json.loads(self.raw[digest], parse_float=D)

    def text(self, digest: str, *, head_pages: int | None = None) -> str:
        key = (digest, head_pages)
        if key not in self.text_cache:
            raw = self.raw[digest]
            if raw.startswith(b"%PDF-"):
                text = _cached_pdf_text(digest, raw, head_pages)
            elif self.refs[digest]["format"] == "ISSUER_DOCUMENT_PUBLIC_REPRINT":
                data = json.loads(raw)["data"]
                text = BeautifulSoup(data["notice_content"], "html.parser").get_text("\n", strip=True)
            else:
                text = BeautifulSoup(raw.decode("utf-8-sig"), "html.parser").get_text("\n", strip=True)
            self.text_cache[key] = compact(text)
        return self.text_cache[key]

    def path(self, digest: str) -> str:
        return str((self.root / self.refs[digest]["file"]).resolve())


def _evidence(source: Originals, inputs: dict) -> dict:
    """所有人工适用性判断必须带可在原件定位的原句，不接收自由布尔正常档。"""
    evidence = inputs["evidence"]
    for key, item in evidence.items():
        if not item.get("quotes") or not item.get("source_refs"):
            raise ValueError("QDII_EVIDENCE_REFERENCE_REQUIRED:" + key)
        for quote in item["quotes"]:
            if not compact(quote) or not any(compact(quote) in source.text(ref) for ref in item["source_refs"]):
                raise ValueError("QDII_EVIDENCE_QUOTE_MISMATCH:" + key)
    return evidence


def _events(source: Originals, inputs: dict) -> dict:
    """从逐项原文审查计算已知事项门控，不从治理分或查询未命中推False。"""
    events = inputs["events"]
    expected = inputs["anchors"]["known_event_inventory"]
    ids = [event["event_id"] for event in events]
    if not expected or sorted(ids) != sorted(expected) or len(set(ids)) != len(ids):
        raise ValueError("QDII_KNOWN_EVENT_INVENTORY_NOT_CLOSED")
    pending = []
    anchors = []
    for event in events:
        evidences = event.get("evidence_refs", [])
        if not evidences or not set(evidences) <= inputs["evidence"].keys():
            raise ValueError("QDII_EVENT_ORIGINAL_EVIDENCE_MISSING")
        refs = {ref for eid in evidences for ref in inputs["evidence"][eid]["source_refs"]}
        if event["available_on"] > inputs["research_on"][:10]:
            raise ValueError("QDII_EVENT_AFTER_RESEARCH")
        if event["review_state"] != "ORIGINAL_REVIEWED" or event["decision"] == "PENDING":
            pending.append(event["event_id"])
        category = event["category"]
        if category == "SUBSCRIPTION_RESTRICTION_REDEMPTION_NORMAL":
            plain = "".join(source.text(ref) for ref in refs)
            if inputs["fund_code"] not in plain or not any(
                phrase in plain for phrase in ["赎回及其他业务仍正常办理", "赎回等其他业务仍照常办理"]
            ):
                raise ValueError("QDII_SUBSCRIPTION_EVENT_REDEMPTION_PROOF")
            if event["decision"] != "RETAIN_BUSINESS_RESTRICTION_NO_HISTORICAL_PERFORMANCE_WITHHOLD":
                raise ValueError("QDII_SUBSCRIPTION_EVENT_REVIEW_MISMATCH")
        elif category == "MANAGER_REGULATORY_REMEDIATION_CONTEXT":
            plain = "".join(source.text(ref) for ref in refs)
            if not all(term in plain for term in ["2025年11月28日", "固定收益类公募基金产品注册申请", "完成整改"]):
                raise ValueError("QDII_MANAGER_REMEDIATION_ORIGINAL_REQUIRED")
            if event["decision"] != "RETAIN_DISCLOSED_GOVERNANCE_REMEDIATION_NO_FUND_FAILURE_INFERRED":
                raise ValueError("QDII_MANAGER_REMEDIATION_REVIEW_MISMATCH")
        elif category == "MANAGER_PERSONNEL_CONTEXT":
            if event["decision"] != "RETAIN_CORPORATE_ROLE_CHANGE_NO_FUND_MANAGER_CHANGE_INFERRED":
                raise ValueError("QDII_PERSONNEL_REVIEW_MISMATCH")
        elif category == "FUND_CONFIRMED_MATERIAL_FAILURE":
            if event["status"] == "UNRESOLVED":
                pending.append(event["event_id"])
        else:
            raise ValueError("QDII_EVENT_UNCLASSIFIED")
        if (
            event["applicability"] == "FUND_CONFIRMED"
            and event["status"] == "UNRESOLVED"
            and event["severity"] in {"MATERIAL", "SEVERE"}
        ):
            pending.append(event["event_id"])
        anchors.append(
            {
                **event,
                "source_refs": sorted(refs),
                "historical_information_scope": event["available_on"] <= inputs["as_of_date"],
            }
        )
    return {
        "material_event_pending": bool(pending),
        "pending_ids": sorted(set(pending)),
        "material_event_anchors": anchors,
        "scope": "仅对显式已知事项及其原文进行适用性审查，不宣称监管全面核查或独立整改验收。",
    }


def _request_identity(source: Originals, inputs: dict, member: str, api: str) -> dict:
    """空记录自身没有基金代码，必须从实际保存的请求回执绑定原始回包摘要。"""
    item = inputs["members"][member]
    field = "dividend_ref" if api == "fund_div" else "nav_ref"
    receipt = source.json(item[field + "_request_ref"])
    expected = {"ts_code": item["ts_code"]}
    if api == "fund_nav":
        expected.update(
            start_date=inputs["window_start"].replace("-", ""), end_date=inputs["as_of_date"].replace("-", "")
        )
    response_ref = item[field]
    if (
        receipt.get("api_name") != api
        or receipt.get("params") != expected
        or receipt.get("subject_code") != member
        or receipt.get("state") != "RECEIVED"
        or receipt.get("source_code") != "TUSHARE_PRO_FUND"
        or receipt.get("status_code") != 200
        or receipt.get("sha256") != response_ref
        or receipt.get("bytes") != len(source.raw[response_ref])
        or receipt.get("received_at") != source.refs[response_ref]["received_at"]
        or datetime.fromisoformat(receipt["received_at"]) > datetime.fromisoformat(inputs["research_on"])
    ):
        raise ValueError("QDII_ACTUAL_PROVIDER_REQUEST_IDENTITY_REQUIRED:" + member + ":" + api)
    return receipt


def _split_identity(source: Originals, inputs: dict, member: str) -> dict:
    """核对本份额实际拆分查询页及唯一完整空表，不使用刷新成功布尔代替。"""
    meta = inputs["members"][member]
    receipt = source.json(meta["split_request_ref"])
    url = "https://fundf10.eastmoney.com/fhsp_" + member + ".html"
    if (
        receipt.get("url") != url
        or receipt.get("method") != "GET"
        or receipt.get("params") is not None
        or receipt.get("status_code") != 200
        or receipt.get("sha256") != meta["split_ref"]
        or receipt.get("bytes") != len(source.raw[meta["split_ref"]])
        or receipt.get("received_at") != source.refs[meta["split_ref"]]["received_at"]
    ):
        raise ValueError("QDII_ACTUAL_SPLIT_QUERY_IDENTITY_REQUIRED:" + member)
    soup = BeautifulSoup(source.raw[meta["split_ref"]], "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if "(" + member + ")基金分红送配" not in compact(title):
        raise ValueError("QDII_SPLIT_PAGE_SHARE_IDENTITY:" + member)
    tables = [t for t in soup.find_all("table") if "拆分折算日" in t.get_text()]
    if len(tables) != 1:
        raise ValueError("QDII_SPLIT_TABLE_NOT_UNIQUE:" + member)
    rows = [[compact(c.get_text(" ", strip=True)) for c in r.find_all(["th", "td"])] for r in tables[0].find_all("tr")]
    if rows != [["年份", "拆分折算日", "拆分类型", "拆分折算比例"], ["暂无拆分信息!"]]:
        raise ValueError("QDII_NONEMPTY_OR_UNCLOSED_SPLIT_HISTORY_REQUIRES_EVENT_ADAPTER:" + member)
    return {
        "share": member,
        "request_ref": meta["split_request_ref"],
        "response_ref": meta["split_ref"],
        "scope": "实际份额全部历史拆分空表；不是由event_refresh布尔推断无事件。",
    }


def _ndx100_discovery(row: dict) -> bool:
    """名称或同指数95/5参考只用于联合发现；运作路线必须另核原法律文件。"""

    def index_hint(value):
        return bool(
            re.search(
                r"纳斯达克100|纳指100|NASDAQ[-_]?100|(?<![A-Z0-9])NDX(?![A-Z0-9])", compact(str(value or "")).upper()
            )
        )

    benchmark = compact(str(row.get("benchmark") or ""))
    return index_hint(row["name"]) or (index_hint(benchmark) and "95%" in benchmark and "5%" in benchmark)


def _feeder_quota_proved(quote: str) -> bool:
    """接纳两种已见原句顺序，保留90%的法定分母，不改成研究阈值。"""
    patterns = [
        r"(?:本基金)?(?:投资于目标ETF|目标ETF的投资|目标ETF投资)[^。;；]{0,140}?(?:不低于|不少于)[^。;；]{0,80}?90%",
        r"本基金(?:将)?[^。;；]{0,50}?(?:不低于|不少于)[^。;；]{0,50}?90%[^。;；]{0,50}?投资于目标ETF",
    ]
    return any(re.fullmatch(pattern, quote) for pattern in patterns)


def _cohort(source: Originals, inputs: dict) -> dict:
    """从全部真实目录分页重建候选边界，再核法律产品一次及最早人民币代表。"""
    cohort = source.json(inputs["cohort_ref"])
    archive = source.json(cohort["catalog_archive_ref"])
    pages = archive["items"]
    if len(pages) != 5 or set(cohort["catalog_response_refs"]) != {p["receipts"][0]["sha256"] for p in pages}:
        raise ValueError("QDII_COMPLETE_CATALOG_PAGE_INVENTORY_REQUIRED")
    shards, all_rows = {}, {}
    for page in pages:
        ref = page["receipts"][0]["sha256"]
        raw = source.json(ref)
        if raw.get("code") != 0 or not isinstance(raw.get("data", {}).get("items"), list):
            raise ValueError("QDII_CATALOG_PROVIDER_FAILURE")
        data, params = raw["data"], page["params"]
        rows = [dict(zip(data["fields"], r, strict=True)) for r in data["items"]]
        if rows != page["records"] or len(rows) != page["count"]:
            raise ValueError("QDII_CATALOG_ARCHIVE_ORIGINAL_MISMATCH")
        key = (params["market"], params["status"])
        shards.setdefault(key, []).append((params.get("offset", 0), data["has_more"], rows))
        for row in rows:
            if row["market"] != key[0] or row["status"] != key[1] or row["ts_code"] in all_rows:
                raise ValueError("QDII_CATALOG_SHARD_IDENTITY_OR_DUPLICATE")
            all_rows[row["ts_code"]] = row
    if set(shards) != {("O", "L"), ("O", "D"), ("E", "L"), ("E", "D")}:
        raise ValueError("QDII_COMPLETE_CATALOG_MARKET_STATUS_REQUIRED")
    for _key, slices in shards.items():
        slices.sort(key=lambda r: r[0])
        next_offset = 0
        for i, (offset, more, rows) in enumerate(slices):
            if offset != next_offset or more is not (i < len(slices) - 1):
                raise ValueError("QDII_CATALOG_PAGINATION_NOT_CLOSED")
            next_offset += len(rows)
    if len(all_rows) != cohort["discovery"]["catalog_unique_share_rows"]:
        raise ValueError("QDII_CATALOG_UNIVERSE_COUNT_MISMATCH")
    matched = {r["ts_code"] for r in all_rows.values() if _ndx100_discovery(r)}
    products, covered, peer_codes, ids = cohort["products"], [], [], []
    if cohort.get("unresolved") or cohort.get("review_complete") is not True:
        raise ValueError("QDII_LEGAL_COHORT_SOURCE_REVIEW_PENDING")
    for product in products:
        codes = product["catalog_share_ts_codes"]
        covered.extend(codes)
        ids.append(product["legal_product_main_code"])
        plain = source.text(product["primary_ref"], head_pages=16)
        if (
            compact(product["legal_name"]) not in plain
            or compact(product["identity_quote"]) not in plain
            or not all(all_rows[c]["management"] == product["manager"] for c in codes)
        ):
            raise ValueError("QDII_COHORT_LEGAL_PRODUCT_ORIGINAL_IDENTITY")
        family = product["original_route_family"]
        if family == "ETF_FEEDER":
            full = source.text(product["primary_ref"])
            quota = product.get("mandatory_target_ETF_quota_quote")
            if quota:
                quota = compact(quota)
                if quota not in full or not _feeder_quota_proved(quota):
                    raise ValueError("QDII_TARGET_ETF_ROUTE_MANDATE_ORIGINAL_REQUIRED")
            else:
                declaration = compact(product.get("operating_feeder_declaration_quote") or "")
                if declaration not in full or not re.fullmatch(
                    r"本基金为(?:[^。;；]{0,70}?)ETF(?:的)?联接基金", declaration
                ):
                    raise ValueError("QDII_FEEDER_ROUTE_DECLARATION_ORIGINAL_REQUIRED")
        elif family == "DIRECT_STOCK_INDEX":
            full = source.text(product["primary_ref"])
            declaration = compact(product["operating_route_quote"])
            if (
                declaration not in full
                or "完全复制" not in declaration
                or not any(term in declaration for term in ["股票", "成份股"])
            ):
                raise ValueError("QDII_DIRECT_STOCK_ROUTE_DECLARATION_ORIGINAL_REQUIRED")
            additional = {}
            for summary in product["additional_family_share_summaries"]:
                text = source.text(summary["source_ref"])
                if (
                    compact(product["legal_name"]) not in text
                    or "基金代码" + product["legal_product_main_code"] not in text
                    or compact(summary["subcode_quote"]) not in text
                    or "交易币种美元" not in text
                ):
                    raise ValueError("QDII_DISCOVERED_SIBLING_PRIMARY_LEGAL_IDENTITY_REQUIRED")
                additional[summary["ts_code"]] = summary["source_ref"]
            if not all(c[:6] in full or c in additional for c in codes):
                raise ValueError("QDII_ALL_DISCOVERED_SHARE_IDENTITIES_MUST_BE_BOUND")
        elif family != "DIRECT_ETF":
            raise ValueError("QDII_UNKNOWN_OPERATING_ROUTE_MUST_WITHHOLD")
        status = product["status"]
        if status == "ADMITTED_LEGAL_PRODUCT_REPRESENTATIVE":
            cny = [all_rows[c] for c in codes if "USD" not in all_rows[c]["name"] and "美元" not in all_rows[c]["name"]]
            candidates = sorted(cny, key=lambda r: (r["found_date"], r["ts_code"]))
            if not candidates or not candidates[0]["found_date"]:
                raise ValueError("QDII_EARLIEST_SAME_CURRENCY_LEGAL_REPRESENTATIVE_REQUIRED")
            supplementary = {}
            for item in product["additional_share_summaries"]:
                text = source.text(item["source_ref"])
                subcode = item["ts_code"][:6]
                if (
                    compact(product["legal_name"]) not in text
                    or "基金代码" + product["legal_product_main_code"] not in text
                    or not re.search(r"下属基金(?:交易)?代码" + subcode, text)
                    or "交易币种" + ("美元" if item["currency"] == "USD" else "人民币") not in text
                ):
                    raise ValueError("QDII_ORIGINAL_SUBSHARE_TO_LEGAL_PRODUCT_BINDING_REQUIRED")
                supplementary[item["ts_code"]] = item["source_ref"]
            if candidates[0]["found_date"] > inputs["window_start"].replace("-", "") or not all(
                c[:6] in plain or c in supplementary for c in codes
            ):
                raise ValueError("QDII_EARLIEST_SAME_CURRENCY_LEGAL_REPRESENTATIVE_REQUIRED")
            earliest = [r for r in candidates if r["found_date"] == candidates[0]["found_date"]]
            states = {r["ts_code"]: [0, 1] for r in earliest}
            for proof in product["fee_priority_proofs"]:
                text = source.text(proof["source_ref"])
                quote = compact(proof["quote"])
                if quote not in text or proof["ts_code"] not in states:
                    raise ValueError("QDII_ORDINARY_SHARE_PRIORITY_ORIGINAL_REQUIRED")
                if proof["kind"] == "ORDINARY_NO_SALES_SERVICE_FEE":
                    if not (
                        any(q in quote for q in ["不收取销售服务费", "不计提销售服务费"])
                        and any(q in quote for q in ["A类", "人民币A"])
                    ):
                        raise ValueError("QDII_ORDINARY_SHARE_PRIORITY_ORIGINAL_REQUIRED")
                    states[proof["ts_code"]] = [0]
                elif proof["kind"] == "POSITIVE_SALES_SERVICE_FEE":
                    fee = re.search(r"销售服务费(?:固定(?:比例|费率))?(0\.\d+)%", quote)
                    if not fee or D(fee[1]) <= 0 or supplementary.get(proof["ts_code"]) != proof["source_ref"]:
                        raise ValueError("QDII_ORDINARY_SHARE_PRIORITY_ORIGINAL_REQUIRED")
                    states[proof["ts_code"]] = [1]
                else:
                    raise ValueError("QDII_ORDINARY_SHARE_PRIORITY_ORIGINAL_REQUIRED")
            outcomes = {
                min(zip(priorities, states, strict=True), key=lambda r: (r[0], r[1]))[1]
                for priorities in combinations(*states.values())
            }
            if outcomes != {product["representative"]}:
                raise ValueError("QDII_REPRESENTATIVE_NOT_UNIQUE_UNDER_UNRESOLVED_FEE_STATES")
            selected = all_rows[product["representative"]]
            if "supplementary_currency_identity" in product:
                evidence = product["supplementary_currency_identity"]
                text = source.text(evidence["source_ref"])
                if not all(compact(q) in text for q in evidence["quotes"]):
                    raise ValueError("QDII_REPRESENTATIVE_CURRENCY_ORIGINAL_REQUIRED")
            peer_codes.append(selected["ts_code"][:6])
        elif status in {
            "EXCLUDED_DIFFERENT_ACTUAL_CONTRACT_REFERENCE",
            "EXCLUDED_LEGAL_PRODUCT_INCEPTION_AFTER_WINDOW",
        }:
            quote = compact(product["exclusion_quote"])
            if quote not in plain:
                raise ValueError("QDII_COHORT_EXCLUSION_ORIGINAL_REQUIRED")
            if status == "EXCLUDED_DIFFERENT_ACTUAL_CONTRACT_REFERENCE":
                if not any(term in quote for term in ["总收益指数", "以美元计价"]):
                    raise ValueError("QDII_DIFFERENT_REFERENCE_EXCLUSION_MUST_PROVE_DIFFERENCE")
            else:
                formed = re.search(r"基金合同生效日(20\d{2})年(\d{1,2})月(\d{1,2})日", quote)
                if (
                    not formed
                    or date(*map(int, formed.groups())) <= date.fromisoformat(inputs["window_start"])
                    or min(all_rows[c]["found_date"] for c in codes) <= inputs["window_start"].replace("-", "")
                ):
                    raise ValueError("QDII_LATE_PRODUCT_EXCLUSION_CANNOT_DROP_OLDER_SIBLING")
        elif status == "TARGET_LEGAL_PRODUCT_ALL_SHARES_EXCLUDED":
            if inputs["fund_code"] not in [c[:6] for c in codes]:
                raise ValueError("QDII_TARGET_ALL_SHARE_EXCLUSION_REQUIRED")
        elif status == "EXCLUDED_DIRECT_STOCK_REPLICATION_DIFFERENT_OPERATING_ROUTE":
            full = source.text(product["primary_ref"])
            strategy = compact(product["operating_route_quote"])
            quota_text = source.text(product["legal_stock_quota_ref"])
            quota = compact(product["legal_stock_quota_quote"])
            if (
                family != "DIRECT_STOCK_INDEX"
                or strategy not in full
                or "完全复制" not in strategy
                or quota not in quota_text
                or "80%" not in quota
                or not any(q in quota for q in ["股票", "成份股"])
                or not any(q in quota for q in ["不低于", "不得低于"])
            ):
                raise ValueError("QDII_DIRECT_STOCK_ROUTE_QUOTA_ORIGINAL_REQUIRED")
            if compact(product["legal_name"]) not in quota_text:
                raise ValueError("QDII_DIRECT_STOCK_ROUTE_LEGAL_SUBJECT_REQUIRED")
        elif status == "EXCLUDED_DIRECT_ETF_STRUCTURE_DIFFERENT_OPERATING_ROUTE":
            quote = compact(product["operating_route_quote"])
            identity = compact(product["identity_quote"])
            if (
                family != "DIRECT_ETF"
                or quote not in identity
                or "基金主代码" + product["legal_product_main_code"] not in identity
                or not re.fullmatch(r"基金运作方式(?:契约型[,、])?交易型开放式", quote)
            ):
                raise ValueError("QDII_DIRECT_ETF_OPERATION_ORIGINAL_REQUIRED")
        else:
            raise ValueError("QDII_COHORT_UNCLASSIFIED_PRODUCT")
    if len(covered) != len(set(covered)) or set(covered) != matched or len(ids) != len(set(ids)):
        raise ValueError("QDII_ALL_MATCHED_LEGAL_PRODUCTS_MUST_BE_REVIEWED_ONCE")
    target_route = cohort["target_mandatory_ETF_route"]
    target_quote = compact(target_route["quote"])
    if target_quote not in source.text(target_route["source_ref"]) or not _feeder_quota_proved(target_quote):
        raise ValueError("QDII_TARGET_ETF_ROUTE_MANDATE_ORIGINAL_REQUIRED")
    if len(matched) != cohort["union_candidate_share_rows"] or len(products) != cohort["union_legal_products"]:
        raise ValueError("QDII_FULL_UNION_PRODUCT_INVENTORY_NOT_CLOSED")
    if set(peer_codes) != set(cohort["candidate_codes"]) or inputs["fund_code"] in peer_codes:
        raise ValueError("QDII_COHORT_SELECTED_PEERS_NOT_DERIVED_FROM_ALL_PRODUCTS")
    preselected = source.json(cohort["preselection_record_ref"])
    if (
        set(preselected["preselected_peer_shares"]) != {inputs["members"][c]["ts_code"] for c in peer_codes}
        or preselected["target_exact_share"] != inputs["members"][inputs["fund_code"]]["ts_code"]
    ):
        raise ValueError("QDII_PRESELECTED_LEGAL_REPRESENTATIVE_RECORD_REQUIRED")
    for code in peer_codes:
        receipt = _request_identity(source, inputs, code, "fund_nav")
        if datetime.fromisoformat(preselected["fixed_at"]) > datetime.fromisoformat(receipt["received_at"]):
            raise ValueError("QDII_REPRESENTATIVE_FIXED_AFTER_NEW_NAV_FETCH")
    return {
        "peers": cohort["candidate_codes"],
        "legal_product_ids": ids,
        "catalog_rows": len(all_rows),
        "union_candidate_share_rows": len(matched),
        "matched_legal_products": len(products),
        "matched_feeder_products": sum(p["original_route_family"] == "ETF_FEEDER" for p in products),
        "classified_direct_index_products": sum(p["original_route_family"] == "DIRECT_STOCK_INDEX" for p in products),
        "classified_direct_ETF_products": sum(p["original_route_family"] == "DIRECT_ETF" for p in products),
        "historical_survivorship_bias_closed": False,
        "scope": cohort["boundary"],
    }


def _contracts(source: Originals, inputs: dict) -> dict:
    members = inputs["members"]
    cohort = _cohort(source, inputs)
    peers = cohort["peers"]
    code = inputs["fund_code"]
    if sorted(members) != sorted([code, *peers]) or code in peers or not peers:
        raise ValueError("QDII_FROZEN_COHORT_CLOSURE")
    summary = source.text(inputs["identity_ref"])
    if not all(t in summary for t in ["下属基金交易代码014978", "(QDII)C", "交易币种人民币", "基金代码040046"]):
        raise ValueError("QDII_OWN_C_CNY_IDENTITY")
    audit = {}
    for member, item in members.items():
        if set(item["contract_refs"]) != {"2023", "2024", "2025", "2026"}:
            raise ValueError("QDII_HISTORICAL_CONTRACT_CLOSURE")
        audit[member] = {}
        for year, digest in item["contract_refs"].items():
            plain = source.text(digest, head_pages=16)
            if member not in plain[:20000] or compact(item["legal_name"]) not in plain[:20000]:
                raise ValueError("QDII_CONTRACT_MEMBER_IDENTITY")
            clauses = [
                plain[m.end() : m.end() + 500].split("风险收益特征")[0] for m in re.finditer("业绩比较基准", plain)
            ]
            # 华夏原文用“标的指数收益率”指向同页明确的纳斯达克100目标ETF；不捏造原句。
            matches = [
                c
                for c in clauses
                if all(term in c for term in ["95%", "5%", "存款"])
                and ("纳斯达克100" in c or ("标的指数收益率" in c and "纳斯达克100" in item["legal_name"]))
            ]
            if not matches:
                raise ValueError("QDII_MATCHED_EXPOSURE_CONTRACT_REQUIRED")
            try:
                begin, end = report_period(plain)
                period_quote = "STANDARD_ORIGINAL_REPORT_PERIOD"
            except ValueError as period_error:
                # 首年报告可以明示“(基金合同生效日)”，只接受该原文括注。
                m = re.search(
                    r"本报告期(?:自|从|为)?(20\d{2})年(\d{1,2})月(\d{1,2})日(?:\(基金合同生效日\))?(?:起)?至(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日",
                    plain,
                )
                if not m:
                    raise ValueError("QDII_ACTUAL_REPORT_PERIOD_REQUIRED:" + member + ":" + year) from period_error
                begin = date(*map(int, m.groups()[:3]))
                end = date(int(m[4] or m[1]), int(m[5]), int(m[6]))
                period_quote = m[0]
            required_start = max(year + "-01-01", inputs["window_start"])
            required_end = year + ("-06-30" if year == "2026" else "-12-31")
            if str(begin) > required_start or str(end) != required_end:
                raise ValueError("QDII_ACTUAL_REPORT_PERIOD_NOT_COVERED:" + member + ":" + year)
            audit[member][year] = {
                "source_hash": digest,
                "actual_clause": matches[0],
                "actual_report_start": str(begin),
                "actual_report_end": str(end),
                "report_period_evidence": period_quote,
            }
    return {"peers": peers, "contracts": audit, "cohort_admission": cohort}


def _table(raw: dict, *, require_empty_count=False) -> list[dict]:
    if raw.get("code") != 0 or not isinstance(raw.get("data", {}).get("items"), list):
        raise ValueError("QDII_PROVIDER_RESPONSE_FAILED_NOT_EMPTY_SET")
    data = raw["data"]
    if data.get("has_more") is not False:
        raise ValueError("QDII_PROVIDER_PAGINATION_NOT_CLOSED")
    # 本批NAV原响应count=0但items实际有730/731行，不能把count冒充行数。
    # NAV完整性由共同727日逐日闭合；空分红则同时要求成功、无下一页、count真0。
    if require_empty_count and (data.get("count") != 0 or isinstance(data.get("count"), bool) or data["items"]):
        raise ValueError("QDII_PROVIDER_PAGINATION_NOT_CLOSED")
    return [dict(zip(data["fields"], row, strict=True)) for row in data["items"]]


def _history(source: Originals, inputs: dict, peers: list[str]) -> dict:
    code = inputs["fund_code"]
    calendar = source.json(inputs["calendar_ref"])
    if calendar["fund_code"] != code:
        raise ValueError("QDII_TARGET_CALENDAR_IDENTITY")
    # 旧快照只保留固定日历；净值、分红和拆分均使用本次真实请求身份及原回包。
    h = calendar["history"]
    days, months = h["dates"], h["month_ends"]
    if (
        days != sorted(set(days))
        or len(days) != 727
        or days[0] != inputs["window_start"]
        or days[-1] != inputs["as_of_date"]
        or len(months) != 37
        or not set(months) <= set(days)
    ):
        raise ValueError("QDII_FIXED_COMMON_WINDOW_REQUIRED")
    derived_month_ends = {}
    for day in days:
        derived_month_ends[day[:7]] = day
    if months != list(derived_month_ends.values()):
        raise ValueError("QDII_NOT_LAST_ACTUAL_VALUATION_DAY_OF_MONTH")
    actual_months = [date.fromisoformat(day).year * 12 + date.fromisoformat(day).month for day in months]
    if actual_months != list(range(actual_months[0], actual_months[0] + 37)):
        raise ValueError("QDII_MONTH_WINDOW_GAP")
    values, availability, event_proofs = {}, {}, {}
    for member in [code, *peers]:
        meta = inputs["members"][member]
        div_receipt = _request_identity(source, inputs, member, "fund_div")
        nav_receipt = _request_identity(source, inputs, member, "fund_nav")
        dividends = _table(source.json(meta["dividend_ref"]), require_empty_count=True)
        if dividends or div_receipt.get("row_count") != len(dividends):
            raise ValueError("QDII_NONZERO_DIVIDEND_MUST_USE_FROZEN_CSRC_GROWTH_ADAPTER")
        split = _split_identity(source, inputs, member)
        rows = _table(source.json(meta["nav_ref"]))
        if nav_receipt.get("row_count") != len(rows):
            raise ValueError("QDII_ACTUAL_NAV_REQUEST_ROW_COUNT_MISMATCH")
        event_proofs[member] = {
            "fund_div_request_ref": meta["dividend_ref_request_ref"],
            "fund_div_response_ref": meta["dividend_ref"],
            "split": split,
        }
        mapped, dates = {}, {}
        for row in rows:
            day = date.fromisoformat(row["nav_date"]).isoformat()
            announced = date.fromisoformat(row["ann_date"]).isoformat()
            if (
                row["ts_code"] != meta["ts_code"]
                or day in mapped
                or day > inputs["as_of_date"]
                or announced < day
                or announced > inputs["research_on"][:10]
            ):
                raise ValueError("QDII_NAV_IDENTITY_DUPLICATE_OR_FUTURE")
            triple = [D(str(row[k])) for k in ["unit_nav", "accum_nav", "adj_nav"]]
            if min(triple) <= 0 or len(set(triple)) != 1:
                raise ValueError("QDII_ADJUSTMENT_NOT_CLOSED")
            mapped[day], dates[day] = triple[0], announced
        if not set(days) <= mapped.keys():
            raise ValueError("QDII_COMMON_NAV_DAY_MISSING")
        values[member] = [mapped[d] for d in days]
        availability[member] = {d: dates[d] for d in days}
        if member == code:
            prior = {r["nav_date"]: D(str(r["adjusted_nav"])) for r in h["rows"]}
            if any(prior[d] != mapped[d] for d in days):
                raise ValueError("QDII_TARGET_SNAPSHOT_NAV_REVISION_REVIEW_REQUIRED")
    published = source.json(inputs["own_publisher_ref"])
    if len(published) != 1 or published[0]["titleText"] != "收益率走势":
        raise ValueError("QDII_ISSUER_OWN_CURVE_IDENTITY")
    chart = {
        f"{int(r['year']):04}-{int(r['month']) + 1:02}-{int(r['day']):02}": D(str(r["value"]))
        for r in published[0]["chartsDatas"]
    }
    if len(chart) != len(published[0]["chartsDatas"]) or not set(days) <= chart.keys():
        raise ValueError("QDII_ISSUER_OWN_CALENDAR_GAP")
    errors = [abs((v / values[code][0] - 1) * 100 - chart[d]) for d, v in zip(days, values[code], strict=True)]
    if max(errors) > D(".005"):
        raise ValueError("QDII_ISSUER_OWN_CURVE_NAV_CONFLICT")
    protocol = source.json(inputs["dividend_statistic_protocol_ref"])
    if "NAV(previous_trading_day)" not in protocol["direct_formula"] or protocol.get("no_imputation") is not True:
        raise ValueError("QDII_CSRC_DIVIDEND_STATISTIC_PROTOCOL_REQUIRED")
    return {
        "dates": days,
        "month_ends": months,
        "values": values,
        "available_on": availability,
        "publisher_max_error_percentage_point": str(max(errors)),
        "dividend_statistic": "CSRC_STANDARD_GROWTH_NO_CASH_EVENTS_ALGEBRAICALLY_IDENTICAL",
        "actual_event_request_identities": event_proofs,
    }


def _relative_values(history):
    """直接算真实费后收益和各窗口，不构造、传入或填零未知合同基准。"""
    days, months = history["dates"], history["month_ends"]
    annualization = D("365.25") / D((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days)
    result = {}
    for code, values in history["values"].items():
        nav = dict(zip(days, values, strict=True))
        current = {
            "return_1y": str(nav[days[-1]] / nav[months[-13]] - 1),
            "return_3y": str((values[-1] / values[0]) ** annualization - 1),
        }
        for name, step, first in [("monthly_success", 1, 1), ("annual_success", 12, 12)]:
            current[name] = [
                {"start": months[i - step], "end": months[i], "target": str(nav[months[i]] / nav[months[i - step]] - 1)}
                for i in range(first, 37)
            ]
        result[code] = current
    return result


def calculate(inputs: dict, *, root: Path) -> dict:
    """独立于未知基准复算两维；原固定同行、月份、分红和合同门控全部核准。"""
    inputs = deepcopy(inputs)
    if (
        inputs.get("version") != VERSION
        or inputs.get("fund_code") != "014978"
        or inputs.get("share_class") != "C"
        or inputs.get("currency") != "CNY"
        or inputs.get("as_of_date") != "2026-09-30"
        or inputs.get("window_start") != "2023-09-28"
        or inputs.get("basis") != PASSIVE_BASIS
        or set(inputs["requested_dimensions"]) != DIMENSIONS
    ):
        raise ValueError("QDII_EXPLICIT_COMPONENT_SCOPE")
    if any(
        key in inputs for key in ["scores", "dimensions", "observations", "material_event_pending", "complete_score"]
    ):
        raise ValueError("QDII_RAW_ORIGINAL_INPUTS_ONLY")
    source = Originals(inputs, root)
    event_review = _events(source, inputs)
    if event_review["material_event_pending"]:
        raise ValueError("QDII_MATERIAL_EVENT_REVIEW_PENDING:" + ",".join(event_review["pending_ids"]))
    _evidence(source, inputs)
    stability = source.json(inputs["stability_rule_ref"])
    if (
        "median of admitted peer own-share net total returns" not in stability["reference_main"]
        or stability["monthly_windows"] != 36
        or stability["rolling_12month_windows"] != 25
    ):
        raise ValueError("QDII_ORIGINAL_MEDIAN_STABILITY_RULE_REQUIRED")
    cohort = _contracts(source, inputs)
    with localcontext() as context:
        context.prec = 70
        history = _history(source, inputs, cohort["peers"])
        values = _relative_values(history)
        code, peers = inputs["fund_code"], cohort["peers"]
        nodes = {}
        for name in ["return_1y", "return_3y"]:
            nodes[name] = {
                "kind": "rank",
                "direction": "higher",
                "target": values[code][name],
                "expected_products": peers,
                "peers": [{"product_id": peer, "value": values[peer][name]} for peer in peers],
                "metric_basis": "RAW_NET_RETURN_COMMON_REFERENCE_TRANSLATION_CANCELLED",
            }
        for name, count in [("monthly_success", 36), ("annual_success", 25)]:
            nodes[name] = {
                "kind": "windows",
                "expected_windows": count,
                "rows": [
                    {**row, "reference": str(median(D(values[peer][name][i]["target"]) for peer in peers))}
                    for i, row in enumerate(values[code][name])
                ],
            }
    registry = {
        h: {"path": source.path(h), "sha256": h, "published_on": r.get("available_on")} for h, r in source.refs.items()
    }
    packet = {
        "basis": PASSIVE_BASIS,
        "fund_code": code,
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "observations": nodes,
            "target_product_aliases": inputs["target_product_aliases"],
            "material_event_pending": event_review["material_event_pending"],
        },
        "r4_evidence_registry": registry,
        "raw_archive": [{"kind": "ORIGINAL", "sha256": h} for h, r in source.refs.items() if r["kind"] == "ORIGINAL"],
        "quality_public": {
            "dataDates": {
                "nav": inputs["as_of_date"],
                "holdings": "2026-06-30",
                "manager": "2026-06-30",
                "fees": "2026-09-18",
            }
        },
    }
    anchors = {"material_events": {code: event_review["material_event_anchors"]}}
    computed = _calculate_frozen(packet, PROTOCOL, anchors, selected_dimensions=DIMENSIONS)
    availability = {
        member: {
            "latest_available_on": max(days.values()),
            "after_asof_count": sum(day > inputs["as_of_date"] for day in days.values()),
        }
        for member, days in history["available_on"].items()
    }
    return {
        "version": VERSION,
        "fund_code": code,
        "as_of_date": inputs["as_of_date"],
        "research_on": inputs["research_on"],
        "status": "READY_FOR_INTEGRATION",
        "dimensions": computed["dimensions"],
        "subscores": computed["subscores"],
        "score": None,
        "grade": None,
        "publishable": False,
        "observations": nodes,
        "compiled_anchors": anchors,
        "event_review": event_review,
        "statistic_time_contract": {
            "nav_valuation_cutoff": inputs["as_of_date"],
            "research_on": inputs["research_on"],
            "nav_available_on": availability,
            "pretends_available_on_asof": False,
            "historical_statistics_only": True,
        },
        "audit": {
            "contracts": cohort["contracts"],
            "calendar_count": len(history["dates"]),
            "cohort_admission": cohort["cohort_admission"],
            "actual_event_request_identities": history["actual_event_request_identities"],
            "month_ends_rebuilt_from_actual_calendar": history["month_ends"],
            "peer_statistic": "PER_WINDOW_MEDIAN_OF_COMPLETE_FIXED_PEER_GROUP",
            "relative_return_identity": "(target-B)-median(peers-B)=target-median(peers);共同B符号消去，没有填零基准。",
            "dividend_statistic": history["dividend_statistic"],
            "peer_dividend_has_more_false_and_count_exact": True,
            "own_publisher_max_error_percentage_point": history["publisher_max_error_percentage_point"],
        },
        "conditional_dimensions_excluded": {
            "risk": "逐日合同基准口径一手依据待核准，九个区间精度核对不替代该依据。",
            "efficiency": "逐日合同基准口径一手依据待核准。",
        },
        "standard_calculator_revision": PROTOCOL["revision"],
        "standard_dimension_weights": policy(PASSIVE_BASIS)["weights"],
    }


def compile_observations(inputs: dict, *, root: Path) -> dict:
    """重读原件、重新准入后提供标准观察；不能从外部输入接收分值。

    沿用上方已冻结的V5 calculate完成所有来源、合同、份额和事件核验。
    这里重建其同一标准packet，供总指挥将实际三处重放接入snapshot；
    不构造合同基准，不扩展到风险或效率，也不返回完整总分。
    """
    result = calculate(inputs, root=root)
    registry = {
        digest: {
            "path": str((root.resolve() / ref["file"]).resolve()),
            "sha256": digest,
            "published_on": ref.get("available_on"),
        }
        for digest, ref in inputs["source_registry"].items()
    }
    observations = deepcopy(result["observations"])
    packet = {
        "basis": PASSIVE_BASIS,
        "fund_code": inputs["fund_code"],
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "observations": observations,
            "target_product_aliases": deepcopy(inputs["target_product_aliases"]),
            "material_event_pending": result["event_review"]["material_event_pending"],
        },
        "r4_evidence_registry": registry,
        "raw_archive": [
            {"kind": "ORIGINAL", "sha256": digest}
            for digest, ref in inputs["source_registry"].items()
            if ref["kind"] == "ORIGINAL"
        ],
        "quality_public": {
            "dataDates": {
                "nav": inputs["as_of_date"],
                "holdings": "2026-06-30",
                "manager": "2026-06-30",
                "fees": "2026-09-18",
            }
        },
    }
    return {
        "version": VERSION,
        "packet": packet,
        "anchors": deepcopy(result["compiled_anchors"]),
        "observations": observations,
        "audit": deepcopy(result["audit"]),
        "statistic_time_contract": deepcopy(result["statistic_time_contract"]),
        "component_scores": deepcopy(result["dimensions"]),
    }


def typed_source_normalize(root: Path, reference: dict) -> dict:
    """物理归档结果与来源角色分开，防止JSON物理类型覆盖ORIGINAL研究角色。"""
    from app.services.fund_rating_pipeline_prepare import canonical_source

    archived = canonical_source(root, reference)
    if archived.get("sha256") != reference["sha256"]:
        raise ValueError("HUAAN_QDII_CANONICAL_SOURCE_HASH")
    return {**archived, **{key: reference[key] for key in ("kind", "scope", "format", "received_at", "available_on")}}


def check_binding(bundle: dict, inputs: dict) -> None:
    """绑定最新本C人民币部分包与同一H1原件，保留原两项已评分输入。

    该函数同时供prepare前、snapshot新增后重放使用，不能要求仍只有两项点值。
    当前公告全文JSON与新官方PDF是同一H1报告的两种已核载体，保留各自hash、
    名称和(QDII)后缀；不是把JSON假称PDF，亦不更改原identity/report归档。
    """
    if any(
        bundle.get(key)
        for key in (
            "r5_calculation",
            "passive_components_calculation",
            "passive_performance_calculation",
            "gold_components_calculation",
            "bosera_performance_calculation",
        )
    ):
        raise ValueError("HUAAN_QDII_CONFLICTING_PERFORMANCE_INPUTS")
    if (
        bundle.get("family") != "INDEX"
        or bundle.get("fund_code") != "014978"
        or inputs.get("fund_code") != "014978"
        or bundle.get("share_class") != "C"
        or inputs.get("share_class") != "C"
        or inputs.get("currency") != "CNY"
        or str(bundle.get("as_of_date")) != "2026-09-30"
        or inputs.get("as_of_date") != "2026-09-30"
        or str(bundle.get("holding_date")) != "2026-06-30"
        or bundle["identity"]["legal_name"] != inputs["members"]["014978"]["legal_name"] + "(QDII)"
        or bundle["identity"]["manager_name"] != "华安基金管理有限公司"
        or bundle["original_report_archive"]["sha256"]
        != "56f2f7fe545cb9e2f17bccc87975e82402dbda62511b46fbbb7ecde4fab67f22"
        or inputs["members"]["014978"]["contract_refs"]["2026"]
        != "a9cff5881098d4fd5734d8aad1b53a8979c8c38b2e4d988b259d4e68e63766e1"
    ):
        raise ValueError("HUAAN_QDII_RELATIVE_CURRENT_BINDING")
    document = bundle["reviewed_document"]
    for name, point in {"management": "55", "fee": "75"}.items():
        node = document["dimensions"][name]
        values = [node.get(key) for key in ("lower", "upper", "candidate")] if isinstance(node, dict) else [node]
        if any(value is None or D(str(value)) != D(point) for value in values):
            raise ValueError("HUAAN_QDII_EXISTING_TWO_POINTS_CHANGED")


def prepare(session, request: dict, *, root: Path) -> dict:
    """准备局部更新清单；不写数据库、不延长有效期，公共hooks由总指挥接入。

    request.inputs为完整140源输入，来源path允许指向项目内既有原件。
    canonical_source逐字节核验后归档并保留received_at、available_on及用途；
    导入时刻不冒充公告日或历史可得日。正式使用前snapshot必须已接入下述
    huaan_qdii_relative_calculation字段及其原件重放。
    """
    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    code, raw = request["fund_code"], deepcopy(request["inputs"])
    if code != "014978" or request["as_of_date"] != raw["as_of_date"]:
        raise ValueError("HUAAN_QDII_REQUEST_SCOPE")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("HUAAN_QDII_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(
        next(member["bundle"] for member in previous["input_snapshot"]["members"] if member["fund_code"] == code)
    )
    check_binding(bundle, raw)
    if any(
        (node.get("candidate") is not None if isinstance(node, dict) else node is not None)
        for name, node in bundle["reviewed_document"]["dimensions"].items()
        if name not in {"management", "fee"}
    ):
        raise ValueError("HUAAN_QDII_EXPECTED_TWO_POINT_PARTIAL_BUNDLE")
    if bundle.get("basis") != BASIS or bundle.get("r5_calculation") or bundle.get("huaan_qdii_relative_calculation"):
        raise ValueError("HUAAN_QDII_CURRENT_PARTIAL_SNAPSHOT_REQUIRED")
    for digest, ref in raw["source_registry"].items():
        raw["source_registry"][digest] = typed_source_normalize(root, ref)
    calculated = calculate(raw, root=root)
    document = bundle["reviewed_document"]
    for name in sorted(DIMENSIONS):
        reason = request["explanations"][name].strip()
        if len(reason) < 40:
            raise ValueError("HUAAN_QDII_EXPLANATION_REQUIRED")
        point = calculated["dimensions"][name]
        document["dimensions"][name] = {"lower": point, "upper": point, "candidate": point, "reason": reason}
        bundle.setdefault("public_explanations", {})[name] = reason
        bundle.get("public_metrics", {}).pop(name, None)
    # 仅更新本轮确实补齐的净值观察日和实际同类数，不能继续沿用旧缺项的null/0。
    bundle["data_dates"] = {**bundle["data_dates"], "nav": raw["as_of_date"]}
    bundle["comparison_product_count"] = len(calculated["audit"]["cohort_admission"]["peers"])
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        huaan_qdii_relative_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {ref["sha256"] for ref in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for digest, ref in raw["source_registry"].items() if digest not in known)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None or checked["grade"] is not None:
        raise ValueError("HUAAN_QDII_COMPONENTS_MUST_REMAIN_PARTIAL")
    return {
        "basis": BASIS,
        "entries": [{key: value for key, value in bundle.items() if key not in {"basis", "reviewed_document"}}],
    }


def replay_snapshot(bundle: dict, dimensions: dict, *, root: Path) -> dict:
    """从140份真实原件重放两维并核三界，兼容新增后的本份额正式快照。"""
    raw = bundle["huaan_qdii_relative_calculation"]["inputs"]
    check_binding(bundle, raw)
    result = calculate(raw, root=root)
    for name, point in result["dimensions"].items():
        expected = D(str(point))
        node = dimensions.get(name)
        if not expected.is_finite() or not isinstance(node, dict):
            raise ValueError("HUAAN_QDII_SNAPSHOT_DIMENSION:" + name)
        for bound in ("lower", "upper", "candidate"):
            value = node.get(bound)
            if value is None:
                raise ValueError("HUAAN_QDII_SNAPSHOT_BOUND_MISSING:" + name + ":" + bound)
            try:
                actual = D(str(value))
            except ArithmeticError as error:
                raise ValueError("HUAAN_QDII_SNAPSHOT_BOUND_INVALID:" + name + ":" + bound) from error
            if not actual.is_finite() or abs(actual - expected) > D("1e-55"):
                raise ValueError("HUAAN_QDII_SNAPSHOT_RECALCULATION:" + name + ":" + bound)
    return result
