"""004605原4.6资产质量候选编译：只产Q，不发布、不重放已核185同行。"""

import json
import re
import unicodedata
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.material_content import (
    content_digest,
    document_pdfium,
    source_bytes,
    source_exists,
    source_size,
    source_text,
)

VERSION = "004605_ORIGINAL46_ASSET_QUALITY_CICPA_20261030_1_CANDIDATE"
LARGE_VERIFIED_SOURCE = "8a1635486a767d11221ae47c03b2b4ef032bcaa6678c4f91e7f4ffc3bd1bdac7"
CICPA_TABLE = "042407b4ce1030cb133a32980bbaa42b72e8bfec0148a303394aa0a4f4772e63"
CICPA_RELEASE = "80040314cf0c93d9ee566954af871d92c9ba23dcdd2fe180df6beadade9eecdc"
FULL_REPORT = "6a5e2b00ad587228241d15844bf98e49694f0e067b3f9848926b1549f6f7dcdd"
MONEY = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"


def require(condition, reason):
    if not condition:
        raise ValueError("004605_ORIGINAL46_" + reason)


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def verify_registered_original(inputs, reference, *, root):
    """给标准快照的窄范围原件钩子；只核本组件已绑定的相同file/SHA/kind。"""
    digest = reference["sha256"]
    registered = inputs["sources"].get(digest)
    require(
        registered is not None and {k: registered[k] for k in ["file", "sha256", "kind"]} == reference,
        "REGISTERED_ORIGINAL_REF",
    )
    require(registered["kind"] == "pdf" and registered["archive_kind"] == "ORIGINAL", "REGISTERED_PDF_ONLY")
    # 复用同一流式校验及精确单原件例外，普通归档读取上限无须改变。
    Sources({"sources": {digest: registered}}, root)


class Sources:
    """逐次流式核SHA；唯一52.99MB必要年度原件例外精确限定到字节和摘要。"""

    def __init__(self, inputs, root):
        self.root, self.refs = Path(root).resolve(), inputs["sources"]
        self.json_cache = {}
        require(1 <= len(self.refs) <= 700, "SOURCE_COUNT")
        for digest, row in self.refs.items():
            path = self.path(digest)
            size = source_size(path)
            require(0 < size <= (52990039 if digest == LARGE_VERIFIED_SOURCE else 50_000_000), "SOURCE_SIZE")
            if digest == LARGE_VERIFIED_SOURCE:
                require(size == 52990039, "EXACT_LARGE_ORIGINAL")
            require(content_digest(source_bytes(path)) == digest == row["sha256"], "SOURCE_HASH")

    def path(self, digest):
        row = self.refs[digest]
        path = (self.root / row["file"]).resolve()
        require(path.is_relative_to(self.root / "sources") and source_exists(path), "SOURCE_PATH")
        return path

    def json(self, digest):
        if digest not in self.json_cache:
            self.json_cache[digest] = json.loads(source_text(self.path(digest), encoding="utf-8"))
        return self.json_cache[digest]

    def match_pages(self, observation):
        """原件页码和引文逐项匹配；不会仅凭词命中自动选择q档。"""

        digest = observation["source"]["sha256"]
        with document_pdfium(self.path(digest)) as pdf:
            for item in observation["actual_original_pages_replayed"]:
                page = item["pdf_page"]
                require(1 <= page <= len(pdf), "PDF_PAGE_RANGE")
                actual = pdf[page - 1].get_textpage().get_text_range()
                require(compact(actual) == compact(item["text"]), "ORIGINAL_PAGE_QUOTE")


def _own_amounts(source):
    raw = source.json(FULL_REPORT)
    require(raw.get("success") == 1 and raw["data"]["notice_date"][:10] == "2026-08-31", "FUND_REPORT")
    text = raw["data"]["notice_content"]
    require("004605" in text and "2026 年 06 月 30 日" in text, "FUND_REPORT_IDENTITY")
    heads = list(re.finditer(r"所\s*有\s*股\s*票\s*投\s*资\s*明\s*细", text))
    require(bool(heads), "FULL_STOCK_TABLE")
    head = heads[-1]
    part = text[head.end() :]
    end = re.search(r"(?:\d\.4\s*)?(?:报告期内股票|本期股票)", part)
    require(end is not None, "FULL_STOCK_END")
    part = part[: end.start()]
    pattern = r"(?:^|\n)\s*(\d+)\s+(\d{6})\s*.+?\s+([\d,]+)\s+(" + MONEY + r")\s+([\d.]+)"
    rows = [(int(m[1]), m[2], D(m[4].replace(",", ""))) for m in re.finditer(pattern, part)]
    require([r[0] for r in rows] == list(range(1, 51)), "ALL50_FULL_STOCKS")
    amounts = {r[1]: r[2] for r in rows}
    total = sum(amounts.values())
    asset = text[text.rfind("期末基金资产组合情况") :]
    match = re.search(r"其中\s*[:：]\s*股票\s+(" + MONEY + r")", asset)
    require(match is not None and total == D(match[1].replace(",", "")) == D("644607905.63"), "STOCK_TOTAL")
    return amounts, total


def _cicpa_case(source, card):
    """正式第三方年度意见分类确认，明确保留原生报告正文缺页，不伪造KAM。"""

    review = source.json(card["source_review_ref"]["sha256"])
    independent = review["independent_annual_confirmation"]
    require(
        independent["original"]["sha256"] == CICPA_TABLE and independent["release"]["sha256"] == CICPA_RELEASE,
        "CICPA_OFFICIAL_SOURCE",
    )
    require(
        independent["original"]["url"] == "https://cicpa.org.cn/xxfb/news/202604/W020260424541516739546.pdf",
        "CICPA_URL",
    )
    with document_pdfium(source.path(CICPA_TABLE)) as pdf:
        first, target = [pdf[n].get_textpage().get_text_range() for n in [0, 9]]
    require("上市公司2025年度财务报表审计报告情况明细表" in compact(first), "CICPA_YEAR")
    require(
        re.search(r"203\s+603293\s+埃泰克\s+2026-04-15\s+容诚\s+无保留意见\s+--\s+--", target) is not None,
        "CICPA_TARGET_CLASS",
    )
    release = source_text(source.path(CICPA_RELEASE), encoding="utf-8")
    require("W020260424541516739546.pdf" in release and "2026-04-24" in release, "CICPA_RELEASE_BINDING")
    binding = review["FY2025_native_report_binding"]
    require(
        binding["required_report_no"] == "容诚审字[2026]241Z0014号"
        and binding["actual_signed_on"] == "2026-03-24"
        and binding["public_disclosed_on"] == "2026-04-15"
        and binding["native_missing_opinion_pages_recovered"] is False,
        "CICPA_SCOPE_NOT_OVERSTATED",
    )
    require(
        binding["source"]["sha256"] == "cc1e23c2e87f6f94e308d6c1770a28331ade0cb96149608205c6e5c7a33c3555"
        and binding["source"]["sha256"] in source.refs,
        "NATIVE_REPORT_NOT_BOUND",
    )
    source.match_pages(
        {
            "source": review["latest_half_year_original"],
            "actual_original_pages_replayed": review["actual_half_year_pages_read"],
        }
    )
    return {
        "formal_confirmation": CICPA_TABLE,
        "native_opinion_pages_recovered": False,
        "trade_ref": review["real_trade_source_ref"],
        "trade_row": review["real_trade_observation"],
        "owner_independent_acceptance_still_required": not card["independent_approval_complete"],
    }


def _trade(source, code, digest, row):
    """当前交易状态仅核9月30日真实成交；不替代6月30日持仓估值金额。"""
    raw = source.json(digest)
    require(type(raw.get("code")) is int and raw["code"] == 0, "TRADE_SOURCE_SUCCESS")
    data = raw["data"]
    matches = [
        dict(zip(data["fields"], x, strict=True))
        for x in data["items"]
        if x[data["fields"].index("ts_code")].split(".")[0] == code
    ]
    require(
        len(matches) == 1
        and matches[0] == row
        and row["trade_date"] == "20260930"
        and D(str(row["close"])) > 0
        and D(str(row["amount"])) > 0,
        "REAL_TRADE_ROW_BINDING",
    )


def _nonstock_scope(source, inputs, total):
    """非股票范围从完整资产表、真实存款/清算附注及银行信用原件重新闭合。"""
    from bs4 import BeautifulSoup

    review = source.json(inputs["nonstock_review_ref"])
    text = source.json(FULL_REPORT)["data"]["notice_content"]
    require(review["source_ref"] == FULL_REPORT, "NONSTOCK_ACTUAL_REPORT")
    starts = list(re.finditer(r"(?m)^\s*(?:[67]\.1\s*)?资产负债表\s*$", text))
    require(bool(starts), "NONSTOCK_BALANCE_HEAD")
    tail = text[starts[-1].end() :]
    stop = re.search(r"(?m)^\s*(?:[67]\.2\s*)?利润表\s*$", tail)
    require(stop is not None, "NONSTOCK_BALANCE_END")
    balance = unicodedata.normalize("NFKC", tail[: stop.start()])
    labels = {
        "bank_deposits_including_interest": "货币资金",
        "settlement_reserve": "结算备付金",
        "deposit_margin": "存出保证金",
        "stocks": r"(?:其中:)?股票投资",
        "settlement_receivable": "应收清算款",
        "subscription_receivable": "应收申购款",
    }
    actual = {}
    for key, label in labels.items():
        match = re.search(r"(?m)^\s*" + label + r"\s+(" + MONEY + r")", balance)
        require(match is not None, "NONSTOCK_ASSET_ROW:" + key)
        actual[key] = str(D(match[1].replace(",", "")))
        require(D(actual[key]) == D(review["balance_CNY"][key]), "NONSTOCK_AMOUNT_BINDING")
    assets = re.search(r"(?m)^\s*资产总计\s+(" + MONEY + r")", balance)
    require(
        assets is not None and sum(D(v) for v in actual.values()) == D(assets[1].replace(",", "")),
        "NONSTOCK_FULL_ASSETS_SUM",
    )
    require(D(actual["stocks"]) == total, "NONSTOCK_STOCK_DENOMINATOR")
    for label in [
        "基金投资",
        "债券投资",
        "资产支持证券投资",
        "贵金属投资",
        "其他投资",
        "衍生金融资产",
        "买入返售金融资产",
    ]:
        require(
            re.search(r"(?m)^\s*" + label + r"\s+-\s+-\s*$", balance) is not None, "NONSTOCK_EXPLICIT_ZERO:" + label
        )
    for group, names in [
        ("bank_cash_scope", ["deposit_identity_original", "deposit_note_original"]),
        ("clearing_and_receivable_scope", ["credit_note_original"]),
    ]:
        for name in names:
            require(compact(review[group][name]["text"]) in compact(text), "NONSTOCK_ORIGINAL_QUOTE")
    bank = review["bank_cash_scope"]
    credit = bank["credit_source"]
    require(
        credit["sha256"] == "47c1d3227035cc34f7d3a25f0dffeacdb4b9b1b58cc48640b0f31083358d78d1"
        and credit["url"] == "https://accounts2.ccb.com/chn/home/investor/summary/xypj/index.shtml",
        "BANK_CREDIT_ORIGINAL",
    )
    html = source_text(source.path(credit["sha256"]), encoding="utf-8")
    credit_text = compact(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
    require("2026年9月" in credit_text and "穆迪评级A1P-1稳定" in credit_text, "BANK_CREDIT_PERIOD_AND_OBSERVATION")
    # 银行名称折为“股份有”/“限公司”，四列金额插入其中；金额已独立对账，
    # 此处仅去掉真实金额列后恢复名称，不能用这种处理改写任何金额。
    bank_name_row = compact(re.sub(MONEY, "", bank["deposit_identity_original"]["text"]))
    require(
        "中国建设银行股份有限公司" in bank_name_row
        and "中国证券登记结算有限责任公司"
        in compact(review["clearing_and_receivable_scope"]["credit_note_original"]["text"]),
        "BANK_AND_CLEARING_IDENTITIES",
    )
    require(
        review["bank_or_cash_q_100_added"] is False and review["broker_T"] is None and review["settlement_S"] is None,
        "NONSTOCK_UNKNOWN_NOT_PADDED",
    )
    return {
        "full_asset_amounts_replayed": actual,
        "bank_credit_original_replayed": credit["sha256"],
        "broker_T": None,
        "settlement_S": None,
        "cash_quality_score_added": False,
    }


def calculate(inputs, *, root):
    """复算全50项人工候选和资产质量Q；不把人工判断说成自动审计验收。"""
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "004605"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("share_class") == "C"
        and inputs.get("holding_date") == "2026-06-30",
        "INPUT_SCOPE",
    )
    require(not any(k in inputs for k in ["score", "candidate", "point", "dimensions"]), "EXTERNAL_POINT")
    require(len(inputs["case_refs"]) == len(set(inputs["case_refs"])) == 50, "EXACTLY50_CASE_REFS")
    source = Sources(inputs, root)
    amounts, total = _own_amounts(source)
    cards = [source.json(digest) for digest in inputs["case_refs"]]
    require(
        len(cards) == len({c["security"] for c in cards}) == 50 and {c["security"] for c in cards} == set(amounts),
        "CASE_COVERAGE",
    )
    audits, positions = [], []
    with localcontext() as context:
        context.prec = 60
        numerator = D(0)
        for digest, card in zip(inputs["case_refs"], cards, strict=True):
            code, q = card["security"], card["q_candidate"]
            require(
                card["fund_code"] == "004605"
                and card["as_of"] == "2026-09-30"
                and D(card["market_value_CNY"]) == amounts[code],
                "CASE_IDENTITY_AMOUNT",
            )
            require(
                card["substantive_review_closed"] is True
                and not card["remaining"]
                and type(card["independent_approval_complete"]) is bool
                and q in {50, 75},
                "MANUAL_CANDIDATE_SCOPE",
            )
            require(bool(card.get("manual_reason_original_46", "").strip()), "CASE_SUBSTANTIVE_REASON")
            if card["independent_approval_complete"]:
                adopted = source.json(card["root_adoption_ref"]["sha256"])
                require(
                    adopted["security"] == code
                    and adopted["as_of"] == inputs["as_of_date"]
                    and adopted["q"] == q
                    and adopted["decision"].startswith("ROOT_ADOPTED_COMPANY_QUALITY_"),
                    "INDEPENDENT_ADOPTION_BINDING",
                )
            if code == "603293":
                audit = _cicpa_case(source, card)
                _trade(source, code, audit.pop("trade_ref"), audit.pop("trade_row"))
            else:
                require(set(card["original_page_observations"]) == {"FY2025", "H1_2026"}, "NECESSARY_PERIODS")
                for obs in card["original_page_observations"].values():
                    source.match_pages(obs)
                audit = {"source_periods_replayed": ["FY2025", "H1_2026"]}
                trade = card["real_trade_observation"]
                _trade(source, code, trade["source_ref"], trade["row"])
            require(
                set(inputs["case_source_refs"][code]) <= source.refs.keys()
                and digest in inputs["case_source_refs"][code],
                "CASE_SOURCE_GRAPH",
            )
            require(q != 50 or code in {"688392", "002943"}, "Q50_REQUIRES_REVIEWED_MATERIAL_FACTS")
            numerator += amounts[code] * q
            audits.append(
                {
                    "security": code,
                    "review_ref": digest,
                    "q_manual_candidate": q,
                    "independent_approval_complete": card["independent_approval_complete"],
                    **audit,
                }
            )
            positions.append(
                {
                    "security": code,
                    "amount": str(amounts[code]),
                    "state": {50: "BASIC_ROLE_EVIDENCED", 75: "NORMAL_REVIEWED"}[q],
                    "reason": card["manual_reason_original_46"],
                    "judgment_ref": digest,
                    "source_refs": inputs["case_source_refs"][code],
                }
            )
        quality = numerator / total
    nonstock = _nonstock_scope(source, inputs, total)
    return {
        "version": VERSION,
        "fund_code": "004605",
        "asset_quality_candidate": str(quality),
        "stock_total_CNY": str(total),
        "case_audit": audits,
        "nonstock_original_scope_replay": nonstock,
        "asset_quality_observation": {
            "kind": inputs["observation_kind"],
            "stock_total": str(total),
            "positions": positions,
            "source_refs": sorted(source.refs),
        },
        "new_CICPA_confirmation_not_native_opinion_recovery": True,
        "independent_acceptance_complete": all(c["independent_approval_complete"] for c in cards),
        "holdings_point": None,
        "complete_score": None,
        "grade": None,
        "formal_publication": False,
    }
