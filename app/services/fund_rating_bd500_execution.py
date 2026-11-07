"""博道伍佰智航C的股票80%/现金短国债5%公开执行原件适配。

仅从inputs.sources经逐字节校验的原件重建20个公开截面和历史适用条款，
不读取研究目录或预计算分数。复用现行R5配置60%/范围40%的正常执行75档，
不认证逐日内部交易，不把期间曾使用期货说成全期无期货。
"""

import calendar
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_allocation import report_product_net_assets
from app.services.fund_quality_v2_balance_proofs import identify_nonnegative_parts, strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_stock_book import periodic_financial_view
from app.services.fund_rating_active_execution import _assets, _stocks
from app.services.fund_rating_bd500_performance import _pages, plain
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_passive_portfolio import portfolio
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS

VERSION = "R5_BD500_STOCK80_CASH5_FULL_CONSTRAINT_PUBLIC_EXECUTION_20261030_2"
REVIEW_HASH = "c6f0d96b74f48ae8402c6b1040f959952d0efd18a8a688e404c6e94ceacd93a1"
REPORT_HASHES = {
    "AN202608311828785447": ["2026-06-30", "HALF", "95efe28e90f2aac77f8fb1066b8b1aa3d272d024546ab1f68895e28e0bed50ef"],
    "AN202603311820894807": [
        "2025-12-31",
        "ANNUAL",
        "cfcd901bf2d10786936a5f8f8da073bc3b1a2d1b41e29edd0cd6f2e67956a9c5",
    ],
    "AN202508291736330082": ["2025-06-30", "HALF", "cabee267177fc1aa51af4a85a7b61cfb8b07ffe78969984cf0b31ccf4b385eda"],
    "AN202503311649466931": [
        "2024-12-31",
        "ANNUAL",
        "61ff43104a7973ebd4924dffdd09bcb070da5b7e91f5677d7d13089fd4e815c3",
    ],
    "AN202408301639600975": ["2024-06-30", "HALF", "2b3890ea6e65b83a57c0c70951d6a5072af5b1fcecd5188f1f282e48c8789be1"],
    "AN202403291629026629": [
        "2023-12-31",
        "ANNUAL",
        "e735360db46e05daa45d23b2c77ea4b625e2981f973dca9832c00ce12eb68b90",
    ],
    "AN202607211827191993": [
        "2026-06-30",
        "QUARTER",
        "8049e4718727b31e7b4bfc465bfbc8f627ced76b05f75c5c78ce30b71f7c6626",
    ],
    "AN202604221821417705": [
        "2026-03-31",
        "QUARTER",
        "b522e5afd0b51cd92e7d16567b9ecbc9a9d72869e35d7aefe53e6b36fa908e8c",
    ],
    "AN202601221818246183": [
        "2025-12-31",
        "QUARTER",
        "b1d111a58fa1ba086460d27eb7fc240ebad40384c10c55d3c8064e67b94cf854",
    ],
    "AN202510281770462662": [
        "2025-09-30",
        "QUARTER",
        "d77ca26ebde7686140305e4a649c891c17d4c221fced8efd83f100151616ee2e",
    ],
    "AN202507211712825240": [
        "2025-06-30",
        "QUARTER",
        "2f9a46b8abd40fede1ced3badf28163ab0ac53263b284a46a56a3af1d0687f85",
    ],
    "AN202504221660338581": [
        "2025-03-31",
        "QUARTER",
        "a29d9b4f0669ce833f897bcb654de3d7fee62dd01ee4005a00f8fb453015dee5",
    ],
    "AN202501221642432991": [
        "2024-12-31",
        "QUARTER",
        "65bf60ad280dc4e290c4cf28ef3ac20da181df798480f340a12c27651148a09d",
    ],
    "AN202410251640482589": [
        "2024-09-30",
        "QUARTER",
        "718f9cb92a8bd3fdd67612183a65c937718dec9c905f7079944b7f5d173b035e",
    ],
    "AN202407191638078412": [
        "2024-06-30",
        "QUARTER",
        "72a7bfc645b2532db39e21ea312d67b9ef3c57d76e394cf2b98cb87b6dfd708e",
    ],
    "AN202404221630951861": [
        "2024-03-31",
        "QUARTER",
        "ffd156e8a776b777430087467b4116927a4359c921ac994bacb6b96317716f36",
    ],
    "AN202401221617894204": [
        "2023-12-31",
        "QUARTER",
        "8cc1997713ab8e9d9efb2575fb3a1530c471a29a032da1b4a2f13179c406a8ce",
    ],
    "AN202310251603449905": [
        "2023-09-30",
        "QUARTER",
        "fac4b33dafe43c3269b297f085cfe82b98761902f8a37923aac04683210d351a",
    ],
    "AN202308311596887684": ["2023-06-30", "HALF", "81eebaa4b6f90769c4ee1e92a35554e58a106ccccced77de804f673161c8ca8e"],
    "AN202307211592526973": [
        "2023-06-30",
        "QUARTER",
        "7d24b9c2bd4dc9dc92774558f8a2f0cae0206759bfd48918c9f0cca57197d634",
    ],
}
CONTRACT_HASHES = {
    "8c0305247644aaf28e21b51f608c3d74de675b73f539d5e9272b2a86c863888e": "2022-11-05",
    "d86cad2cb4dcd614177298ccea746c6120973fb71d8b2209574e1ac1bc6bdee7": "2023-08-14",
    "cba7ad5f8a338a7ca23510aea36a0960a7204d148968cf6f120dfdb5f4f5d8b4": "2023-11-04",
    "815ef0902a271ff316a2c76c8a34fa9f95a89f295e43c4230c240d7fdfe237f1": "2024-11-01",
    "3fc7b050ec4efd0593e43e2fc82d25e85f344de4bde66235014e6acf1ff34cec": "2025-07-26",
    "fdb230f91b98d1b85d14a5b750259d05e611027b05d822622c6ba4c7f9a94859": "2025-09-26",
    "57a84d053e8366d6f106c343bac1376c7c2ad843ae6a7beb4715632a432bc049": "2026-09-24",
}
TERM_HASHES = {
    "019827": ["565c5a644f8ba8baebaac4c229108f9852fcc6890e1b07cd1f099a21008b932a", "2027-01-15"],
    "019792": ["8663270668287ade5ac23eb6c5e3c603d6e9afd6ed4b12808991e200ac26413d", "2026-10-15"],
    "019835": ["f65102f7eac3bade064ae16e95f00b79551a433f034c5e75f4b4d43f300cfc6c", "2027-04-25"],
    "019785": ["df9c30482eaf16481a534ab8276d5282f55c542026ed535b5bef4035b32176f8", "2026-07-25"],
    "019766": ["e464b259c77f8c930d63a9f6ad4264e9cf814cea7d9e97d9a974d6a226805a71", "2026-01-15"],
    "019749": ["6b4d313f20146bdfc6e8932bdea176700ba3f8e2fc17a2cc223e11f298dea5c4", "2025-07-25"],
    "019740": ["2dcdca9e25a7557acb17b901f07b8ab563796304c70809abbde1bec8e45a9cf5", "2025-04-25"],
    "019709": ["d5db9febb9b9b71fd50aa902eff644eb33efafad85ed962ee3f7c2f2f11d94a2", "2024-07-25"],
    "019727": ["5ba3f01fa4341f623e178ae48328bbcb9689f3262cea35db34be3b796089b480", "2024-10-25"],
    "019678": ["a89873f6e8170bed3325a2b7b0708ffa7cb1e233bf7b8f0019224f7fa05db4fd", "2024-06-25"],
    "019703": ["82aa3d2ef2ad20bfde62bffb04377a464bfd3ed71ce82892f10105651615ae43", "2024-04-25"],
    "019679": ["e668ee5832faac58968efe97cd5f8b9d76ce6ac1e87fda775527a3a09be38cb4", "2023-07-15"],
    "019638": ["2552672ef06ed6b21d582f05f13aef1db7585d5b2cb660b47ee6f01c573369c3", "2023-07-02"],
    "019773": ["d842e8ad6088d35c009efc64549b54c0e26f913c13e59788356b8a5ea3a4029f", "2026-04-15"],
}
REGISTRY_HASH = "36b7b544ac4cfe6772078b84a38eeb7f53c60e8bd7cacc4a71e2b0a4f05172b4"
LEGAL = "博道伍佰智航股票型证券投资基金"
MANAGER = "博道基金管理有限公司"
MONEY = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"


def require(condition, code):
    if not condition:
        raise ValueError("R5_BD500_EXECUTION_" + code)


def section(text, start, stop):
    hits = list(re.finditer(start, text))
    require(bool(hits), "SECTION:" + start)
    tail = text[hits[-1].end() :]
    end = re.search(stop, tail)
    require(end is not None, "SECTION_END:" + stop)
    return tail[: end.start()]


def body(source, digest):
    require(digest in source.raw, "SOURCE_MISSING")
    ref = source.sources[digest]
    if ref["file"].lower().endswith((".html", ".htm")):
        return BeautifulSoup(source.raw[digest], "html.parser").get_text(" ", strip=True)
    return "\n".join(_pages(source, digest))


def asset_layout(text):
    # 只恢复原第7行跨行的标签顺序；不改金额、百分数或未知符号。
    pattern = rf"(?m)^[ \t]*银行存款和结算备付金合\s+7\s+计\s+({MONEY}|-)\s+([\d.]+|-)"
    matches = list(re.finditer(pattern, text))
    require(len(matches) <= 1, "ASSET_LAYOUT_AMBIGUOUS")
    if matches:
        m = matches[0]
        text = text[: m.start()] + "7 银行存款和结算备付金合计 " + m[1] + " " + m[2] + text[m.end() :]
    return re.sub(r"银行存款和结算备付金合\s+计", "银行存款和结算备付金合计", text)


def stock_book(text, period):
    try:
        return portfolio([text], holding_date=period)
    except ValueError:
        table = section(
            text,
            r"[78]\.3(?:\.1)?\s*期末按公允价值占基金资产净值比例大小排序的所有股票投资明细",
            r"[78]\.4\s*报告期内股票",
        )
        rows = list(
            re.finditer(
                r"(?m)^\s*(\d{1,4})\s+(\d{6})\s+(.+?)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})\s*$", table
            )
        )
        require(bool(rows) and [int(m[1]) for m in rows] == list(range(1, len(rows) + 1)), "STOCK_ROWS")
        require(len({m[2] for m in rows}) == len(rows), "DUPLICATE_STOCK")
        return {
            "positions": [
                {
                    "stock_code": m[2] + (".SH" if m[2][0] == "6" else ".SZ"),
                    "stock_name": m[3],
                    "market_value_cny": m[5].replace(",", ""),
                    "quantity_shares": m[4].replace(",", ""),
                }
                for m in rows
            ],
            "stock_total": str(sum(D(m[5].replace(",", "")) for m in rows)),
        }


def bonds(text):
    hits = list(
        re.finditer(
            r"([578])\.([56])\s*(?:期末|报告期末)按公允价值占基金资产净值比例大小排序的前五名债券投资明细", text
        )
    )
    require(bool(hits), "BOND_DETAIL_SECTION")
    hit = hits[-1]
    tail = text[hit.end() :]
    end = re.search(r"(?m)^\s*" + re.escape(hit[1] + "." + str(int(hit[2]) + 1)) + r"\s*(?:期末|报告期末)", tail)
    require(end is not None, "BOND_DETAIL_END")
    rows = list(
        re.finditer(
            r"(?m)^\s*(\d)\s+(\d{6})\s+(.+?)\s+([\d,]+)\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})\s*$", tail[: end.start()]
        )
    )
    require(bool(rows) and [int(m[1]) for m in rows] == list(range(1, len(rows) + 1)) and len(rows) <= 5, "BOND_RANKS")
    require(len({m[2] for m in rows}) == len(rows), "DUPLICATE_BOND")
    return [{"security": m[2], "name": m[3], "amount_cny": m[5].replace(",", ""), "source_row": m[0]} for m in rows]


def material_review(inputs, source):
    digest = inputs.get("material_event_review_ref")
    require(digest == REVIEW_HASH and digest in source.raw, "MATERIAL_REVIEW_REQUIRED")
    review = source.json(digest)
    require(
        review.get("fund_code") == "007832"
        and review.get("as_of_date") == inputs["as_of_date"]
        and review.get("primary_ref") == inputs["primary_ref"]
        and review.get("review_closed") is True
        and review.get("blocking_execution_event") is False,
        "MATERIAL_REVIEW_SCOPE",
    )
    catalog = {}
    for h in review["catalog_refs"]:
        raw = source.json(h)
        require(raw["result"]["status"]["code"] == 0, "CATALOG_RESPONSE")
        for row in raw["result"]["data"]["data"] or []:
            require(row["fbsj"][:10] <= inputs["as_of_date"], "FUTURE_CATALOG_EVENT")
            require(row["id"] not in catalog or catalog[row["id"]] == row, "CATALOG_CONFLICT")
            catalog[row["id"]] = row
    events = review["events"]
    require(
        len(events) == 64 and len({e["id"] for e in events}) == 64 and set(catalog) == {e["id"] for e in events},
        "ALL_EVENTS_REQUIRED",
    )
    for event in events:
        row = catalog[event["id"]]
        require(
            event["reviewed_for_execution"] is True
            and event["blocks_execution"] is False
            and event["execution_reason"]
            and event["claims"],
            "UNKNOWN_EVENT",
        )
        require(
            event["published_on"] == row["fbsj"][:10] and plain(event["title"]) == plain(row["title"]),
            "EVENT_DATE_TITLE",
        )
        for claim in event["claims"]:
            require(plain(claim["quote"]) in plain(body(source, claim["source_hash"])), "EVENT_QUOTE")
    return {
        "review_hash": digest,
        "events_reviewed": 64,
        "scope": "配置与投资范围执行；不关闭发行人资产质量或全部经营事项",
    }


def contracts(inputs, source):
    result = []
    for item in inputs["contracts"]:
        h = item["source_hash"]
        ref = source.sources[h]
        p = plain(body(source, h))
        require(CONTRACT_HASHES.get(h) == item["published_on"], "REVIEWED_LEGAL_VERSION_REQUIRED")
        require(
            ref["published_on"] == item["published_on"] and LEGAL in p[:18000] and MANAGER in p[:18000],
            "LEGAL_IDENTITY_DATE",
        )
        require("A类基金份额" in p and "C类基金份额" in p, "LEGAL_SHARE_CLASSES")
        for clause in [
            "股票资产占基金资产的比例不低于80%",
            "到期日在一年以内的政府债券",
            "结算备付金",
            "应收申购款",
            "国内依法发行上市的股票",
            "国债",
        ]:
            require(clause in p, "LEGAL_CLAUSE:" + clause)
        require(re.search(r"基金资产净值(?:的)?5%", p) is not None, "LEGAL_CASH5")
        require(
            re.search(r"持有一家公司发行的证券[,，]其市值不超过基金资产净值的10%", p) is not None,
            "LEGAL_ISSUER_SECURITIES10",
        )
        parent = body(source, item["parent_ref"])
        require(item["published_on"] in parent and item["title"] in parent, "LEGAL_PUBLICATION_BINDING")
        raw_parent = source.raw[item["parent_ref"]].decode("utf-8")
        links = re.findall(r"(?:https?:)?//[^\s\"\x27<>]+\.pdf", raw_parent, re.I)
        require(
            item["pdf_url"] in {urljoin(item["publication_page_url"], link) for link in links},
            "LEGAL_ACTUAL_ATTACHMENT_LINK",
        )
        result.append(
            {
                **item,
                "stock_min_total_asset_pct": "80",
                "cash_short_gov_min_nav_pct": "5",
                "single_legal_issuer_securities_max_nav_pct": "10",
            }
        )
    require(len(result) == 7 and len({r["published_on"] for r in result}) == 7, "LEGAL_VERSION_SEQUENCE")
    return sorted(result, key=lambda r: r["published_on"])


def maturity_terms(inputs, source):
    result = {}
    for code, item in inputs["government_bond_terms"].items():
        require(TERM_HASHES.get(code) == [item["source_hash"], item["maturity"]], "REVIEWED_GOVERNMENT_TERMS_REQUIRED")
        p = plain(body(source, item["source_hash"]))
        maturity = date.fromisoformat(item["maturity"])
        require("固定利率附息债" in p, "GOVERNMENT_FIXED_COUPON")
        hits = list(re.finditer(rf"{maturity.year}年0?{maturity.month}月0?{maturity.day}日", p))
        require(any("偿还本金" in p[m.end() : m.end() + 55] for m in hits), "GOVERNMENT_MATURITY_CLAUSE")
        result[code] = {**item, "actual_terms_checked": True}
    require(len(result) == 14, "GOVERNMENT_TERMS_COUNT")
    return result


def _report(inputs, source, item, profiles, terms, issuer_ids):
    h = item["source_hash"]
    article = source.json(h)["data"]
    entry = item["entry"]
    raw = article["notice_content"]
    require(
        article["art_code"] == entry["id"]
        and article["notice_date"][:10] == entry["published_on"]
        and plain(article["notice_title"]) == plain(entry["title"]),
        "REPORT_PUBLISHER_BINDING",
    )
    require({"007831", "007832"} <= {s["stock"] for s in article["security"]}, "REPORT_SHARE_BINDING")
    period = entry["period_end"]
    require(str(report_period(raw)[1]) == period, "REPORT_PERIOD")
    p = plain(raw)
    require(LEGAL in p[:18000] and "007832" in p[:18000], "REPORT_IDENTITY")
    applicable = [r for r in profiles if r["published_on"] <= period]
    require(bool(applicable), "APPLICABLE_LEGAL_VERSION")
    text = asset_layout(raw)
    assets, total = _assets(text)
    proof = identify_nonnegative_parts(
        str(total), {k: None if v["explicit_dash_zero"] else v["amount_cny"] for k, v in assets.items()}
    )
    require(proof["state"] == "POINT_IDENTIFIED", "ASSET_LEDGER_NOT_IDENTIFIED")
    stock = D(assets["权益投资"]["amount_cny"])
    require(stock / total * 100 >= 80, "STOCK80_NOT_DEMONSTRATED")
    manager = section(text, r"4\.2\s*管理人对报告期内本基金运作遵规守信情况的说明", r"4\.3")
    mp = plain(manager)
    require(
        "基金合同" in manager
        and (
            re.search(r"(?:未发生|不存在|无)损害基金(?:份额)?持有人利益", mp)
            or "基金投资管理符合有关法律法规和基金合同的规定" in mp
        ),
        "ACTUAL_MANAGER_EXECUTION",
    )
    full = entry["kind"] != "QUARTER"
    row = {
        "source_hash": h,
        "period_end": period,
        "kind": entry["kind"],
        "applicable_contract_ref": applicable[-1]["source_hash"],
        "stock_cny": str(stock),
        "total_assets_cny": str(total),
        "stock_total_asset_pct": str(stock / total * 100),
        "asset_equation": proof,
        "manager_quote": manager,
        "qualified_cash_pct": None,
        "daily_compliance_certified": False,
    }
    if not full:
        match = re.search(r"4\.期末基金资产净值\s+((?:" + MONEY + r"\s+){1,8}" + MONEY + r")", text)
        require(match is not None, "QUARTER_NAV_COLUMNS")
        values = re.findall(MONEY, match[1])
        require(len(values) == 2, "QUARTER_AC_COUNT")
        nav = sum(D(v.replace(",", "")) for v in values)
        require(nav > 0, "QUARTER_NAV_POSITIVE")
        top = _stocks(text, False)
        require(len(top) == 10 and max(D(s["amount_cny"]) for s in top) / nav * 100 <= 10, "QUARTER_TOP10_LIMIT")
        row.update(nav_cny=str(nav), quarter_cash_principal=None)
        return row
    view, _ = periodic_financial_view([raw])
    try:
        balance = strict_balance_values(view, observed_on=period)
        require(balance["point_book"] is not None, "NAV_POINT_NOT_IDENTIFIED")
        nav = D(balance["point_book"]["net_assets"])
    except ValueError as error:
        if str(error) != "QUALITY_BALANCE_PROOF_ROW_MISSING_OR_AMBIGUOUS:货币资金":
            raise
        balance = report_product_net_assets([raw], observed_on=period, total_assets=str(total))
        nav = D(balance["net_assets_cny"])
    vtext = "\n".join(view)
    note = section(vtext, r"6\.4\.7\.1\s*(?:货币资金|银行存款)", r"6\.4\.7\.2")
    match = re.search(
        r"(?:活期存款|银行存款)\s+[\d,]+\.\d{2}(?:\s+[\d,]+\.\d{2})?\s+等于[:：]本金\s+([\d,]+\.\d{2})", note
    )
    require(match is not None, "BANK_PRINCIPAL_REQUIRED")
    bank = D(match[1].replace(",", ""))
    restriction = section(vtext, r"6\.4\.12\s*期末", r"6\.4\.13\s*金融工具风险")
    rp = plain(restriction)
    require(
        re.search(r"部分基金资产流通暂时受限制不能自由转让的情况参见附注[67]\.4\.12", p) is not None,
        "RESTRICTION_COMPLETE_SCOPE",
    )
    require(
        not any(s in rp for s in ["货币资金", "银行存款", "其他存款", "现金冻结"]), "BANK_RESTRICTION_REVIEW_REQUIRED"
    )
    for market in ["银行间市场", "交易所市场"]:
        require(
            re.search(market + r"债券正回购(?:本基金本报告期末)?无[。.]", rp) is not None
            or "本基金本报告期末无从事" + market + "债券正回购交易形成的卖出回购证券款余额。" in rp,
            "REPO_PLEDGE_SCOPE",
        )
    govt = D(0)
    eligible = []
    for b in bonds(raw):
        if b["security"] not in terms:
            continue
        term = terms[b["security"]]
        maturity = term["maturity"]
        day = date.fromisoformat(period)
        if period < maturity <= date(day.year + 1, day.month, day.day).isoformat():
            require(source.sources[term["source_hash"]]["published_on"] <= period, "GOVERNMENT_TERMS_AFTER_HOLDING")
            govt += D(b["amount_cny"])
            eligible.append({"position": b, "actual_terms": term})
    ratio = (bank + govt) / nav * 100
    require(ratio >= 5, "QUALIFIED_CASH_SHORT_GOV5_NOT_DEMONSTRATED")
    book = stock_book(raw, period)
    require(D(book["stock_total"]) == stock, "FULL_STOCK_RECONCILIATION")
    mapped = defaultdict(D)
    unknown = D(0)
    for position in book["positions"]:
        code = position["stock_code"]
        require(re.fullmatch(r"(?:00[0-3]|30[01]|60[0135]|68[89])\d{3}\.(?:SH|SZ)", code) is not None, "STOCK_SCOPE")
        identity = issuer_ids.get(code[:6])
        amount = D(position["market_value_cny"])
        if identity:
            mapped[identity] += amount
        else:
            unknown += amount
    categories = section(raw, r"[78]\.5\s*期末按债券品种分类的债券投资组合", r"[78]\.6")
    match = re.search(r"1\s+国家债券\s+([\d,]+\.\d{2})", categories)
    require(match is not None, "GOVERNMENT_CATEGORY")
    non_gov = D(assets["固定收益投资"]["amount_cny"]) - D(match[1].replace(",", ""))
    require(non_gov >= 0, "NON_GOVERNMENT_NEGATIVE")
    upper = (max(mapped.values(), default=D(0)) + unknown + non_gov) / nav * 100
    require(upper <= 10, "LEGAL_ISSUER_SECURITIES10_NOT_DEMONSTRATED")
    require(
        all(D(assets[k]["amount_cny"]) == 0 for k in ["基金投资", "贵金属投资", "金融衍生品投资", "买入返售金融资产"]),
        "PERIOD_END_OTHER_ASSET_SCOPE",
    )
    trustee = section(text, r"5\.2\s*托管人对报告期内本基金投资运作", r"5\.3")
    require(
        re.search(r"未发现基金管理人有损害基金(?:份额)?持有人利益的行为", plain(trustee)) is not None,
        "ACTUAL_TRUSTEE_EXECUTION",
    )
    require(
        re.search(r"前十名股票(?:中[,，]?)?(?:没有|未)超出基金合同(?:规定|约定)的备选股票库(?:之外)?", p) is not None,
        "PUBLIC_TOP10_SCOPE",
    )
    futures = section(raw, r"[78]\.10\s*本基金投资股指期货的投资政策", r"[78]\.11")
    row.update(
        nav_cny=str(nav),
        nav_proof=balance,
        bank_principal_cny=str(bank),
        bank_quote=note,
        eligible_government_lower_bound_cny=str(govt),
        eligible_government_positions=eligible,
        qualified_cash_pct=str(ratio),
        issuer_unknown_upper_bound_cny=str(unknown),
        all_non_government_bonds_upper_bound_cny=str(non_gov),
        single_issuer_securities_upper_bound_nav_pct=str(upper),
        full_stock_positions=len(book["positions"]),
        trustee_quote=trustee,
        period_futures_policy_and_actual_use_quote=futures,
        restrictions=restriction,
    )
    return row


FULL_LEGAL_ORIGINALS = {
    "1706321": [
        "2019-11-08",
        "FULL_HTML_PUBLIC_REPRINT_NO_PDF_ATTACHMENT_LINKED",
        "bc676f1b2a7955a41b92b523b46af6f7814cfd6ad4c8e309b7272cf3f21b194a",
        "bc676f1b2a7955a41b92b523b46af6f7814cfd6ad4c8e309b7272cf3f21b194a",
        "博道伍佰智航股票型证券投资基金托管协议",
    ],
    "3433686": [
        "2023-08-12",
        "PDF_ORIGINAL_PUBLIC_REPRINT",
        "14a30f743823504ccb4a43b5d33449526cd6c52d5bef03edd724d22d39772d9c",
        "5e594e9f0245e2867d5321e85567d0213f5156168120394a080ad9d75ae70967",
        "博道伍佰智航股票型证券投资基金托管协议",
    ],
    "5568578": [
        "2026-09-24",
        "PDF_ORIGINAL_PUBLIC_REPRINT",
        "f626e41db044c3dca734365d2168998b09201e3a2625d5cad54565c363299006",
        "2f3f9cc0b9822ffbae0333bbd5c4ea224f5f9ef526fdf7b632a1f1ec5d1e7931",
        "博道基金管理有限公司博道伍佰智航股票型证券投资基金基金合同",
    ],
    "2108696": [
        "2020-11-07",
        "PDF_ORIGINAL_PUBLIC_REPRINT",
        "f729a6c281d617cb03708cce5d197eaaae9186c7175d12db5c8b4cf157fa8b19",
        "294ac5a6366e6dc20bb80429c52d5ebd5289d1540ec49a0addb2acd8db666005",
        "博道伍佰智航股票型证券投资基金托管协议",
    ],
    "3433711": [
        "2023-08-12",
        "PDF_ORIGINAL_PUBLIC_REPRINT",
        "01a5a95e49839390902312a0f1fcb8e77c67f1a81b6a96ae26cff4e43f607a47",
        "139ca010276f2eba6b157bfec0f4cbfd7287c0cbfda839d9f5f46aeb5de4e88b",
        "博道基金管理有限公司博道伍佰智航股票型证券投资基金基金合同",
    ],
    "2108690": [
        "2020-11-07",
        "PDF_ORIGINAL_PUBLIC_REPRINT",
        "e9610a3ee552089621a627ff3e267dc73437ed62e7c03efec88e2069a21f6b3c",
        "48dd3a8e36477232aed03ebc0e039fc5967c6d0c915a6f242e679f1c7da37871",
        "博道基金管理有限公司博道伍佰智航股票型证券投资基金基金合同",
    ],
    "1706322": [
        "2019-11-08",
        "FULL_HTML_PUBLIC_REPRINT_NO_PDF_ATTACHMENT_LINKED",
        "0eef4327e0f19f9b1c759f6a72f28f2447c6eba1350ec3d328cba91c95c83ade",
        "0eef4327e0f19f9b1c759f6a72f28f2447c6eba1350ec3d328cba91c95c83ade",
        "博道伍佰智航股票型证券投资基金基金合同",
    ],
}


def contract_clauses(pages):
    """仅去除明确重复页眉和独占页码；正文金额和比例一字不改。"""
    cleaned = []
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        if lines and lines[0] == "博道伍佰智航股票型证券投资基金基金合同":
            lines.pop(0)
            if lines and re.fullmatch("\\d{1,3}", lines[0]):
                lines.pop(0)
        cleaned.append("\n".join(lines))
    text = plain("\n".join(cleaned))
    start = text.find("1、组合限制")
    end = text.find("2、禁止行为", start)
    require(0 <= start < end, "FULL_CONSTRAINT_SOURCE_CHECK_23")
    section = text[start:end]
    combination = section[: section.index("除上述")]
    heads = list(re.finditer("\\((\\d{1,2})\\)", combination))
    require(len(heads) in (19, 20), "FULL_CONSTRAINT_SOURCE_CHECK_28")
    require([int(m[1]) for m in heads] == list(range(1, len(heads) + 1)), "FULL_CONSTRAINT_SOURCE_CHECK_29")
    tail = section[heads[-1].end() :]
    tail_end = tail.find("除上述")
    require(tail_end > 0, "FULL_CONSTRAINT_SOURCE_CHECK_32")
    clauses = {
        str(i + 1): section[m.end() : heads[i + 1].start()] if i + 1 < len(heads) else tail[:tail_end]
        for i, m in enumerate(heads)
    }
    return (clauses, section)


def last_section(text, start, end):
    hits = list(re.finditer(start, text))
    require(hits, "FULL_CONSTRAINT_SOURCE_CHECK_18")
    tail = text[hits[-1].end() :]
    stop = re.search(end, tail)
    require(stop is not None, "FULL_CONSTRAINT_SOURCE_CHECK_21")
    return tail[: stop.start()]


def actual_quarter_nav_view(raw):
    """仅给标准读取器正文表，不让目录或业绩文字引用冒充第二张财务表。

    截取原封面、唯一正文产品概况至3.2标题；不更改表格任何金额、单位或日期。
    原始字节、段落坐标仍在外层引用中保留。
    """
    starts = list(re.finditer("(?m)^[ \\t]*§2[ \\t]*基金产品概况[ \\t]*$", raw))
    require(len(starts) == 1, "FULL_CONSTRAINT_SOURCE_CHECK_32")
    s = starts[0].start()
    ends = list(re.finditer("(?m)3\\.2[ \\t]*基金净值表现[ \\t]*$", raw[s:]))
    require(len(ends) == 1, "FULL_CONSTRAINT_SOURCE_CHECK_36")
    e = s + ends[0].end()
    cover_end = raw.index("§1")
    view = raw[:cover_end] + "\n" + raw[s:e]
    require(len(list(re.finditer("3\\.1\\s*主要财务指标", view))) == 1, "FULL_CONSTRAINT_SOURCE_CHECK_40")
    return (
        view,
        {
            "method": "UNCHANGED_COVER_AND_UNIQUE_BODY_PRODUCT_PROFILE_FINANCIAL_TABLE_SLICES",
            "cover_end": cover_end,
            "body_start": s,
            "body_end": e,
            "no_amount_date_unit_or_share_order_edits": True,
        },
    )


def quarter_nav(view, period, total):
    """核A/C同产品净资产两列；本原文日期有04/01前导零，原样匹配而非改金额。

    共用读取器当前拒绝带零日期，局部适配仍要求确切季度、人民币元、列序及
    唯一当期金额行，绝不从比例或比较列反推本产品净资产。
    """
    observed = date.fromisoformat(period)
    q = observed.month // 3
    require(
        q in (1, 2, 3, 4) and observed.day == calendar.monthrange(observed.year, observed.month)[1],
        "FULL_CONSTRAINT_SOURCE_CHECK_54",
    )
    text = plain(view)
    require(f"{observed.year}年第{q}季度报告" in text[:2500], "FULL_CONSTRAINT_SOURCE_CHECK_56")
    financial = view[view.index("3.1") :]
    compact = plain(financial)
    pattern = (
        f"报告期\\({observed.year}年0?{(q - 1) * 3 + 1}月0?1日[-至—～]+{observed.year}年0?{q * 3}月{observed.day}日\\)"
    )
    matched = re.search(pattern, compact)
    require("人民币元" in compact and matched, "FULL_CONSTRAINT_SOURCE_CHECK_62")
    basic = text[: text.index("3.1主要财务指标")]
    names = re.search("下属分级基金的基金简称(.{2,70}?A)(.{2,70}?C)下属分级基金的交易代码", basic)
    codes = re.search("下属分级基金的交易代码(\\d{6})(\\d{6})(?!\\d)", basic)
    require(names and codes and (list(codes.groups()) == ["007831", "007832"]), "FULL_CONSTRAINT_SOURCE_CHECK_66")
    require(names[1] + names[2] in compact[: compact.index("1.本期已实现收益")], "FULL_CONSTRAINT_SOURCE_CHECK_67")
    money = "(\\d[\\d,]*\\.\\d{2})"
    rows = list(re.finditer(f"(?m)^[ \\t]*4\\.[ \\t]*期末基金资产净值[ \\t]+{money}[ \\t]+{money}[ \\t]*$", financial))
    require(len(rows) == 1, "FULL_CONSTRAINT_SOURCE_CHECK_70")
    parts = [part.replace(",", "") for part in rows[0].groups()]
    nav = sum(D(part) for part in parts)
    require(0 < nav <= D(total), "FULL_CONSTRAINT_SOURCE_CHECK_73")
    return {
        "net_assets_cny": str(nav),
        "observed_on": period,
        "unit": "CNY_YUAN",
        "proof": {
            "basis": "EXPLICIT_CURRENT_A_C_NET_ASSETS_SUM",
            "share_codes": list(codes.groups()),
            "share_names": list(names.groups()),
            "share_net_assets_cny": parts,
            "original_row": rows[0][0],
            "period_excerpt": matched[0],
            "local_zero_padded_calendar_date_compatibility": True,
        },
    }


def full_legal_profiles(inputs, source):
    require(len(inputs.get("full_legal_documents", [])) == 7, "SEVEN_FULL_LEGAL_ORIGINALS_REQUIRED")
    profiles = []
    contracts_by_date = {}
    seen = set()
    for item in inputs["full_legal_documents"]:
        ident = item["id"]
        expected = FULL_LEGAL_ORIGINALS.get(ident)
        require(expected is not None and ident not in seen, "FULL_LEGAL_IDENTITY")
        seen.add(ident)
        require(
            [item[k] for k in ["published_on", "source_format", "source_hash", "parent_ref", "title"]] == expected,
            "FULL_LEGAL_FROZEN_VERSION",
        )
        h = item["source_hash"]
        parent = item["parent_ref"]
        require(
            source.sources[h]["published_on"] == item["published_on"] <= inputs["as_of_date"], "FULL_LEGAL_PUBLIC_DATE"
        )
        original = body(source, h)
        text = plain(original)
        parent_body = body(source, parent)
        require(LEGAL in text and MANAGER in text, "FULL_LEGAL_PRODUCT")
        require(item["published_on"] in parent_body and item["title"] in parent_body, "FULL_LEGAL_PARENT_DATE_TITLE")
        if item["source_format"] == "PDF_ORIGINAL_PUBLIC_REPRINT":
            links = re.findall(r"(?:https?:)?//[^\s\"\x27<>]+\.pdf", source.raw[parent].decode("utf-8"), re.I)
            require(
                item["source_url"] in {urljoin(item["parent_url"], link) for link in links},
                "FULL_LEGAL_ATTACHMENT_LINK",
            )
            pages = _pages(source, h)
        else:
            require(
                item["source_format"] == "FULL_HTML_PUBLIC_REPRINT_NO_PDF_ATTACHMENT_LINKED" and h == parent,
                "FULL_LEGAL_HTML_FORM",
            )
            pages = [original]
        if "基金合同" in item["title"]:
            clauses, section = contract_clauses(pages)
            require(len(clauses) == (19 if ident == "1706322" else 20), "FULL_LEGAL_CLAUSE_COUNT")
            contracts_by_date[item["published_on"]] = clauses
            profiles.append(
                {
                    **item,
                    "clauses": clauses,
                    "adjustment_exceptions_and_ten_day_window": section[section.index("除上述") :],
                    "source_type": "FULL_CONTRACT",
                    "not_signed_original_claim": True,
                }
            )
        else:
            quotes = {}
            for anchor in ["所有流通受限证券", "同一流通受限证券", "期货投资托管操作三方备忘录"]:
                at = text.find(anchor)
                require(at >= 0, "CUSTODY_ADDITIONAL_SCOPE:" + anchor)
                quotes[anchor] = text[max(0, at - 25) : at + 180]
            require(re.search(r"所有流通受限证券.{0,80}15%", text) is not None, "CUSTODY_RESTRICTED_TOTAL15")
            require(re.search(r"同一流通受限证券.{0,80}10%", text) is not None, "CUSTODY_SINGLE_RESTRICTED10")
            profiles.append(
                {
                    **item,
                    "source_type": "FULL_CUSTODY",
                    "additional_actual_quotes": quotes,
                    "scope": "流通受限证券定义不含普通停牌、已发行未上市及质押券；仍保留更广主动流动性受限约束。",
                }
            )
    require(
        set(contracts_by_date) == {"2019-11-08", "2020-11-07", "2023-08-12", "2026-09-24"},
        "FULL_CONTRACT_VERSION_SEQUENCE",
    )
    require(
        contracts_by_date["2020-11-07"] == contracts_by_date["2023-08-12"] == contracts_by_date["2026-09-24"],
        "APPLICABLE_TWENTY_FULL_CONSTRAINTS_DIFFER",
    )
    return profiles


def full_constraint_replay(inputs, source, rows, profiles):
    records = []
    quarters = {}
    by_key = {(r["period_end"], r["kind"]): r for r in rows}
    for item in inputs["reports"]:
        entry = item["entry"]
        row = by_key[(entry["period_end"], entry["kind"])]
        raw = source.json(item["source_hash"])["data"]["notice_content"]
        text = plain(raw)
        nav = D(row["nav_cny"])
        gross = D(row["total_assets_cny"])
        abs_quote = last_section(text, r"(?:所有|前十名)资产支持证券投资明细", r"[578]\.\d+(?:报告期末|期末)")
        require(abs_quote in ["无。", "本基金本报告期末未持有资产支持证券。"], "ACTUAL_ABS_HOLDING_COLUMN_REQUIRED")
        result = {
            "source_hash": item["source_hash"],
            "period_end": entry["period_end"],
            "kind": entry["kind"],
            "ABS_no_holding_actual_quote": abs_quote,
            "manager_actual_quote": row["manager_quote"],
            "gross_asset_pct_NAV": str(gross / nav * 100),
            "all_liabilities_upper_pct_NAV": str((gross - nav) / nav * 100),
        }
        require(gross / nav * 100 < 140 and (gross - nav) / nav * 100 < 40, "GROSS140_REPO40_PUBLIC_UPPER_BOUND")
        require(D(row["asset_equation"]["point_values"]["买入返售金融资产"]) == 0, "REVERSE_REPO_ACTUAL_ASSET_SCOPE")
        if entry["kind"] == "QUARTER":
            future = last_section(text, r"5\.9报告期末本基金投资的股指期货交易情况说明", r"5\.10")
            other = last_section(text, r"5\.11\.3其他资产构成", r"5\.11\.4")
            if "本基金本报告期内未进行股指期货投资。" in future:
                status = "EXPLICIT_NO_FUTURES_IN_QUARTER"
            else:
                require(entry["period_end"] == "2023-09-30", "NEW_FUTURES_USE_REVIEW_REQUIRED")
                require(
                    "5.9.1报告期末本基金投资的股指期货持仓和损益明细" in future
                    and "代码名称持仓量合约市公允价值变风险说明(买/卖)值(元)动(元)------" in future,
                    "FUTURES_ACTUAL_NOMINAL_HOLDINGS_TABLE",
                )
                require(
                    "股指期货投资本期收益(元)-18,264.46" in future
                    and "本报告期内,本基金投资股指期货符合既定的投资政策和投资目的。" in future,
                    "FUTURES_PERIOD_USE_PRESERVED",
                )
                require(
                    "1存出保证金-" in other and "5应收申购款992,328.89" in other and "8合计992,328.89" in other,
                    "FUTURES_MARGIN_NONNEGATIVE_BOOK_CLOSURE",
                )
                status = "ACTUAL_PERIOD_USE_EMPTY_END_NOMINAL_TABLE_MARGIN_ZERO_IDENTIFIED"
            result["futures"] = {
                "scope": status,
                "actual_quote": future,
                "other_asset_actual_quote": other,
                "fair_value_zero_alone_not_used": True,
                "not_daily_trade_turnover_certification": True,
            }
            quarters[entry["period_end"]] = result["futures"]
            view, layout = actual_quarter_nav_view(raw)
            nav_proof = quarter_nav(view, entry["period_end"], row["total_assets_cny"])
            require(D(nav_proof["net_assets_cny"]) == nav, "ACTUAL_TWO_SHARE_QUARTER_NAV")
            result["product_nav_proof"] = nav_proof
            result["unchanged_body_view"] = layout
        else:
            risk = last_section(text, r"[67]\.4\.13\.3流动性风险", r"[67]\.4\.13\.4")
            match = re.search(
                r"于(\d{4})年(\d+)月(\d+)日,本基金持有的流动性受限资产的估值占基金资产净值的比例为(\d+\.\d+)%", risk
            )
            require(
                match is not None and "-".join([match[1], match[2].zfill(2), match[3].zfill(2)]) == entry["period_end"],
                "RESTRICTED_RATIO_EXACT_PERIOD",
            )
            upper = D(match[4]) + D("0.005")
            require(upper < 15, "RESTRICTED_TOTAL15_UPPER_BOUND")
            result["restricted_actual_ratio_quote"] = match[0]
            result["restricted_upper_pct_NAV"] = str(upper)
            require(
                all(k in risk for k in ["全部投资组合", "15%", "30%", "质押品", "投资范围保持一致"]),
                "CROSS_PORTFOLIO_AND_COLLATERAL_ACTUAL_CONTROL_DISCLOSURE",
            )
            result["cross_portfolio_and_reverse_repo_actual_control_quote"] = risk
            assets = row["asset_equation"]["point_values"]
            securities = (
                D(assets["权益投资"]) + D(assets["固定收益投资"]) - D(row["eligible_government_lower_bound_cny"])
            )
            require(securities / nav * 100 <= 95, "FUTURES_FINANCING_ALL_SECURITIES95_UPPER_BOUND")
            result["all_non_short_government_securities_upper_pct_NAV"] = str(securities / nav * 100)
            require(
                D(row["single_issuer_securities_upper_bound_nav_pct"]) <= 10, "SINGLE_RESTRICTED_WITHIN_ALL_ISSUER10"
            )
            result["single_issuer_all_securities_upper_pct_NAV"] = row["single_issuer_securities_upper_bound_nav_pct"]
            result["trustee_actual_quote"] = row["trustee_quote"]
        contract_id = "2108690" if entry["period_end"] < "2023-08-12" else "3433711"
        custody_id = "2108696" if entry["period_end"] < "2023-08-12" else "3433686"
        result["applicable_full_contract"] = next(r["source_hash"] for r in profiles if r["id"] == contract_id)
        result["applicable_full_custody"] = next(r["source_hash"] for r in profiles if r["id"] == custody_id)
        records.append(result)
    require(len(quarters) == 13 and len(records) == 20, "FULL_CONSTRAINT_ALL_SNAPSHOTS")
    for result in records:
        if result["kind"] != "QUARTER":
            result["same_date_futures_nominal_scope"] = quarters[result["period_end"]]
    return {
        "records": records,
        "all_twenty_combination_constraints_reviewed": True,
        "quantitative_public_snapshot_checks_rebuilt_from_originals": True,
        "qualitative_scope": {
            "manager_all_portfolio_limits": [4, 13],
            "IPO_order_process": [10],
            "repo_term_and_futures_intraday_turnover_or_three_party_memo": [11, 12],
            "general_rules": [20],
            "basis": "据跨期管理、托管及64项公开事项评价；不宣称内部日账或监管全面认证。",
        },
        "not_private_daily_compliance_certification": True,
    }


def calculate(inputs, *, root):
    require(inputs.get("version") == VERSION, "VERSION")
    require(
        not set(inputs)
        & {"score", "state", "scores", "dimensions", "observations", "allocation_state", "mandate_state"},
        "RAW_INPUTS_ONLY",
    )
    require(
        inputs.get("fund_code") == "007832"
        and inputs.get("share_class") == "C"
        and inputs.get("legal_name") == LEGAL
        and inputs.get("manager_name") == MANAGER
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30",
        "TARGET_SHARE_PERIOD",
    )
    require(1 <= len(inputs["sources"]) <= 300, "SOURCE_COUNT")
    source = Originals(inputs, Path(root))
    primary = plain(body(source, inputs["primary_ref"]))
    require(LEGAL in primary[:18000] and "007832" in primary[:18000], "PRIMARY_IDENTITY")
    event_review = material_review(inputs, source)
    profiles = contracts(inputs, source)
    terms = maturity_terms(inputs, source)
    require(inputs["issuer_registry_ref"] == REGISTRY_HASH, "REVIEWED_REGISTRY_REQUIRED")
    registry = source.json(REGISTRY_HASH)
    issuer_ids = {r["code"]: r["orgId"] for r in registry["stockList"]}
    quarters = {
        f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
        for y in range(2023, 2027)
        for m in [3, 6, 9, 12]
        if "2023-06-30" <= f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}" <= "2026-06-30"
    }
    full = {p for p in quarters if p[5:7] in ["06", "12"]}
    seen = set()
    require(len(inputs["reports"]) == 20, "ALL_PUBLIC_SNAPSHOTS_REQUIRED")
    for item in inputs["reports"]:
        entry = item["entry"]
        key = (entry["period_end"], entry["kind"])
        require(key not in seen, "DUPLICATE_REPORT")
        seen.add(key)
        require(
            REPORT_HASHES.get(entry["id"]) == [entry["period_end"], entry["kind"], item["source_hash"]],
            "REVIEWED_REPORT_REQUIRED",
        )
        require(
            entry["published_on"] == source.sources[item["source_hash"]]["published_on"]
            and entry["published_on"] <= inputs["as_of_date"],
            "REPORT_DATE",
        )
    require(
        {p for p, k in seen if k == "QUARTER"} == quarters and {p for p, k in seen if k != "QUARTER"} == full,
        "NECESSARY_PERIODS_MISSING",
    )
    with localcontext() as context:
        context.prec = 70
        rows = [_report(inputs, source, item, profiles, terms, issuer_ids) for item in inputs["reports"]]
        full_rows = [r for r in rows if r["kind"] != "QUARTER"]
        quarter_rows = {r["period_end"]: r for r in rows if r["kind"] == "QUARTER"}
        for row in full_rows:
            other = quarter_rows[row["period_end"]]
            require(all(row[k] == other[k] for k in ["nav_cny", "stock_cny", "total_assets_cny"]), "SAME_DATE_CONFLICT")
        require(
            "本报告期内按照相关法律法规的规定"
            in plain(
                next(r for r in full_rows if r["period_end"] == "2023-12-31")[
                    "period_futures_policy_and_actual_use_quote"
                ]
            ),
            "FY2023_ACTUAL_FUTURES_USE_MUST_BE_PRESERVED",
        )
        full_profiles = full_legal_profiles(inputs, source)
        full_constraints = full_constraint_replay(inputs, source, rows, full_profiles)
        refs = sorted(h for h, r in source.sources.items() if r.get("origin_type") != "REVIEW_EVIDENCE")
        observations = {
            key: {
                "kind": "state",
                "state": "NORMAL_PUBLIC_EXECUTION",
                "source_refs": refs,
                "reason": "已核完整合同、托管约束及公开截面，重放仓位、现金、集中、杠杆、期货和受限比例。",
            }
            for key in ["allocation", "mandate"]
        }
        packet = {
            "fund_code": "007832",
            "basis": ACTIVE_BASIS,
            "as_of_date": inputs["as_of_date"],
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_metrics": {
                "target_product_aliases": ["007831", "007832"],
                "material_event_pending": False,
                "observations": observations,
            },
            "r4_evidence_registry": {
                h: {**source.sources[h], "path": str((Path(root) / source.sources[h]["file"]).resolve())} for h in refs
            },
            "quality_public": {"dataDates": {"holdings": inputs["holding_date"]}},
            "raw_archive": [{"kind": "ORIGINAL", "sha256": h} for h in refs],
        }
        result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"efficiency"})
    require(result["score"] is None and result["grade"] is None, "COMPONENT_ONLY")
    return {
        "version": VERSION,
        "score": result["dimensions"]["efficiency"],
        "dimensions": result["dimensions"],
        "subscores": result["subscores"],
        "packet": packet,
        "report_ledger": rows,
        "contract_profiles": profiles,
        "material_event_review": event_review,
        "full_legal_profiles": full_profiles,
        "full_constraint_ledger": full_constraints,
        "complete_score": None,
        "grade": None,
        "limits": [
            "仅公开截面及实际履职范围，非逐日内部合规认证。",
            "季报未单列的合格现金本金保持未知；不借邻期本金、结算备付金或利息补数。",
            "保留2023年度期间股指期货使用；期末衍生余额为零不解释成全期无期货。",
            "2023年6月两份报告为36个月观察起点的额外基线，不删任何应有季度或全文。",
        ],
    }
