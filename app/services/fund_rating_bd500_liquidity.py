"""博道伍佰智航C流动性：完整持仓、60日成交、受限和持有人原件编译。

运行时只读取inputs.sources核准的字节，不读研究缓存、不联网、不发布。
同行准入复用已正式核准的固定七产品原件规则，重新核预算而不计算业绩三维。
6月30日持仓配9月30日止60日市场窗口，不能解释为9月30日实时持仓或兑付保证。
"""

import json
import re
import unicodedata
from collections import OrderedDict, defaultdict
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.fund_quality_v2_balance_proofs import identify_nonnegative_parts, strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_restricted_layout import fixed_width_stock_restrictions
from app.services.fund_quality_v2_stock_book import _stock_parser_view, periodic_financial_view
from app.services.fund_rating_bd500_performance import (
    PEERS,
    _calendar,
    _claim,
    _discovery,
    _pages,
    _verify_reports,
    plain,
)
from app.services.fund_rating_bd500_performance import VERSION as COHORT_VERSION
from app.services.fund_rating_bd500_performance import _originals as cohort_originals
from app.services.fund_rating_cmf_liquidity import COMPATIBLE_TYPES, FOUNDATION_GATES
from app.services.fund_rating_cmf_liquidity import PROTOCOL as HOLDER_PROTOCOL
from app.services.fund_rating_direct_index import Originals, _flows, _liabilities
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_passive_portfolio import portfolio
from app.services.fund_rating_public_market import compact, section
from app.services.fund_rating_public_stock_tables import parse
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_r5_rules import holders_score, payment_score
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, _node
from app.services.fund_rating_rules import decimal

VERSION = "R5_BD500_LIQUIDITY_ORIGINALS_20261029_1"
PRIMARY = "1b9355d3a9d82937bfc6a2a0f05019fe0481d383aaa5aadd87942a7222d397fd"
PROSPECTUS = "73a7a6ee727c934636cec6c36230602c43af3c7e1a6c780ae169c59cfd979462"
PDF = "bcf5b03127e65b872e54e2211c8b79732e2f75b1e4750c58f79c2ba4411dfea9"
SCOPE_REVIEW_HASH = "911ee3d8d4a194308c5e1171bab7ffbd62651b81606b20df2177454b2e21bcc7"
MARKET_HASHES = {
    "2026-07-08": "68a4b84462becffd7f404b44820c13535b20f4ee30a1c18c5f388e808e2f607e",
    "2026-07-09": "b889a2feba49f905df6607b939b8f65023c38e268087b507f42ca72ef3047ecd",
    "2026-07-10": "8fb5d99be60f3c1149ab7246f6fb0762b6ed1dc0866cff2e315298beee4f69b8",
    "2026-07-13": "27b73bccd50709e7a69018583aafcac098ee124ed2e41113084462d96702df7e",
    "2026-07-14": "3b53b74a9f21450dc564c690519ae696b5270c51ce1ef6fcb6c2b794042ca055",
    "2026-07-15": "5b3130a455344c73428572ca6ef490ed726a914ca887bf2ef31f35322d68b8ef",
    "2026-07-16": "f07dba62b22daf32ac183c835fa65dce4677494e8da8c92f75d109927e133c1f",
    "2026-07-17": "c4130cf15f7f266ec8d1e349e693961ef4a22f748f8ec04b936a99a27e5271c6",
    "2026-07-20": "a768d341226e4ee6b66e44dbd1e1dcd5e168bda1e150db85c07da35505f48329",
    "2026-07-21": "4ce861f5403f2ef2a7d60b296bfcfa268581302b4248915b36145ea008b4b4fc",
    "2026-07-22": "852674daaac7171f63fa3537c7b6b3f8e37861754bc6eb57554acbb0c9c4aa1d",
    "2026-07-23": "f8555f14e413b2af9e1ef19407283f35c5d43367df484698153117cd422d75f3",
    "2026-07-24": "1bef66d158a66aff935d8df26febe5c6eb3d08a000cc2ab72cdaedc26ed57bbc",
    "2026-07-27": "1d8c2b8a334336770952c70f48461d9691eb5de76952c1b6c9e026a96f31df70",
    "2026-07-28": "95cc166f63a30e0da64e964d6915fe69e664c6d9822110957146c780daa736ae",
    "2026-07-29": "e816d2285243c7ee8f793b9af02787c6c50f6efae9b2c07081a4130ff7a2f2ae",
    "2026-07-30": "2ffc6f452482ec6ffb86b6c5f87d9171a538e13e6c5fcbc9bcc5353ba3532ffd",
    "2026-07-31": "8db0482a644799e5bc9a728f3616151c2e99f913445ec39052d3df111d3c5151",
    "2026-08-03": "9bcadd2aa9e5ff5a0bec68c627f749802aeb053ba16e710d49067727822013b7",
    "2026-08-04": "abcd1633e9120acf27f58b697c8f99a62cea058a43f64ac0db593fb32e0ee4b2",
    "2026-08-05": "7da67d4a929eeb14063e9c81bb44ecbabb8bbbecdda1fcbb4f9fb6286a1f82cd",
    "2026-08-06": "a6fc735cf644c120b775ce0096cdd37e646ce61fe78ff9f325d247f36eec4534",
    "2026-08-07": "fe2f05f75d004fb2f18e9c72990cde585153ab2d4b23736c7fd4abf6bc6914d5",
    "2026-08-10": "cb3a92d809d33d9565e628ed147cf6b61ff0ce37e781df91910e063317a04300",
    "2026-08-11": "d397529bf8aa356f6c4546781910b4a88be5d4738eb9bfce477411443466d994",
    "2026-08-12": "0cb6b756089a3c3c06708389ac301c26a5c9d017cea9a8226d9602394288e95d",
    "2026-08-13": "214070352076ff684124a32e26af5e408ff39efc15c2a5373a35922413171d04",
    "2026-08-14": "fc0bc2cbf5337b26463d71d5f5483f165c2f898a56367c0746aabc2c251230a6",
    "2026-08-17": "d0f91d895ffe04ce49b500f15ce8df21c1897cbf1294c6de333296f9bbb0fe06",
    "2026-08-18": "cab8478ce53d979be5f6f3de4ab365cc31c248a8f0c9c6e84215c3b3e958b0e0",
    "2026-08-19": "3b3b15c154c1b9dbcc641848135a591c79fa3a4230d67228303c80da01a3bd55",
    "2026-08-20": "eafa872c9c84ba6879b2d19ac1bd4a0dc1ad80557d10be0f210e61b693b59a62",
    "2026-08-21": "57020db6eeda871c74bf69e373450a4ace49fa943e0b3ba539adc7a11a95f1fd",
    "2026-08-24": "52f827ecb16b502e9828caaca80c1e4853842a5348c94a68722f4959dc8f1897",
    "2026-08-25": "97356f6db98089afe8325a4f7fbae179a49f5bd68eecea889a7323f235bb895e",
    "2026-08-26": "c83774a6d5b021ad3527a6faf5082e60b4c67b5e07c3d944fc0de4fcada53c34",
    "2026-08-27": "a6912742e225f3128f7d2930cc3a0dcdd2a2d577b445e718c2090a1c0186832d",
    "2026-08-28": "04fbac113a58a48adccdaa3421ef1e300e8c0d245f2905643d87841553d96bf7",
    "2026-08-31": "073ceb77a75b5a99f612d106dd7c6b780a924bebd8b768b57a24b4a57590802e",
    "2026-09-01": "1274072f811be0be21420cf585697a20b43cd8923450795d1c7226b4ef996bf3",
    "2026-09-02": "4efd602628a2167b6855bbb7a97a51cd65677065d621988881958251820d0985",
    "2026-09-03": "ddfb104d39128d645b11f87b8472dca6c59f3ee186749105e76804f9c10e57d3",
    "2026-09-04": "726b9ff5dfc1cf8f951b9691e062a3c9348bc75a3ebbabc4187e211b52b3eb3f",
    "2026-09-07": "6b308d0d529c0dc6b2faa9c35aae025d0d3a7115ca3c9477a2c635a797b989a3",
    "2026-09-08": "2d6a046a671d5837ed1d476dcdecd4061d7eb1f32d05eeca5ccc4876e1b8532d",
    "2026-09-09": "7b0ccf997fb319ef0e7bbc3623d489d817cf90c7375401f1d22814ac141ed5cc",
    "2026-09-10": "e81a6db7d17d63e406a55f0f3511463d1e42c78a98e0d6ceb93b1a7c30c77f54",
    "2026-09-11": "47a9083f06f1bf6f2ec721d15ffe476de32a8c8459bb8433629fc43f0058fe32",
    "2026-09-14": "3ad576a390dbe3bef08ddadc31ddddcada631f92ba771de38f519fc9728c21c0",
    "2026-09-15": "82db5633f58a5e792d08a00d3a31c4ea04297d9ab92bf70a54411bcbffa55be9",
    "2026-09-16": "afe47645962a51d2b51f0bfeccd8bc03bf645a33194a34953f05d9b2d36db50f",
    "2026-09-17": "0be0f9f45b5424050caaba0d57112abb4513fee17c2c78d97e8c2ce1b37a414e",
    "2026-09-18": "db0f0e62d2b552b9a2d7b2403f069ffe2d06493df51129087ecb37f89d23b100",
    "2026-09-21": "d8b902270c656a89d54df52b186a277b9b41ced2996526947c60f1158a39fdbc",
    "2026-09-22": "2106a2367bca35af62a8b4f88532e6c6cd04aca4fc4f784cdf6e7ec8fbcd7dc1",
    "2026-09-23": "859432b2a2eb20a49e9f46d0cab886b0611121bfdaca899c69032d2838f477d3",
    "2026-09-24": "23cd218c46c304caef77cc2939cc9364d0fc6b91c88732cb77a5e252781f1bde",
    "2026-09-28": "dc19f4e5f7156da4978356b557aa0ea5e8e89da97e6c8c32969300e8a26b69d2",
    "2026-09-29": "88d8fe0bc5c272bd7cea2424af0a781f625d2f7a16a38e633cb1db2f9f8f438c",
    "2026-09-30": "aeee7f4b79ab6aba2466c249b7f67912fec1ecbb5a67dc60b6872981541a667b",
}
_CACHE = OrderedDict()


def require(value, code):
    if not value:
        raise ValueError("R5_BD500_LIQUIDITY_" + code)


def literal_section(text, start, stop):
    begin = text.rfind(start)
    end = text.find(stop, begin + len(start))
    require(begin >= 0 and end > begin, "FULL_LITERAL_SECTION")
    return text[begin:end]


def originals(inputs, root):
    """源时点分开处理：原始价格至研究日，基础公告至评级日，底稿不伪装公告。"""
    require(inputs.get("version") == VERSION, "VERSION")
    require(
        not set(inputs) & {"score", "scores", "dimensions", "observations", "state", "complete_score"},
        "RAW_ORIGINALS_ONLY",
    )
    require(
        inputs.get("fund_code") == "007832"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("research_on") == "2026-10-29"
        and inputs.get("primary_ref") == PRIMARY,
        "TARGET_SHARE_PERIOD",
    )
    require(inputs.get("scope_review_ref") == SCOPE_REVIEW_HASH, "REVIEW_BINDING")
    require(
        {r["date"]: r["source_hash"] for r in inputs["market_refs"]} == MARKET_HASHES
        and len(inputs["market_refs"]) == 60,
        "FIXED_60_MARKET_ORIGINALS",
    )
    perf = inputs["cohort_inputs"]
    require(
        perf["version"] == COHORT_VERSION
        and perf["fund_code"] == "007832"
        and all(
            perf[k] == inputs[k]
            for k in ["as_of_date", "holding_date", "research_on", "primary_ref", "legal_name", "manager_name"]
        ),
        "ACCEPTED_COHORT_BINDING",
    )
    require(
        all(h in inputs["sources"] and inputs["sources"][h] == ref for h, ref in perf["sources"].items()),
        "COHORT_SOURCE_BINDING",
    )
    require(len(inputs["sources"]) <= 450, "BOUNDED_SOURCE_COUNT")
    for h, ref in inputs["sources"].items():
        require(
            ref.get("sha256") == h and ref.get("acquired_at") and ref["acquired_at"][:10] <= inputs["research_on"],
            "SOURCE_ACQUISITION",
        )
        if h in perf["sources"]:
            continue
        if h == SCOPE_REVIEW_HASH:
            require(
                ref.get("origin_type") == "REVIEW_EVIDENCE" and not ref.get("published_on"),
                "REVIEW_NOT_ISSUER_PUBLICATION",
            )
        elif h in MARKET_HASHES.values():
            require(not ref.get("published_on") or ref["published_on"] <= inputs["research_on"], "MARKET_PUBLICATION")
        else:
            require(ref.get("published_on") and ref["published_on"] <= inputs["as_of_date"], "FUNDAMENTAL_PUBLICATION")
    cohort = cohort_originals(perf, root)
    source = Originals({**inputs, "as_of_date": inputs["research_on"]}, Path(root))
    return source, cohort


def scope_review(inputs, source):
    """重验完整64项基金事项与停复牌/上市解释，不继承其他维度的布尔结论。"""
    r = source.json(SCOPE_REVIEW_HASH)
    require(
        r.get("fund_code") == "007832"
        and r.get("as_of_date") == inputs["as_of_date"]
        and r.get("primary_ref") == PRIMARY
        and r.get("holder_pressure_scope_complete") is True
        and r.get("confirmed_pressure_effects") == []
        and r.get("unknown_single_holder_ratio_preserved") is True,
        "REVIEW_SCOPE",
    )
    catalog = {}
    for h in r["catalog_refs"]:
        raw = source.json(h)
        require(raw["result"]["status"]["code"] == 0, "CATALOG_STATUS")
        for row in raw["result"]["data"]["data"] or []:
            require(row["fbsj"][:10] <= inputs["as_of_date"], "FUTURE_EVENT")
            require(row["id"] not in catalog or catalog[row["id"]] == row, "CATALOG_CONFLICT")
            catalog[row["id"]] = row
    require(
        len(r["events"]) == 64 and len(catalog) == 64 and {e["id"] for e in r["events"]} == set(catalog),
        "ALL_EVENTS_REQUIRED",
    )
    for e in r["events"]:
        row = catalog[e["id"]]
        require(
            e.get("reviewed_for_liquidity") is True
            and e.get("confirmed_holder_pressure_effect") is False
            and e.get("liquidity_reason")
            and e.get("claims"),
            "EVENT_NOT_REVIEWED",
        )
        require(
            e["published_on"] == row["fbsj"][:10] and plain(e["title"]) == plain(row["title"]), "EVENT_CATALOG_BINDING"
        )
        for claim in e["claims"]:
            _claim(source, claim)
    return r


def _date_in_quote(day, text):
    y, m, d = day.split("-")
    return re.search(y + r"年0?" + str(int(m)) + r"月0?" + str(int(d)) + r"日", plain(text)) is not None


def verified_gap_days(review, source, days):
    """只有原文引用和固定审阅容器共同绑定的停牌/上市日期才处理缺失成交日。"""
    halts = defaultdict(set)
    prelisting = defaultdict(set)
    for row in review["gap_records"]:
        code = row["security"]
        events = []
        for e in row["suspension_events"]:
            text = "\n".join(_pages(source, e["source_hash"]))
            require(code in plain(text)[:6000] and plain(e["clause"]) in plain(text), "GAP_ISSUER_OR_QUOTE")
            require(e["kind"] in {"停牌", "复牌"} and _date_in_quote(e["date"], e["clause"]), "GAP_ACTUAL_DATE")
            events.append((e["date"], e["kind"]))
        events = sorted(set(events))
        for day in row["suspension_days"]:
            before = [(d, k) for d, k in events if d <= day]
            after = [(d, k) for d, k in events if d > day]
            # 不能强求评级日之后的复牌公告：吸收合并公告可明确“连续停牌、
            # 不再交易”。单日停牌则仅准入原文那一天，不向后推演成持续停牌。
            clauses = [
                plain(e["clause"])
                for e in row["suspension_events"]
                if before and e["date"] == before[-1][0] and e["kind"] == "停牌"
            ]
            continuing = any("连续停牌" in q and "不再交易" in q for q in clauses)
            single_day = bool(
                before and before[-1][0] == day and any("停牌一日" in q or "停牌一天" in q for q in clauses)
            )
            require(
                day in days
                and before
                and before[-1][1] == "停牌"
                and (any(k == "复牌" for d, k in after) or continuing or single_day),
                "SUSPENSION_INTERVAL_NOT_CLOSED",
            )
            halts[code].add(day)
        pre = row.get("prelisting_original")
        if row["prelisting_days"]:
            require(pre is not None, "LISTING_ORIGINAL_REQUIRED")
            text = "\n".join(_pages(source, pre["source_hash"]))
            require(
                plain(pre["clause"]) in plain(text)
                and code in plain(text)[:8000]
                and _date_in_quote(pre["date"], pre["clause"])
                and "上市" in pre["clause"],
                "LISTING_ORIGINAL_CLAUSE",
            )
            require(row["prelisting_days"] == [d for d in days if d < pre["date"]], "PRELISTING_DAYS")
            prelisting[code].update(row["prelisting_days"])
    return halts, prelisting


def market_capacity(inputs, source, cohort, review):
    """重建8份完整股票和60个实际市场响应；未知缺行拒收，不筛除难算同行。"""
    perf = inputs["cohort_inputs"]
    days, _ = _calendar(perf, cohort)
    days = days[-60:]
    require(
        days == sorted(MARKET_HASHES) and days[0] == "2026-07-08" and days[-1] == "2026-09-30", "REAL_60_DAY_CALENDAR"
    )
    discovery = _discovery(perf, cohort)
    _, admission = _verify_reports(perf, cohort)
    books = {}
    for member in [perf["target"]] + perf["peers"]:
        refs = [r for r in member["reports"] if r["period_end"] == "2026-06-30" and r["kind"] != "QUARTER"]
        require(len(refs) == 1, "ONE_CURRENT_COMPLETE_REPORT")
        h = refs[0]["source_hash"]
        book = portfolio(_pages(source, h), holding_date=inputs["holding_date"])
        require(
            book["positions"] and all(p["stock_code"].endswith((".SH", ".SZ", ".BJ")) for p in book["positions"]),
            "MAINLAND_COMPLETE_BOOK",
        )
        require(sum(D(p["market_value_cny"]) for p in book["positions"]) == D(book["stock_total"]), "FULL_STOCK_AMOUNT")
        books[member["code"]] = {**book, "source_hash": h}
    require(set(books) == {"007832", *PEERS}, "ALL_SEVEN_PEERS_RETAINED")
    amounts = defaultdict(dict)
    for row in inputs["market_refs"]:
        day = row["date"]
        raw = source.json(row["source_hash"])
        data = raw.get("data", {})
        require(
            type(raw.get("code")) is int and raw["code"] == 0 and data.get("has_more") is False,
            "MARKET_RESPONSE_COMPLETE",
        )
        fields = data.get("fields")
        require(fields and len(fields) == len(set(fields)), "MARKET_FIELDS")
        records = [dict(zip(fields, r, strict=True)) for r in data["items"]]
        require(len({r["ts_code"] for r in records}) == len(records), "MARKET_DUPLICATE_SECURITY")
        for r in records:
            require(date.fromisoformat(r["trade_date"]).isoformat() == day, "MARKET_DAY_IDENTITY")
            if r["amount"] is None:
                continue
            amount = D(str(r["amount"])) * 1000  # 授权行情amount单位为千元，统一成人民币元。
            require(amount >= 0, "NEGATIVE_MARKET_AMOUNT")
            amounts[r["ts_code"]][day] = amount
    halts, prelisting = verified_gap_days(review, source, days)
    audit = {}
    for code, book in books.items():
        total = D(book["stock_total"])
        rows = []
        for p in book["positions"]:
            security = p["stock_code"]
            short = security[:6]
            values = []
            missing = []
            zeros = []
            excluded = []
            for day in days:
                if day in amounts[security]:
                    values.append(amounts[security][day])
                elif day in halts[short]:
                    values.append(D(0))
                    zeros.append(day)
                elif day in prelisting[short]:
                    excluded.append(day)
                else:
                    missing.append(day)
            require(
                not missing and len(values) >= 30 and sum(values) > 0,
                "UNEXPLAINED_MARKET_GAP:" + code + ":" + security + ":" + ",".join(missing),
            )
            value = D(p["market_value_cny"])
            adv = sum(values) / len(values)
            rows.append(
                {
                    "security": security,
                    "value_cny": str(value),
                    "adv_cny": str(adv),
                    "observations": len(values),
                    "verified_suspension_zero_days": zeros,
                    "prelisting_excluded_days": excluded,
                    "contribution": str(value / total * value / adv),
                }
            )
        audit[code] = {
            "positions": len(rows),
            "stock_total_cny": str(total),
            "source_hash": book["source_hash"],
            "K": str(sum(D(r["contribution"]) for r in rows)),
            "rows": rows,
        }
    peers = sorted(PEERS)
    node = {
        "kind": "rank",
        "target": audit["007832"]["K"],
        "direction": "lower",
        "expected_products": ["PUBLIC:" + c for c in peers],
        "peers": [{"product_id": "PUBLIC:" + c, "value": audit[c]["K"]} for c in peers],
        "source_refs": sorted(
            {r["source_hash"] for r in inputs["market_refs"]} | {b["source_hash"] for b in books.values()}
        ),
    }
    return node, {"funds": audit, "window": days, "discovery": discovery, "cohort_admission": admission}


def calculate(inputs, *, root):
    source, cohort = originals(inputs, root)
    key = (str(Path(root).resolve()), json.dumps(inputs, ensure_ascii=False, sort_keys=True))
    if key in _CACHE:
        return deepcopy(_CACHE[key])
    review = scope_review(inputs, source)
    with localcontext() as ctx:
        ctx.prec = 70
        capacity, cap_audit = market_capacity(inputs, source, cohort, review)
        restricted = restriction(source, inputs)
        holder = holder_foundation(source, inputs)
        nodes = {
            "market_capacity": capacity,
            "restricted": restricted["restricted_node"],
            "known_payment": restricted["payment_node"],
            "holders": holder["node"],
        }
        refs = sorted(h for h, r in source.sources.items() if r.get("origin_type") != "REVIEW_EVIDENCE")
        packet = {
            "fund_code": "007832",
            "basis": ACTIVE_BASIS,
            "as_of_date": inputs["as_of_date"],
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_metrics": {
                "target_product_aliases": ["PUBLIC:007831", "PUBLIC:007832"],
                "material_event_pending": False,
                "observations": nodes,
            },
            "r4_evidence_registry": {
                h: {**r, "path": str((Path(root) / r["file"]).resolve())}
                for h, r in source.sources.items()
                if h in refs
            },
            "quality_public": {"dataDates": {"holdings": inputs["holding_date"]}},
            "raw_archive": [{"kind": "ORIGINAL", "sha256": h} for h in refs],
        }
        result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"liquidity"})
    require(result["score"] is None and result["grade"] is None, "COMPONENT_ONLY")
    out = {
        "version": VERSION,
        "score": result["dimensions"]["liquidity"],
        "dimensions": result["dimensions"],
        "subscores": result["subscores"],
        "packet": packet,
        "capacity_audit": cap_audit,
        "restriction_audit": restricted,
        "holder_audit": holder,
        "material_review_hash": SCOPE_REVIEW_HASH,
        "material_events_reviewed": 64,
        "complete_score": None,
        "grade": None,
        "unknown_single_holder_ratio_preserved": True,
        "limits": [
            "持仓、受限及支付为6月30日公开截面；成交窗口截至9月30日，不代替实时持仓或兑付保证。",
            "未披露的单一持有人比例保持未知；75为有完整基础证据时两种可适用结构同分，不是推定分散。",
            "发行人资产质量另核，本组件不生成Q、持仓整维或完整总分。",
        ],
    }
    _CACHE[key] = deepcopy(out)
    while len(_CACHE) > 3:
        _CACHE.popitem(last=False)
    return out


def check_binding(bundle, inputs):
    require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == "007832"
        and bundle.get("share_class") == "C"
        and str(bundle.get("as_of_date")) == inputs["as_of_date"] == "2026-09-30"
        and str(bundle.get("holding_date")) == inputs["holding_date"] == "2026-06-30"
        and bundle["identity"]["legal_name"] == inputs["legal_name"]
        and bundle["identity"]["manager_name"] == inputs["manager_name"]
        and bundle["original_report_archive"]["sha256"] == inputs["primary_ref"] == PRIMARY,
        "CURRENT_BUNDLE_BINDING",
    )


def check_snapshot(bundle, *, root):
    raw = bundle["bd500_liquidity_calculation"]["inputs"]
    check_binding(bundle, raw)
    result = calculate(raw, root=root)
    value = bundle["reviewed_document"]["dimensions"]["liquidity"]
    require(
        all(D(str(value[k])) == D(result["score"]) for k in ["lower", "upper", "candidate"]),
        "SNAPSHOT_ORIGINAL_RECALCULATION",
    )
    return result


def prepare(session, request, *, root):
    """供协调审查后接入；仅替换流动性并保留其余七维，本工作者不调用。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    require(request.get("fund_code") == "007832" and request.get("as_of_date") == "2026-09-30", "REQUEST_SCOPE")
    current = current_inputs(session, ["007832"]).get("007832")
    require(current and str(current["as_of_date"]) == "2026-09-30", "LATEST_CURRENT")
    bundle = deepcopy(next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == "007832"))
    raw = deepcopy(request["inputs"])
    check_binding(bundle, raw)
    require(bundle.get("basis") == BASIS and not bundle.get("r5_calculation"), "REVIEWED_BASIS")
    before = deepcopy(bundle["reviewed_document"]["dimensions"])
    raw["sources"] = {h: canonical_source(Path(root), r) for h, r in raw["sources"].items()}
    raw["cohort_inputs"]["sources"] = {h: deepcopy(raw["sources"][h]) for h in raw["cohort_inputs"]["sources"]}
    result = calculate(raw, root=root)
    reason = request["explanation"].strip()
    require(len(reason) >= 40, "BUSINESS_EXPLANATION")
    bundle["reviewed_document"]["dimensions"]["liquidity"] = {
        "lower": result["score"],
        "upper": result["score"],
        "candidate": result["score"],
        "reason": reason,
    }
    require(
        all(bundle["reviewed_document"]["dimensions"][k] == v for k, v in before.items() if k != "liquidity"),
        "OTHER_SEVEN_CHANGED",
    )
    bundle.setdefault("public_explanations", {})["liquidity"] = reason
    bundle.get("public_metrics", {}).pop("liquidity", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(Path(root), bundle["reviewed_document"]),
        bd500_liquidity_calculation={"inputs": raw, "input_archive": archive_json(Path(root), raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(r for h, r in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=Path(root))
    checked = snapshot_calculate(bundle, root=Path(root))
    check_snapshot(bundle, root=root)
    require(
        checked["score"] is None and checked["grade"] is None and checked["status"] == "PARTIAL", "MUST_REMAIN_PARTIAL"
    )
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


# ORIGINAL_HELPERS_BEGIN
STRUCTURE_POLICY = {
    "version": "QUALITY_GROSS_ASSET_INDUSTRY_20261017_1",
    "denominator": "资产负债表资产总计；负债单列，不抵扣货币资金或制造负行业权重",
    "equity": "完整股票金额按同日CAPCO门类A-S；基金完整行业表逐项金额与全股票金额对账",
    "non_equity": "国债、货币资金、结算备付金、存出保证金及各项应收分别成类；其他资产未知即拒收",
    "target": "合同业绩基准明示的指数权重乘同期完整指数行业权重，活期存款权重记货币资金",
    "rounding": "公开指数舍入残差保留，不归一；半L1原值不截断",
    "arithmetic": "十进制40位有效数字；类别代码排序后逐项求和，不依赖哈希种子或调用方精度",
    "zero": "完整账簿与完整目标的并集外权重按已明确不存在记0；缺行、缺来源不可记0",
    "meaning": "行业及非权益资产类别结构参照偏离；不是合同强制仓位、发行人复制或合规判断",
    "scope": "境内直接股票及国债；基金、信用债、衍生品、回购和其他未穿透投资暂不适配",
}


def gross_asset_book(pages: list[str]) -> dict:
    """固定十进制精度后重建账簿，避免调用方上下文改变同一原件的权重。"""
    with localcontext() as context:
        context.prec = 40
        return _gross_asset_book(pages)


def _gross_asset_book(pages: list[str]) -> dict:
    """逐项对账全资产，再拆出股票行业和非权益类别；未知投资不放入残差桶。

    国债品种表的所有类别必须齐全，只有国家债券允许非零；前五债券仅当逐行
    合计恰好覆盖全部债券余额时才能认作完整债券表。表内“－”保留未知，
    仅由完整、互斥非负科目精确守恒唯一识别的余额另留证明；缺行不作零。
    所有金额单位为人民币元。
    """
    balance_proof = strict_balance_values(pages, observed_on=str(report_period("\n".join(pages))[1]))
    b = balance_proof["point_book"]
    if b is None:
        raise ValueError("QUALITY_STRUCTURE_EXACT_BALANCE_POINT_NOT_IDENTIFIED")
    text = "\n".join(pages)
    stock = parse(pages, b["stock_amount"], b["net_assets"], require_industry=True)
    assets = {key: decimal(value) for key, value in b["assets"].items()}
    if any(v < 0 for v in assets.values()) or decimal(b["fund_amount"]) != 0:
        raise ValueError("QUALITY_STRUCTURE_ASSET_VALUE_OR_FUND_UNSUPPORTED")
    if assets["交易性金融资产"] != decimal(b["stock_amount"]) + decimal(b["bond_amount"]):
        raise ValueError("QUALITY_STRUCTURE_TRADING_ASSETS_NOT_FULLY_DECOMPOSED")
    unsupported = {
        "衍生金融资产",
        "买入返售金融资产",
        "递延所得税资产",
        "其他资产",
        "债权投资",
        "其他债权投资",
        "其他权益工具投资",
    }
    if any(assets.get(key, 0) != 0 for key in unsupported):
        raise ValueError("QUALITY_STRUCTURE_OTHER_ASSETS_NOT_LOOKED_THROUGH")
    # 报告也会将主语写在报告期之前，均是明确当期无衍生资产及负债；
    # 一般投资政策或只披露资产为零仍不能替代负债范围。
    # 同期完整衍生资产/负债附注明示“无”，其语义与未持有相同；只接纳
    # 有准确前后章号的完整独立小节，目录、资产单边为零或孤立横杠不能通过。
    explicit_empty_derivatives = re.search(
        r"6\.4\.7\.3衍生金融资产/负债无[。.]?6\.4\.7\.4买入返售金融资产", compact(text)
    )
    if not (
        re.search(r"(?:本报告期末本基金|本基金本报告期末)未持有衍生金融资产/负债", compact(text))
        or explicit_empty_derivatives
    ):
        raise ValueError("QUALITY_STRUCTURE_DERIVATIVE_LIABILITIES_UNVERIFIED")
    bond_section = section(text, r"(?m)^7\.5\s+期末按债券品种", r"(?m)^7\.6\s")
    rows = {}
    money = r"(?:\d[\d,]*\.\d{2}|[-－—])"
    for m in re.finditer(rf"(?m)^\s*(\d+)\s+([^\n\d]+?)\s+({money})\s+({money})\s*$", bond_section):
        rank, name, amount, _ = m.groups()
        if int(rank) in rows:
            raise ValueError("QUALITY_STRUCTURE_DUPLICATE_BOND_CATEGORY")
        rows[int(rank)] = (compact(name), None if amount in "-－—" else decimal(amount.replace(",", "")))
    labels = [
        "国家债券",
        "央行票据",
        "金融债券",
        "企业债券",
        "企业短期融资券",
        "中期票据",
        "可转债（可交换债）",
        "同业存单",
        "其他",
        "合计",
    ]
    if set(rows) != set(range(1, 11)) or [unicodedata.normalize("NFKC", rows[i][0]) for i in range(1, 11)] != [
        unicodedata.normalize("NFKC", label) for label in labels
    ]:
        raise ValueError("QUALITY_STRUCTURE_BOND_CATEGORIES_INCOMPLETE")
    bond_total = decimal(b["bond_amount"])
    bond_proof = identify_nonnegative_parts(
        str(bond_total), {str(i): str(rows[i][1]) if rows[i][1] is not None else None for i in range(1, 10)}
    )
    if bond_proof["state"] != "POINT_IDENTIFIED":
        raise ValueError("QUALITY_STRUCTURE_BOND_CATEGORIES_POINT_UNIDENTIFIED")
    identified_bonds = {int(key): decimal(value) for key, value in bond_proof["point_values"].items()}
    if (
        identified_bonds[1] != bond_total
        or rows[10][1] is not None
        and rows[10][1] != bond_total
        or any(identified_bonds[i] != 0 for i in range(2, 10))
    ):
        raise ValueError("QUALITY_STRUCTURE_NON_GOVERNMENT_BOND_UNSUPPORTED")
    details = section(text, r"(?m)^7\.6\s+期末按公允价值", r"(?m)^7\.7\s")
    bonds = []
    for m in re.finditer(r"(?m)^\s*(\d+)\s+(\d{6})\s+(.*?)\s+([\d,]+)\s+([\d,]+\.\d{2})\s+([\d.]+)\s*$", details):
        rank, code, name, qty, amount, weight = m.groups()
        bonds.append(
            {
                "rank": int(rank),
                "security_code": code,
                "name": compact(name),
                "quantity_bonds": qty.replace(",", ""),
                "value_cny": amount.replace(",", ""),
                "nav_percent": weight,
                "excerpt": m[0].strip(),
            }
        )
    if (
        len({r["security_code"] for r in bonds}) != len(bonds)
        or sum(decimal(r["value_cny"]) for r in bonds) != bond_total
    ):
        raise ValueError("QUALITY_STRUCTURE_BOND_DETAIL_DOES_NOT_COVER_FULL_AMOUNT")
    industries = {}
    for row in stock["reported_industries"]:
        key = "EQUITY:" + row["industry_code"]
        industries[key] = industries.get(key, 0) + decimal(row["value_cny"])
    amounts = {**industries, "GOVERNMENT_BONDS": bond_total}
    for key, value in assets.items():
        if key != "交易性金融资产":
            amounts["ASSET:" + key] = value
    total = decimal(b["total_assets"])
    if total <= 0 or sum(amounts.values()) != total:
        raise ValueError("QUALITY_STRUCTURE_GROSS_ASSET_LEDGER_NOT_RECONCILED")
    return {
        "balance": b,
        "balance_proof": balance_proof,
        "bond_category_proof": bond_proof
        | {
            "printed_total_cell": str(rows[10][1]) if rows[10][1] is not None else None,
            "parent_bond_total_cny": str(bond_total),
            "original_excerpt": bond_section,
        },
        "positions": stock["holdings"],
        "disclosed_industries": stock["reported_industries"],
        "bonds": bonds,
        "bond_category_excerpt": bond_section,
        "amounts_cny": {k: str(v) for k, v in sorted(amounts.items())},
        "weights": {k: str(v / total) for k, v in sorted(amounts.items())},
        "denominator_cny": str(total),
        "liabilities_cny": b["liabilities"],
        "policy": STRUCTURE_POLICY,
    }


def restriction(source, inputs):
    """核原始字节后返回受限与支付子项；不评价同行承接、不生成流动性整维。"""
    reference = {"sha256": PRIMARY}
    text = source.json(PRIMARY)["data"]["notice_content"]
    plain = normalized_text(text)
    if not all(v in plain[:16000] for v in ["007832", "博道伍佰智航股票型证券投资基金", "2026年06月30日"]):
        raise ValueError("RESTRICTION_REPORT_IDENTITY")
    scope = re.search(
        r"本基金所持部分证券在证券交易所上市[,，]其余亦可在银行间同业市场交易[,，]部分基金资产流通暂时受限制不能自由转让的情况参见附注6\.4\.12",
        plain,
    )
    if not scope:
        raise ValueError("RESTRICTION_FULL_ASSET_REFERENCE_REQUIRED")
    view, trace = periodic_financial_view([text])
    gross = gross_asset_book(view)
    # 所有用于换分的金额均从这份原件重放，外部预解析值仅可用作一致性检查。
    from datetime import date

    from app.services.fund_quality_v2_accounting import accounting_observations

    actual_liabilities = _liabilities(view, gross["balance"]["liabilities"])
    actual_accounting = accounting_observations(view, date(2026, 6, 30))
    liability_proof = None
    accounting = None
    if liability_proof is not None and liability_proof["point_values"] != actual_liabilities["point_values"]:
        raise ValueError("RESTRICTION_LIABILITIES_REPLAY_CONFLICT")
    if accounting is not None and any(
        accounting["observations"][k]["value"] != actual_accounting["observations"][k]["value"]
        for k in ["demand_deposits_principal", "total_liabilities", "net_assets"]
    ):
        raise ValueError("RESTRICTION_ACCOUNTING_REPLAY_CONFLICT")
    liability_proof = actual_liabilities
    accounting = actual_accounting
    positions = gross["positions"]
    stock_view, edits = _stock_parser_view(view, positions)
    stock_text = "\n".join(stock_view)
    note = section(stock_text, r"6\.4\.12\s+期末", r"6\.4\.13\s+金融工具风险")
    stock_part = section(note, r"6\.4\.12\.1\s*因认购", r"6\.4\.12\.2")
    # 完整附注必须恰为本原件的1.1股票表、2停牌、3两个市场正回购；
    # 额外证券类别、未知子节或缺终止标题不被静默吞掉。
    subheadings = re.findall(r"(?m)^\s*(6\.4\.12(?:\.\d+)+)\s+", note)
    if subheadings != ["6.4.12.1", "6.4.12.1.1", "6.4.12.2", "6.4.12.3", "6.4.12.3.1", "6.4.12.3.2"]:
        raise ValueError("RESTRICTION_NOTE_COMPLETE_LAYOUT_REQUIRED")
    if "受限证券类别:股票" not in normalized_text(stock_part) or "受限证券类别:债券" in normalized_text(stock_part):
        raise ValueError("RESTRICTION_STOCK_TABLE_CATEGORY")
    stocks = fixed_width_stock_restrictions(stock_part.replace("金额单位:人民币元", "金额单位：人民币元"), positions)[
        "rows"
    ]
    suspended = section(note, r"6\.4\.12\.2\s*期末持有的暂时停牌等流通受限股票", r"6\.4\.12\.3")
    if normalized_text(suspended) != "6.4.12.2期末持有的暂时停牌等流通受限股票无。":
        raise ValueError("RESTRICTION_SUSPENDED_EXPLICIT_NONE_REQUIRED")
    pledge = normalized_text(note[note.index("6.4.12.3 期末债券") :])
    expected = (
        "6.4.12.3期末债券正回购交易中作为抵押的债券6.4.12.3.1银行间市场债券正回购无。6.4.12.3.2交易所市场债券正回购无。"
    )
    if pledge != expected:
        raise ValueError("RESTRICTION_PLEDGE_EXPLICIT_NONE_REQUIRED")
    by_code = {r["stock_code"][:6]: r for r in positions}
    for code, position in by_code.items():
        own = [r for r in stocks if r["code"] == code]
        if sum(D(r["quantity"]) for r in own) > D(position["quantity_shares"]) or sum(D(r["amount"]) for r in own) > D(
            position["market_value_cny"]
        ):
            raise ValueError("RESTRICTION_TOTAL_EXCEEDS_HOLDING:" + code)
    amount = sum(D(r["amount"]) for r in stocks)
    nav = D(gross["balance"]["net_assets"])
    # 报告总资产受限0.30%只作公开舍入范围交叉核对，精确金额来自完整38行，
    # 绝不将0.30%当作未公开精确数字，也不从比例反推现金冻结为0。
    aggregate = re.search(
        r"于2026年6月30日[,，]本基金持有的流动性受限资产的估值占基金资产净值的比例为(\d+\.\d+)%", plain
    )
    if not aggregate or not D(aggregate[1]) - D(".005") <= amount / nav * 100 < D(aggregate[1]) + D(".005"):
        raise ValueError("RESTRICTION_TOTAL_DISCLOSED_ROUNDING_CONFLICT")
    restricted = {
        "kind": "restricted",
        "restricted_amount": str(amount),
        "net_assets": str(nav),
        "scope_closed": True,
        "source_refs": [reference["sha256"]],
    }
    restricted_value = _node(restricted, aliases={"PUBLIC:007831"})
    # 两个本金均由本期货币资金完整附注明示，非收息、应收、备付金或国债变现。
    # 证券专户本金仅用于证券交收；本期应付清算款实际已唯一识别为0。
    cash = section(text, r"6\.4\.7\.1\s+货币资金", r"6\.4\.7\.2")
    other = re.search(r"其他存款\s+[\d,]+\.\d{2}\s+等于[:：]本金\s+([\d,]+\.\d{2})", cash)
    if not other or "其他存款为存放在证券经纪商基金专用证券账户的证券交易结算资金" not in normalized_text(cash):
        raise ValueError("RESTRICTION_BROKER_PRINCIPAL_PURPOSE_REQUIRED")
    obs = accounting["observations"]
    payment = {
        "kind": "purpose_covered_payment",
        "as_of": "2026-06-30",
        "currency": "CNY",
        "bank_principal": obs["demand_deposits_principal"]["value"],
        "broker_settlement_principal": other[1].replace(",", ""),
        "settlement_liabilities": liability_proof["point_values"]["应付清算款"],
        "all_book_liabilities": gross["balance"]["liabilities"],
        "scope_closed": True,
        "restriction_amount_pending": False,
        "disclosed_restrictions": [],
        "source_refs": [reference["sha256"]],
        "scope_basis": "完整原表与全资产受限引用闭合，已列受限均为股票；证券专户本金仅用于证券交收。",
    }
    payment_value, payment_audit = payment_score(payment)
    return {
        "status": "ORIGINAL_RESTRICTION_AND_PAYMENT_REPLAYED",
        "restricted_node": restricted,
        "restricted_subscore": str(D(restricted_value.numerator) / D(restricted_value.denominator)),
        "payment_node": payment,
        "payment_subscore": str(payment_value),
        "payment_audit": payment_audit,
        "restricted_stock_rows": stocks,
        "full_asset_scope_quote": scope[0],
        "absent_lending_section_not_assigned_a_zero_balance": True,
        "complete_scope_basis": "原报告以完整附注覆盖全部暂时受限事项，非股票为现金和国债；不以旧版格式增设评分门槛。",
        "public_rounding_crosscheck": {"reported_percent": aggregate[1], "calculated_percent": str(amount / nav * 100)},
        "complete_liquidity_dimension": False,
        "peer_capacity_still_required": True,
    }


def holder_foundation(source, inputs):
    """原件字节、A/C结构守恒、普通/巨额机制和实际履职齐备才建候选节点。"""
    primary = source.json(PRIMARY)["data"]["notice_content"]
    contract = source.json(PROSPECTUS)["data"]["notice_content"]
    primary_path = source.sources[PRIMARY]["file"]
    prospectus_path = source.sources[PROSPECTUS]["file"]
    pdf_path = source.sources[PDF]["file"]
    p = plain(primary)
    if not all(v in p[:14000] for v in ["博道伍佰智航股票型证券投资基金", "007831", "007832", "契约型开放式"]):
        raise ValueError("HOLDER_PRODUCT_IDENTITY")
    structure = literal_section(primary, "8.1 期末基金份额持有人户数及持有人结构", "8.2")
    manager = literal_section(primary, "4.2 管理人对报告期内本基金运作遵规守信情况的说明", "4.3")
    trustee = literal_section(primary, "5.2 托管人对报告期内本基金投资运作", "5.3")
    ordinary = literal_section(contract, "（五）申购和赎回的程序", "（六）")
    large = literal_section(contract, "（十）巨额赎回的情形及处理方式", "（十一）")
    if not all(t in plain(structure) for t in ["持有人结构", "机构投资者", "个人投资者", "份额单位:份"]):
        raise ValueError("HOLDER_COMPLETE_CATEGORY_HEADERS")
    # 数字取自原PDF72/73页已逐页目视核对的单元格；原件固定哈希，金额
    # 同时与正文逐项核对、与同产品份额流动守恒。没有单一持有人比例赋值。
    cells = [
        {
            "share": "A",
            "holders": 4262,
            "average": "62348.68",
            "institution": "145180671.75",
            "individual": "120549399.22",
        },
        {
            "share": "C",
            "holders": 13350,
            "average": "53128.37",
            "institution": "344319892.38",
            "individual": "364943815.38",
        },
    ]
    numeric = structure.replace(",", "")
    flows, flow_proof = _flows([primary])
    for row, flow in zip(cells, flows, strict=True):
        assert row["share"] == flow["share"]
        for key in ["institution", "individual"]:
            if row[key] not in numeric:
                raise ValueError("HOLDER_CATEGORY_ORIGINAL_CELL")
        total = D(row["institution"]) + D(row["individual"])
        if total != D(flow["closing"]):
            raise ValueError("HOLDER_CATEGORY_SHARE_FLOW_CONFLICT")
        if not D(row["average"]) - D(".005") <= total / row["holders"] < D(row["average"]) + D(".005"):
            raise ValueError("HOLDER_COUNT_AVERAGE_ROUNDING_CONFLICT")
    for key, total in [("institution", "489500564.13"), ("individual", "485493214.60")]:
        if sum(D(row[key]) for row in cells) != D(total) or total not in numeric:
            raise ValueError("HOLDER_TOTAL_CATEGORY_CONFLICT")
    if not all(s in plain(ordinary) for s in ["T+7日", "基金份额登记机构", "赎回款项", "巨额赎回"]):
        raise ValueError("HOLDER_ORDINARY_MECHANISM")
    if not all(
        s in plain(large) for s in ["单个开放日", "单个基金份额持有人", "基金总份额的10%", "延期办理", "无优先权"]
    ):
        raise ValueError("HOLDER_LARGE_ACCOUNT_MECHANISM")
    if "未发生损害基金份额持有人利益的行为" not in plain(
        manager
    ) or "未发现基金管理人有损害基金份额持有人利益的行为" not in plain(trustee):
        raise ValueError("HOLDER_CURRENT_ACTUAL_EXECUTION")
    node = {
        "kind": "holder_execution_r5",
        "structure_condition_protocol": HOLDER_PROTOCOL,
        "structure_type": None,
        "compatible_structure_types": COMPATIBLE_TYPES,
        "structure_basis": "OPEN_END_CASH_LEGAL_PRODUCT_AND_COMPLETE_REPORTED_CATEGORY_TABLE",
        "foundation_gate_checks": {k: True for k in FOUNDATION_GATES},
        "structure_closed": True,
        "share_flow_closed": True,
        "actual_execution_verified": True,
        "redemption_mechanism_verified": True,
        "confirmed_pressure_effects": [],
        "actual_redemption_shares": str(sum(D(r["redeemed"]) for r in flows)),
        "source_refs": [PRIMARY, PROSPECTUS, PDF],
        "individual_largest_holder_ratio": None,
    }
    value, audit = holders_score(node)
    return {
        "status": "ORIGINAL_FOUNDATION_REPLAYED_STRUCTURE_UNCLASSIFIED",
        "node": node,
        "subscore": str(value),
        "audit": audit,
        "proof": {
            "category_cells": cells,
            "flows": flows,
            "flow_proof": flow_proof,
            "structure_quote": structure,
            "ordinary_quote": ordinary,
            "large_account_quote": large,
            "manager_quote": manager,
            "trustee_quote": trustee,
            "manual_original_page_review": ["holder-original-page72.png", "holder-original-page73.png"],
            "sources": [str(primary_path), str(prospectus_path), str(pdf_path)],
        },
        "not_complete_liquidity_by_itself": True,
    }
