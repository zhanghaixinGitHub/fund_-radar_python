"""004605持仓及流动性的可移植原件适配候选，仅编译部分维度。

全仓、行业及成交量重新读取原始字节；公司质量保留逐公司人工序档判断，
不把人工判断说成审计结论。模块不联网、不登记、不发布，不接受外部总分。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from decimal import localcontext

from app.services.fund_rating_r5 import _calculate_frozen
from app.services.material_content import (
    content_digest,
    document_pdfium,
    location_exists,
    source_bytes,
    source_exists,
    source_size,
)

VERSION = "R5_004605_HOLDINGS_ORIGINAL46_CICPA_20261030_2_CANDIDATE"
FIELD = "active_004605_holdings_calculation"
FROZEN_COHORT = "2d12743a08362ccba9cf36fccd15e28697b43d5b06e1e16c22f3ef7e4a7febf3"
PRIMARY = "2ce51f77ed65ae0f62e2aacc025b343998eea47dcbc91b3150c06346b2d1c43e"
PROTOCOL_HASH = "abc11cd7e58357e6be0b9dafd4c36c37a3d4f7d27505094fe22210540c5e1023"


def require(condition, reason):
    if not condition:
        raise ValueError("ACTIVE_PORTFOLIO_" + reason)


def compact(value):
    """只归一版式字符；金额、证券身份及正文内容均不替换。"""
    value = unicodedata.normalize("NFKC", value.replace("\\r", "").replace("\\n", ""))
    return re.sub(r"\s+", "", value).replace("，", ",").replace("：", ":")


class Sources:
    """每次调用核全部已绑定字节；缓存限本次复算，不跨请求沿用结论。"""

    def __init__(self, inputs, root):
        self.root, self.refs = root.resolve(), inputs["sources"]
        self.text_cache, self.page_cache = {}, {}
        require(1 <= len(self.refs) <= 1500, "SOURCE_COUNT")
        for digest, ref in self.refs.items():
            require(digest == ref["sha256"], "SOURCE_IDENTITY")
            self.bytes(digest)
            if ref.get("published_on"):
                require(ref["published_on"] <= inputs["as_of_date"], "SOURCE_AFTER_CUTOFF")

    def bytes(self, digest):
        ref = self.refs[digest]
        path = (self.root / ref["file"]).resolve()
        require(path.is_relative_to(self.root / "sources") and source_exists(path), "SOURCE_PATH")
        require(0 < source_size(path) <= 50000000, "SOURCE_SIZE")
        body = source_bytes(path)
        require(content_digest(body) == digest, "SOURCE_HASH")
        return body

    def json(self, digest):
        return json.loads(self.bytes(digest))

    def pages(self, digest):
        if digest not in self.page_cache:
            raw = self.bytes(digest)
            require(raw.startswith(b"%PDF"), "ACTUAL_PDF_REQUIRED")
            with document_pdfium(raw) as document:
                self.page_cache[digest] = [page.get_textpage().get_text_range() for page in document]
        return self.page_cache[digest]

    def text(self, digest):
        if digest not in self.text_cache:
            raw = self.bytes(digest)
            if raw.startswith(b"%PDF"):
                body = "\n".join(self.pages(digest))
            elif self.refs[digest].get("encoding") == "gb18030":
                body = raw.decode("gb18030")
            else:
                data = json.loads(raw)
                require(data.get("success") == 1 and bool(data.get("data", {}).get("notice_content")), "NOTICE_BODY")
                body = data["data"]["notice_content"]
            self.text_cache[digest] = body
        return self.text_cache[digest]


def check_binding(bundle, inputs):
    """限制为同一份额、时期、原报告及已采用参照；禁止覆盖其他类型组件。"""
    require(
        bundle.get("basis") == "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3"
        and bundle.get("family") == "ACTIVE_EQUITY"
        and inputs.get("version") == VERSION
        and inputs.get("fund_code") == bundle["fund_code"] == "004605"
        and inputs.get("share_class") == bundle["share_class"] == "C"
        and inputs.get("as_of_date") == bundle["as_of_date"] == "2026-09-30"
        and inputs.get("holding_date") == bundle["holding_date"] == "2026-06-30"
        and inputs.get("legal_name") == bundle["identity"]["legal_name"]
        and inputs.get("manager_name") == bundle["identity"]["manager_name"]
        and inputs.get("primary_ref") == bundle["original_report_archive"]["sha256"] == PRIMARY,
        "BUNDLE_IDENTITY",
    )
    require(
        not any(
            bundle.get(k) for k in ["r5_calculation", "passive_components_calculation", "passive_holdings_calculation"]
        ),
        "CONFLICTING_COMPONENT",
    )
    packets = bundle.get("active_performance_calculations", [])
    require(
        bool(packets) and all(p["inputs"]["cohort_ref"] == inputs["cohort_ref"] == FROZEN_COHORT for p in packets),
        "ADOPTED_COHORT",
    )
    require(
        all(p["inputs"]["material_event_review_ref"] == inputs["material_event_review_ref"] for p in packets),
        "ADOPTED_MATERIAL_REVIEW",
    )


def _peers(source, inputs):
    frozen = source.json(inputs["cohort_ref"])
    require(inputs["cohort_ref"] == FROZEN_COHORT, "FROZEN_COHORT_HASH")
    group = frozen["holdings_liquidity"]
    require(group["thresholds"] == {"mean_gap_pp": 5, "max_gap_pp": 10, "domestic_min_pct": 100}, "FROZEN_THRESHOLDS")
    rows = group["rows"]
    peers = sorted(row["code"] for row in rows if row["state"] == "ADMITTED")
    require(len(peers) == len(set(peers)) == 185, "FIXED_PEER_COUNT")
    require([row["code"] for row in rows if "PENDING" in row["state"]] == ["010064"], "PENDING_COHORT_CHANGED")
    proof = source.json(inputs["exclusion_review_ref"])
    digest = proof["source"]["sha256"]
    primary = source.json(digest)
    require(primary.get("success") == 1 and primary["data"]["notice_date"][:10] == "2024-04-22", "EXCLUSION_ORIGINAL")
    require({"010064", "010065"} <= {str(v["stock"]) for v in primary["data"]["security"]}, "EXCLUSION_IDENTITY")
    body = compact(primary["data"]["notice_content"])
    equity = re.search(r"权益投资([\d,]+\.\d{2})", body)
    nav = re.search(r"期末基金资产净值([\d,]+\.\d{2})([\d,]+\.\d{2})", body)
    require(equity is not None and nav is not None, "EXCLUSION_ORIGINAL_AMOUNTS")
    stock = D(equity[1].replace(",", ""))
    total = sum(D(nav[i].replace(",", "")) for i in [1, 2])
    gap = abs(D(frozen["target_quarter_exposure"]["2024-03-31"]) - stock / total * 100)
    require(
        stock == D(proof["equity_CNY"]) and total == D(proof["all_share_NAV_CNY"]) and gap > 10,
        "STRICT_EXPOSURE_EXCLUSION",
    )
    require(sorted(inputs["peer_codes"]) == peers and not {"004604", "004605", "010064"} & set(peers), "PEER_IDENTITY")
    return peers


def _portfolios(source, inputs, peers):
    require(set(inputs["portfolios"]) == set(peers) | {"004605"}, "FULL_PORTFOLIO_COVERAGE")
    portfolios = {}
    for code, report in inputs["portfolios"].items():
        digest = report["source_ref"]
        require(report["period"] == "2026-06-30" and report["published_on"] <= "2026-09-30", "PORTFOLIO_PERIOD")
        raw = source.bytes(digest)
        if raw.startswith(b"%PDF"):
            require(code == "004666", "UNSUPPORTED_PORTFOLIO_PDF")
            text = "\n".join(
                line
                for line in source.text(digest).splitlines()
                if not re.fullmatch(
                    r"长城久嘉创新成长混合\s*2026\s*年中期报告|第\s*\d+\s*页\s*共\s*\d+\s*页", line.strip()
                )
            )
            require(content_digest(text.encode("utf-8")) == report["text_derivation_ref"], "PDF_TEXT_DERIVATION")
            require(
                "2026年6月30日" in compact(text)[:6000] and "送出日期:2026年8月31日" in compact(text)[:6000],
                "PDF_PERIOD_PUBLICATION",
            )
        else:
            notice = json.loads(raw)
            require(
                notice.get("success") == 1 and notice["data"]["notice_date"][:10] == report["published_on"],
                "PORTFOLIO_PUBLICATION",
            )
            require(code in {str(v["stock"]) for v in notice["data"]["security"]}, "PORTFOLIO_LEGAL_IDENTITY")
            text = notice["data"]["notice_content"]
        require(re.search(r"2026年0?6月30日", compact(text)[:10000]) is not None, "PORTFOLIO_HOLDING_DATE")
        try:
            parsed = parse_stock_text(code, text)
        except (AssertionError, ValueError, AttributeError):
            parsed = parse_wrapped_text(code, text)
        require(parsed["full_table_reconciled"] and D(parsed["total_stock_value"]) > 0, "FULL_TABLE_RECONCILIATION")
        portfolios[code] = parsed
    return portfolios


def _rank_matches(node, peers, values):
    expected = ["CN_PUBLIC_FUND:" + c for c in peers]
    require(
        node["kind"] == "rank" and node["direction"] == "lower" and node["expected_products"] == expected,
        "FIXED_RANK_UNIVERSE",
    )
    require(
        len(node["peers"]) == len(peers) and {r["product_id"] for r in node["peers"]} == set(expected),
        "FIXED_RANK_COVERAGE",
    )
    require(abs(D(node["target"]) - values["004605"]) < D("1e-55"), "TARGET_OBSERVATION")
    for row in node["peers"]:
        require(abs(D(row["value"]) - values[row["product_id"].split(":")[1]]) < D("1e-55"), "PEER_OBSERVATION")


def parse_stock_text(code, text):
    """沿用已核全表解析与版式恢复；独立总额、净值及证券集合不闭合即拒绝。"""
    match = list(re.finditer("(?:所有|全部)股票投资明细", text))
    assert match, "NO_FULL_STOCK_TABLE"
    part = text[match[-1].end() :]
    end = re.search("\\d\\.4\\s*(?:报告期内股票|本期股票)", part)
    assert end, "TABLE_END_MISSING"
    part = part[: end.start()]
    part = re.sub(
        "([\\d,]+\\.\\d)\\s+([\\d]+\\.\\d{2})\\s*\\n\\s*(\\d)(?=\\s|$)", lambda m: m[1] + m[3] + " " + m[2], part
    )
    pattern = (
        "(?:^|\\n)[ \\t]*([\\d,]+)[ \\t]+(\\d{6})[ \\t]+(.+?)[ \\t]+"
        "([\\d,]+(?:\\.\\d+)?)[ \\t]+([\\d,]+\\.\\d{2})[ \\t]+([\\d.]+|-)"
    )
    stocks = []
    for m in re.finditer(pattern, part):
        stocks.append(
            {
                "rank": int(m[1].replace(",", "")),
                "security": m[2],
                "name": re.sub("\\s+", "", m[3]),
                "quantity": m[4],
                "market_value": m[5].replace(",", ""),
                "reported_nav_pct": m[6],
            }
        )
    assert stocks and len({s["security"] for s in stocks}) == len(stocks), ("SECURITY_DUPLICATE", code)
    assert stocks[0]["rank"] == 1 and all((a["rank"] <= b["rank"] for a, b in zip(stocks, stocks[1:], strict=False))), (
        "STOCK_RANKS",
        code,
        len(stocks),
    )
    compact = re.sub("\\s+", "", text)
    asset = compact[compact.rfind("期末基金资产组合情况") :]
    asset = re.sub(
        "([\\d,]+\\.\\d)\\s+([\\d]+\\.\\d{2})\\s*\\n\\s*(\\d)(?=\\s|$)",
        lambda m: m[1] + m[3] + " " + m[2],
        text[text.rfind("期末基金资产组合情况") :],
    )
    stock = re.search("其中[：:]股票\\s+([\\d,]+\\.\\d{2})", asset)
    assert stock
    stock_total = D(stock[1].replace(",", ""))
    assert sum(D(s["market_value"]) for s in stocks) == stock_total, (
        "FULL_STOCK_SUM",
        code,
        len(stocks),
        sum(D(s["market_value"]) for s in stocks),
        stock_total,
    )
    nav_matches = list(re.finditer("(?:净资产合计|所有者权益合计)([\\d,]+\\.\\d{2})", compact))
    assert nav_matches
    nav = D(nav_matches[0][1].replace(",", ""))
    assert 0 < stock_total / nav < D("1.5")
    return {
        "code": code,
        "stocks": stocks,
        "total_stock_value": str(stock_total),
        "nav": str(nav),
        "full_table_reconciled": True,
        "period": "2026-06-30",
    }


def parse_wrapped_text(code, text):
    """沿用已核全表解析与版式恢复；独立总额、净值及证券集合不闭合即拒绝。"""
    compact = re.sub("\\s+", "", text)
    nav = D(re.search("(?:净资产合计|所有者权益合计)([\\d,]+\\.\\d{2})", compact)[1].replace(",", ""))
    head = list(re.finditer("(?:所\\s*有|全\\s*部)\\s*股\\s*票\\s*投\\s*资\\s*明\\s*细", text))[-1]
    part = text[head.end() :]
    stop = re.search("(?:\\d\\.4\\s*)?(?:报告期内|本期)股票投资组合的重大变动", part)
    assert stop, "BOUNDARY"
    part = part[: stop.start()]
    table = text[text.rfind("期末基金资产组合情况") :]
    stock_total = D(re.search("其中[：:]\\s*股票\\s+([\\d,]+\\.\\d{2})", table)[1].replace(",", ""))
    pattern = "(?m)^[ \\t]*(\\d{1,4})[ \\t]+(\\d{6})[ \\t]*(.*)$"
    starts = list(re.finditer(pattern, part))
    rows = []
    edits = []
    for i, m in enumerate(starts):
        chunk = m[3] + part[m.end() : starts[i + 1].start() if i + 1 < len(starts) else len(part)]
        dash = re.search("([\\d,]+)\\s+[-－—]\\s+0\\.00\\s*$", m[3])
        if dash:
            rows.append(
                {
                    "rank": int(m[1]),
                    "security": m[2],
                    "name": m[3].split()[0],
                    "market_value": None,
                    "quantity": dash[1],
                    "reported_nav_pct": "0.00",
                    "original_amount_dash": True,
                    "original_row": chunk,
                }
            )
            continue
        tokens = re.findall("(?<![A-Za-z\\d])(?:\\d[\\d,]*(?:\\.\\d*)?|\\.\\d+)(?![A-Za-z\\d])", chunk)
        if not tokens:
            raise ValueError("EMPTY_NUMBERS")
        choices = []
        for offset in [0]:
            tail = tokens[offset:]
            if len(tail) == 3:
                choices.append((tail[0], tail[1], tail[2], "UNCHANGED"))
            if len(tail) == 4 and re.fullmatch("\\d[\\d,]*", tail[1]) and re.fullmatch("\\.\\d{2}", tail[3]):
                choices.append((tail[0], tail[1] + tail[3], tail[2], "AMOUNT_CENT_FRACTION_AFTER_PERCENT_COLUMN"))
            if len(tail) == 4 and re.fullmatch("\\d[\\d,]*\\.\\d", tail[1]) and re.fullmatch("\\d", tail[3]):
                choices.append((tail[0], tail[1] + tail[3], tail[2], "AMOUNT_LAST_CENT_AFTER_PERCENT_COLUMN"))
            if len(tail) == 4 and re.fullmatch("\\d[\\d,]*\\.", tail[1]) and re.fullmatch("\\d{2}", tail[3]):
                choices.append((tail[0], tail[1] + tail[3], tail[2], "AMOUNT_CENTS_AFTER_PERCENT_COLUMN"))
            if len(tail) == 4 and len(m[1]) == 3 and re.fullmatch("\\d", tail[3]):
                candidate_rank = int(m[1] + tail[3])
                if candidate_rank >= 1000 and re.fullmatch("\\d[\\d,]*\\.\\d{2}", tail[1]):
                    choices.append((tail[0], tail[1], tail[2], "WRAPPED_FOUR_DIGIT_RANK_SUFFIX"))
        valid = []
        for quantity, amount, weight, rule in choices:
            try:
                a = D(amount.replace(",", ""))
                w = D(weight)
                if (
                    re.fullmatch("\\d[\\d,]*\\.\\d{2}", amount)
                    and a >= 0
                    and (0 <= w <= 100)
                    and (abs(a / nav * 100 - w) <= D(".0050000001"))
                ):
                    valid.append((quantity, amount, weight, rule))
            except Exception:
                pass
        assert len(set(valid)) == 1, ("AMBIGUOUS_CELL", m[2], tokens, valid)
        qty, amount, weight, rule = valid[0]
        name = re.split("\\d", m[3], maxsplit=1)[0].strip()
        rows.append(
            {
                "rank": int(m[1]),
                "security": m[2],
                "name": name,
                "market_value": str(D(amount.replace(",", ""))),
                "quantity": qty,
                "reported_nav_pct": weight,
            }
        )
        if rule != "UNCHANGED":
            edits.append({"security": m[2], "rule": rule, "original": chunk, "cells": [qty, amount, weight]})
    assert rows and rows[0]["rank"] == 1 and (len({r["security"] for r in rows}) == len(rows))
    known_total = sum(D(r["market_value"]) for r in rows if r["market_value"] is not None)
    assert known_total == stock_total, ("SUM_MISMATCH", code, len(rows), str(known_total), str(stock_total))
    for r in rows:
        if r["market_value"] is None:
            r.update(market_value="0", zero_proof="完整非负股票表已知金额等于独立股票总额，剩余横杠金额唯一为零")
    return {
        "code": code,
        "stocks": rows,
        "total_stock_value": str(stock_total),
        "nav": str(nav),
        "full_table_reconciled": True,
        "period": "2026-06-30",
    }


def parse_classifier(pages):
    """沿用已核全表解析与版式恢复；独立总额、净值及证券集合不闭合即拒绝。"""
    out = {}
    for page_number, page in enumerate(pages, 1):
        matches = list(re.finditer("(?m)^\\s*(\\d{6})\\s+", page))
        for i, m in enumerate(matches):
            text = re.sub(
                "\\s+", " ", page[m.start() : matches[i + 1].start() if i + 1 < len(matches) else len(page)]
            ).strip()
            c = re.search("^\\d{6}\\s+(.+?)\\s+([A-S])\\s+(.+?)(?:\\s+([A-Z]{2})\\s+(.+?))?\\s+(\\d{2})\\s+(.+)", text)
            assert c, ("CLASSIFICATION_ROW_PARSE", text)
            code = m[1]
            row = {
                "security": code,
                "name": c[1].replace(" ", ""),
                "section_code": c[2],
                "section_name": c[3].replace(" ", ""),
                "secondary_code": c[4],
                "secondary_name": c[5].replace(" ", "") if c[5] else None,
                "industry_code": c[2] + c[6],
                "industry_name": re.sub("\\s+", "", c[7]),
                "pdf_page": page_number,
                "raw_row": text,
            }
            assert code not in out
            out[code] = row
    return out


def _structure(source, inputs, peers, portfolios, nodes):
    """原件逐证券合并发行人及同口径行业；序档仍由明确的逐公司研究判断给出。"""
    meta = inputs["classification"]
    require(meta["published_on"] == "2026-09-30", "CLASSIFICATION_DATE")
    require(
        source.refs[meta["pdf_ref"]]["url"] in source.bytes(meta["article_ref"]).decode("utf-8"),
        "CLASSIFICATION_ATTACHMENT",
    )
    classes = parse_classifier(source.pages(meta["pdf_ref"]))
    require(len(classes) == 5521, "CLASSIFICATION_COMPLETENESS")
    for item in meta["ipo_supplements"]:
        pages = source.pages(item["source_ref"])
        require(
            item["source_type"] == "ISSUER_IPO_DOCUMENT_SAME_GRAIN_SUPPLEMENT_NOT_CAPCO_LIST_ASSIGNMENT",
            "IPO_CLASSIFICATION_PROVENANCE",
        )
        require(
            item["security"] in compact("".join(pages))
            and any(item["industry_code"] in compact(pages[p - 1]) for p in item["reviewed_pdf_pages"]),
            "IPO_CLASSIFICATION_ORIGINAL",
        )
        classes[item["security"]] = item
    issuers = {row["code"]: row for row in source.json(inputs["issuer_identity_ref"])["stockList"]}
    for portfolio in portfolios.values():
        by_issuer, by_industry = defaultdict(D), defaultdict(D)
        total = D(portfolio["total_stock_value"])
        for stock in portfolio["stocks"]:
            amount, security = D(stock["market_value"]), stock["security"]
            if amount == 0:
                continue
            require(amount > 0 and bool(issuers[security]["orgId"]), "ISSUER_IDENTITY")
            by_issuer[issuers[security]["orgId"]] += amount
            by_industry[classes[security]["industry_code"]] += amount
        portfolio["issuer_concentration"] = sum((v / total) ** 2 for v in by_issuer.values())
        portfolio["industry_concentration"] = sum((v / total) ** 2 for v in by_industry.values())
    for name in ["issuer_concentration", "industry_concentration"]:
        _rank_matches(nodes[name], peers, {c: p[name] for c, p in portfolios.items()})
    return {"portfolios": len(portfolios), "fixed_peers": len(peers), "issuer_and_industry_originals_replayed": True}


def calculate(inputs, *, root):
    """按原35/35/30重算持仓；新Q来自逐页原件编译，不读取输入中的旧质量分。"""
    from app.services.fund_rating_004605_asset_quality import calculate as quality_compile
    from app.services.fund_rating_active_performance import _require_material_review

    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "004605"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("primary_ref") == PRIMARY
        and inputs.get("selected_dimensions") == ["holdings"],
        "INPUT_SCOPE",
    )
    require(not any(k in inputs for k in ["score", "candidate", "point", "grade", "dimensions"]), "EXTERNAL_POINT")
    source = Sources(inputs, root)
    material = _require_material_review(inputs, root)
    with localcontext() as context:
        context.prec = 60
        peers = _peers(source, inputs)
        portfolios = _portfolios(source, inputs, peers)
        packet = deepcopy(inputs["packet"])
        require(
            packet["fund_code"] == "004605"
            and packet["as_of_date"] == "2026-09-30"
            and packet["quality_metrics"]["material_event_pending"] is material["pending"],
            "PACKET_SCOPE",
        )
        nodes = packet["quality_metrics"]["observations"]
        require(set(nodes) == {"issuer_concentration", "industry_concentration"}, "LEGACY_QUALITY_NODE_FORBIDDEN")
        structure = _structure(source, inputs, peers, portfolios, nodes)
        quality = quality_compile(inputs["asset_quality_inputs"], root=root)
        require(
            D(quality["stock_total_CNY"]) == D(portfolios["004605"]["total_stock_value"]), "QUALITY_FULL_HOLDINGS_MATCH"
        )
        nodes["asset_quality"] = quality["asset_quality_observation"]
        sources = {**inputs["sources"], **inputs["asset_quality_inputs"]["sources"]}
        packet["r4_evidence_registry"] = {
            h: {**r, "path": str((root / r["file"]).resolve())} for h, r in sources.items()
        }
        packet["raw_archive"] = [{"sha256": h, "kind": r["archive_kind"]} for h, r in sources.items()]
        require(inputs["protocol_ref"] == PROTOCOL_HASH, "FROZEN_PROTOCOL")
        result = _calculate_frozen(packet, source.json(PROTOCOL_HASH), {}, selected_dimensions={"holdings"})
    require(result["score"] is None and result["grade"] is None, "COMPONENT_ONLY")
    return {
        "version": VERSION,
        "dimensions": result["dimensions"],
        "subscores": result["subscores"],
        "structure_audit": structure,
        "asset_quality_audit": quality,
        "material_event_review": material,
        "packet": packet,
        "independent_acceptance_complete": False,
        "native_603293_opinion_pages_recovered": False,
        "complete_score": None,
        "grade": None,
    }


def _canonical_existing(ref, *, root):
    """只复用本worker已有核SHA原件；PDF硬链接，派生JSON独立封存，不写共享目录。"""

    root = root.resolve()
    path = (root / ref["file"]).resolve()
    require(path.is_relative_to(root / "sources") and source_exists(path), "CANONICAL_PATH")
    digest = ref["sha256"]
    content = source_bytes(path)
    require(content_digest(content) == digest, "CANONICAL_HASH")
    destination = root / "sources/pipeline" / (digest + path.suffix.lower())
    destination.parent.mkdir(exist_ok=True)
    if not location_exists(destination):
        from app.services.material_content import write_source

        # 复用来源只登记库内别名，禁止重新硬链接/复制成C盘原件。
        write_source(destination, content)
    else:
        require(content_digest(source_bytes(destination)) == digest, "CANONICAL_CHANGED")
    return {**ref, "file": destination.relative_to(root).as_posix(), "path": destination.relative_to(root).as_posix()}


def validate_input_archive(bundle, *, root, archive_reader):
    """标准快照钩子：先核封存输入与内嵌输入一致，再核基金和历史批次范围。"""
    node = bundle[FIELD]
    require(archive_reader(node["input_archive"], root) == node["inputs"], "INPUT_ARCHIVE")
    check_binding(bundle, node["inputs"])


def replay_snapshot(bundle, dimensions, *, root):
    """持仓组件真实原件回放；不重放既有流动性容量数据。"""
    from app.services.fund_rating_reviewed_snapshot import _read

    validate_input_archive(bundle, root=root, archive_reader=_read)
    result = calculate(bundle[FIELD]["inputs"], root=root)
    require(
        all(
            dimensions["holdings"][k] is not None
            and abs(D(dimensions["holdings"][k]) - D(result["dimensions"]["holdings"])) <= D("1e-55")
            for k in ["lower", "upper", "candidate"]
        ),
        "SNAPSHOT_POINT",
    )
    return result


def compile_bundle(bundle, request, *, root):
    """只追加持仓研究和专属组件；保留原六维、原组件输入及来源历史。"""
    from app.services.fund_rating_pipeline import archive_json

    bundle, raw = deepcopy(bundle), deepcopy(request["inputs"])
    require(FIELD not in bundle, "COMPONENT_ALREADY_PRESENT")
    check_binding(bundle, raw)
    require(request["fund_code"] == raw["fund_code"] and request["as_of_date"] == raw["as_of_date"], "REQUEST_IDENTITY")
    raw["sources"] = {h: _canonical_existing(r, root=root) for h, r in raw["sources"].items()}
    raw["asset_quality_inputs"]["sources"] = {
        h: _canonical_existing(r, root=root) for h, r in raw["asset_quality_inputs"]["sources"].items()
    }
    result = calculate(raw, root=root)
    reason = request["explanation"].strip()
    require(len(reason) >= 40, "PUBLIC_EXPLANATION")
    value = result["dimensions"]["holdings"]
    bundle["reviewed_document"]["dimensions"]["holdings"] = {
        "lower": value,
        "upper": value,
        "candidate": value,
        "reason": reason,
    }
    bundle[FIELD] = {"inputs": raw, "input_archive": archive_json(root, raw)}
    bundle.update(research_format="single", research_archive=archive_json(root, bundle["reviewed_document"]))
    sources = {**raw["sources"], **raw["asset_quality_inputs"]["sources"]}
    extras = {r["sha256"]: r for r in bundle.get("extra_archives", [])}
    extras.update({h: {k: r[k] for k in ["file", "sha256", "kind"]} for h, r in sources.items()})
    bundle["extra_archives"] = list(extras.values())
    bundle.setdefault("public_explanations", {})["holdings"] = reason
    return bundle, result


def prepare_dossier(session, request, *, root):
    """只读当前批次prepare；该候选不登记、不发布，并以专属回放核研究记录。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS

    current = current_inputs(session, [request["fund_code"]]).get(request["fund_code"])
    require(
        current is not None
        and current["batch_id"] == request["prior_rating_ref"]
        and str(current["as_of_date"]) == request["as_of_date"],
        "CURRENT_REF",
    )
    base = next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == request["fund_code"])
    updated, _ = compile_bundle(base, request, root=root)
    replay_snapshot(updated, updated["reviewed_document"]["dimensions"], root=root)
    return {"basis": BASIS, "entries": [{k: v for k, v in updated.items() if k not in ["basis", "reviewed_document"]}]}


def verify_registered_original(bundle, reference, *, root):
    """供标准快照额外归档核验使用，仅接受已绑定的资产质量原PDF。"""
    from app.services.fund_rating_004605_asset_quality import verify_registered_original as verify

    verify(bundle[FIELD]["inputs"]["asset_quality_inputs"], reference, root=root)
