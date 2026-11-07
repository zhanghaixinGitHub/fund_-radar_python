"""010737原件编译与现R5执行/收益分项复算候选。

所有实际原件仅由inputs.sources及Originals(root)逐次校字节读取；不联网、不发布。
本组件不复制其他基金事实；收益参照适用范围仍待统筹裁定，不据候选分自动开通。
"""

import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_accounting import accounting_observations
from app.services.fund_quality_v2_allocation import report_product_net_assets
from app.services.fund_quality_v2_balance_proofs import strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_restrictions import (
    full_asset_restriction_scope,
    restricted_securities,
    restriction_page_header_view,
)
from app.services.fund_quality_v2_stock_book import periodic_financial_view
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_public_stock_tables import parse_disclosed
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import document_pdfium
from app.services.material_content import document_reader as PdfReader

VERSION = "R5_EFUNDS_010737_COMPONENTS_ORIGINALS_20261029_1"
LEGAL = "易方达沪深300指数精选增强型证券投资基金"
MANAGER = "易方达基金管理有限公司"
EXPECTED = [
    "2023-09-30",
    "2023-12-31",
    "2024-03-31",
    "2024-06-30",
    "2024-09-30",
    "2024-12-31",
    "2025-03-31",
    "2025-06-30",
    "2025-09-30",
    "2025-12-31",
    "2026-03-31",
    "2026-06-30",
]
FULL = ["2023-12-31", "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31", "2026-06-30"]
MATERIAL_REVIEW_HASH = "8d608c2da27008487f2f96e50f3b00ce6792f0a3bff10fdd8a9a82c7b2871056"
CALENDAR_ORIGINAL_HASH = "829285154cfdfba402b60707b0bf32e0fd38c962c49979d5604a472544d2a23c"
TIME_CONTRACT = {
    "kind": "RETROSPECTIVE_HISTORICAL_QUALITY_RESEARCH",
    "valuation_window_end": "as_of_date",
    "nav_and_market_publication_boundary": "research_on",
    "fundamental_disclosure_boundary": "as_of_date",
    "not_historical_asof_tradable_signal": True,
}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def norm(text):
    return re.sub(r"\s+", "", text)


def _iso(value):
    _require(isinstance(value, str) and re.fullmatch(r"(?:\d{8}|\d{4}-\d{2}-\d{2})", value), "EFUNDS_DATE_REQUIRED")
    return date.fromisoformat(value).isoformat()


def load_source(source, ref, first_pages=None):
    """仅解析本次Originals已验证的原始字节；不按输入旧path另开文件。"""
    raw = source.raw[ref["sha256"]]
    if raw.startswith(b"%PDF-"):
        if first_pages is None:
            return [p.extract_text() or "" for p in PdfReader(raw).pages]
        # 原件部分中文字体映射仅PDFium可恢复；不用乱码或缓存摘要代原文。
        doc = document_pdfium(raw)
        pages = []
        try:
            for i in range(min(first_pages, len(doc))):
                page = doc[i]
                tp = page.get_textpage()
                try:
                    pages.append(tp.get_text_range())
                finally:
                    tp.close()
                    page.close()
        finally:
            doc.close()
        return pages
    doc = json.loads(raw)
    return [BeautifulSoup(doc["data"]["notice_content"], "html.parser").get_text("\n", strip=True)]


def _source_guard(inputs, root):
    """区分研究日可得的历史NAV与截止前已公开的基本面，缺日拒收。"""
    _require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "010737"
        and inputs.get("share_class") == "C"
        and inputs.get("legal_name") == LEGAL
        and inputs.get("manager_name") == MANAGER
        and inputs.get("basis") == ACTIVE_BASIS
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("time_contract") == TIME_CONTRACT,
        "EFUNDS_IDENTITY_OR_TIME_CONTRACT",
    )
    research = _iso(inputs.get("research_on"))
    _require(research == "2026-10-29", "EFUNDS_RESEARCH_DATE")
    selected = inputs.get("selected_dimensions")
    _require(selected in [["execution"], ["execution", "return"]], "EFUNDS_SELECTED_DIMENSIONS")
    market = set(inputs["execution_inputs"]["index_sources"].values())
    if "return" in selected:
        market.add(inputs["target"]["provider_hash"])
        market.add(inputs["target"]["div_hash"])
        market.update(m[k] for m in inputs["peers"].values() for k in ["nav_hash", "div_hash"])
    review = inputs.get("material_event_review_ref")
    calendar_ref = inputs.get("calendar_ref")
    catalogs = set(inputs.get("catalog_source_hashes", []))
    for h, ref in inputs["sources"].items():
        if h in [review, calendar_ref]:
            continue
        if h in catalogs:
            _require(
                _iso((ref.get("acquired_at") or ref.get("received_at") or "")[:10]) <= research,
                "EFUNDS_CATALOG_RESEARCH_DATE",
            )
            continue
        if h in market:
            if ref.get("published_on"):
                _require(_iso(ref["published_on"]) <= research, "EFUNDS_MARKET_AFTER_RESEARCH")
        else:
            _require(_iso(ref.get("published_on")) <= inputs["as_of_date"], "EFUNDS_FUNDAMENTAL_AFTER_CUTOFF")
    checked = {
        **inputs,
        "as_of_date": research,
        "sources": {
            h: {**r, **({"published_on": _iso(r["published_on"])} if r.get("published_on") else {})}
            for h, r in inputs["sources"].items()
        },
    }
    source = Originals(checked, Path(root))
    _require(calendar_ref == CALENDAR_ORIGINAL_HASH, "EFUNDS_FROZEN_CALENDAR_SOURCE")
    cal = source.json(calendar_ref)
    _require(cal.get("calendar_hash") == inputs.get("calendar_hash"), "EFUNDS_CALENDAR_HASH")
    sessions = tuple(date.fromisoformat(d) for d in cal["sessions"])
    _require(len(sessions) == len(set(sessions)) and list(sessions) == sorted(sessions), "EFUNDS_CALENDAR_ORDER")
    actual = [str(d) for d in sessions if "2023-09-28" <= str(d) <= "2026-09-30"]
    _require(len(actual) == 727 and actual[0] == "2023-09-28" and actual[-1] == "2026-09-30", "EFUNDS_CALENDAR_WINDOW")
    _require(
        inputs["primary_ref"] == inputs["execution_inputs"]["reports"]["2026-06-30-full"]["source_hash"],
        "EFUNDS_REAL_PRIMARY_REPORT",
    )
    return source, sessions, cal["calendar_hash"]


def _nav_time_audit(inputs, source, sessions):
    """每行估值日/原公告日各自核验；晚公告保留日期，不冒称月末已知。"""
    required = {str(d) for d in sessions if "2023-09-28" <= str(d) <= "2026-09-30"}
    out = {}
    for code in ["010737", *PEERS]:
        spec = inputs["target"] if code == "010737" else inputs["peers"][code]
        if code == "010737":
            raw = source.json(spec["provider_hash"])
            _require(raw.get("fund_code") == code, "EFUNDS_NAV_OTHER_SHARE")
            rows = raw["history"]["rows"]
        else:
            rows = api(source.raw[spec["nav_hash"]], source.sources[spec["nav_hash"]], code)
        _require(isinstance(rows, list) and rows, "EFUNDS_NAV_ROWS")
        dates = []
        late = []
        for r in rows:
            day = _iso(r.get("nav_date"))
            ann = _iso(r.get("ann_date"))
            _require(day <= inputs["as_of_date"] and day <= ann <= inputs["research_on"], "EFUNDS_NAV_DATE_BOUNDARY")
            dates.append(day)
            if ann > inputs["as_of_date"]:
                late.append({"nav_date": day, "ann_date": ann, "known_on_valuation_cutoff": False})
        _require(len(dates) == len(set(dates)) and required <= set(dates), "EFUNDS_NAV_MISSING_OR_DUPLICATE_DAY")
        dividends = api(source.raw[spec["div_hash"]], source.sources[spec["div_hash"]], code)
        for row in dividends:
            ex = row.get("net_ex_date") or row.get("ex_date")
            if ex and "20230928" < ex <= "20260930":
                announced = _iso(row.get("ann_date"))
                _require(
                    announced <= _iso(ex) <= inputs["as_of_date"] and announced <= inputs["research_on"],
                    "EFUNDS_USED_DIVIDEND_DATE_REQUIRED",
                )
        out[code] = {"rows": len(rows), "late_announcements": late, "knowledge_boundary": inputs["research_on"]}
    return out


PEERS = [
    "000176",
    "000312",
    "000512",
    "003015",
    "004190",
    "004513",
    "004788",
    "005113",
    "005137",
    "005248",
    "005530",
    "005850",
    "006020",
    "007044",
    "007143",
    "008238",
    "008592",
    "009059",
    "010854",
    "010872",
    "010908",
    "011545",
    "012206",
    "015387",
    "016204",
    "018257",
    "110030",
    "165310",
]


def restricted_upper_bound(pages, positions, bond_amount, nav):
    """金额表断行时，仅证明15%阈值所需的保守上界，不伪造精确受限金额。

    原表证券代码末位跨行时，保留五位前缀的全部可能持仓；每只候选将其完整
    股票仓位而非受限小部分计入上界。债券也将全部余额计入，不能据此生成
    流动性的精确受限比例。任何无法界定的证券行均拒绝，不依赖搜索未命中。
    """
    text, header_view = restriction_page_header_view(pages)
    compact = norm(text)
    assert full_asset_restriction_scope(text), "EXECUTION_RESTRICTION_REMAINDER_SCOPE"
    m = re.search(r"[56]\.4\.12\.1\s*因认购(.*?)[56]\.4\.12\.2", text, re.S)
    assert m
    stockpart = m[1].partition("受限证券类别：债券")[0]
    assert all(label in norm(stockpart) for label in ["证券", "代码", "名称", "估值", "总额", "受限证券类别：股票"])
    code_rows = list(re.finditer(r"(?m)^\s*(\d{5,6})(?!\d)(?=[ \t]|$)[^\n]*", stockpart))
    assert code_rows, "EXECUTION_RESTRICTION_NO_IDENTIFIABLE_ROWS"
    matched = set()
    claims = []
    for row in code_rows:
        prefix = row[1]
        candidates = [p for p in positions if p["stock_code"].split(".")[0].startswith(prefix)]
        assert candidates, ("EXECUTION_RESTRICTION_UNKNOWN_PREFIX", prefix)
        # 原件行必须同时包含所持证券简称的一部分，避免将数量列误作代码。
        chars = re.sub(r"[^\u4e00-\u9fff]", "", row[0])
        if chars:
            assert any(set(re.sub(r"[^\u4e00-\u9fff]", "", p["stock_name"])) & set(chars) for p in candidates)
        matched.update(p["stock_code"] for p in candidates)
        claims.append(
            {"source_row": row[0], "prefix": prefix, "all_possible_positions": [p["stock_code"] for p in candidates]}
        )
    suspended = re.search(r"[56]\.4\.12\.2(.*?)[56]\.4\.12\.3", compact)
    assert suspended
    if not re.search(r"(?:未持有|无).*?暂时停牌", suspended[1]):
        for p in positions:
            if p["stock_code"].split(".")[0] in suspended[1]:
                matched.add(p["stock_code"])
        assert any(p["stock_code"].split(".")[0] in suspended[1] for p in positions), "EXECUTION_SUSPENDED_UNKNOWN"
    lending = re.search(r"[56]\.4\.12\.4(.*?)[56]\.4\.13", compact)
    assert lending
    assert re.search(r"(?:未持有|无|未参与).*?转融通证券出借业务", lending[1])
    upper = sum(D(p["market_value_cny"]) for p in positions if p["stock_code"] in matched) + D(bond_amount)
    assert upper / nav <= D(".15")
    return {
        "method": "ALL_POSSIBLE_RESTRICTED_STOCK_WHOLE_POSITION_PLUS_ALL_BONDS_UPPER",
        "amount_upper": str(upper),
        "ratio_upper": str(upper / nav),
        "exact_restricted_amount": None,
        "matched_positions": sorted(matched),
        "identity_rows": claims,
        "original_stock_section": stockpart,
        "suspended_section": suspended[0],
        "lending_section": lending[0],
        "header_view": header_view,
        "only_supports_contract_15pct_state": True,
    }


def _execution(inputs, source, sessions, calendar_hash):
    """纯原件回放入口；只读取指定归档根目录，返回观察及标准R5单维复算。

    输入只携带来源和报告/证券身份绑定，不接受缓存现金、受限金额或维度分。
    所有数字重新从同日原件提取；函数不联网、不写文件或数据库。
    """
    _require(
        inputs.get("fund_code") == "010737"
        and inputs.get("share_class") == "C"
        and inputs.get("basis") == ACTIVE_BASIS
        and inputs.get("as_of_date") == "2026-09-30",
        "EFUNDS_EXECUTION_SCOPE",
    )
    sources = {h: {**ref, "path": str(source.root / ref["file"])} for h, ref in source.sources.items()}
    reports = {key: {**r, "source": sources[r["source_hash"]]} for key, r in inputs["reports"].items()}
    assert len(reports) == 18
    assert sorted({r["period"] for r in reports.values()}) == EXPECTED
    assert sorted(r["period"] for k, r in reports.items() if k.endswith("full")) == FULL
    contracts = [{**r, "source": sources[r["source_hash"]]} for r in inputs["contracts"]]
    assert sorted(c["published_on"] for c in contracts) == ["2021-03-30", "2024-12-12", "2025-03-19"]
    contract_observations = []
    for c in contracts:
        pages = load_source(source, c["source"])
        t = norm("".join(pages))
        assert LEGAL in norm(pages[0]), "EXECUTION_CONTRACT_LEGAL_IDENTITY"
        required = [
            "本基金股票资产占基金资产的比例不低于80%",
            "投资于标的指数成份股及备选成份股的资产不低于非现金基金资产的80%",
            "其中港股通股票不超过股票资产的50%",
            "本基金持有一家公司发行的证券（同一家公司在境内和香港同时上市的A+H股合计计算）",
            "本基金资产总值不超过基金资产净值的140%",
            "现金不包括结算备付金、存出保证金",
            "本基金主动投资于流动性受限资产的市值合计不得超过该基金资产净值的15%",
            "基金财产参与股票发行申购",
            "可转换债券",
        ]
        missing = [s for s in required if s not in t]
        if missing:
            raise ValueError("EXECUTION_CONTRACT_CLAUSE:" + repr(missing))
        assert "本基金股票资产占基金资产的比例为80%-95%" in t
        contract_observations.append(
            {
                "source": c["source"],
                "published_on": c["published_on"],
                "limits_same_across_all_acquired_versions": True,
                "clauses": [
                    {"text": s, "physical_pages": [i + 1 for i, p in enumerate(pages) if s in norm(p)]}
                    for s in required
                ],
            }
        )

    ah = {}
    for link in inputs["ah_links"]:
        link = {**link, "source": sources[link["source_hash"]]}
        pages = load_source(source, link["source"], first_pages=20)
        assert any(
            link["a_code"] in norm(p)
            and re.search(r"(?<!\d)0*" + re.escape(link["hk_code"].lstrip("0")) + r"(?!\d)", norm(p))
            and any(s in p for s in ["股票", "股份代号", "股份代號"])
            for p in pages
        ), "EXECUTION_AH_STOCK_IDENTITY"
        ah[link["hk_code"]] = link["a_code"]
    observations = []
    full = []
    source_refs = []
    for key, r in sorted(reports.items()):
        pages = load_source(source, r["source"])
        text = "\n".join(pages)
        t = norm(text)
        assert LEGAL in t and "010737" in t
        assert str(report_period(text)[1]) == r["period"], (key, "ECONOMIC_PERIOD")
        if r["source"].get("published_on", "0000") > "2026-09-30":
            raise ValueError("EXECUTION_FUTURE_REPORT")
        source_refs.append(r["source"]["sha256"])
        start = list(re.finditer("期末基金资产组合情况", t))[-1].end()
        tail = t[start:]
        end = re.search(r"[5678]\.2(?:报告期末)?按行业", tail)
        assert end
        block = tail[: end.start()]
        stocks = re.findall(r"其中[:：]股票([\d,]+\.\d{2})[\d.]+", block)
        hks = re.findall(r"通过港股通交易机制投资的港股(?:市值|金额)(?:为人民币|为|人民币)?([\d,]+\.\d{2})元", block)
        totals = re.findall(r"合计([\d,]+\.\d{2})100\.00", block)
        assert len(stocks) == len(hks) == len(totals) == 1, (key, "ASSET_TABLE")
        stock, hk, total = [D(x.replace(",", "")) for x in [stocks[0], hks[0], totals[0]]]
        try:
            navproof = report_product_net_assets(pages, observed_on=r["period"], total_assets=str(total))
        except ValueError as error:
            if str(error) != "ALLOCATION_NAV_EXPLICIT_TWO_SHARE_CLASS_HEADER_REQUIRED":
                raise
            financial = t[t.find("3.1主要财务指标") :]
            m = re.search(r"4\.期末基金资产净值((?:[\d,]+\.\d{2}){3})5\.期末基金份额净值", financial)
            assert m and "010736010737022914" in t[t.find("下属分级基金的交易代码") :][:500]
            values = re.findall(r"\d{1,3}(?:,\d{3})+\.\d{2}", m[1])
            assert len(values) == 3
            assert all(s in financial[:700] for s in ["精选增强A", "精选增强C", "精选增强Y"])
            navproof = {
                "net_assets_cny": str(sum(D(v.replace(",", "")) for v in values)),
                "proof": {"source_row": m[0], "share_codes": ["010736", "010737", "022914"], "values": values},
            }
        nav = D(navproof["net_assets_cny"])
        assert D(0) <= hk <= stock <= total and D(0) < nav <= total
        assert stock / total >= D(".8") and stock / nav <= D(".95") and hk / stock <= D(".5")
        # 配置比率即使分母解释在净/总资产之间，亦不跨80/95阈值；无需取中点。
        row = {
            "period": r["period"],
            "key": key,
            "stock_amount": str(stock),
            "total_assets": str(total),
            "net_assets": str(nav),
            "stock_ratio_bounds": [str(stock / total), str(stock / nav)],
            "hk_stock_ratio": str(hk / stock),
            "asset_source_quote": block,
            "nav_proof": navproof,
            "source": r["source"],
        }
        managers = list(re.finditer(r"[34]\.2管理人对报告期内本基金运作遵规守信情况的说明(.*?)[34]\.3", t))
        manager = managers[-1] if managers else None
        assert manager and "基金运作合法合规" in manager[1], (key, "MANAGER_STATEMENT")
        row["manager_statement"] = manager[0]
        # 取实际投资结论而非目录或合同投资政策，不从账面衍生余额推期货名义头寸。
        for kind in ["股指期货", "国债期货"]:
            m = re.search(r"本基金本报告期(?:末)?(?:未投资|未参与|未持有)" + kind, t)
            assert m, (key, "ACTUAL_FUTURES", kind)
            row[kind + "_actual"] = m[0]
        observations.append(row)
        if not key.endswith("full"):
            continue
        view, trace = periodic_financial_view([text])
        view = [re.sub(r"(?m)^[ \t]+7\.[1-9](?:\.[1-9])?[^\n]*", lambda m: m[0].lstrip(), p) for p in view]
        b = strict_balance_values(view, observed_on=r["period"])["point_book"]
        assert b
        positions = parse_disclosed(view, b["stock_amount"], b["net_assets"])["holdings"]
        assert D(b["stock_amount"]) == stock and D(b["net_assets"]) == nav and D(b["total_assets"]) == total
        ixref = sources[inputs["index_sources"][r["period"]]]
        ix = source.json(ixref["sha256"])
        assert ixref["api_name"] == "index_weight" and ixref["params"]["index_code"] == "000300.SH"
        assert ix.get("code") == 0
        ixrows = [dict(zip(ix["data"]["fields"], v, strict=True)) for v in ix["data"]["items"]]
        assert len(ixrows) == 300 and len({v["con_code"] for v in ixrows}) == 300
        assert {v["index_code"] for v in ixrows} == {"000300.SH"}
        expected_session = max(str(d) for d in sessions if str(d) <= r["period"]).replace("-", "")
        dates = {v["trade_date"] for v in ixrows}
        assert dates == {expected_session}
        members = {v["con_code"] for v in ixrows}
        eligible = sum(D(p["market_value_cny"]) for p in positions if p["stock_code"] in members)
        accounting = accounting_observations(view, date.fromisoformat(r["period"]))
        principal = D(accounting["observations"]["demand_deposits_principal"]["value"])
        # 全总资产减普通银行本金是非现金分母的保守上界，未核备选只能提高分子。
        assert eligible / (total - principal) >= D(".8")
        assert (principal - D(b["assets"]["存出保证金"])) / nav >= D(".05")
        issuers = defaultdict(D)
        for p in positions:
            sec = p["stock_code"]
            short = sec.split(".")[0]
            issuers[ah.get(short, short)] += D(p["market_value_cny"])
        # 所有已披露债券全部加在最大股票发行人上，仍不足10%则无需假设债券发行人。
        largest = max(issuers.values())
        bond = D(b["bond_amount"])
        assert (largest + bond) / nav <= D(".1")
        assert total / nav <= D("1.4")
        try:
            restrictions = restricted_securities(view, positions, b["bond_amount"])
            assert D(restrictions["restricted_amount"]) / nav <= D(".15")
        except ValueError as error:
            if not str(error).startswith(("RATING_MARKET_RESTRICTED_VALUE:", "RATING_MARKET_RESTRICTED_IDENTITY:")):
                raise
            restrictions = restricted_upper_bound(view, positions, b["bond_amount"], nav)
            restrictions["exact_parser_failure_preserved"] = str(error)
        bondhead = list(re.finditer(r"[78]\.6期末按公允价值占基金资产净值比例大小排序的前五名债券投资明细", t))[-1]
        bondtail = t[bondhead.end() :]
        end = re.search(r"[78]\.7期末", bondtail)
        assert end
        bondtable = bondtail[: end.start()]
        if bond == 0:
            assert "未持有债券" in bondtable
        else:
            # 下方已逐项审阅的实际债券全表，按发行股票代码不猜信用等级。
            expected = {
                "2025-06-30": {"123254": "759161.25"},
                "2025-12-31": {"123254": "1014827.98", "127110": "208483.44"},
            }[r["period"]]
            assert sum(D(v) for v in expected.values()) == bond and all(code in bondtable for code in expected)
            assert all(v.replace(",", "") in bondtable.replace(",", "") for v in expected.values())
        assert D(b["fund_amount"]) == 0
        assert all("本基金本报告期末未持有" + s in t for s in ["资产支持证券", "贵金属", "权证"])
        trustees = list(
            re.finditer(r"5\.2托管人对报告期内本基金投资运作遵规守信、净值计算、利润分配等情况的说明(.*?)5\.3", t)
        )
        trustee = trustees[-1] if trustees else None
        assert trustee and any(s in trustee[1] for s in ["未发现", "不存在"]), (key, "TRUSTEE")
        actual_te = re.findall(r"年化跟踪误差[为是：:约达达到了]*([\d.]+)%", t)
        assert len(set(actual_te)) == 1
        full.append(
            {
                "period": r["period"],
                "positions": positions,
                "balance": b,
                "index_source": ixref,
                "index_member_amount": str(eligible),
                "index_non_cash_ratio_lower": str(eligible / (total - principal)),
                "cash_after_all_deposited_guarantees_lower": str((principal - D(b["assets"]["存出保证金"])) / nav),
                "max_same_issuer_including_all_bonds_upper": str((largest + bond) / nav),
                "gross_asset_nav": str(total / nav),
                "restrictions": restrictions,
                "bond_full_table": bondtable,
                "trustee_actual_statement": trustee[0],
                "reported_annualized_TE_percent": actual_te[0],
                "financial_parser_trace": trace,
                "source": r["source"],
            }
        )

    reason = (
        "18份原件的12个经济时点股票80%—95%及港股不高于股票50%均在净/总资产两端闭合；"
        "六期完整股票、全部资产类型、同发行人10%、总资产140%、已披露受限15%、"
        "正式成份/非现金至少80%及银行本金扣全部存出保证金后至少5%均核准；"
        "两类期货报告期未投资有实际原句，经理与托管实际履职逐期核对。"
        "仅评价原协议公开披露粒度的配置/范围正常执行75，不保证逐日及全管理人合计合规。"
    )
    nodes = {
        k: {"kind": "state", "state": "NORMAL_PUBLIC_EXECUTION", "reason": reason, "source_refs": source_refs}
        for k in ["allocation", "mandate"]
    }
    packet = {
        "fund_code": "010737",
        "basis": ACTIVE_BASIS,
        "as_of_date": "2026-09-30",
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "observations": nodes,
            "target_product_aliases": ["010736", "010737", "022914"],
            "material_event_pending": inputs["material_event_pending"],
        },
        "quality_public": {"dataDates": {"holdings": "2026-06-30", "nav": "2026-09-30"}},
        "r4_evidence_registry": {
            h: {"path": ref["path"], "published_on": ref.get("published_on")} for h, ref in sources.items()
        },
        "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in sources],
    }
    standard = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"efficiency"})
    assert standard["dimensions"]["efficiency"] == "75" and standard["score"] is None and standard["grade"] is None
    output = {
        "at": datetime.now(UTC).isoformat(),
        "fund_code": "010737",
        "as_of": "2026-09-30",
        "contracts": contract_observations,
        "reports": observations,
        "full_reports": full,
        "calendar_hash": calendar_hash,
        "observations": nodes,
        "packet": packet,
        "standard_replay": standard,
        "score": "75",
        "complete_score": None,
        "scope_limit": [
            "按原协议12季/6全文公开粒度，不声称三年每日合规；2026Q3未在截止日前公开。",
            "全管理人合计持仓及每日交易不由公开定报反推为0。",
            "年化跟踪误差5.41/6.96/7.25/8.02/9.26/15.46%已保存；超过6%力求目标不隐去。"
            "原主动60/40评价配置/范围，不擅改为被动TE公式或自造异常扣档。",
        ],
        "formal_writes": 0,
    }
    return output


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def api(raw, ref, code):
    data = json.loads(raw, parse_float=D)
    if ref.get("params", {}).get("ts_code") != code + ".OF" or data.get("code") != 0 or data["data"].get("has_more"):
        raise ValueError("ENHANCED_RETURN_API_SCOPE")
    rows = [dict(zip(data["data"]["fields"], r, strict=True)) for r in data["data"]["items"]]
    if any(r["ts_code"] != code + ".OF" for r in rows):
        raise ValueError("ENHANCED_RETURN_OTHER_SHARE")
    return rows


def quarter(text):
    """从全部行业合计对上全部股票金额；公布百分数保留完整舍入区间。"""
    heads = [s for s in re.finditer(r"5\.2(?=[^\d.])", text) if "行业" in text[s.end() : s.end() + 60]]
    if not heads:
        raise ValueError("ENHANCED_RETURN_INDUSTRY_HEADER")
    start = heads[-1].start()
    ends = [m for m in re.finditer(r"5\.3(?=[^\d.])", text) if m.start() > start]
    if not ends:
        raise ValueError("ENHANCED_RETURN_INDUSTRY_END")
    block = text[start : ends[0].start()]
    assetheads = [m for m in re.finditer("期末基金资产组合情况", text) if m.start() < start]
    if not assetheads:
        raise ValueError("ENHANCED_RETURN_STOCK_HEADER")
    asset = text[assetheads[-1].end() : start]
    amounts = re.findall(r"其中[:：]?(?:股票|普通股)([\d,]+\.\d{2})", asset)
    if len(amounts) != 1:
        raise ValueError("ENHANCED_RETURN_STOCK_AMOUNT")
    stock = D(amounts[0].replace(",", ""))
    lower_scale = D(1)
    if "其中:普通股" in asset:
        equities = re.findall(r"权益投资([\d,]+\.\d{2})", asset)
        receipts = re.findall(r"存托凭证([\d,]+\.\d{2})", asset)
        if len(equities) != 1 or len(receipts) > 1 or not receipts and "存托凭证--" not in asset:
            raise ValueError("ENHANCED_RETURN_CDR_SCOPE")
        cdr = D(receipts[0].replace(",", "")) if receipts else D(0)
        equity = D(equities[0].replace(",", ""))
        if stock + cdr != equity:
            raise ValueError("ENHANCED_RETURN_CDR_RECONCILIATION")
        lower_scale = stock / equity
        stock = equity
    totals = list(re.finditer(r"合计([\d,]+\.\d{2})(\d+\.\d{2})", block))
    if not totals or sum(D(m[1].replace(",", "")) for m in totals) != stock:
        raise ValueError("ENHANCED_RETURN_ALL_STOCK_RECONCILIATION")
    point = sum(D(m[2]) for m in totals)
    error = D(".005") * len(totals)
    return (max(D(0), point - error) * lower_scale, point + error)


def growth(nav_rows, dividend_rows, days):
    """按披露净值增长口径逐日连乘，分红日分母为前日净值减每份分红。

    单位净值与累计净值的差额变化逐日核分配，重复事件仅在金额和日期相同
    时去重；不依赖存在异常变更的供应商复权列，不修改该原始列。
    """
    rows = {r["nav_date"]: r for r in nav_rows}
    if len(rows) != len(nav_rows) or not set(days) <= rows.keys() or any(not days[0] <= d <= days[-1] for d in rows):
        raise ValueError("ENHANCED_RETURN_NAV_CALENDAR")
    events = {}
    for row in dividend_rows:
        ex = row.get("net_ex_date") or row.get("ex_date")
        if not ex or not days[0] < ex <= days[-1]:
            continue
        if (
            row["div_proc"] != "实施"
            or D(row["div_cash"]) <= 0
            or row.get("net_ex_date")
            and row.get("ex_date")
            and row["net_ex_date"] != row["ex_date"]
        ):
            raise ValueError("ENHANCED_RETURN_DISTRIBUTION_STATE")
        cash = D(row["div_cash"])
        if ex in events and events[ex] != cash:
            raise ValueError("ENHANCED_RETURN_DISTRIBUTION_CONFLICT")
        events[ex] = cash
    ordered = sorted(rows)
    series = {ordered[0]: D(1)}
    for a, b in zip(ordered, ordered[1:], strict=False):
        prior = D(rows[a]["unit_nav"])
        now = D(rows[b]["unit_nav"])
        cash = events.get(b, D(0))
        if min(prior, now, prior - cash) <= 0:
            raise ValueError("ENHANCED_RETURN_NONPOSITIVE_NAV")
        delta = (D(rows[b]["accum_nav"]) - now) - (D(rows[a]["accum_nav"]) - prior)
        if delta != cash:
            raise ValueError("ENHANCED_RETURN_CUMULATIVE_DIVIDEND_CONFLICT")
        series[b] = series[a] * now / (prior - cash)
    return rows, series, events


def _returns(inputs, source, own_execution, sessions, cal_hash):
    """独立重读原件并回放28产品准入及回报；返回单维候选，不写任何文件。"""
    _require(sorted(inputs["peers"]) == PEERS, "EFUNDS_FROZEN_COHORT")
    financial_docs = set(inputs["financial_document_hashes"])
    expected_docs = {
        r["source_hash"] for spec in [inputs["target"], *inputs["peers"].values()] for r in spec["reports"].values()
    }
    _require(financial_docs == expected_docs, "EFUNDS_DOCUMENT_SCOPE")
    sources = {h: {**ref, "path": str(source.root / ref["file"])} for h, ref in source.sources.items()}
    raws = source.raw
    own_budget = {
        r["period"]: D(r["stock_amount"]) / D(r["net_assets"]) * 100
        for r in own_execution["reports"]
        if r["key"].endswith("quarter")
    }
    dates = [str(d) for d in sessions if "2023-09-28" <= str(d) <= "2026-09-30"]
    days = [d.replace("-", "") for d in dates]
    audit = {}
    metrics = {}
    all_refs = set()
    with localcontext() as ctx:
        ctx.prec = 70
        for code in ["010737", *PEERS]:
            spec = inputs["target"] if code == "010737" else inputs["peers"][code]
            if code == "010737":
                provider = json.loads(raws[spec["provider_hash"]], parse_float=D)
                if provider["fund_code"] != code:
                    raise ValueError("ENHANCED_RETURN_TARGET_SHARE")
                nr = [
                    {
                        "nav_date": r["nav_date"].replace("-", ""),
                        "unit_nav": D(r["unit_nav"]),
                        "accum_nav": D(r["accumulated_nav"]),
                    }
                    for r in provider["history"]["rows"]
                ]
                div = api(raws[spec["div_hash"]], sources[spec["div_hash"]], code)
                all_refs.add(spec["provider_hash"])
            else:
                nr = api(raws[spec["nav_hash"]], sources[spec["nav_hash"]], code)
                div = api(raws[spec["div_hash"]], sources[spec["div_hash"]], code)
                all_refs.add(spec["nav_hash"])
            all_refs.add(spec["div_hash"])
            nav, series, events = growth(nr, div, days)
            reports = spec["reports"]
            expected_reports = sorted([p + "-quarter" for p in EXPECTED] + [p + "-full" for p in FULL])
            if sorted(reports) != expected_reports:
                raise ValueError("ENHANCED_RETURN_REPORT_WINDOW")
            budget = []
            checks = []
            distribution_evidence = []
            for key, item in sorted(reports.items()):
                digest = item["source_hash"]
                ref = sources[digest]
                kind = item["kind"]
                period = item["period"]
                pages = load_source(source, ref, first_pages=20)
                text = compact("".join(pages))
                all_refs.add(digest)
                if code not in text:
                    raise ValueError("ENHANCED_RETURN_REPORT_SHARE")
                yr, mo, dd = map(int, period.split("-"))
                if not re.search(rf"{yr}(?:年|[-/.])0?{mo}(?:月|[-/.])0?{dd}(?:日)?", compact("".join(pages[:2]))):
                    raise ValueError("ENHANCED_RETURN_REPORT_PERIOD:" + code + ":" + key)
                if not any(
                    all(q in text[max(0, m.start() - 40) : m.end() + 420] for q in ["沪深300", "95%", "活期", "5%"])
                    for m in re.finditer("业绩比较基准", text)
                ):
                    raise ValueError("ENHANCED_RETURN_HISTORICAL_BENCHMARK")
                title = compact("".join(pages[:2]))[:500]
                product = ""
                for match in re.finditer("投资目标", text):
                    segment = text[match.start() : match.start() + 3500]
                    if "投资策略" in segment and "业绩比较基准" in segment:
                        end = segment.find("基金管理人", segment.find("业绩比较基准"))
                        product = segment[:end] if end > 0 else segment[:1800]
                        break
                role = title + product
                if (
                    not (
                        any(v in role for v in ["指数增强", "增强型指数", "增强指数型", "量化增强", "增强策略"])
                        or re.search(r"沪深300[^。;]{0,12}增强", role)
                    )
                    or "联接基金" in title
                    or "联接型基金" in title
                    or re.search(r"本基金(?:为|是)[^。;]{0,35}(?:ETF联接基金|联接型基金|联接基金)", product)
                ):
                    raise ValueError("ENHANCED_RETURN_HISTORICAL_ROLE")
                if kind == "quarter" and code != "010737":
                    lo, hi = quarter(text)
                    target = own_budget[period]
                    budget.append(
                        {
                            "period": period,
                            "low_gap_pp": str(D(0) if lo <= target <= hi else min(abs(target - lo), abs(target - hi))),
                            "high_gap_pp": str(max(abs(target - lo), abs(target - hi))),
                        }
                    )
                if kind == "full":
                    for match in re.finditer(
                        r"(?:3\.3过去三年基金的利润分配情况|4\.[78]管理人对报告期内基金利润分配情况的说明)", text
                    ):
                        quote = text[match.start() : match.start() + 1100]
                        if "......" not in quote[:150]:
                            distribution_evidence.append({"period": period, "source_hash": digest, "quote": quote})
                    if period == FULL[0]:
                        continue
                    marker = "基金份额净值增长率及其与同期业绩比较基准收益率的比较"
                    start = text.find(marker)
                    section = text[start:] if start >= 0 else ""
                    if code == "010737":
                        # 目标仅允许本C份额，不借A或Y净值。
                        heads = list(re.finditer(r"易方达沪深300(?:指数)?精选增强C", section))
                        if not heads:
                            raise ValueError("ENHANCED_RETURN_TARGET_C_TABLE")
                        section = section[heads[0].start() :]
                    match = re.search(
                        r"过去(?:六|6)个月([-+]?\d+(?:\.\d+)?)%([-+]?\d+(?:\.\d+)?)%([-+]?\d+(?:\.\d+)?)%", section
                    )
                    if not match:
                        raise ValueError("ENHANCED_RETURN_REPORT_RETURN_ROW")
                    header = section[: match.start()]
                    if code != "010737" and not re.search(r"A[^ABC]*阶段", header):
                        if not (code == "005850" or code == "110030" and period < "2026-06-30"):
                            raise ValueError("ENHANCED_RETURN_FIRST_TABLE_NOT_A")
                    boundary = period[:4] + ("0101" if period[5:7] == "06" else "0701")
                    a = max(d for d in series if d < boundary)
                    b = period.replace("-", "")
                    actual = (series[b] / series[a] - 1) * 100
                    if abs(actual - D(match[1])) > D(".005"):
                        raise ValueError("ENHANCED_RETURN_REPORT_NAV_CONFLICT:" + code + ":" + period)
                    checks.append(
                        {
                            "period": period,
                            "source_hash": digest,
                            "share_header": header,
                            "start": a,
                            "end": b,
                            "reported_percent": match[1],
                            "reconstructed_percent": str(actual),
                            "within_original_published_rounding": True,
                        }
                    )
            if code != "010737":
                if (
                    len(budget) != 12
                    or sum(D(r["high_gap_pp"]) for r in budget) / 12 > 5
                    or max(D(r["high_gap_pp"]) for r in budget) > 10
                ):
                    raise ValueError("ENHANCED_RETURN_PEER_STOCK_BUDGET")
            if len(checks) != 5:
                raise ValueError("ENHANCED_RETURN_FIVE_REPORT_CHECKS")
            # 两项实际分配须在本产品年报的分配表及管理人说明中直接出现，
            # 对没有分配者仍保留年度原文；日历内累计/单位净值差额另逐日闭合。
            if events:
                annual = "".join(r["quote"] for r in distribution_evidence if r["period"] == "2025-12-31")
                expected = "2.000" if code == "005137" else "0.841" if code == "005850" else None
                if len(events) != 1 or not expected or expected not in annual:
                    raise ValueError("ENHANCED_RETURN_DISTRIBUTION_REPORT_BINDING")
            span = D((date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days)
            metrics[code] = {
                "return_1y": str(series[days[-1]] / series["20250930"] - 1),
                "return_3y": str((series[days[-1]] / series[days[0]]) ** (D("365.25") / span) - 1),
            }
            audit[code] = {
                "reports_checked": len(reports),
                "half_year_return_crosschecks": checks,
                "stock_budget_distances": budget,
                "actual_dividend_events": {k: str(v) for k, v in events.items()},
                "distribution_original_evidence": distribution_evidence,
                "calendar_observations": len(days),
                "extra_actual_nav_dates": sorted(set(nav) - set(days)),
            }
    registry = {h: {"path": r["path"], "published_on": r.get("published_on")} for h, r in sources.items()}
    observations = {
        name: {
            "kind": "rank",
            "target": metrics["010737"][name],
            "peers": [{"product_id": code, "value": metrics[code][name]} for code in PEERS],
            "expected_products": PEERS,
            "direction": "higher",
            "source_refs": sorted(all_refs),
        }
        for name in ["return_1y", "return_3y"]
    }
    packet = {
        "fund_code": "010737",
        "basis": ACTIVE_BASIS,
        "as_of_date": "2026-09-30",
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "observations": observations,
            "target_product_aliases": ["010736", "010737", "022914"],
            "material_event_pending": inputs["material_event_pending"],
        },
        "quality_public": {"dataDates": {"nav": "2026-09-30", "holdings": "2026-06-30"}},
        "r4_evidence_registry": registry,
        "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in sources],
    }
    result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"return"})
    if any(result[k] is not None for k in ["score", "grade", "raw_score"]):
        raise ValueError("ENHANCED_RETURN_TOTAL_LEAK")
    return {
        "score": result["dimensions"]["return"],
        "standard_replay": result,
        "metrics": metrics,
        "audit": audit,
        "calendar_hash": cal_hash,
        "observations": observations,
        "packet": packet,
        "cohort_scope": "28个同目标直接增强法律产品；12季预算沿既有均差5pp/最大10pp准入，须统筹核现行适用范围。",
        "complete_score": None,
        "grade": None,
        "formal_writes": 0,
    }


def _material_review(inputs, source):
    """核逐事项底稿及原话，审阅完成不等于事件结束或监管独立验收。"""
    digest = inputs.get("material_event_review_ref")
    _require(digest == MATERIAL_REVIEW_HASH and digest in source.raw, "EFUNDS_MATERIAL_REVIEW_REQUIRED")
    review = source.json(digest)
    _require(
        review.get("version") == "010737_COMPONENT_MATERIAL_REVIEW_V1"
        and review.get("fund_code") == "010737"
        and review.get("share_class") == "C"
        and review.get("as_of_date") == inputs["as_of_date"]
        and review.get("legal_name") == LEGAL
        and review.get("manager_name") == MANAGER
        and review.get("primary_ref") == inputs["primary_ref"]
        and review.get("review_complete") is True
        and review.get("material_event_pending") is False
        and inputs.get("material_event_pending") is False,
        "EFUNDS_MATERIAL_REVIEW_SCOPE",
    )
    _require(
        review.get("manager_governance_state") == "REMEDIATION_DISCLOSED"
        and review.get("independent_regulatory_acceptance") is False,
        "EFUNDS_GOVERNANCE_FACT_PRESERVATION",
    )
    evidence = review.get("evidence")
    events = review.get("events")
    _require(
        isinstance(evidence, list) and evidence and isinstance(events, list) and events,
        "EFUNDS_EVENT_EVIDENCE_REQUIRED",
    )
    checked = set()
    for item in evidence:
        h = item["source_hash"]
        key = item["key"]
        _require(h in source.raw and key not in checked and item.get("quote"), "EFUNDS_EVENT_ORIGINAL_REQUIRED")
        _require(_iso(item.get("published_on")) <= inputs["as_of_date"], "EFUNDS_EVENT_PUBLICATION_DATE")
        if item.get("role") == "CATALOG_ROW":
            _require(h in inputs["catalog_source_hashes"], "EFUNDS_EVENT_CATALOG_SOURCE")
            candidates = [
                r
                for r in source.json(h)["aaData"]
                if r["uploadInfoId"] == item["upload_info_id"] and r["uploadInfoDetailId"] == item["upload_detail_id"]
            ]
            _require(
                len(candidates) == 1
                and candidates[0]["reportName"] == item["quote"]
                and (candidates[0].get("reportSendDate") or candidates[0]["uploadDate"]) == item["published_on"],
                "EFUNDS_EVENT_CATALOG_ROW_BINDING",
            )
        else:
            _require(
                _iso(source.sources[h].get("published_on")) == _iso(item["published_on"]), "EFUNDS_EVENT_ORIGINAL_DATE"
            )
        raw = source.raw[h]
        if raw.startswith(b"%PDF-"):
            body = "".join(load_source(source, source.sources[h], first_pages=2000))
        else:
            payload = source.json(h)
            if isinstance(payload.get("data"), dict) and payload["data"].get("notice_content"):
                body = "".join(load_source(source, source.sources[h]))
            else:
                body = json.dumps(payload, ensure_ascii=False)
        _require(norm(item["quote"]) in norm(body), "EFUNDS_EVENT_QUOTE_MISMATCH")
        checked.add(key)
    for event in events:
        _require(
            event.get("key")
            and event.get("evidence_keys")
            and set(event["evidence_keys"]) <= checked
            and event.get("review_complete") is True
            and event.get("blocks_selected_components") is False
            and event.get("reason")
            and event.get("status")
            in {
                "ROUTINE_CALENDAR_OR_SALES_ARRANGEMENT",
                "KNOWN_MANAGER_EVENT_PRESERVED",
                "ORIGINAL_NAV_ACCOUNTING_PRESERVED",
                "NO_SELECTED_COMPONENT_IMPACT",
                "NOT_THIS_PRODUCT",
            },
            "EFUNDS_UNKNOWN_OR_UNREVIEWED_EVENT",
        )
    _require(len({e["key"] for e in events}) == len(events), "EFUNDS_DUPLICATE_EVENT")
    for catalog in review.get("catalogs", []):
        _require(
            catalog.get("complete") is True
            and catalog.get("source_hashes")
            and set(catalog["source_hashes"]) <= source.raw.keys(),
            "EFUNDS_EVENT_CATALOG_REQUIRED",
        )
        for month in catalog.get("months", []):
            rows = []
            for h in month["source_hashes"]:
                payload = source.json(h)
                _require(
                    payload.get("success") is not False and payload["iTotalDisplayRecords"] == month["count"],
                    "EFUNDS_CATALOG_COUNT_CHANGED",
                )
                rows.extend(payload["aaData"])
            _require(
                len(rows) == month["count"]
                and len({(r["uploadInfoId"], r["uploadInfoDetailId"]) for r in rows}) == len(rows),
                "EFUNDS_CATALOG_INCOMPLETE_OR_DUPLICATE",
            )
            _require(
                all(
                    r.get("organName") == "易方达"
                    and month["start"] <= (r.get("reportSendDate") or r["uploadDate"]) <= month["end"]
                    for r in rows
                ),
                "EFUNDS_CATALOG_ISSUER_OR_DATE",
            )
    _require(
        any(c.get("period") == "2026-07-01/2026-09-30" for c in review.get("catalogs", [])),
        "EFUNDS_POST_REPORT_EVENT_REVIEW_REQUIRED",
    )
    return {
        "source_hash": digest,
        "events_reviewed": len(events),
        "evidence_checked": len(evidence),
        "manager_governance_state": review["manager_governance_state"],
        "independent_regulatory_acceptance": False,
        "material_event_pending": False,
    }


def calculate(inputs, *, root):
    """每次重验原件、编译观察，再交现R5选维计算；完整分与等级恒空。"""
    source, sessions, cal_hash = _source_guard(inputs, root)
    event_review = _material_review(inputs, source)
    ex = {**inputs["execution_inputs"], "material_event_pending": event_review["material_event_pending"]}
    audit = {"material_event_review": event_review, "calendar_hash": cal_hash}
    if "return" in inputs["selected_dimensions"]:
        audit["nav_availability"] = _nav_time_audit(inputs, source, sessions)
    execution = _execution(ex, source, sessions, cal_hash)
    dimensions = {"execution": execution["score"]}
    observations = dict(execution["observations"])
    audit["execution"] = execution
    if "return" in inputs["selected_dimensions"]:
        performance = _returns(
            {**inputs, "material_event_pending": event_review["material_event_pending"]},
            source,
            execution,
            sessions,
            cal_hash,
        )
        dimensions["return"] = performance["score"]
        observations.update(performance["observations"])
        audit["return"] = performance
    # 返回的是原观察重放的分项，无调用方score、未知维度或完整分的默认值。
    return {
        "dimensions": dimensions,
        "observations": observations,
        "audit": audit,
        "complete_score": None,
        "grade": None,
        "formal_writes": 0,
        "cohort_scope_review_required": "return" in dimensions,
    }


def check_binding(bundle, inputs):
    """本C份额、管理人、名义截止、持仓期和真实主报告缺一即拒收。"""
    _require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code") == "010737"
        and bundle.get("share_class") == inputs.get("share_class") == "C"
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date") == "2026-09-30"
        and str(bundle.get("holding_date")) == inputs.get("holding_date") == "2026-06-30"
        and bundle["identity"]["legal_name"] == inputs.get("legal_name") == LEGAL
        and bundle["identity"]["manager_name"] == inputs.get("manager_name") == MANAGER
        and bundle["original_report_archive"]["sha256"] == inputs.get("primary_ref"),
        "EFUNDS_CURRENT_BINDING",
    )


def prepare(session, request, *, root):
    """只读最新current生成待审追加包；保留其他维度，不登记、不发布。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    code = request["fund_code"]
    raw = deepcopy(request["inputs"])
    previous = current_inputs(session, [code]).get(code)
    _require(previous and str(previous["as_of_date"]) == request["as_of_date"], "EFUNDS_CURRENT_PERIOD_REQUIRED")
    _require(request.get("expected_current_ref") == previous["batch_id"], "EFUNDS_STALE_CURRENT_REF")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    check_binding(bundle, raw)
    _require(
        bundle.get("basis") == BASIS
        and not any(
            bundle.get(k)
            for k in [
                "r5_calculation",
                "passive_components_calculation",
                "passive_execution_calculation",
                "passive_performance_calculation",
                "gold_components_calculation",
                "enhanced_performance_calculation",
                "efunds_enhanced_components_calculation",
            ]
        ),
        "EFUNDS_CONFLICTING_COMPONENTS",
    )
    _require(request.get("selected_dimensions") == raw["selected_dimensions"], "EFUNDS_REQUEST_SELECTED_DIMENSIONS")
    # 本候选不能把研究冻结的参照适配自行升格为已批准。执行单维不受此缺口阻塞。
    _require("return" not in raw["selected_dimensions"], "EFUNDS_RETURN_COHORT_SCOPE_DECISION_REQUIRED")
    old = deepcopy(bundle["reviewed_document"]["dimensions"])
    for key in raw["selected_dimensions"]:
        point = old[key]
        _require(
            point is None or point.get("candidate") is None or point.get("lower") != point.get("upper"),
            "EFUNDS_DIMENSION_ALREADY_FORMALLY_PUBLISHED:" + key,
        )
    raw["sources"] = {h: canonical_source(root, ref) for h, ref in raw["sources"].items()}
    result = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    for key in raw["selected_dimensions"]:
        reason = request["explanations"][key].strip()
        _require(len(reason) >= 40, "EFUNDS_EXPLANATION_REQUIRED")
        score = result["dimensions"][key]
        doc["dimensions"][key] = {"lower": score, "upper": score, "candidate": score, "reason": reason}
        bundle.setdefault("public_explanations", {})[key] = reason
        bundle.get("public_metrics", {}).pop(key, None)
    for key in set(old) - set(raw["selected_dimensions"]):
        _require(old[key] == doc["dimensions"][key], "EFUNDS_UNSELECTED_DIMENSION_CHANGED")
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        efunds_enhanced_components_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(r for h, r in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=Path(root))
    checked = snapshot_calculate(bundle, root=Path(root))
    _require(checked["status"] == "PARTIAL" and checked["score"] is None, "EFUNDS_CANNOT_CLAIM_COMPLETE")
    # 统筹须在正式快照加入本字段的calculate/check_binding调度；当前仅准备可审包。
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
