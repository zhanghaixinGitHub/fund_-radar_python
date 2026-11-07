"""010737完整持仓维度的独立原件重放组件。

按已冻结的增强适配，与自身同日沪深300比较发行人及行业额外集中；
107家人工质量判断绑定不可变研究及实际原件。代码只读，不联网、不写库。
"""

import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services import fund_rating_efunds_components as base
from app.services.fund_quality_v2_stock_book import full_stock_book
from app.services.material_content import content_digest, source_bytes, source_exists, source_size, source_text

VERSION = "R7_EFUNDS_010737_COMPLETE_HOLDINGS_ORIGINALS_20261030_1"
BASE_SHA = "c214a325e8c4d58a48c13d4d47ffde31fd365c34f71ff6f43f536c0a600faf33"
# 原计算基线指纹保留；另固定只改资料读取方式的适配版本，数值和冻结研究仍逐项复算。
MATERIAL_ADAPTER_BASE_SHA = "2189751a098c6d235ca15459295d826c52aec9bef358e95e0311a1781b171a7a"
LARGE_ORIGINALS = {
    "8a1635486a767d11221ae47c03b2b4ef032bcaa6678c4f91e7f4ffc3bd1bdac7": 52990039,
    "01819e1c7daad939d1779a8aa729f50f02151192e609cb28c2c405634a8f343d": 79925886,
    "9962a41bf695316e960de23e32a8a31cd1cc4d4e6822d98a8ad4f6f46e3baf7f": 86097392,
}
# 以下摘要由同目录生成器填入，冻结逐公司人工判断及字段适配，不接受调用者改分。
PINS = {
    "quality_review_text": "68b4867aab3c21dfab8575795855eb12c8fc28d3f45356d960a8fb8ff7e53e79",
    "financial_identity_review_text": "19069af9a9414cf74f5ca600b98ff45c8ccc8575d42ef784f21a4aa3b086f50a",
    "non_stock_review_text": "f39ab9cb867795f41971e11788b418609f72710a488742f98233f328f5d94b44",
    "adaptation_text": "a5598ac5d0683accfc0045ab7c5a2a531cacf17c4af3d81b90cbb0b800785d26",
    "structure_bindings": "17adef86d45aa05b53d20b564551b85c0347cfe66fc736024e9e73220930a788",
    "source_hashes": "f862bb8cc2f79963c1db24961af73346da6a350166107536ba3f7bcb3b8cef46",
    "quality_original_raw_sha256": "c6c1f7975db28923483035a2ce45e1e7e6b5d5eff28cadcf55b149f21940d51b",
}


def require(condition, code):
    if not condition:
        raise ValueError("COMPLETE_HOLDINGS_" + code)


def canonical_hash(value):
    return content_digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


class Archive:
    def __init__(self, inputs, root):
        self.root = Path(root).resolve()
        self.sources = inputs["sources"]
        self.raw = {}

    def read(self, sha):
        if sha not in self.raw:
            ref = self.sources[sha]
            path = (self.root / ref["file"]).resolve()
            require(path.is_relative_to(self.root / "sources") and source_exists(path), "SOURCE_PATH")
            # 仅这3份真实大PDF按精确摘要/大小例外，不放宽普通50MB导入限制。
            size = source_size(path)
            require(size <= 50_000_000 or LARGE_ORIGINALS.get(sha) == size, "SOURCE_SIZE")
            data = source_bytes(path)
            require(content_digest(data) == sha == ref["sha256"], "SOURCE_HASH")
            self.raw[sha] = data
        return self.raw[sha]

    def pages(self, sha):
        self.read(sha)
        return base.load_source(self, self.sources[sha], first_pages=500)


def check_binding(bundle, inputs):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "010737"
        and inputs.get("selected_dimensions") == ["holdings"],
        "VERSION_OR_SCOPE",
    )
    require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == "010737"
        and bundle.get("share_class") == "C"
        and str(bundle.get("as_of_date")) == inputs["as_of_date"]
        and str(bundle.get("holding_date")) == inputs["holding_date"]
        and bundle["identity"]["legal_name"] == base.LEGAL
        and bundle["identity"]["manager_name"] == base.MANAGER
        and bundle["original_report_archive"]["sha256"] == inputs["structure_bindings"]["fund_report_hash"],
        "BUNDLE_IDENTITY_OR_DATES",
    )


def calculate(inputs, *, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "010737"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("selected_dimensions") == ["holdings"],
        "IDENTITY_OR_SELECTED_DIMENSIONS",
    )
    require(
        content_digest(source_bytes(Path(base.__file__))) in {BASE_SHA, MATERIAL_ADAPTER_BASE_SHA},
        "BASE_MODULE_VERSION",
    )
    for key in ["quality_review_text", "financial_identity_review_text", "non_stock_review_text", "adaptation_text"]:
        require(content_digest(inputs[key].encode("utf-8")) == PINS[key], "FROZEN_REVIEW:" + key)
    require(canonical_hash(inputs["structure_bindings"]) == PINS["structure_bindings"], "STRUCTURE_BINDINGS")
    require(canonical_hash(sorted(inputs["sources"])) == PINS["source_hashes"], "REGISTERED_SOURCE_SET")
    quality = json.loads(inputs["quality_review_text"])
    identity = json.loads(inputs["financial_identity_review_text"])
    non_stock = json.loads(inputs["non_stock_review_text"])
    adaptation = json.loads(inputs["adaptation_text"])
    require(
        quality["known_issuer_count"] == quality["total_issuer_count"] == 107
        and quality["remaining_unjudged_issuers"] == 0,
        "INCOMPLETE_COMPANY_QUALITY",
    )
    require(
        identity["passed"]
        and identity["financial_documents"] == 214
        and identity["quality_input_sha256"] == PINS["quality_original_raw_sha256"],
        "FINANCIAL_IDENTITY_OR_PERIOD_AUDIT",
    )
    require(adaptation["revision"] == "R7_ENHANCED_STRUCTURE_OWN_INDEX", "ENHANCED_STRUCTURE_PROTOCOL")
    require(
        non_stock["accounting_amount_reconciliation"]["sum_equals_reported_assets"]
        and non_stock["non_stock_review_state"] == "AMOUNTS_CLOSED_AND_PUBLIC_COUNTERPARTY_CREDIT_SCOPE_REVIEWED"
        and non_stock["cash_q_assigned"] is None
        and non_stock["cash_not_given_100_or_added_to_stock_Q"],
        "NONSTOCK_SUBSTANTIVE_SCOPE",
    )
    archive = Archive(inputs, root)
    # 全部原件先核真实字节；不将研究备注/公司卡当公开经济原件。
    for sha in archive.sources:
        archive.read(sha)
        archive.raw.clear()
    binding = inputs["structure_bindings"]
    pages = archive.pages(binding["fund_report_hash"])
    book = full_stock_book(pages, report_end=date(2026, 6, 30))
    require(
        len(book["positions"]) == 108 and D(book["balance"]["stock_amount"]) == D("5933694488.10"), "FULL_STOCK_BOOK"
    )
    registry = {r["code"]: r for r in json.loads(archive.read(binding["issuer_registry_hash"]))["stockList"]}
    ah = {}
    for link in binding["ah_identity"]:
        sha = link["source"]["sha256"]
        text = base.compact("".join(archive.pages(sha)))
        require(link["a_code"] in text and link["hk_code"] in text, "AH_LEGAL_IDENTITY")
        ah[link["hk_code"]] = link["a_code"]
    require(ah == {"00981": "688981", "00883": "600938", "02611": "601211"}, "AH_MAPPING")
    raw_index = json.loads(archive.read(binding["index_weight_hash"]), parse_float=D)
    require(raw_index["code"] == 0 and not raw_index["data"].get("has_more"), "INDEX_RESPONSE")
    index = [dict(zip(raw_index["data"]["fields"], r, strict=True)) for r in raw_index["data"]["items"]]
    require(
        len(index) == 300
        and len({r["con_code"] for r in index}) == 300
        and {r["trade_date"] for r in index} == {"20260630"}
        and {r["index_code"] for r in index} == {"000300.SH"},
        "SAME_DATE_OWN_INDEX_300",
    )
    class_pages = archive.pages(binding["classification_hash"])
    class_pages = [base.compact(t) for t in class_pages]
    cls = binding["classification_rows"]
    positions = []
    index_positions = []
    needed = {ah.get(p["stock_code"].split(".")[0], p["stock_code"].split(".")[0]) for p in book["positions"]}
    needed |= {r["con_code"].split(".")[0] for r in index}
    for code in needed:
        row = cls[code]
        require(row["security"] == code, "CLASSIFICATION_IDENTITY")
        if row["original_source"]["sha256"] != binding["classification_hash"]:
            # 三只不在同期统一清单的新股沿用已核同粒度发行文件，不能把
            # 真实行业缺项误填成统一清单行，也不放宽其他发行人的分类检查。

            from app.services.material_content import document_reader as PdfReader

            require(
                code in {"001248": "D44", "001399": "C39", "688797": "C39"}
                and row["industry_code"] == {"001248": "D44", "001399": "C39", "688797": "C39"}[code]
                and row["source_type"] == "ISSUER_IPO_DOCUMENT_SAME_GRAIN_SUPPLEMENT_NOT_CAPCO_LIST_ASSIGNMENT",
                "IPO_CLASSIFICATION_SCOPE",
            )
            source_hash = row["original_source"]["sha256"]
            original = PdfReader(archive.read(source_hash))
            actual_text = base.compact("".join(p.extract_text() or "" for p in original.pages))
            require(
                base.compact(row["quote"]) in actual_text and row["industry_code"] in row["quote"],
                "IPO_CLASSIFICATION_ACTUAL_QUOTE",
            )
            continue
        require(row["original_source"]["sha256"] == binding["classification_hash"], "CLASSIFICATION_LIST_SOURCE")
        require(base.compact(row["raw_row"]) in class_pages[row["pdf_page"] - 1], "CLASSIFICATION_ACTUAL_ROW")
        require(
            re.fullmatch(r"[A-S]\d{2}", row["industry_code"])
            and row["industry_code"][0] == row["section_code"]
            and re.search(r"(?<!\d)" + row["industry_code"][1:] + r"(?!\d)", row["raw_row"])
            and base.compact(row["industry_name"]) in base.compact(row["raw_row"]),
            "CLASSIFICATION_SECTION_AND_SUBINDUSTRY",
        )
    for p in book["positions"]:
        code = ah.get(p["stock_code"].split(".")[0], p["stock_code"].split(".")[0])
        positions.append(
            {
                "security": p["stock_code"],
                "issuer_code": code,
                "issuer": registry[code]["orgId"],
                "industry": cls[code]["industry_code"],
                "amount": p["market_value_cny"],
            }
        )
    for row in index:
        code = row["con_code"].split(".")[0]
        index_positions.append(
            {
                "security": row["con_code"],
                "issuer": registry[code]["orgId"],
                "industry": cls[code]["industry_code"],
                "amount": str(row["weight"]),
            }
        )

    def concentration(rows):
        total = sum(D(r["amount"]) for r in rows)
        issuers, industries = defaultdict(D), defaultdict(D)
        require(total > 0 and all(D(r["amount"]) > 0 for r in rows), "POSITIVE_AMOUNTS")
        for row in rows:
            issuers[row["issuer"]] += D(row["amount"]) / total
            industries[row["industry"]] += D(row["amount"]) / total
        return {
            "total": total,
            "issuer": sum(v * v for v in issuers.values()),
            "industry": sum(v * v for v in industries.values()),
            "issuer_count": len(issuers),
            "industry_count": len(industries),
        }

    with localcontext() as ctx:
        ctx.prec = 70
        target, reference = concentration(positions), concentration(index_positions)
        require(abs(reference["total"] - 100) <= D(".02"), "PUBLISHED_INDEX_ROUNDING_CLOSURE")
        require(target["issuer_count"] == 107, "LEGAL_ISSUER_COUNT")
        actual = defaultdict(D)
        for p in positions:
            actual[p["issuer_code"]] += D(p["amount"])
        records = quality["records"]
        require(
            len({r["code"] for r in records}) == len(records) == 107 and set(actual) == {r["code"] for r in records},
            "QUALITY_FULL_LEGAL_SCOPE",
        )
        weighted = D(0)
        for company in records:
            require(D(company["amount_CNY"]) == actual[company["code"]], "QUALITY_OWN_AMOUNT:" + company["code"])
            require(
                (company["state"], company["q"]) in [("NORMAL_REVIEWED", 75), ("LOCAL_UNCERTAINTY", 50)],
                "MANUAL_QUALITY_STATE",
            )
            for source in company["financial_sources"].values():
                require(
                    source["sha256"] in archive.sources and source.get("published_on") <= "2026-09-30",
                    "QUALITY_FINANCIAL_SOURCE",
                )
            weighted += actual[company["code"]] * D(company["q"])
        q = weighted / target["total"]
        require(str(q) == quality["asset_quality_score"], "QUALITY_INDEPENDENT_RECALCULATION")
        points = {}
        for key in ["issuer", "industry"]:
            x, m = target[key], reference[key]
            points[key] = (D(75) + D(25) * (m - x) / (m + x)) * (1 - max(x - m, D(0)))
        point = D(".35") * points["issuer"] + D(".35") * points["industry"] + D(".30") * q
        sensitivity = []
        for scenario in quality["judgment_sensitivity_scenarios"]:
            delta = (
                D(scenario["alternative_q"] - scenario["actual_q"]) * actual[scenario["issuer_code"]] / target["total"]
            )
            sensitivity.append(
                {
                    "issuer_code": scenario["issuer_code"],
                    "actual_q": scenario["actual_q"],
                    "alternative_q": scenario["alternative_q"],
                    "conditional_holdings_delta": str(D(".30") * delta),
                    "conditional_holdings_score": str(point + D(".30") * delta),
                }
            )
    return {
        "dimensions": {"holdings": str(point)},
        "complete_score": None,
        "grade": None,
        "audit": {
            "full_stock_positions": 108,
            "legal_issuers": 107,
            "index_members": 300,
            "target": {k: str(v) for k, v in target.items()},
            "reference": {k: str(v) for k, v in reference.items()},
            "issuer_subscore": str(points["issuer"]),
            "industry_subscore": str(points["industry"]),
            "stock_quality": str(q),
            "stock_quality_weight_coverage": "1",
            "subweights": [".35", ".35", ".30"],
            "non_stock_reviewed_not_added_to_stock_Q": True,
            "financial_original_identity_period_checks": 214,
            "source_count": len(archive.sources),
            "judgment_sensitivity": sensitivity,
        },
        "formal_writes": 0,
    }


def verify_registered_original(bundle, reference, *, root):
    """仅核本基金固定原件集中的同一PDF，普通50MB上限及三件精确例外均保留。"""
    inputs = bundle["efunds_complete_holdings_calculation"]["inputs"]
    check_binding(bundle, inputs)
    require(canonical_hash(sorted(inputs["sources"])) == PINS["source_hashes"], "REGISTERED_SOURCE_SET")
    require(inputs["sources"].get(reference["sha256"]) == reference, "ORIGINAL_ARCHIVE_BINDING")
    require(reference.get("kind") == "pdf" and reference.get("archive_kind") == "ORIGINAL", "ORIGINAL_PDF_KIND")
    content = Archive(inputs, root).read(reference["sha256"])
    require(content.startswith(b"%PDF-"), "ORIGINAL_PDF_SIGNATURE")


def canonical_source_for_prepare(root, reference):
    """大PDF只复用根已归档的三件，核字节与标准导入回执，不复制大原件。"""
    from datetime import datetime

    from app.services.fund_rating_pipeline_prepare import canonical_source

    sha = reference["sha256"]
    if sha not in LARGE_ORIGINALS:
        return canonical_source(root, reference)
    root = Path(root).resolve()
    relative = "sources/pipeline/" + sha + ".pdf"
    path = (root / relative).resolve()
    require(
        path.is_relative_to(root / "sources" / "pipeline")
        and source_exists(path)
        and (source_size(path) == LARGE_ORIGINALS[sha]),
        "LARGE_ORIGINAL_ALREADY_ARCHIVED_EXACT_SIZE",
    )
    raw = source_bytes(path)
    require(raw.startswith(b"%PDF-") and content_digest(raw) == sha, "LARGE_ORIGINAL_PDF_AND_HASH")
    receipt_path = root / "pipeline" / "imports" / (sha + ".json")
    require(source_exists(receipt_path), "LARGE_ORIGINAL_IMPORT_RECEIPT_REQUIRED")
    receipt = json.loads(source_text(receipt_path, "utf-8"))
    require(receipt.get("sha256") == sha and bool(receipt.get("imported_at")), "LARGE_ORIGINAL_IMPORT_RECEIPT_BINDING")
    imported_at = datetime.fromisoformat(receipt["imported_at"])
    require(imported_at.tzinfo is not None, "LARGE_ORIGINAL_IMPORT_RECEIPT_TIMEZONE")
    return {**reference, "file": relative, "path": relative, "imported_at": receipt["imported_at"]}


def prepare(session, request, *, root):
    """保留当前正式四维，只追加持仓；保存发布由根在实际服务完成。"""
    from datetime import UTC, datetime

    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    require(
        "efunds_complete_holdings_calculation" in snapshot_calculate.__code__.co_consts,
        "SNAPSHOT_DISPATCH_NOT_INSTALLED",
    )
    require(request["selected_dimensions"] == ["holdings"], "REQUEST_SCOPE")
    previous = current_inputs(session, ["010737"]).get("010737")
    require(previous and previous["batch_id"] == request["expected_current_ref"], "STALE_CURRENT_REF")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == "010737"))
    require(bundle.get("basis") == BASIS, "CURRENT_BASIS")
    raw = deepcopy(request["inputs"])
    check_binding(bundle, raw)
    raw["sources"] = {h: canonical_source_for_prepare(root, ref) for h, ref in raw["sources"].items()}
    result = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    old = deepcopy(doc["dimensions"])
    reason = request["explanations"]["holdings"].strip()
    require(len(reason) >= 40, "BUSINESS_EXPLANATION_REQUIRED")
    value = result["dimensions"]["holdings"]
    doc["dimensions"]["holdings"] = {"lower": value, "upper": value, "candidate": value, "reason": reason}
    require(all(old[k] == doc["dimensions"][k] for k in old if k != "holdings"), "UNREQUESTED_DIMENSION_CHANGED")
    bundle.setdefault("public_explanations", {})["holdings"] = reason
    bundle.get("public_metrics", {}).pop("holdings", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        efunds_complete_holdings_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=Path(root))
    checked = snapshot_calculate(bundle, root=Path(root))
    require(checked["status"] == "PARTIAL" and checked["score"] is None, "NO_COMPLETE_SCORE_CLAIM")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
