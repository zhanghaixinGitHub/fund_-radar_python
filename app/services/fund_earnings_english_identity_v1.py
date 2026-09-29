"""英文原件的身份补证；较晚译文不继承中文版的早期公开日期。"""

import re
from datetime import date, timedelta

from app.services.fund_earnings_batch_v4 import issuer_identity
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_earnings_yoy_amount_v1 import unique
from app.services.fund_information_history_v1 import normalize


def _name(value):
    """公司名称只忽略大小写、空白及末尾句点，不翻译、不删除组成词。"""
    return normalize(value).casefold().rstrip(".")


def review_english_identity(spec, document, reference=None, reference_spec=None):
    """按冻结页码逐字核对代码、双语法定名称和期间，仅返回身份结论。

    年报在自身公司简介中闭合身份；无代码的一季报可依同组织此前已核实的
    英文年报。参考不得来自未来，也不能用译文封面日期倒填目录公开日。
    两种载体均不因此成为新的独立业绩事件，金额、翻译一致性需另行核验。
    """
    row, pages = document["row"], document["pages"]
    public = date.fromisoformat(row["published_date"])
    year, mode = spec["year"], spec["mode"]
    if (
        row["announcementId"] != spec["document_id"]
        or row["secCode"] != spec["stock"]
        or not re.fullmatch(r"\d{6}", spec["stock"])
        or not row.get("orgId")
        or document.get("revision_issues")
    ):
        raise ValueError("ENGLISH_IDENTITY_CATALOG_OR_REVISION_CONFLICT")
    explicit = re.findall(r"Stock\s*[Cc]ode\s*:?\s*(\d{6})(?!\d)", "\n".join(pages[:15]))
    if (
        any(code != spec["stock"] for code in explicit)
        or issuer_identity(row, pages).get("reason") == "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG"
    ):
        raise ValueError("EXPLICIT_ISSUER_CONFLICT")
    if not re.fullmatch(r"[\u4e00-\u9fff（）()]+(?:股份有限公司|有限公司)", spec["chinese_name"]):
        raise ValueError("FULL_CHINESE_LEGAL_NAME_REQUIRED")
    titles = {
        "ANNUAL_PROFILE": f"{year}年年度报告（英文版）",
        "Q1_EARLIER_PROFILE": f"{year}年第一季度报告全文（英文版）",
    }
    if mode not in titles or normalize(row["title_plain"]) not in {titles[mode], spec["chinese_name"] + titles[mode]}:
        raise ValueError("EXACT_ENGLISH_CARRIER_REQUIRED")
    allowed_titles = (
        {f"{year} Annual Report", f"The {year} Annual Report", f"Annual Report {year}"}
        if mode == "ANNUAL_PROFILE"
        else {f"Interim Report for the First Quarter {year}"}
    )
    end = date(year, 12, 31) if mode == "ANNUAL_PROFILE" else date(year, 3, 31)
    if spec["title_quote"] not in allowed_titles or public <= end:
        raise ValueError("ENGLISH_REPORT_TITLE_OR_PERIOD_INVALID")
    if _name(spec["cover_name_quote"]) != _name(spec["english_name"]):
        raise ValueError("COVER_ENGLISH_NAME_MISMATCH")
    # 原文词边界拒绝 LongerMidea 等名称子串；允许 PDF 的换行和空白变化。
    pattern = (
        r"(?<![A-Za-z])"
        + r"\s*".join(re.escape(c) for c in spec["cover_name_quote"] if not c.isspace())
        + r"(?![A-Za-z])"
    )
    if not re.search(pattern, pages[0]):
        raise ValueError("EXACT_COVER_NAME_REQUIRED")
    anchors = [occurrence(pages, 1, spec["cover_name_quote"]), occurrence(pages, 1, spec["title_quote"])]
    reference_proof = None
    if mode == "ANNUAL_PROFILE":
        cn_heads = {"Name of the Company in Chinese", "Full Chinese name"}
        en_heads = {"Name of the Company in English (if any)", "Full English name"}
        if spec["chinese_heading"] not in cn_heads or spec["english_heading"] not in en_heads:
            raise ValueError("EXPLICIT_BILINGUAL_PROFILE_HEADINGS_REQUIRED")
        for heading, name in [
            (spec["chinese_heading"], spec["chinese_name"]),
            (spec["english_heading"], spec["english_name"]),
        ]:
            anchors.append(unique(pages, spec["profile_page"], heading + name))
        code_quote = spec["code_quote"]
        if not re.fullmatch(r"Stock[Cc]ode:?" + spec["stock"], normalize(code_quote)):
            raise ValueError("EXPLICIT_STOCK_CODE_REQUIRED")
        a = unique(pages, spec["code_page"], code_quote)
        tail = normalize(pages[spec["code_page"] - 1])[a["offset"] + len(a["text"]) :]
        if tail and tail[0].isdigit():
            raise ValueError("STOCK_CODE_TOKEN_BOUNDARY")
        anchors.append(a)
    else:
        if reference is None or reference_spec is None or reference_spec["mode"] != "ANNUAL_PROFILE":
            raise ValueError("EARLIER_INTRINSIC_ENGLISH_PROFILE_REQUIRED")
        other = reference["row"]
        if (
            other["announcementId"] == row["announcementId"]
            or other["secCode"] != row["secCode"]
            or other["orgId"] != row["orgId"]
            or date.fromisoformat(other["published_date"]) > public
            or reference_spec["chinese_name"] != spec["chinese_name"]
            or _name(reference_spec["english_name"]) != _name(spec["english_name"])
        ):
            raise ValueError("ENGLISH_PROFILE_REFERENCE_CONFLICT")
        reference_proof = review_english_identity(reference_spec, reference)
    return {
        "document_id": row["announcementId"],
        "stock": row["secCode"],
        "passed": True,
        "rule": mode,
        "chinese_name": spec["chinese_name"],
        "english_name": spec["english_name"],
        "anchors": anchors,
        "reference_proof": reference_proof,
        "source_sha256": document["receipt"]["sha256"],
        "published_date": str(public),
        "available_at": str(public + timedelta(days=1)) + "T08:00:00+08:00",
        "published_date_unchanged": True,
        "earlier_chinese_release_date_inherited": False,
        "translation_content_equivalence_verified": False,
        "independent_earnings_event": False,
        "semantic_verified": False,
        "version_history_verified": False,
        "old_stop_entry_preserved": True,
        "permission_to_redownload": False,
        "training_ready": False,
    }
