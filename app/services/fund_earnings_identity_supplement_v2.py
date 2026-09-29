"""补证已有 PDF 的主体和载体，不修改旧停止登记，不把审计附件当新业绩事件。"""

from app.services.fund_earnings_batch_v4 import issuer_identity
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import normalize


def review_explicit_identity(spec, document, reference=None):
    """按登记的页码核验主体、证券代码、年度及报告类型的连续原文锚点。

    两种年报只确认原 PDF 内主体和会计年度。审计附件必须同时找到同公司、
    同公开日且已通过身份核验的年报；仅作为财务审计佐证，不增加独立披露数。
    年报信息页可以位于末尾；不以原下载时间或独立历史网页存档为附加要求。
    """
    row, pages = document["row"], document["pages"]
    if row["announcementId"] != spec["document_id"] or row["secCode"] != spec["stock"]:
        raise ValueError("IDENTITY_CATALOG_CONFLICT")
    if (
        document.get("revision_issues")
        or issuer_identity(row, pages).get("reason") == "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG"
    ):
        raise ValueError("IDENTITY_SOURCE_CONFLICT")
    year, legal, short, stock = spec["year"], spec["legal_name"], spec["short_name"], spec["stock"]
    anchors = []

    def anchor(page, quote):
        value = occurrence(pages, page, quote)
        if len(value["offsets"]) != 1:
            raise ValueError("SUPPLEMENT_ANCHOR_NOT_UNIQUE")
        end = value["offsets"][0] + len(value["text"])
        source = normalize(pages[page - 1])
        if value["text"].endswith(stock) and end < len(source) and source[end].isdigit():
            raise ValueError("SECURITY_CODE_TOKEN_BOUNDARY")
        anchors.append(value)

    mode = spec["mode"]
    title = normalize(row["title_plain"])
    if mode in {"ANNUAL_CHINESE_YEAR", "ANNUAL_BODY_PERIOD"}:
        if title not in {f"{year}年年度报告", f"{short}{year}年年度报告", f"{legal}{year}年年度报告"}:
            raise ValueError("FULL_ANNUAL_CATALOG_TITLE_REQUIRED")
        p = spec["profile_page"]
        if mode == "ANNUAL_CHINESE_YEAR":
            chinese = "".join("零一二三四五六七八九"[int(c)] for c in str(year))
            anchor(spec["title_page"], chinese + "年年报")
            anchor(p, chinese + "年年报" + legal)
            headings = ["法定名称中文／英文全称", "法定名称中文╱英文全称"]
            matched = [h for h in headings if h + legal in normalize(pages[p - 1])]
            if len(matched) != 1:
                raise ValueError("EXPLICIT_LEGAL_NAME_HEADING_REQUIRED")
            anchor(p, matched[0] + legal)
            anchor(p, "证券简称及代码A股" + short + stock)
        else:
            anchor(p, "公司的中文名称" + legal)
            anchor(p, "股票简称" + short + "股票代码" + stock)
            anchor(spec["period_page"], f"报告期指{year}年1月1日—{year}年12月31日")
            anchor(
                spec["annual_page"], "公司董事会、监事会及董事、监事、高级管理人员保证年度报告内容的真实、准确、完整"
            )
        purpose = "ANNUAL_REPORT_IDENTITY_ONLY"
        reference_proof = None
    elif mode == "EXPLICIT_CATALOG_TITLE_VARIANT":
        # 只登记这三类可逐字解释的标题差异，摘要不得去掉，预告不得变成实际报告。
        variants = {
            "FORECAST_YEAR_WORD": (
                f"{legal}关于{year}年业绩预告的公告",
                f"{legal}关于{year}年度业绩预告的公告",
            ),
            "INCREASE_YEAR_WORD": (
                f"{short}{year}年年度业绩预增公告",
                f"{legal}{year}年度业绩预增公告",
            ),
            "SUMMARY_PARENTHESES": (
                f"{legal}{year}年年度报告(摘要)",
                f"{legal}{year}年年度报告摘要",
            ),
        }
        if spec["variant"] not in variants:
            raise ValueError("UNREGISTERED_TITLE_VARIANT")
        catalog_title, body_title = variants[spec["variant"]]
        if title != catalog_title or not issuer_identity(row, pages)["stock_verified"]:
            raise ValueError("EXPLICIT_CODE_AND_CATALOG_VARIANT_REQUIRED")
        anchor(1, body_title)
        purpose = "TITLE_AND_ISSUER_IDENTITY_ONLY"
        reference_proof = None
    elif mode == "AUDIT_ATTACHMENT_SAME_DAY_ANNUAL":
        if title != f"{year}年度报告审计报告（含经审计的财务报告及附注）" or reference is None:
            raise ValueError("AUDIT_ATTACHMENT_REFERENCE_REQUIRED")
        other = reference["row"]
        if (
            other["announcementId"] == row["announcementId"]
            or other["secCode"] != stock
            or other["published_date"] != row["published_date"]
            or normalize(other["title_plain"]) != f"{year}年年度报告"
            or reference.get("revision_issues")
            or not issuer_identity(other, reference["pages"])["passed"]
        ):
            raise ValueError("AUDIT_ATTACHMENT_REFERENCE_CONFLICT")
        anchor(1, legal + f"财务报表及审计报告{year}年12月31日止年度")
        reference_proof = occurrence(reference["pages"], spec["reference_legal_page"], legal)
        purpose = "AUDIT_ATTACHMENT_SUPPORT_ONLY_NOT_INDEPENDENT_EARNINGS_EVENT"
    else:
        raise ValueError("UNREGISTERED_IDENTITY_MODE")
    return {
        **spec,
        "passed": True,
        "rule": mode,
        "anchors": anchors,
        "reference_anchor": reference_proof,
        "source_sha256": document["receipt"]["sha256"],
        "reference_sha256": reference["receipt"]["sha256"] if reference else None,
        "published_date": row["published_date"],
        "published_date_unchanged": True,
        "purpose": purpose,
        "semantic_verified": False,
        "version_history_verified": False,
        "old_stop_entry_preserved": True,
        "permission_to_redownload": False,
        "training_ready": False,
    }
