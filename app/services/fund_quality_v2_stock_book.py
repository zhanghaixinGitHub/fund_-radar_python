"""V2研究用完整股票账簿：跨年度/中期重建，不改变正式评级解析器。

原文件及页号始终保留。年度报告的7节财报/8节组合仅在带有明确标题
的解析视图中映射到既有6/7节规则，表内金额、股数和证券代码不改写。
完整股票、股票受限类别、资产负债表及范围声明缺一则不认定账簿齐全。
债券及其他非股票受限事实不在本候选范围，不把未评估的金额记为零。
"""

import hashlib
import re
from datetime import date

from app.services.fund_quality_v2_balance_proofs import strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope, restricted_securities
from app.services.fund_rating_public_stock_tables import parse

VERSION = "QUALITY_FULL_STOCK_BOOK_20261017_6"


def annual_parser_view(pages: list[str]) -> tuple[list[str], list[dict]]:
    """仅对有明确年度财报及组合标题的原文建立解析视图，记录每处章号变更。

    7.40等小数不会匹配多级附注引用；财务列中的7.4.7.1是章节引用，
    保留改写前后行。缺少成对年度标题则不映射，交由原规则明确拒收。
    """
    text = "\n".join(pages)
    financial = re.search(r"(?m)^\s*7\.1\s*资产负债表", text)
    stock = re.search(r"(?m)^\s*8\.3\s*(?:期末|报告期末)[^\n]*股票\s*投\s*资\s*明\s*细", text)
    titles = {
        "资产负债表": "6.1",
        "利润表": "6.2",
        "报表附注": "6.4",
        "期末基金资产组合情况": "7.1",
        "报告期末按行业分类的股票投资组合": "7.2",
        "期末按公允价值占基金资产净值比例大小排序的股票投资明细": "7.3",
        "期末按公允价值占基金资产净值比例大小排序的所有股票投资明细": "7.3",
        "报告期内股票投资组合的重大变动": "7.4",
    }
    # 部分完整转载没有Word自动编号，按实际独占一行的完整标题设置内部标记。
    # 必须存在财报及完整股票表成对边界，不能从目录截断或缺标题中推定。
    standalone = {line.strip() for page in pages for line in page.splitlines()}
    unnumbered = {"资产负债表", "利润表", "报告期内股票投资组合的重大变动"} <= standalone and any(
        title in standalone and marker == "7.3" for title, marker in titles.items()
    )
    if not (financial and stock) and not unnumbered:
        return list(pages), []
    output, edits = [], []
    for page, source in enumerate(pages, 1):
        lines = []
        for line_number, original in enumerate(source.splitlines(), 1):
            # 附注引用至少两级，例如7.4.7.1，绝不匹配单一小数7.40。
            line = re.sub(r"(?<![\d.])7\.4\.(\d+(?:\.\d+)*)(?![\d.])", r"6.4.\1", original)
            line = re.sub(r"^\s*7(\.(?:1|2|4))(?=\s*[\u4e00-\u9fff])\s*", r"6\1 ", line)
            line = re.sub(r"^\s*8(\.\d+(?:\.\d+)*)(?=\s*[\u4e00-\u9fff])\s*", r"7\1 ", line)
            # 多级附注标题的章号已改，补上确实缺少的分隔空格。
            line = re.sub(r"^(\s*6\.4\.\d+(?:\.\d+)*)(?=[\u4e00-\u9fff])", r"\1 ", line)
            if unnumbered and original.strip() in titles:
                line = titles[original.strip()] + " " + original.strip()
            if line != original:
                edits.append({"page": page, "line": line_number, "before": original, "after": line})
            lines.append(line)
        output.append("\n".join(lines))
    return output, edits


def _stock_parser_view(pages: list[str], positions: list[dict]) -> tuple[list[str], list[dict]]:
    """只为确已在完整股票表中的证券补代码/名称分隔，不猜代码尾数或股数。"""
    known = {p["stock_code"].split(".")[0] for p in positions}
    output, edits = [], []
    for page, source in enumerate(pages, 1):
        lines = []
        for number, original in enumerate(source.splitlines(), 1):
            match = re.match(r"^(\s*)(\d{6})(?=[\u4e00-\u9fff])", original)
            line = original
            if match and match[2] in known:
                line = original[: match.end()] + " " + original[match.end() :]
            # 表尾成本与估值各明确保留两位小数，但转载会去掉两列间空格。
            # 只分开紧接备注横杠的两个完整金额，不推导或补入任何数字。
            line = re.sub(
                r"(?<![\d,.])((?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2})"
                r"((?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2})(?=[-－—](?:\s|$))",
                r"\1 \2",
                line,
            )
            if line != original:
                edits.append({"page": page, "line": number, "before": original, "after": line})
            lines.append(line)
        output.append("\n".join(lines))
    return output, edits


def _balance_parser_view(pages: list[str]) -> tuple[list[str], list[dict]]:
    """恢复当期金额尾数跨到下一行、上期金额已完整的唯一两列版式。

    仅资产负债表中两位金额的可见尾数，两个候选均可能合法时拒绝恢复。
    这不是由资产总额差额补值，所有数字必须已经出现在这两行原文中。
    """
    text = "\n".join(pages)
    starts = list(re.finditer(r"6\.1\s*资产负债表", text))
    if not starts:
        return pages, []
    a = starts[-1].start()
    stop = re.search(r"6\.2\s*利润表", text[a:])
    if not stop:
        return pages, []
    b = a + stop.start()
    block = text[a:b]
    pattern = (
        r"(?m)^(?P<left>[ \t]*[\u4e00-\u9fff：:（）()、]{2,40}"
        r"(?:[ \t]+[56]\.4\.\d+(?:\.\d+)*)?[ \t]+)"
        r"(?P<current>[\d,]+\.\d?)[ \t]+(?P<prior>[\d,]+\.\d{2})[ \t]*"
        r"\r?\n(?:[ \t]*\r?\n)?[ \t]*(?P<tail>\d{1,2})[ \t]*$"
    )
    edits = []

    def restore(match):
        current = match["current"]
        if len(current.split(".")[1]) + len(match["tail"]) != 2:
            return match[0]
        replacement = match["left"] + current + match["tail"] + " " + match["prior"]
        edits.append(
            {"before": match[0], "after": replacement, "rule": "VISIBLE_CURRENT_AMOUNT_TAIL_AFTER_PRIOR_COLUMN"}
        )
        return replacement

    restored = re.sub(pattern, restore, block)
    return ([text[:a] + restored + text[b:]], edits) if edits else (pages, [])


def periodic_financial_view(pages: list[str]) -> tuple[list[str], dict]:
    """为年度主报告复用已验证的财务章号与可见金额断行适配。

    返回的是内部解析页及完整修改轨迹，绝不是改写后的原件。业务摘录仍由
    调用者在原始pages中定位；不调整任何证券、股数或金额的数字次序。
    """
    view, edits = annual_parser_view(pages)
    view, money_edits = _balance_parser_view(view)
    return view, {
        "chapter_edits": edits,
        "visible_money_tail_edits": money_edits,
        "original_text_sha256": hashlib.sha256("\n".join(pages).encode("utf-8")).hexdigest(),
        "parser_text_sha256": hashlib.sha256("\n".join(view).encode("utf-8")).hexdigest(),
        "scope": "ORIGINAL_DIGITS_PRESERVED_PARSER_VIEW_NOT_ORIGINAL_TEXT",
    }


def full_stock_book(pages: list[str], *, report_end: date) -> dict:
    """从有身份绑定的真实报告页重新重建全表，不读取任何旧分项或布尔核验值。

    调用者负责报告份额/产品身份及原字节哈希。本函数核报告日期、资产和
    全部股票金额守恒、全部股票受限类别及范围声明，金额单位为人民币元。
    允许保留港股/BJ原事实，但下游沪深成交候选须单独拒收超范围产品。
    """
    text = "\n".join(pages)
    if report_period(text)[1] != report_end:
        raise ValueError("STOCK_BOOK_ORIGINAL_PERIOD_MISMATCH")
    view, edits = annual_parser_view(pages)
    view, money_edits = _balance_parser_view(view)
    balance_proof = strict_balance_values(view, observed_on=str(report_end))
    b = balance_proof["point_book"]
    if b is None:
        raise ValueError("STOCK_BOOK_EXACT_BALANCE_POINT_NOT_IDENTIFIED")
    stock = parse(view, b["stock_amount"], b["net_assets"], require_industry=False)
    restriction_view, stock_edits = _stock_parser_view(view, stock["holdings"])
    restricted = restricted_securities(restriction_view, stock["holdings"], b["bond_amount"], scope="STOCKS_ONLY")
    # 范围声明来自原文，不能用规范化视图中的人工章号充当原文摘录。
    scope = full_asset_restriction_scope(text)
    if not scope:
        raise ValueError("STOCK_BOOK_FULL_ASSET_RESTRICTION_SCOPE_NOT_EXPLICIT")
    return {
        "version": VERSION,
        "report_date": str(report_end),
        "balance": b,
        "balance_proof": balance_proof,
        "positions": stock["holdings"],
        "stock_row_normalization": stock["normalization"],
        "restricted_stock": restricted,
        "restriction_scope_excerpt": scope[0],
        "chapter_view": {
            "edits": edits,
            "visible_money_tail_edits": money_edits,
            "restriction_layout_edits": stock_edits,
            "original_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parser_text_sha256": hashlib.sha256("\n".join(view).encode("utf-8")).hexdigest(),
            "scope": "ORIGINAL_DIGITS_PRESERVED_PARSER_VIEW_NOT_ORIGINAL_TEXT",
        },
    }
