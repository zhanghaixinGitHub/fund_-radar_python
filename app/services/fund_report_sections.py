"""恢复公开转载中丢失的章节号，仅修改解析副本，不改原文与证券数字。"""

import re


def numbered_pages(pages: list[str]) -> list[str]:
    """为独占一行的标准正文标题补章节号；目录省略号和普通段落不匹配。

    部分管理人的 PDF 章节号是独立绘图对象，公开转载的文本只有标题。
    每个标题在整份文本中必须唯一；有多个候选时保留原样，由下游拒绝，
    避免把附注中的同名文字或目录误当成财务表、期末股票表。
    """
    text = "\n".join(pages)
    headings = {
        "6.1": r"资产负债表",
        "6.2": r"利润表",
        "7.1": r"(?:报告)?期末基金资产组合情况",
        "7.2": r"(?:报告)?期末按行业分类的股票投资组合",
        "7.3": r"期末按公允价值占基金资产净值比例大小排序的股票投资明细",
        "7.3.1": r"期末按公允价值占基金资产净值比例大小排序的(?:所有|全部)股票投资明细",
        "7.4": r"报告期内股票投资组合的重大变动",
    }
    for number, title in headings.items():
        pattern = r"(?m)^[ \t]*(" + title + r")[ \t]*$"
        if len(list(re.finditer(pattern, text))) == 1:
            text = re.sub(pattern, lambda m, section_number=number: section_number + " " + m[1], text)
    return [text]
