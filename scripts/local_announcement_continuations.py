"""为可确定的跨页续句添加双向出处依赖，不依据问题或模型答案补凑证据。"""

import re
from copy import deepcopy


def link_continuations(units: list[dict]) -> list[dict]:
    """只关联相邻页正文的数字/单位断点或未闭合括号，保留所有原文及顺序。

    末尾没有句号本身不足以证明续句；新章节、非连续页和表格边界均不推断。
    这里的关联只说明阅读上下文相接，不能证明整个回答得到这些引文支持。
    """
    result = deepcopy(units)
    body = [u for u in result if u.get("kind") != "page_layout"]
    for left, right in zip(body, body[1:], strict=False):
        if left.get("kind") != "text" or right.get("kind") != "text":
            continue
        page = left.get("page")
        if not isinstance(page, int) or right.get("page") != page + 1:
            continue
        tail = re.sub(r"\s+", "", left["text"])
        head = re.sub(r"\s+", "", right["text"])
        if not tail or not head or re.match(r"(?:[一二三四五六七八九十]+[、．.]|第.+[章节]|[（(]\d+[）)])", head):
            continue
        split_unit = re.search(r"\d(?:[\d,.]*\d)?$", tail) and re.match(r"(?:亿|万)?(?:元|股|%|％|亿元|万元)", head)
        # 只检查最后一个句号之后的括号，不把前文引用标题中的括号跨页传播。
        ending = re.split(r"[。！？!?]", tail)[-1]
        opened = ending.count("（") + ending.count("(") - ending.count("）") - ending.count(")")
        closing = re.search(r"[）)]", head)
        split_bracket = opened == 1 and closing is not None and closing.start() <= 80
        if not (split_unit or split_bracket):
            continue
        reason = "number_unit_across_pages" if split_unit else "parenthesis_across_pages"
        for source, target in [(left, right), (right, left)]:
            source.setdefault("continuation_ids", []).append(target["id"])
            source.setdefault("continuation_audit", []).append({"target": target["id"], "reason": reason})
    return result
