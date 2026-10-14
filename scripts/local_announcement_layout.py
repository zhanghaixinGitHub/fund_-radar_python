"""仅根据 PDF 几何信息恢复版面角色与简单下标，保留原字符供追溯。

不包含公司、公告、业务金额或评测答案规则；无法可靠识别的版面保持原样。
"""

import re
from collections import Counter


def repeated_headers(pages: list) -> set[str]:
    """只识别至少两页重复、处于顶端 8.5% 区域的短文字行。"""
    counts = Counter()
    for page in pages:
        labels = {
            re.sub(r"\s+", "", line["text"])
            for line in page.extract_text_lines()
            if line["bottom"] <= page.height * 0.085 and len(line["text"]) >= 4
        }
        counts.update(labels)
    return {label for label, count in counts.items() if count >= 2}


def character_key(char: dict) -> tuple:
    return char["x0"], char["top"], char["x1"], char["bottom"], char["text"]


def running_page_numbers(pages: list) -> set[int]:
    """同一位置连续出现的物理页序可佐证页脚，避免仅靠字体大小漏判。

    仍要求底端10%、居中、窄行及恰好等于该页页序；至少两页位置一致才采用。
    不按业务金额、公司或待答问题判断，页码字符仍完整留存。
    """
    candidates = []
    for n, page in enumerate(pages, 1):
        for line in page.extract_text_lines():
            center = (line["x0"] + line["x1"]) / 2 / page.width
            if (
                line["text"].strip() == str(n)
                and line["top"] >= page.height * 0.9
                and 0.4 <= center <= 0.6
                and line["x1"] - line["x0"] < page.width * 0.1
            ):
                candidates.append((n, center, line["top"] / page.height))
    return {
        n
        for n, x, y in candidates
        if any(m != n and abs(x - xx) < 0.01 and abs(y - yy) < 0.015 for m, xx, yy in candidates)
    }


def margin_roles(page, number: int, headers: set[str], footer_pages: set[int] | None = None) -> dict[tuple, str]:
    """页码需位于底端、水平居中且等于 PDF 页序，避免误判正文的独立数字。

    角色只改变展示和事实引用资格，不删除任何字符。页序偏移或无法识别时保留正文。
    """
    roles = {}
    sizes = sorted(c.get("size", 0) for c in page.chars if c["text"].strip())
    body_size = sizes[len(sizes) // 2] if sizes else 0
    for line in page.extract_text_lines():
        label = re.sub(r"\s+", "", line["text"])
        center = (line["x0"] + line["x1"]) / 2
        role = None
        if line["bottom"] <= page.height * 0.085 and label in headers:
            role = "header"
        elif (
            label == str(number)
            and line["top"] >= page.height * 0.9
            and (
                number in (footer_pages or set())
                or line["top"] >= page.height * 0.94
                or max(c.get("size", 0) for c in line["chars"]) <= body_size * 0.85
            )
            and page.width * 0.4 <= center <= page.width * 0.6
            and line["x1"] - line["x0"] < page.width * 0.1
        ):
            role = "footer"
        if role:
            roles.update((character_key(c), role) for c in line["chars"])
    return roles


def align_subscripts(chars: list[dict]) -> tuple[list[dict], list[dict]]:
    """将紧邻拉丁/希腊变量且字体较小、基线较低的单个数字渲染成 _{数字}。

    只处理几何上唯一的相邻变量。原字形不改写；返回的副本用于展示，变换记录含
    原始坐标。普通数字、年份、正文脚注及无法唯一对应的符号不会被改作下标。
    """
    rendered, changes = [], []
    for char in chars:
        copied = dict(char)
        matches = []
        if re.fullmatch(r"[0-9]", char["text"]):
            for base in chars:
                if not re.fullmatch(r"[A-Za-zΑ-ω]", base["text"]):
                    continue
                height = base["bottom"] - base["top"]
                if (
                    char.get("size", 0) <= base.get("size", 0) * 0.8
                    and -0.5 <= char["x0"] - base["x1"] <= height * 0.3
                    and height * 0.3 <= char["top"] - base["top"] <= height * 0.8
                    and char["bottom"] <= base["bottom"] + height * 0.4
                ):
                    matches.append(base)
        if len(matches) == 1:
            base = matches[0]
            copied.update(top=base["top"], bottom=base["bottom"], text="_{" + char["text"] + "}")
            changes.append(
                {
                    "base": base["text"],
                    "subscript": char["text"],
                    "bbox": [char[k] for k in ("x0", "top", "x1", "bottom")],
                }
            )
        rendered.append(copied)
    return rendered, changes


def layout_blocks(chars: list[dict], prefix: str, page: int, role: str = "body") -> list[dict]:
    """按阅读顺序在完整分句后分块，保留原文字、排版展示、坐标及下标变换。

    分组使用恢复基线后的字符，避免下标与所属公式进入不同片段。跨页句子仍用
    不同原始引用编号，不拼接成伪造的连续引文。上下文中会保留相邻页正文。
    """
    from pdfplumber.utils import extract_text

    if not chars:
        return []
    aligned, _ = align_subscripts(chars)
    lines = []
    for index in sorted(range(len(chars)), key=lambda i: (aligned[i]["top"], aligned[i]["x0"])):
        if not lines or abs(aligned[index]["top"] - lines[-1][0]) > 3:
            lines.append((aligned[index]["top"], []))
        lines[-1][1].append(index)
    # 只在完整句/分句结束处切块，不在达到字数时截断物理行。括号中的例外条件
    # 与主句留在一起；长句可超过目标长度，整篇上下文预算另由请求入口严格限制。
    chunks, current, depth = [], [], 0
    for _, indexes in lines:
        # 同一行的中文、数字可能有轻微字体顶边偏移。行归并后必须重新按横坐标
        # 排序，否则标点会先于较低的数字被访问，导致金额被切到下一块。
        for index in sorted(indexes, key=lambda i: aligned[i]["x0"]):
            current.append(index)
            for char in aligned[index]["text"]:
                if char in "（(":
                    depth += 1
                elif char in "）)":
                    depth = max(0, depth - 1)
            if len(current) >= 260 and depth == 0 and aligned[index]["text"] in "。；！？":
                chunks.append(current)
                current = []
    if current:
        chunks.append(current)
    blocks = []
    for index, indexes in enumerate(chunks, 1):
        raw_chars = [chars[i] for i in indexes]
        display_chars, transforms = align_subscripts(raw_chars)
        raw_text = extract_text(raw_chars) or ""
        display = extract_text(display_chars) or ""
        if not raw_text.strip():
            continue
        if role != "body":
            display = f"【{'页眉' if role == 'header' else '页脚页码'}，仅定位，不作为业务数字】{display}"
        box = [
            min(c["x0"] for c in raw_chars),
            min(c["top"] for c in raw_chars),
            max(c["x1"] for c in raw_chars),
            max(c["bottom"] for c in raw_chars),
        ]
        blocks.append(
            {
                "id": f"{prefix}B{index}",
                "kind": "text" if role == "body" else "page_layout",
                "page": page,
                "role": role,
                "text": raw_text,
                "rendered": display,
                "bbox": box,
                "band": [box[1], box[3]],
                "typography_transforms": transforms,
            }
        )
    if role == "body":
        # 多重集只能发现丢字/重字，不能发现“字都在但顺序错了”。同时与整段
        # 几何提取的展示顺序核对，发现差异直接拒绝，不能把错序文字交给模型。
        expected = re.sub(r"\s+", "", extract_text(aligned) or "")
        actual = re.sub(r"\s+", "", "".join(b["rendered"] for b in blocks))
        if actual != expected:
            raise ValueError(f"{prefix}:READING_ORDER_MISMATCH")
    return blocks


def table_context(rows: list[dict]) -> dict[str, dict]:
    """从同表真实表头与合并格生成列名提示；不跨表、跨页猜测表头。

    顶行包含纯数值时不推断其为表头；多层表头只依据顶行明确的纵向合并范围。
    继承值必须来自覆盖该行的真实合并格，记录原引用编号，不填补普通空白格。
    """
    if not rows:
        return {}
    first = rows[0]["cells"]
    has_number = any(re.fullmatch(r"[\d,.%+-]+", re.sub(r"\s+", "", c["text"])) for c in first)
    has_paragraph = any(len(re.sub(r"\s+", "", c["text"])) > 50 for c in first)
    header_end = 0 if has_number or has_paragraph else max((c.get("row_end", 1) for c in first), default=1)
    if header_end > 3:
        header_end = 0
    headers = [c for r in rows[:header_end] for c in r["cells"]]
    active_header_ids = [r["id"] for r in rows[:header_end]]
    first_label = re.sub(r"\s+", "", first[0]["text"]) if first else ""
    result = {}
    for number, row in enumerate(rows, 1):
        # 财务表在同一网格中可能从“期末资产”切换为“期间收入”。遇到重新声明
        # 的同名表头，后续只能使用新日期，不能把最上方日期强加给整张表。
        row_cells = row["cells"]
        repeated_header = (
            header_end > 0
            and number > header_end
            and len(row_cells) >= 2
            and first_label
            and re.sub(r"\s+", "", row_cells[0]["text"]) == first_label
            and all(not re.fullmatch(r"[\d,.%+-]+", re.sub(r"\s+", "", c["text"])) for c in row_cells)
        )
        # 同一网格可能先列公司概况，再出现“项目 + 两个期间”的财务子表。
        # 后者的表头不一定处于第一列，前面还可能有跨多行的单位格。必须使用
        # 这一行实际写出的期间；不能继续把“经营范围”等业务内容当作列名。
        labels = [re.sub(r"\s+", "", c["text"]) for c in row_cells]
        dated_header = (
            number > header_end
            and any(label in {"项目", "指标", "财务指标"} for label in labels)
            and sum(
                c["column_start"] == c["column_end"]
                and len(label) <= 80
                and bool(re.search(r"(?:19|20)\d{2}年", label))
                for c, label in zip(row_cells, labels, strict=True)
            )
            >= 2
            and not any(re.fullmatch(r"[\d,.%+-]+", label) for label in labels)
        )
        if repeated_header or dated_header:
            headers = row_cells
            active_header_ids = [row["id"]]
            header_end = number
        inherited = []
        if number > header_end:
            for earlier in rows[: number - 1]:
                for cell in earlier["cells"]:
                    if header_end < cell.get("row_start", 0) < number <= cell.get("row_end", 0):
                        inherited.append({**cell, "source_row_id": earlier["id"]})
        columns = {}
        for cell in row["cells"] + inherited:
            if number <= header_end or cell["column_start"] != cell["column_end"]:
                continue
            column = cell["column_start"]
            names = [
                re.sub(r"\s+", "", h["text"])
                for h in headers
                if h["column_start"] <= column <= h["column_end"] and h["text"].strip()
            ]
            columns[column] = "/".join(dict.fromkeys(names))
        result[row["id"]] = {
            "header_rows": active_header_ids.copy(),
            "column_labels": columns,
            "context_cells": inherited,
        }
    return result
