"""按原件字符列位恢复跨行的财务日期；不移动金额、不从报告标题猜日期。

限于原文同时明示本期末与比较期、两个年份位于同一印刷行、人民币元单位
以及可唯一恢复的两个完整自然日。未知文字、缺少日期片段或列序冲突均拒收。
"""

import re
import unicodedata
from datetime import date

VERSION = "QUALITY_DATE_COLUMNS_20261017_3"


def _offsets(value: str, measure: str) -> list[int] | None:
    """构建原字符边界；显示宽度只是可复算排版约定，不冒充PDF像素坐标。"""
    result = [0]
    for char in value:
        if char == "\t" and measure != "CODEPOINT":
            return None  # 原件未声明Tab宽度，不能给显示列猜一个制表位置。
        width = 2 if measure != "CODEPOINT" and unicodedata.east_asian_width(char) in {"W", "F"} else 1
        result.append(result[-1] + width)
    return result


def _column_parts(value: str, left: int, middle: int, measure: str) -> tuple | None:
    offsets = _offsets(value, measure)
    if offsets is None:
        return None
    boundaries = []
    for boundary in (left, middle):
        if boundary >= offsets[-1]:
            boundaries.append(len(value))
        elif boundary in offsets:
            boundaries.append(offsets.index(boundary))
        else:
            return None  # 不能把一个中文字符切成两部分后凑日期。
    a, b = boundaries
    return value[a:b], value[b:], [a, b], [b, len(value)]


def _decode_headers(lines: list[str], measure: str) -> list[dict] | None:
    """同一量尺核完全部重复表头；只还原原文日期，暂不按考察日筛选量尺。"""
    candidates = []
    for start, line in enumerate(lines):
        current = list(re.finditer("本期末", line))
        previous = list(re.finditer("上年度末|上期末", line))
        if not current or not previous:
            continue
        if len(current) != 1 or len(previous) != 1 or current[0].start() >= previous[0].start():
            return None
        # 不越过8个非空表头行寻找年份；不能在后面的金额或说明文字中找补。
        frame = []
        for index in range(start, len(lines)):
            if lines[index].strip():
                frame.append((index, lines[index]))
            if len(frame) >= 8:
                break
        year_rows = [(index, value, list(re.finditer(r"20\d{2}", value))) for index, value in frame]
        year_rows = [(index, value, matches) for index, value, matches in year_rows if matches]
        if not year_rows or len(year_rows[0][2]) != 2:
            return None
        year_index, year_line, years = year_rows[0]
        label_offsets, year_offsets = _offsets(line, measure), _offsets(year_line, measure)
        if label_offsets is None or year_offsets is None:
            return None
        left = min(label_offsets[current[0].start()], year_offsets[years[0].start()])
        middle = min(label_offsets[previous[0].start()], year_offsets[years[1].start()])
        if not left <= year_offsets[years[0].start()] < middle <= year_offsets[years[1].start()]:
            return None
        fragments = {"current": [], "comparison": []}
        trace, ignored = [], []
        decoded = False
        for index, value in frame:
            stub = re.sub(r"\s+", "", value)
            if index < year_index and stub in {"资产附注号", "资产", "附注号", "项目"}:
                ignored.append({"line": index + 1, "original_line": value, "meaning": "ROW_LABELS_NOT_DATE_COLUMNS"})
                trace.append({"line": index + 1, "original_line": value, "stub_row": True})
                continue
            parts = _column_parts(value, left, middle, measure)
            if parts is None:
                return None
            first, second, current_span, comparison_span = parts
            fragments["current"].append(re.sub(r"\s+|本期末", "", first))
            fragments["comparison"].append(re.sub(r"\s+|上年度末|上期末", "", second))
            trace.append({"line": index + 1, "original_line": value,
                          "current_fragment": first, "comparison_fragment": second,
                          "current_char_span": current_span, "comparison_char_span": comparison_span})
            current_text, previous_text = ("".join(fragments[key]) for key in ("current", "comparison"))
            stamp = r"(20\d{2})年(\d{1,2})月(\d{1,2})日"
            a, b = re.fullmatch(stamp, current_text), re.fullmatch(stamp, previous_text)
            if a and b:
                try:
                    actual, comparison = date(*map(int, a.groups())), date(*map(int, b.groups()))
                except ValueError as exc:
                    raise ValueError("ACCOUNTING_INVALID_EXPLICIT_HEADER_DATE") from exc
                decoded = True
                candidates.append({"version": VERSION, "rule": "EXPLICIT_UNIQUE_DATE_COLUMNS",
                        "current_date": str(actual), "comparison_date": str(comparison),
                        "column_measure": measure,
                        "current_left_column": left, "comparison_left_column": middle,
                        "original_header": "\n".join(lines[start:index + 1]), "fragments": trace,
                        "ignored_stub_rows": ignored, "currency": "CNY_YUAN"})
                break
        if not decoded:
            return None
    return candidates or None


def date_column_evidence(text: str, observed_on: date) -> dict | None:
    """只接纳所有完整排版解释一致的日期，不按考察日挑一个有利的解释。

    原件既有按字符数量对齐的转载，也有中文按两列宽度排版的表格。分别
    用两种量尺重读全部重复表头；任一完整解释有日期矛盾即拒收。切片仍
    保存原字符下标和原片段，不改日期数字，更不移动金额或填缺失单元格。
    """
    if len(text) > 100_000:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    units = re.findall(r"单位:\s*([^\r\n]{1,20})", normalized)
    if not units or any(unit.strip() != "人民币元" for unit in units):
        return None
    interpretations = []
    for measure in ("CODEPOINT", "EAST_ASIAN_DISPLAY_WIDTH"):
        try:
            decoded = _decode_headers(normalized.splitlines(), measure)
        except ValueError:
            return None
        if decoded:
            interpretations.append(decoded)
    pairs = {(item["current_date"], item["comparison_date"]) for model in interpretations for item in model}
    if not interpretations or len(pairs) != 1:
        return None
    actual, comparison = next(iter(pairs))
    if actual != str(observed_on) or comparison >= actual:
        return None
    chosen = interpretations[0]
    return chosen[0] | {"concordant_header_count": len(chosen), "repeated_headers": chosen[1:],
                        "equivalent_column_measures": [model[0]["column_measure"] for model in interpretations]}
