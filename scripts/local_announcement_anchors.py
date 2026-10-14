"""逐字引文定位：只匹配资料中实际出现的片段，不根据答案数字补凑出处。"""

import re
from bisect import bisect_right
from copy import deepcopy

from scripts.local_announcement_reader import compact


def resolve_quotes(parsed: dict, units: list[dict]) -> tuple[dict, dict[int, dict]]:
    """把模型摘录映射回原文编号；允许排版空白变化和最多三个相邻证据单元。

    原文文字与带列号的表格展示分别索引，绝不把两种展示混接。定位单元必须唯一，
    不能模糊改字、修数字、扩大到附近段落，亦不使用 answer 内容作为检索条件。
    返回兼容旧检查器的副本及定位轨迹；模型原始响应保持不变。
    """
    business = [u for u in units if u.get("kind") != "page_layout"]
    aliases = {u["citation_id"]: u["id"] for u in business if "citation_id" in u}

    def locator_text(text):
        # 中英文引号是展示字形差异，原始引文另行保留；数字、标点边界及限定词
        # 一律不改。额外的E编号包装必须已知且与最终定位相符，不能猜错编号。
        return compact(text).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))

    indexes = []
    for key in ("text", "rendered"):
        pieces = [locator_text(u.get(key, u["text"])) for u in business]
        ends, total = [], 0
        for piece in pieces:
            total += len(piece)
            ends.append(total)
        indexes.append(("".join(pieces), ends))
    normalized = deepcopy(parsed)
    audits = {}
    candidates = normalized.get("answers", []) if isinstance(normalized, dict) else []
    if not isinstance(candidates, list):
        return normalized, audits
    for item in candidates:
        if not isinstance(item, dict) or type(item.get("question_number")) is not int:
            continue
        snippets = item.get("evidence_quotes")
        refs, anchors, issues = [], [], []
        if not isinstance(snippets, list) or not 1 <= len(snippets) <= 12:
            snippets = []
            issues.append("QUOTE_LIST_INVALID")
        for snippet in snippets:
            if not isinstance(snippet, str) or not 8 <= len(compact(snippet)) <= 800:
                issues.append("QUOTE_LENGTH_INVALID")
                continue
            needle = locator_text(snippet)
            prefix = re.match(r"^\[(E\d+)\]", needle)
            declared = aliases.get(prefix[1]) if prefix else None
            if declared:
                needle = needle[prefix.end() :]
                if len(needle) < 8:
                    issues.append("QUOTE_LENGTH_INVALID")
                    continue
            matches = set()
            for text, ends in indexes:
                start = text.find(needle)
                while start >= 0:
                    first = bisect_right(ends, start)
                    last = bisect_right(ends, start + len(needle) - 1)
                    if last - first < 3:
                        matches.add(tuple(u["id"] for u in business[first : last + 1]))
                    start = text.find(needle, start + 1)
            if len(matches) != 1:
                if declared:
                    matches = {m for m in matches if declared in m}
            if len(matches) == 1 and declared not in next(iter(matches)) and declared:
                issues.append("QUOTE_ID_MISMATCH")
                continue
            if len(matches) != 1:
                issues.append("QUOTE_NOT_FOUND" if not matches else "QUOTE_LOCATION_AMBIGUOUS")
                anchors.append({"quote": snippet, "matches": [list(m) for m in sorted(matches)]})
                continue
            matched = list(next(iter(matches)))
            anchors.append({"quote": snippet, "source_ids": matched, "declared_source_id": declared})
            refs.extend(ref for ref in matched if ref not in refs)
        # 即使输入另带任意 evidence_ids，也不能绕过逐字定位。
        item["evidence_ids"] = refs
        audits[item["question_number"]] = {
            "model_evidence_quotes": deepcopy(snippets),
            "quote_anchors": anchors,
            "resolved_evidence_ids": refs,
            "anchor_issues": list(dict.fromkeys(issues)),
        }
    return normalized, audits
