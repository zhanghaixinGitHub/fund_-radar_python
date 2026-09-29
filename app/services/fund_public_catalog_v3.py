"""公开栏目分页核验：栏目记录与正文网址分开计数，避免把别名当漏页。"""


def entry_key(row):
    """同一正文可由多个媒体标题收录；完整记录由网址、显示日期、标题共同标识。"""
    return row["url"], row["display_date"], row["title"]


def append_catalog(accumulated, rows, group, total):
    """验证原站单条前瞻重叠；保留不同标题记录，拒绝重复记录及跨页漂移。

    返回的是栏目记录，不是经济事件。后续正文按网址去重，标题别名必须单独
    审查，不能把栏目中的两种媒体名称计成两条独立新闻。
    """
    expected = min(45 if group == 0 else 46, total - group * 45)
    keys = [entry_key(r) for r in rows]
    if expected <= 0 or len(rows) != expected or len(set(keys)) != len(keys):
        raise ValueError("PAGE_COUNT_OR_DUPLICATE_RECORD")
    if group >= 2:
        if not accumulated or rows[0] != accumulated[-1]:
            raise ValueError("LOOKAHEAD_BOUNDARY_CHANGED")
        rows = rows[1:]
    if {entry_key(r) for r in rows} & {entry_key(r) for r in accumulated}:
        raise ValueError("REPEATED_RECORD_OUTSIDE_BOUNDARY")
    accumulated.extend(rows)
