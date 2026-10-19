"""读取正式报告账面基础事实，独立于国内股票表章号与质量参考区间。

金额仅代表披露日的人民币账面观察，不等于可动用现金、合格现金篮子或
策略容量。本模块不生成计分指标，不给未知横杠赋零，不使用上年比较列。
"""

import re
import unicodedata
from datetime import date
from decimal import Decimal

from app.services.fund_quality_v2_header_columns import date_column_evidence


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _net_asset_cent_suffix(section: str, label: str) -> dict | None:
    """识别原文净资产当前金额末位换行；不从会计等式或比较列补金额。

    只处理当前金额已有一位小数、比较金额完整两位小数的净资产合计行。
    前一行必须是未分配利润，后一有效行只有一个末位数字，再后一行必须
    是负债和净资产总计。末位须按原文本字符列或中英文显示宽度列与当前
    金额右边缘严格对齐，且落在比较金额左侧。这是文本版式规则，不是PDF
    像素坐标证明；不能吞页码或合并另一个单元格，另保留原文和对账结果。
    """
    lines = [match for match in re.finditer(r"(?m)^[^\r\n]+", section) if match[0].strip()]
    candidates = [i for i, match in enumerate(lines) if _compact(match[0]).startswith(label)]
    if len(candidates) != 1:
        return None
    i = candidates[0]
    if not 0 < i < len(lines) - 2:
        return None
    head, suffix, closing = lines[i][0], lines[i + 1][0], lines[i + 2][0]
    if (unicodedata.normalize("NFKC", head) != head
            or not lines[i - 1][0].strip().startswith("未分配利润")
            or not closing.strip().startswith("负债和净资产总计")):
        return None
    integer = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    partial = re.fullmatch(rf"[ \t]*{re.escape(label)}[ \t]+(?P<current>{integer}\.\d)"
                           rf"[ \t]+(?P<comparison>{integer}\.\d{{2}})[ \t]*", head)
    digit = re.fullmatch(r"[ \t]*(?P<digit>[0-9])[ \t]*", suffix)
    if not partial or not digit or "\t" in head + suffix:
        return None

    def offset(value: str, end: int, wide: bool) -> int:
        return sum(2 if wide and unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in value[:end])

    measures = []
    for wide, name in ((False, "CODEPOINT"), (True, "EAST_ASIAN_DISPLAY_WIDTH")):
        current_end = offset(head, partial.end("current"), wide)
        left = offset(head, partial.start("current"), wide)
        comparison_start = offset(head, partial.start("comparison"), wide)
        suffix_left, suffix_end = (offset(suffix, digit.start("digit"), wide), offset(suffix, digit.end("digit"), wide))
        if left <= suffix_left and suffix_end == current_end < comparison_start:
            measures.append({"column_measure": name, "current_right_edge": current_end,
                             "suffix_right_edge": suffix_end, "comparison_left_edge": comparison_start})
    if not measures:
        return None
    return {"current": partial["current"] + digit["digit"], "comparison": partial["comparison"],
            "original_block": section[lines[i].start():lines[i + 1].end()],
            "evidence": {"rule": "EXPLICIT_SINGLE_CENT_SUFFIX_CURRENT_CELL_RIGHT_EDGE",
                         "original_fragments": [head, suffix], "following_original_row": closing,
                         "current_fragment_char_span": list(partial.span("current")),
                         "suffix_char_span": list(digit.span("digit")), "matching_column_measures": measures}}


def _current_cny_header(text: str, observed_on: date) -> bool:
    """核实当期列的实际年月日和原文金额单位，兼容明确写出的前导零。"""
    if len(text) > 100_000:
        return False
    normalized = unicodedata.normalize("NFKC", text)
    units = re.findall(r"单位:\s*([^\r\n]{1,20})", normalized)
    stamp = r"(20\d{2})年(\d{1,2})月(\d{1,2})日"
    previous = r"(?:上年度末|上期末)"
    # PDF常按列抽取“本期末/日期/上年度末/日期”，完整转载则常按行抽取
    # “本期末/上年度末/日期/日期”。两种都必须明示两列及其完整日期；
    # 不从报告标题、下载日或孤立年份猜列序，比较期不得晚于或等于当期。
    # 两列日期可能各自完整印在括号内。只剥除包住完整年月日的成对括号，
    # 不改动正文、不补日期、不删除两列之间的未知文字；原始表头另行保存。
    dated_header = re.sub(rf"\(({stamp})\)", lambda match: match[1], _compact(text))
    pairs = [match.groups() for pattern in (
        rf"本期末{stamp}{previous}{stamp}", rf"本期末{previous}{stamp}{stamp}",
    ) for match in re.finditer(pattern, dated_header)]
    if not units or any(unit.strip() != "人民币元" for unit in units):
        return False
    if pairs:
        # 已匹配到明确日期但存在错期/列序冲突时不得改用几何分列“挽救”。
        return all(date(*map(int, pair[:3])) == observed_on and date(*map(int, pair[3:])) < observed_on
                   for pair in pairs)
    return date_column_evidence(text, observed_on) is not None


def _cash_note_columns(text: str, observed_on: date) -> int:
    """按原文表头确定资金附注是一列还是两列，不由数据行缺单元格猜列数。"""
    if _current_cny_header(text, observed_on):
        return 2
    normalized = unicodedata.normalize("NFKC", text)
    compact = _compact(text)
    units = re.findall(r"单位:\s*([^\r\n]{1,20})", normalized)
    # 单列只适用于附注；资产负债表继续要求完整的当期/比较期两列。
    # 出现比较列标记但日期未完整解码时拒收，不能把残缺两列表当作单列表。
    if not units or units[0].strip() != "人民币元" or any(
        label in compact for label in ("上年度末", "上期末", "比较期", "上年同期")
    ):
        return 0
    stamps = re.findall(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", compact)
    stamp = r"20\d{2}年\d{1,2}月\d{1,2}日"
    # 单列模板可能在“本期末”和唯一日期之间单独印一行“项目”。只在
    # 日期表头视图中剔除这条完整行，其他文字及全部日期仍按原文核验。
    date_header = re.sub(r"(?m)^[ \t]*项目[ \t]*$", "", normalized)
    declared_date = re.search(rf"本期末(?:{stamp}|\({stamp}\))", _compact(date_header))
    return int(bool(declared_date) and bool(stamps) and all(
        date(*map(int, stamp)) == observed_on for stamp in stamps
    ))


def accounting_observations(pages: list[str], observed_on: date) -> dict:
    """从同一报告资产负债表及货币资金附注读取明示的当期金额。

    调用方须先验证报告身份、原件摘要及公开时点。本函数再核表内日期、
    人民币元单位及主表两列/附注明示一至两列版式，返回原文行、页索引及逐项缺口。只在
    原文都有明确数值时检查会计恒等式；矛盾、错期、单位及列序不明即拒绝。
    """
    text = "\n".join(pages)
    day = f"{observed_on.year}年{observed_on.month}月{observed_on.day}日"
    headers = list(re.finditer(r"(?m)^\s*([567])\.1\s*资产负债表[ \t]*\r?$", text))
    # 有些发行人PDF不输出Word自动章号。只接纳独占行完整标题，且前方
    # 紧邻财务报告父标题；目录点线、一般提及或缺父节的孤立标题均不接纳。
    # 不给原件虚构“7.1”，保留financial_chapter=None和真实标题位置。
    for match in re.finditer(r"(?m)^[ \t]*资产负债表[ \t]*\r?$", text):
        if re.search(r"(?m)^[ \t]*(?:年度|半年度|中期)财务报告[ \t]*\r?$",
                     text[max(0, match.start() - 120):match.start()]):
            headers.append(match)
    sections = []
    for header in headers:
        tail = text[header.end():]
        chapter = header[1] if header.lastindex else None
        stop = re.search(rf"(?m)^\s*{chapter}\.2\s*利润表" if chapter else
                         r"(?m)^[ \t]*利润表[ \t]*\r?$", tail)
        if stop:
            section = tail[:stop.start()]
            if _current_cny_header(section, observed_on):
                sections.append((header, section))
    if len(sections) != 1:
        raise ValueError("ACCOUNTING_BALANCE_DATE_CNY_UNIT_OR_UNIQUE_BODY_UNVERIFIED")
    header, body = sections[0]
    observations, unresolved = {}, {}
    money = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
    slot = rf"(?:{money}|[-－—])"

    def row(name: str, label: str, section: str, columns: int = 2) -> None:
        candidates = [line for line in section.splitlines() if _compact(line).startswith(label)]
        if len(candidates) != 1:
            unresolved[name] = {"reason": "ACCOUNTING_ROW_MISSING_OR_AMBIGUOUS", "row_label": label}
            return
        original = candidates[0]
        line = unicodedata.normalize("NFKC", original).strip()
        values = rf"({slot})\s+({slot})" if columns == 2 else rf"({slot})"
        match = re.fullmatch(rf"{re.escape(label)}\s*(?:[567]\.4\.\d+(?:\.\d+)*)?\s*{values}", line)
        continuation = _net_asset_cent_suffix(section, label) if not match and name == "net_assets" else None
        if continuation and any(continuation["original_block"] in page for page in pages):
            match = (None, continuation["current"], continuation["comparison"])
            original = continuation["original_block"]
        if not match:
            unresolved[name] = {
                "reason": "ACCOUNTING_DECLARED_COLUMNS_UNRESOLVED", "declared_columns": columns,
                "original_row": original,
            }
            return
        if match[1] in {"-", "－", "—"}:
            unresolved[name] = {"reason": "ACCOUNTING_CURRENT_DASH_NOT_ASSIGNED_ZERO", "original_row": original}
            return
        value = Decimal(match[1].replace(",", ""))
        if value < 0:
            raise ValueError("ACCOUNTING_NONNEGATIVE_BALANCE_DOMAIN:" + name)
        observations[name] = {
            "value": str(value), "unit": "CNY_YUAN", "observed_on": str(observed_on),
            "original_row": original, "pages": [i + 1 for i, page in enumerate(pages) if original in page],
            "column": "本期末" + day, "declared_columns": columns,
            "comparison_cell_not_used": match[2] if columns == 2 else None,
            **({"monetary_continuation_evidence": continuation["evidence"]} if continuation else {}),
        }

    for name, labels in {
        "money_funds": ("货币资金", "银行存款"), "total_assets": ("资产总计",),
        "net_assets": ("净资产合计", "所有者权益合计"), "total_liabilities": ("负债合计",),
    }.items():
        present = [label for label in labels if any(_compact(line).startswith(label) for line in body.splitlines())]
        row(name, present[0] if len(present) == 1 else labels[0], body)
    notes, reference_conflicts = [], []
    balance_reference = re.search(
        r"[567]\.4\.\d+\.1", observations.get("money_funds", {}).get("original_row", "")
    )
    for match in re.finditer(r"(?m)^\s*([567]\.4\.\d+\.1)\s*(?:货币资金|银行存款)[ \t]*\r?$", text):
        # 跨境报告的资金附注可能是6.4.6.1。优先与主表明示的科目附注号
        # 相同；主表未印引用时仍要求资金标题、日期、单位及正文唯一。
        conflict = balance_reference and balance_reference[0] != match[1]
        if conflict:
            # 正式报告或转载本身也可能出现主表/附注编号不一致。保留两处
            # 原文作为待核证据；不能静默改号，也不能拿活期小计冒充货币资金。
            reference_conflicts.append({
                "balance_reference": balance_reference[0], "note_reference": match[1],
                "original_heading": match[0].strip(),
                "heading_pages": [i + 1 for i, page in enumerate(pages) if match[0].strip() in page],
            })
        tail = text[match.end():]
        next_note = match[1].rsplit(".", 1)[0] + ".2"
        stop = re.search(rf"(?m)^\s*{re.escape(next_note)}(?:\s|$)", tail)
        note = tail[:stop.start()] if stop else tail
        active_start = re.search(r"(?m)^\s*活期存款", note)
        if active_start:
            columns = _cash_note_columns(note[:active_start.start()], observed_on)
            if columns:
                binding = {"state": "EXPLICIT_REFERENCE_MATCH_OR_UNIQUE_UNREFERENCED_NOTE"}
                if conflict:
                    # 原件确有章号不一致时，不能只凭相似标题拼接。只允许同一
                    # 已核身份报告内日期/币种明确、正文唯一且附注“合计”明示金额
                    # 与主表货币资金完全相同的绑定，并永久保留两个原始章号。
                    # 比较的是整个附注合计，绝不拿活期存款小计作全额证明。
                    totals = [line for line in note.splitlines() if _compact(line).startswith("合计")]
                    total_cells = rf"({slot})\s+({slot})" if columns == 2 else rf"({slot})"
                    proof = re.fullmatch(rf"合计\s*{total_cells}", unicodedata.normalize("NFKC", totals[0]).strip()) \
                        if len(totals) == 1 else None
                    main = observations.get("money_funds")
                    if not proof or proof[1] in {"-", "－", "—"} or not main or (
                        Decimal(proof[1].replace(",", "")) != Decimal(main["value"])
                    ):
                        continue
                    binding = {
                        "state": "EXPLICIT_NOTE_TOTAL_RECONCILED_ORIGINAL_REFERENCE_CONFLICT_RETAINED",
                        "balance_reference": balance_reference[0], "note_reference": match[1],
                        "explicit_note_total": {"value": str(Decimal(proof[1].replace(",", ""))),
                                                "unit": "CNY_YUAN", "observed_on": str(observed_on),
                                                "original_row": totals[0],
                                                "pages": [i + 1 for i, page in enumerate(pages) if totals[0] in page]},
                        "balance_money_funds_value": main["value"],
                    }
                notes.append((note, columns, match[1], binding))
    if len(notes) == 1:
        note, columns, cash_note_reference, binding = notes[0]
        start = re.search(r"(?m)^\s*活期存款", note)
        stop = re.search(r"(?m)^\s*定期存款", note[start.end():]) if start else None
        if start and stop:
            active = note[start.start():start.end() + stop.start()]
            row("demand_deposits_total", "活期存款", active, columns)
            row("demand_deposits_principal", "等于:本金", active, columns)
            row("demand_deposits_accrued_interest", "加:应计利息", active, columns)
            if any(_compact(line).startswith("减:坏账准备") for line in active.splitlines()):
                row("demand_deposits_credit_loss", "减:坏账准备", active, columns)
        else:
            unresolved["demand_deposits_detail"] = {"reason": "ACCOUNTING_ACTIVE_DEPOSIT_SECTION_UNRESOLVED"}
    else:
        unresolved["demand_deposits_detail"] = (
            {"reason": "ACCOUNTING_BALANCE_NOTE_REFERENCE_CONFLICT",
             "conflicting_original_references": reference_conflicts}
            if reference_conflicts and not notes else
            {"reason": "ACCOUNTING_CASH_NOTE_DATE_UNIT_OR_BODY_UNVERIFIED"}
        )
    reconciliations = []
    for total, left, right in (
        ("total_assets", "net_assets", "total_liabilities"),
        ("demand_deposits_total", "demand_deposits_principal", "demand_deposits_accrued_interest"),
    ):
        if all(name in observations for name in (total, left, right)):
            difference = Decimal(observations[total]["value"]) - sum(
                Decimal(observations[name]["value"]) for name in (left, right)
            )
            deduction = "demand_deposits_credit_loss" if total == "demand_deposits_total" else None
            # 坏账准备原文有明确金额时按减项处理。原文为横杠时保留未知，
            # 三个明示金额的差额只作算术观察，不能反填为坏账准备或可用现金。
            if deduction in unresolved and abs(difference) > Decimal("0.01"):
                reconciliations.append({"total": total, "parts": [left, right], "difference_cny": str(difference),
                                        "state": "CREDIT_LOSS_UNRESOLVED_NO_IMPUTATION"})
                continue
            if deduction in observations:
                difference += Decimal(observations[deduction]["value"])
            if abs(difference) > Decimal("0.01"):
                raise ValueError("ACCOUNTING_EXPLICIT_AMOUNTS_DO_NOT_RECONCILE:" + total)
            reconciliations.append({
                "total": total, "parts": [left, right], "difference_cny": str(difference),
                **({"explicit_deduction": deduction} if deduction in observations else {}),
                **({"state": "DISCLOSED_AMOUNTS_MATCH_CREDIT_LOSS_DASH_STILL_UNASSIGNED"}
                   if deduction in unresolved else {}),
            })
    if all(name in observations for name in ("money_funds", "demand_deposits_total")) and (
        Decimal(observations["demand_deposits_total"]["value"]) > Decimal(observations["money_funds"]["value"])
    ):
        raise ValueError("ACCOUNTING_DEMAND_DEPOSITS_EXCEED_MONEY_FUNDS")
    return {
        "observations": observations, "unresolved": unresolved, "reconciliations": reconciliations,
        "balance_date_column_evidence": date_column_evidence(body, observed_on),
        "financial_chapter": header[1] if header.lastindex else None,
        "original_balance_heading": header[0].strip(),
        "balance_heading_pages": [i + 1 for i, page in enumerate(pages) if header[0].strip() in page],
        "observed_on": str(observed_on),
        "cash_note_reference": notes[0][2] if len(notes) == 1 else None,
        "cash_note_binding": notes[0][3] if len(notes) == 1 else None,
        "cash_note_date_header": notes[0][0].split("活期存款", 1)[0] if len(notes) == 1 else None,
        "scope": "DISCLOSED_ACCOUNTING_BALANCES_NOT_FREE_CASH_CAPACITY_OR_SCORE_METRICS",
        "unknowns_not_assigned": [
            "本模块未评估保证金/结算备付金及其可用性", "期货保证金义务", "合格政府债券", "现金合理目标", "策略容量边界",
        ],
    }
