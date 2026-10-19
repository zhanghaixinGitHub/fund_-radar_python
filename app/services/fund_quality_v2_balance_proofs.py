"""严格账簿与会计守恒证明；未知横杠不默认作零，不修改正式V1读取器。

研究适配限定为人民币、当期/比较期两列、完整列示的资产负债表。明确
数值与由非负完整账目唯一识别的数值分别保存，原横杠单元格始终为None。
有剩余金额且多个分量未知、缺行、未知资产科目、差额或错期均不生成点账簿。
"""

import re
import unicodedata
from datetime import date
from decimal import Decimal, localcontext

from app.services.fund_quality_v2_accounting import _current_cny_header
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_restrictions import restriction_page_header_view

VERSION = "QUALITY_BALANCE_PROOFS_20261017_6"
CORE = ("货币资金", "结算备付金", "存出保证金", "交易性金融资产", "衍生金融资产",
        "买入返售金融资产", "应收清算款", "应收股利", "应收申购款", "递延所得税资产", "其他资产")
OPTIONAL = ("债权投资", "其他债权投资", "其他权益工具投资")
CHILDREN = ("其中:股票投资", "基金投资", "债券投资", "资产支持证券投资", "贵金属投资", "其他投资")
DEBT_CHILDREN = ("其中:债券投资", "资产支持证券投资", "其他投资")
MONEY = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
SLOT = rf"(?:{MONEY}|[-—－])"


def identify_nonnegative_parts(total: str, parts: dict[str, str | None]) -> dict:
    """用完整、互斥非负分量与明示合计求唯一值，精确到原文的分，不容差补零。

    None表示实际打印横杠的单元格，缺行应由调用方提前拒绝。剩余为0时
    所有未知分量只能为0；剩余为正且仅一项未知时才唯一可解。原单元格
    不回填，推导值及理由另列。两个未知分量分摊同一正余额时保持区间。
    """
    with localcontext() as context:
        context.prec = 50
        parent = Decimal(total)
        original = {key: None if value is None else Decimal(value) for key, value in parts.items()}
        if not original or parent < 0 or any(value is not None and value < 0 for value in original.values()):
            raise ValueError("QUALITY_BALANCE_NONNEGATIVE_PARTS_DOMAIN")
        residual = parent - sum((value for value in original.values() if value is not None), Decimal(0))
        unknown = sorted(key for key, value in original.items() if value is None)
        if residual < 0 or not unknown and residual != 0:
            raise ValueError("QUALITY_BALANCE_EXACT_EQUATION_CONFLICT")
        derived = {}
        if unknown and (residual == 0 or len(unknown) == 1):
            derived = {key: str(residual) for key in unknown}
        identified = len(derived) == len(unknown)
        return {"state": "POINT_IDENTIFIED" if identified else "MULTIPLE_NONNEGATIVE_SOLUTIONS",
                "explicit_total": str(parent), "explicit_parts": parts, "derived_parts": derived,
                "remaining_amount": str(residual), "unknown_parts": unknown,
                "derivation": "ZERO_UNIQUELY_IDENTIFIED_BY_COMPLETE_NONNEGATIVE_SUM"
                if unknown and residual == 0 else "SINGLE_UNKNOWN_BY_EXACT_SUM"
                if derived else "EXPLICIT_EXACT_SUM" if not unknown else "UNRESOLVED_POSITIVE_RESIDUAL",
                "point_values": {key: str(value) if value is not None else derived.get(key)
                                 for key, value in original.items()},
                "rounding_tolerance_used": "0"}


def strict_balance_values(pages: list[str], *, observed_on: str) -> dict:
    """逐行读取资产表并构建完整性与零值证明；本研究函数不提供评分。

    调用方验证原件身份和字节摘要；此处核经济期、当期列、币种、所有必需
    行及资产/金融资产分解。额外科目须显式适配，不能偷偷放入零或残差桶。
    所有行都保留原文与比较列，横杠仅作为未知输入交给精确会计约束。
    这里只提供文本解析事实，不伪造原字节哈希；指标包必须由上层原件目录绑定。
    """
    observed = date.fromisoformat(observed_on)
    if report_period("\n".join(pages))[1] != observed:
        raise ValueError("QUALITY_BALANCE_PROOF_SOURCE_OR_PERIOD")
    normalized = [unicodedata.normalize("NFKC", page) for page in pages]
    text, header_trace = restriction_page_header_view(normalized)
    blocks = []
    for header in re.finditer(r"(?m)^\s*([567])\.1\s*资产负债表[ \t]*$", text):
        tail = text[header.end():]
        stop = re.search(rf"(?m)^\s*{header[1]}\.2\s*利润表", tail)
        if stop and _current_cny_header(tail[:stop.start()], observed):
            blocks.append(tail[:stop.start()])
    if len(blocks) != 1:
        raise ValueError("QUALITY_BALANCE_PROOF_CURRENT_CNY_TWO_COLUMN_BODY_UNRESOLVED")
    body = blocks[0]
    rows = {}

    def read_row(label, *, start=0, stop=None, output_key=None):
        # 仅允许已知科目名按原字顺序折行，不跨行拼接金额或猜缺失单元格。
        pattern = r"[ \t\r\n]*".join(map(re.escape, label))
        expression = re.compile(rf"(?m)^[ \t]*{pattern}[ \t]+(?:[567]\.4\.\d+(?:\.\d+)*[ \t]+)?"
                                rf"({SLOT})[ \t]+({SLOT})[ \t]*$")
        matches = list(expression.finditer(body, start, len(body) if stop is None else stop))
        layout_prefix, layout_suffix = label, ""
        if not matches:
            wrapped = []
            # 原PDF可能按“资产支持证 / 两个金额 / 券投资”抽取。仅把
            # 已知科目名的完整前后缀绑定到原位的两个金额，后缀必须独占
            # 紧接的下一非空行；金额及列序绝不拼接、移动或猜测。
            for split in range(1, len(label)):
                prefix, suffix = label[:split], label[split:]
                left = r"[ \t\r\n]*".join(map(re.escape, prefix))
                right = r"[ \t]*".join(map(re.escape, suffix))
                expression = re.compile(rf"(?m)^[ \t]*{left}[ \t]+({SLOT})[ \t]+({SLOT})[ \t]*"
                                        rf"\r?\n(?:[ \t]*\r?\n)*[ \t]*{right}[ \t]*$")
                wrapped.extend((match, prefix, suffix) for match in expression.finditer(
                    body, start, len(body) if stop is None else stop))
            if len(wrapped) == 1:
                match, layout_prefix, layout_suffix = wrapped[0]
                matches = [match]
        if len(matches) != 1:
            raise ValueError("QUALITY_BALANCE_PROOF_ROW_MISSING_OR_AMBIGUOUS:" + label)
        match = matches[0]
        value = None if match[1] in {"-", "—", "－"} else Decimal(match[1].replace(",", ""))
        if value is not None and value < 0:
            raise ValueError("QUALITY_BALANCE_PROOF_NEGATIVE_VALUE:" + label)
        row = {"value": str(value) if value is not None else None, "raw_current_cell": match[1],
               "comparison_cell_not_used": match[2], "original_row": match[0],
               "label_prefix_before_cells": layout_prefix, "label_suffix_after_cells": layout_suffix,
               "row_start": match.start(), "row_end": match.end(),
               "state": "EXPLICIT_AMOUNT" if value is not None else "PRINTED_DASH_UNASSIGNED"}
        rows[output_key or label] = row
        return row

    for label in (*CORE, "资产总计", "负债合计", "净资产合计"):
        read_row(label)
    for label in OPTIONAL:
        if re.search(rf"(?m)^[ \t]*{re.escape(label)}[ \t]+", body):
            read_row(label)
    # 同一表内，交易性金融资产和债权投资各有“资产支持证券投资”等子项。
    # 以实际父行及下一个一级资产行限定子表，不取第一次/最后一次来猜归属。
    roots = [*CORE, *(label for label in OPTIONAL if label in rows), "资产总计"]

    def parent_range(label):
        start = rows[label]["row_end"]
        ends = [rows[key]["row_start"] for key in roots if rows[key]["row_start"] >= start]
        if not ends:
            raise ValueError("QUALITY_BALANCE_PROOF_PARENT_SCOPE_UNRESOLVED:" + label)
        return start, min(ends)

    trading_start, trading_stop = parent_range("交易性金融资产")
    for label in CHILDREN:
        read_row(label, start=trading_start, stop=trading_stop)
    if "债权投资" in rows:
        debt_start, debt_stop = parent_range("债权投资")
        for label in DEBT_CHILDREN:
            read_row(label, start=debt_start, stop=debt_stop, output_key="债权投资/" + label)
    # 资产行内的未知额外科目，即使只印横杠，也会改变完整性，必须拒收。
    top = re.search(r"(?m)^[ \t]*资[ \t]*产[ \t]*:[ \t]*$", body)
    if not top:
        raise ValueError("QUALITY_BALANCE_PROOF_ASSET_SECTION_START_UNRESOLVED")
    asset_region = body[top.end():rows["资产总计"]["row_start"]]
    for line in re.finditer(r"(?m)^[^\r\n]+", asset_region):
        if re.search(rf"[ \t]+{SLOT}[ \t]+{SLOT}[ \t]*$", line[0]):
            position = top.end() + line.start()
            if not any(item["row_start"] <= position < item["row_end"] for item in rows.values()):
                raise ValueError("QUALITY_BALANCE_PROOF_UNREVIEWED_ASSET_ROW:" + line[0].strip())
    if any(rows[label]["value"] is None for label in ("资产总计", "负债合计", "净资产合计", "交易性金融资产")):
        raise ValueError("QUALITY_BALANCE_PROOF_EXPLICIT_TOTAL_REQUIRED")
    if Decimal(rows["净资产合计"]["value"]) <= 0:
        raise ValueError("QUALITY_BALANCE_PROOF_NONPOSITIVE_NET_ASSETS")
    if Decimal(rows["资产总计"]["value"]) != sum(Decimal(rows[label]["value"]) for label in ("负债合计", "净资产合计")):
        raise ValueError("QUALITY_BALANCE_PROOF_NET_ASSET_EQUATION_CONFLICT")
    asset_labels = [*CORE, *(label for label in OPTIONAL if label in rows)]
    asset_proof = identify_nonnegative_parts(rows["资产总计"]["value"],
                                             {label: rows[label]["value"] for label in asset_labels})
    trading_proof = identify_nonnegative_parts(rows["交易性金融资产"]["value"],
                                              {label: rows[label]["value"] for label in CHILDREN})
    identified = asset_proof["state"] == trading_proof["state"] == "POINT_IDENTIFIED"
    debt_proof = None
    if "债权投资" in rows:
        parent = asset_proof["point_values"]["债权投资"]
        if parent is not None:
            debt_proof = identify_nonnegative_parts(parent, {
                label: rows["债权投资/" + label]["value"] for label in DEBT_CHILDREN
            })
            identified = identified and debt_proof["state"] == "POINT_IDENTIFIED"
        else:
            identified = False
    book = None
    if identified:
        book = {"assets": asset_proof["point_values"], "total_assets": rows["资产总计"]["value"],
                "liabilities": rows["负债合计"]["value"], "net_assets": rows["净资产合计"]["value"],
                "stock_amount": trading_proof["point_values"]["其中:股票投资"],
                "fund_amount": trading_proof["point_values"]["基金投资"],
                "bond_amount": trading_proof["point_values"]["债券投资"]}
    return {"version": VERSION, "observed_on": observed_on,
            "currency": "CNY", "state": "POINT_IDENTIFIED" if identified else "PARTIAL_BALANCE_NO_POINT_BOOK",
            "original_rows": rows, "asset_equation": asset_proof, "trading_equation": trading_proof,
            "debt_equation": debt_proof,
            "point_book": book, "header_trace": header_trace,
            "scope": "EXACT_COMPLETE_NONNEGATIVE_ACCOUNTING_PROOF_NOT_LIQUIDITY_OR_QUALITY_SCORE",
            "score_or_grade_created": False}


def strict_balance_proof(pages: list[str], *, observed_on: str, source_hash: str) -> dict:
    """为上层已核字节的原件绑定严格账簿；传入摘要本身不能替代调用方核文件。"""
    if not re.fullmatch(r"[a-f0-9]{64}", source_hash):
        raise ValueError("QUALITY_BALANCE_PROOF_SOURCE_OR_PERIOD")
    return strict_balance_values(pages, observed_on=observed_on) | {"source_hashes": [source_hash]}
