"""本地实验的确定性事实账本；计算、列绑定和区间名称不由模型自由改写。

原文事实保留原始偏移、来源哈希、单位与条件。仅对明确语法或已有结构字段计算，
不猜纯文本表格的空列，不把规则提取当成所有公告语义已验证，也不接入正式服务。
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from app.services.direction_1d_protocol import calendar, digest

from scripts.local_fund_repair_run import read, save, sha
from scripts.local_fund_semantics import compact

SCALES = {
    "元": ("YUAN", 1),
    "万元": ("YUAN", 10000),
    "亿元": ("YUAN", 100000000),
    "股": ("SHARE", 1),
    "万股": ("SHARE", 10000),
    "亿股": ("SHARE", 100000000),
    "美元": ("USD", 1),
    "万美元": ("USD", 10000),
    "亿美元": ("USD", 100000000),
    "%": ("PERCENT", 1),
}


def literal_matches(body: str, pattern: str) -> list[dict]:
    """只忽略排版空白，结果回指原始正文；不删标点或跨字段补字。"""
    positions = [i for i, char in enumerate(body) if not char.isspace()]
    text = compact(body)
    rows = []
    for match in re.finditer(pattern, text):
        start, end = positions[match.start()], positions[match.end() - 1] + 1
        rows.append({"start": start, "end": end, "raw": body[start:end], "groups": match.groupdict()})
    return rows


def quantity_facts(body: str) -> list[dict]:
    """准确换算明确出现的数值与单位，不擅自确定主体、年度或表格所属列。

    数值上下文仍需关系核对；规范值只服务程序比较，不把原文替换成规范单位。
    带斜杠的每股价格不按总金额处理，括号内数字也不补到没有数值的合计列。
    """
    units = "|".join(sorted(SCALES, key=len, reverse=True))
    number = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    pattern = rf"(?<![\d.,])(?P<value>{number})[】]?\s*(?P<unit>{units})(?![/股])"
    result = []
    for row in literal_matches(body, pattern):
        value = Decimal(row["groups"]["value"].replace(",", ""))
        unit = row["groups"]["unit"]
        dimension, scale = SCALES[unit]
        # 单位“元”只决定十进位换算，不自行指定币种；美元则在原单位中明确。
        prefix = compact(body[max(0, row["start"] - 12) : row["start"]])
        currency = "USD" if dimension == "USD" else None
        if dimension == "YUAN":
            currency = "CNY" if re.search(r"人民币(?:约|共计|合计)?[【]?$", prefix) else "NOT_BOUND"
        result.append(
            {
                **row,
                "value": str(value),
                "unit": unit,
                "dimension": dimension,
                "currency": currency,
                "base_value": str(value * scale),
                "context": body[max(0, row["start"] - 60) : row["end"] + 60],
                "subject_period_relation": "REQUIRES_SOURCE_BINDING",
            }
        )
    return result


def annual_average_bases(body: str) -> list[dict]:
    """保留年均基数的连续原文；不从缺失的各年利润生成最低分红金额。"""
    return literal_matches(body, r"(?P<basis>(?:该)?三年实现的年均可分配利润)")


def threshold_tables(body: str) -> list[dict]:
    """仅解析连续明确的Am/An表头与“归属期-年-两数同单位”行，不猜列序。

    表头先触发值或先目标值均按原顺序绑定。预留批次的重复年度保留为独立行，
    保留前文范围而不合并；不能确认行/列或公式时不推导实际归属结果。
    """
    pattern = r"(?P<first>目标值|触发值)[（(](?P<s1>Am|An)[）)](?P<second>目标值|触发值)[（(](?P<s2>Am|An)[）)]"
    headers = literal_matches(body, pattern)
    result = []
    row_pattern = (
        r"(?P<installment>第[一二三四五六七八九十]+个归属期)(?P<year>20\d{2})年"
        r"(?P<v1>\d+(?:\.\d+)?)(?P<u1>%|亿元|万元)"
        r"(?P<v2>\d+(?:\.\d+)?)(?P<u2>%|亿元|万元)"
    )
    for index, header in enumerate(headers):
        h = header["groups"]
        if (
            h["first"] == h["second"]
            or h["s1"] != {"目标值": "Am", "触发值": "An"}[h["first"]]
            or h["s2"] != {"目标值": "Am", "触发值": "An"}[h["second"]]
        ):
            continue
        stop = headers[index + 1]["start"] if index + 1 < len(headers) else len(body)
        tail = body[header["end"] : stop]
        section_end = re.search(r"考核指标|各归属期对应", tail)
        if section_end:
            tail = tail[: section_end.start()]
        rows = []
        for row in literal_matches(tail, row_pattern):
            row["start"] += header["end"]
            row["end"] += header["end"]
            g = row["groups"]
            values = {h["first"]: g["v1"], h["second"]: g["v2"]}
            verified = g["u1"] == g["u2"] and Decimal(values["触发值"]) <= Decimal(values["目标值"])
            rows.append(
                {
                    **row,
                    "year": g["year"],
                    "unit": g["u1"],
                    "target": values["目标值"],
                    "trigger": values["触发值"],
                    "status": "BOUND_LITERAL_ROW" if verified else "REQUIRES_REVIEW",
                    "preceding_scope_context": body[max(header["start"], row["start"] - 130) : row["start"]],
                }
            )
        formulas = literal_matches(body[header["start"] : stop], r"(?P<condition>An≤A<Am|Am＞A≥An)X=A/Am")
        for formula in formulas:
            formula["start"] += header["start"]
            formula["end"] += header["start"]
        result.append(
            {
                "header": header,
                "rows": rows,
                "partial_vesting_formulas": formulas,
                "metric_basis_context": body[max(0, header["start"] - 150) : header["start"]],
                "semantics": "ONLY_LITERAL_HEADER_ROW_AND_FORMULA_BINDING",
            }
        )
    return result


def vesting_ratio(actual: Decimal, target: Decimal, trigger: Decimal) -> Decimal:
    """明确给出An≤A<Am、X=A/Am协议时使用；不自动适用于其他激励条款。

    三个数须同一指标、同一年度、同一单位。违反正目标或阈值关系直接失败。
    返回0至1比例；恰到触发值可以部分归属，只有恰到目标值才达到100%。
    """
    if not all(v.is_finite() for v in (actual, target, trigger)) or not 0 <= trigger <= target or target <= 0:
        raise ValueError("INVALID_VESTING_THRESHOLDS")
    if actual < trigger:
        return Decimal(0)
    return Decimal(1) if actual >= target else actual / target


def table_facts(units: list[dict]) -> list[dict]:
    """依据真实PDF单列表头和单元格输出账本；空格及合并格不借用别列数字。

    多个表头给出同一列不同含义时降为歧义。合计行未列投入总额就明确无可采用
    数值，即使分项可以相加，也不将自行计算说成原表披露合计。
    """
    catalog = {u["id"]: u for u in units}
    result = []
    for row in units:
        if row.get("kind") != "table_row" or row["id"] in row.get("header_rows", []):
            continue
        candidates = {}
        for ref in row.get("header_rows", []):
            for cell in catalog.get(ref, {}).get("cells", []):
                if cell["column_start"] != cell["column_end"]:
                    continue
                header = compact(cell["text"])
                match = re.fullmatch(r"(?P<label>.+?)[（(](?P<unit>亿元|万元|元|亿股|万股|股)[）)]", header)
                if match:
                    candidates.setdefault(cell["column_start"], []).append((ref, match.groupdict()))
        for column, headers in candidates.items():
            labels = {(h["label"], h["unit"]) for _, h in headers}
            cells = [c for c in row["cells"] if c["column_start"] == c["column_end"] == column]
            raw = cells[0]["text"] if len(cells) == 1 else None
            numeric = compact(raw).replace(",", "") if raw is not None else ""
            if len(labels) != 1:
                status = "AMBIGUOUS_HEADER"
                value = None
            else:
                status = "SOURCE_NUMERIC" if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric) else "NO_NUMERIC_SOURCE"
                value = numeric if status == "SOURCE_NUMERIC" else None
            result.append(
                {
                    "row_id": row["id"],
                    "page": row.get("page"),
                    "column": column,
                    "headers": headers,
                    "raw": raw,
                    "value": value,
                    "status": status,
                    "merged_cells": [
                        c
                        for c in row["cells"]
                        if c["column_start"] <= column <= c["column_end"] and c["column_start"] != c["column_end"]
                    ],
                }
            )
    return result


def total_table_text(facts: list[dict]) -> str:
    """直接展示原表合计行，合并格未列金额时明确说明；不自行补算。"""
    total_ids = {r["row_id"] for r in facts if any(compact(c["text"]) == "合计" for c in r["merged_cells"])}
    lines = []
    for row in facts:
        if row["row_id"] not in total_ids or len(row["headers"]) != 1:
            continue
        header = row["headers"][0][1]
        value = row["raw"] + header["unit"] if row["status"] == "SOURCE_NUMERIC" else "原合计行未列出可采用数值"
        lines.append(header["label"] + "：" + value + f"（第{row['page']}页，{row['row_id']}第{row['column']}列）")
    return "\n".join(lines)


def metrics_from_frozen_navs(
    rows: list[dict], start: str, end: str, as_of: str, sessions: tuple[str, ...] | None = None
) -> dict:
    """从冻结复权净值独立重算；重复、非正值或截止后记录不能静默删除。

    收益采用末收盘/首收盘-1，回撤采用区间峰值至后续低点，20日窗口可重叠。
    不用未复权净值替代，不推导次日胜率。交易日完整性还须与原快照计数核对；
    本函数不把已存在日期列表冒充独立交易日历。
    """
    cutoff = datetime.fromisoformat(as_of)
    selected = []
    for row in rows:
        available = datetime.fromisoformat(row["updated_at"])
        if row["nav_date"] > str(cutoff.date()) or row["ann_date"] > str(cutoff.date()) or available > cutoff:
            raise ValueError("NAV_AFTER_FROZEN_CUTOFF")
        if start <= row["nav_date"] <= end:
            selected.append(row)
    selected.sort(key=lambda r: r["nav_date"])
    if len(selected) < 2 or len({r["nav_date"] for r in selected}) != len(selected):
        raise ValueError("NAV_MISSING_OR_DUPLICATE")
    excluded = []
    if sessions is not None:
        expected = [day for day in sessions if start <= day <= end]
        actual = {r["nav_date"]: r for r in selected}
        if len(expected) < 2 or any(day not in actual for day in expected):
            raise ValueError("TRADING_SESSION_NAV_MISSING")
        excluded = [r["nav_date"] for r in selected if r["nav_date"] not in expected]
        selected = [actual[day] for day in expected]
    values = [Decimal(r["adjusted_nav"]) for r in selected]
    if any(not v.is_finite() or v <= 0 for v in values):
        raise ValueError("ADJUSTED_NAV_INVALID")
    peak, drawdown = values[0], Decimal(0)
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
    windows = [values[i] / values[i - 20] - 1 for i in range(20, len(values))]

    def pct(value):
        return str((value * 100).quantize(Decimal("0.0001")))

    return {
        "start": selected[0]["nav_date"],
        "end": selected[-1]["nav_date"],
        "observations": len(values),
        "return_pct": pct(values[-1] / values[0] - 1),
        "max_drawdown_pct": pct(drawdown),
        "positive_20d_pct": pct(Decimal(sum(v > 0 for v in windows)) / len(windows)) if windows else None,
        "rolling_windows": len(windows),
        "short_sample": len(values) - 1 < 252,
        "calendar_completeness": "VERIFIED_FROZEN_CALENDAR"
        if sessions is not None
        else "CHECK_AGAINST_FROZEN_SOURCE_COUNTS",
        "excluded_non_session_dates": excluded,
    }


def manager_facts(source: dict, payload: dict, sessions: tuple[str, ...], calendar_version: str) -> dict:
    """经理数据仅使用10月10日独立快照；所有统计项带固定区间与原来源哈希。

    全期和独立期逐项重算后与保存的结构指标比较。共同管理子区间是本轮程序
    新算出的统计，明确标为派生数据，不能说此前模型已得到这些区间的原始指标。
    """
    context = payload["manager_context"]
    if payload["window"]["calendar_version"] != calendar_version:
        raise ValueError("FROZEN_CALENDAR_VERSION_CHANGED")
    if context["as_of"] != source["as_of"] or not context["current_verified"]:
        raise ValueError("MANAGER_SNAPSHOT_SCOPE_MISMATCH")
    if context["managers"] != source["facts"]["manager:performance"]["metrics"]:
        raise ValueError("MANAGER_STRUCTURED_SOURCE_DISAGREEMENT")
    body = source["facts"]["manager:change"]["text"]
    transfer = literal_matches(body, r"转任本公司其他工作岗位的说明(?P<value>.*?)是否已按规定")
    reasons = literal_matches(body, r"离任原因(?P<value>.*?)离任日期")
    if len(transfer) != 1 or len(reasons) != 1:
        raise ValueError("MANAGER_DISCLOSURE_FIELD_NOT_UNIQUE")
    status = "NOT_DISCLOSED" if transfer[0]["groups"]["value"] in ("-", "—", "–") else "REQUIRES_REVIEW"
    metrics = []
    for manager in context["managers"]:
        for field, label in (("tenure", "参与管理全期（包含共同与独立管理）"), ("solo", "独立管理区间")):
            old = manager[field]
            if not old or old.get("status") != "AVAILABLE":
                continue
            new = metrics_from_frozen_navs(context["navs"], old["start"], old["end"], source["as_of"], sessions)
            for key in ("start", "end", "observations", "rolling_windows"):
                if new[key] != old[key]:
                    raise ValueError("MANAGER_FROZEN_COUNT_OR_PERIOD_MISMATCH:" + key)
            for key in ("return_pct", "max_drawdown_pct", "positive_20d_pct"):
                if Decimal(str(new[key])) != Decimal(str(old[key])):
                    raise ValueError("MANAGER_FROZEN_METRIC_MISMATCH:" + key)
            metrics.append(
                {
                    "manager": manager["name"],
                    "source_field": field,
                    "scope": label,
                    "metric_origin": "INDEPENDENT_RECALCULATION",
                    **new,
                }
            )
        if manager.get("solo_start") and manager.get("co_managed"):
            boundary = (datetime.fromisoformat(manager["solo_start"]) - timedelta(days=1)).date()
            selected = [r for r in context["navs"] if manager["begin_date"] <= r["nav_date"] <= str(boundary)]
            if any(
                not any(
                    a["manager_name"] != manager["name"]
                    and a["begin_date"] <= r["nav_date"]
                    and (not a.get("end_date") or r["nav_date"] <= a["end_date"])
                    for a in context["assignments"]
                )
                for r in selected
            ):
                raise ValueError("CO_MANAGED_INTERVAL_HAS_UNVERIFIED_GAPS")
            new = metrics_from_frozen_navs(
                context["navs"], manager["begin_date"], str(boundary), source["as_of"], sessions
            )
            metrics.append(
                {
                    "manager": manager["name"],
                    "source_field": "co_managed_derived",
                    "scope": "共同管理子区间",
                    "metric_origin": "NEW_DERIVED_SUBINTERVAL",
                    **new,
                }
            )
    return {
        "as_of": source["as_of"],
        "source_hash": source["source_hash"],
        "transfer_status": status,
        "transfer_source": transfer[0],
        "reason_source": reasons[0],
        "metrics": metrics,
        "mixed_into_old_prediction": False,
        "calendar_version": calendar_version,
        "limits": [
            "缺少可比基准和个人贡献证据，不能确认超额能力",
            "历史正收益窗口占比不是预测胜率",
            "按原快照同版本交易日历核对，非交易日记录明确排除并保留日期",
        ],
    }


def manager_text(facts: dict) -> str:
    """直接从已核对的类型字段生成说明，不接受模型提供的区间名或数值。"""
    transfer = "转任其他岗位情况未披露。" if facts["transfer_status"] == "NOT_DISCLOSED" else "转任字段需复核。"
    lines = ["公开离任原因：" + facts["reason_source"]["groups"]["value"] + "。" + transfer]
    for row in facts["metrics"]:
        lines.append(
            f"{row['manager']}{row['scope']}（{row['start']}至{row['end']}，{row['observations']}个观测）："
            f"区间收益{row['return_pct']}%，最大回撤{row['max_drawdown_pct']}%，"
            f"历史20日正收益窗口占比{row['positive_20d_pct']}%（{row['rolling_windows']}个重叠窗口）。"
        )
    lines.extend(facts["limits"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = read(args.repair / "snapshot.json")
    cutoff = datetime.fromisoformat(snapshot["as_of"])
    if any(datetime.fromisoformat(doc["available_at"]) > cutoff for doc in snapshot["documents"]):
        raise ValueError("DOCUMENT_AFTER_FROZEN_CUTOFF")
    docs = [
        {
            "document_id": doc["id"],
            "source_hash": doc["source_hash"],
            "body_scope": doc["body_scope"],
            "body_truncated": doc["body_truncated"],
            "quantities": quantity_facts(doc["body"]),
            "annual_average_bases": annual_average_bases(doc["body"]),
            "threshold_tables": threshold_tables(doc["body"]),
        }
        for doc in snapshot["documents"]
    ]
    source = read(args.repair / "manager-source.json")
    manager_path = Path(source["source_path"])
    manager_input = read(manager_path)
    # 此字段是项目canonical payload哈希，不是外层JSON文件字节哈希。
    if manager_input["hash"] != source["source_hash"] or digest(manager_input["payload"]) != source["source_hash"]:
        raise ValueError("MANAGER_ORIGINAL_SOURCE_HASH_CHANGED")
    days, version = calendar()
    # 日历是原快照引用的同一哈希版本；只使用经理截止日及以前的交易日期。
    sessions = tuple(str(day) for day in days if str(day) <= source["as_of"][:10])
    manager = manager_facts(source, manager_input["payload"], sessions, version)
    task = next(t for t in read(args.benchmark / "tasks.json") if t["id"] == "pdf-table")
    value = {
        "version": "LOCAL_VERIFIED_FACT_LEDGER_V3",
        "as_of": snapshot["as_of"],
        "documents": docs,
        "pdf_table_cells": table_facts(task["units"]),
        "pdf_total_deterministic_text": total_table_text(table_facts(task["units"])),
        "manager_separate": manager,
        "manager_deterministic_text": manager_text(manager),
        "input_sha256": sha(args.repair / "snapshot.json"),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "semantic_coverage": "LIMITED_EXPLICIT_NUMERIC_AND_TYPED_FIELDS",
        "model_calls": 0,
        "production_switched": False,
    }
    save(args.output, value)
    print(
        json.dumps(
            {
                "documents": len(docs),
                "quantities": sum(len(d["quantities"]) for d in docs),
                "table_cells": len(value["pdf_table_cells"]),
                "manager_metrics": len(manager["metrics"]),
                "model_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
