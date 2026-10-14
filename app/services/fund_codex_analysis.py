"""002112 聊天分析的本机导入：完整输入留存、原文引用核对、追加结果、不调用模型。"""

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.schemas.fund_information_analysis import PROTOCOL, validate_analysis, validate_events
from app.services.direction_1d_protocol import canonical, digest, window
from app.services.fund_exposure_common import read, save
from app.services.fund_information_contracts import event_facts, group_events, reference_catalog, time_context
from app.services.fund_information_snapshot import prepare

SOURCE = "CODEX_ASSISTED_REVIEW"


def freeze_packet(directory: Path) -> dict:
    """冻结数据库时钟对应的完整公共输入；分段仅方便读取，正文不截断、不改写。

    本函数不采集数据、不调用供应商、不生成预测。目录必须全新，防止覆盖已有
    输入和首次结果。每段记录原文字符范围，审阅结果不得虚构已读全文的覆盖范围。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = prepare(repo.clock())
    if data["window"]["status"] != "OPEN":
        raise ValueError("MISSED_DEADLINE")
    packet = {"format": "CODEX_002112_INPUT_V1", "input_hash": digest(data), "input": data}
    save(directory / "input.json", packet)
    documents = directory / "documents"
    documents.mkdir()
    index = []
    for number, document in enumerate(data["documents"], 1):
        body, chunks = document["body"], []
        for part, start in enumerate(range(0, len(body), 12000), 1):
            end = min(start + 12000, len(body))
            name = f"{number:03d}-{part:03d}.txt"
            (documents / name).write_text(body[start:end], encoding="utf-8")
            chunks.append({"file": name, "start": start, "end": end})
        index.append(
            {
                **{key: value for key, value in document.items() if key != "body"},
                "characters": len(body),
                "chunks": chunks,
            }
        )
    save(directory / "documents.json", index)
    return {
        "target_date": data["window"]["target_nav_date"],
        "input_hash": packet["input_hash"],
        "documents": len(index),
        "directory": str(directory.resolve()),
    }


def checked_packet(path: Path) -> dict:
    """核对文件摘要和完整输入身份；原文全文及资料清单始终保留在冻结包中。"""
    packet = read(Path(path))
    if packet.get("format") != "CODEX_002112_INPUT_V1" or digest(packet.get("input")) != packet.get("input_hash"):
        raise ValueError("MANUAL_INPUT_HASH_MISMATCH")
    data = packet["input"]
    if (
        data.get("fund_code") != "002112"
        or data["window"]["target_nav_date"] != window(repo.clock())["target_nav_date"]
    ):
        raise ValueError("WINDOW_CHANGED")
    return packet


def reviewed_events(data: dict, review: dict) -> tuple[list[dict], list[dict]]:
    """逐份核对覆盖范围和引用；未读或仅读片段必须单独记录，不能冒称解析全文。"""
    documents = {document["id"]: document for document in data["documents"]}
    rows = review.get("documents", [])
    if len(rows) != len(documents) or {row.get("id") for row in rows} != set(documents):
        raise ValueError("MANUAL_DOCUMENT_COVERAGE_REQUIRED")
    events, limitations = [], []
    for row in rows:
        document = documents[row["id"]]
        body, scope, ranges = document["body"], row.get("scope"), row.get("ranges", [])
        if scope not in {"FULL_TEXT", "EXCERPTS", "NOT_REVIEWED"}:
            raise ValueError("MANUAL_REVIEW_SCOPE_INVALID")
        if scope == "NOT_REVIEWED" and (ranges or row.get("events")):
            raise ValueError("MANUAL_UNREAD_DOCUMENT_HAS_EVENTS")
        previous = 0
        for span in ranges:
            if (
                set(span) != {"start", "end"}
                or type(span["start"]) is not int
                or type(span["end"]) is not int
                or not 0 <= span["start"] < span["end"] <= len(body)
                or span["start"] < previous
            ):
                raise ValueError("MANUAL_REVIEW_RANGE_INVALID")
            previous = span["end"]
        if scope == "FULL_TEXT":
            end = 0
            for span in ranges:
                if span["start"] != end:
                    raise ValueError("MANUAL_FULL_REVIEW_INCOMPLETE")
                end = span["end"]
            if end != len(body) or document.get("body_scope") != "FULL_TEXT":
                raise ValueError("MANUAL_FULL_REVIEW_INCOMPLETE")
        elif scope == "EXCERPTS" and not ranges:
            raise ValueError("MANUAL_REVIEW_RANGE_INVALID")
        if scope != "FULL_TEXT":
            limitations.append(
                {
                    "documentId": document["id"],
                    "title": document["title"],
                    "reasonCode": "MANUAL_EXCERPT_REVIEW" if scope == "EXCERPTS" else "MANUAL_NOT_REVIEWED",
                }
            )
        for item in row.get("events", []):
            for quote in item.get("quotes", []):
                start = body.find(quote)
                if start < 0 or not any(
                    span["start"] <= start and start + len(quote) <= span["end"] for span in ranges
                ):
                    raise ValueError("MANUAL_QUOTE_OUTSIDE_REVIEW")
        if row.get("events"):
            events.extend(validate_events({"items": [{"id": document["id"], "events": row["events"]}]}, [document]))
    return events, limitations


def catalog(data: dict, review: dict) -> tuple[dict, list[dict], list[dict]]:
    """合并程序事实与已核对事件；引用编号由原身份确定，不用摘要相似度替换原文。"""
    events, coverage = reviewed_events(data, review)
    facts = dict(data["facts"])
    # 只由冻结净值计算最近收盘变化；明确是历史观察，不用旧收益推定本期必涨。
    if len(data.get("nav", [])) >= 2:
        previous, latest = data["nav"][-2:]
        change = (Decimal(latest["unit_nav"]) / Decimal(previous["unit_nav"]) - 1) * 100
        facts["nav:latest_observed_change"] = {
            "category": "基金净值",
            "text": f"冻结净值：{previous['nav_date']}单位净值{previous['unit_nav']}，"
            f"{latest['nav_date']}单位净值{latest['unit_nav']}，区间变化{change:+.2f}%。"
            "这是已经公布的比较基准日变化，不是目标日实际结果或预期收益。",
            "source": None,
        }
    facts.update(event_facts(group_events(events), data["companies"]))
    return reference_catalog(facts), events, coverage


def import_review(packet_path: Path, review_path: Path) -> dict:
    """导入当前窗口判断并追加成功作业，供已有 Java 认证、关注范围及留档链路接收。

    此入口仅由本机任务执行；不接收账号、仓位或网页指定模型。先完整验证冻结来源、
    文档覆盖、引用、方向依据和限制，再存储。相同审阅重复执行复用原结果；不更新旧
    作业或供应商台账。Java 回执仍是页面可见及提前预测验收的最终凭据。
    """
    if get_settings().fund_002112_analysis_mode != "CODEX_MANUAL":
        raise ValueError("MANUAL_MODE_REQUIRED")
    packet = checked_packet(packet_path)
    data, review = packet["input"], read(Path(review_path))
    if review.get("input_hash") != packet["input_hash"] or review.get("generation_source") != SOURCE:
        raise ValueError("MANUAL_REVIEW_IDENTITY_INVALID")
    cutoff, now = datetime.fromisoformat(data["as_of"]), repo.clock()
    if cutoff > now or now >= datetime.fromisoformat(data["window"]["deadline_at"]):
        raise ValueError("MISSED_DEADLINE")
    # 重新按原冻结时刻读取已保存来源，阻止源文件或目录被替换后仍沿用原判断。
    if digest(prepare(cutoff)) != packet["input_hash"]:
        raise ValueError("MANUAL_FROZEN_SOURCE_CHANGED")
    facts, events, coverage = catalog(data, review)
    result = validate_analysis(review["analysis"], facts, time_context(data))
    if review.get("audit") != {"valid": True, "issues": []}:
        raise ValueError("MANUAL_REVIEW_FAILED")
    review_hash = digest(review)
    frozen = deepcopy(data)
    frozen["codex_review"] = deepcopy(review)

    def runner(*args, skipped_documents, **kwargs):
        skipped_documents.extend(coverage)
        partial = sum(row["reasonCode"] == "MANUAL_EXCERPT_REVIEW" for row in coverage)
        unread = len(coverage) - partial
        gaps = []
        if partial:
            gaps.append(f"本次有{partial}份资料只核对选定原文片段，未声称完整阅读全文；未读章节可能包含其他重要条件。")
        if unread:
            gaps.append(f"本次有{unread}份资料未完成审阅，未作为具体事件依据。")
        return result, facts, gaps

    from app.services.fund_information_analysis import _infer_current

    output = _infer_current(
        data["window"]["target_nav_date"],
        frozen_data=frozen,
        reviewed_events=events,
        review_runner=runner,
        generation_source=SOURCE,
        review_hash=review_hash,
    )
    key = f"CODEX_MANUAL:002112:{data['window']['target_nav_date']}:{review_hash}"
    payload = {
        "fund_code": "002112",
        "target_nav_date": data["window"]["target_nav_date"],
        "protocol": PROTOCOL,
        "generation_source": SOURCE,
        "review_hash": review_hash,
    }
    with get_engine().begin() as connection:
        connection.execute(
            text("""INSERT INTO direction_1d_job
            (job_id,task_key,kind,state,payload,result,finished_at,lease_until)
            VALUES(:id,:key,'FORECAST','SUCCEEDED',CAST(:payload AS jsonb),CAST(:result AS jsonb),
                   clock_timestamp(),clock_timestamp())
            ON CONFLICT(task_key) DO NOTHING"""),
            {"id": uuid4(), "key": key, "payload": canonical(payload), "result": canonical(output)},
        )
        job = (
            connection.execute(
                text("SELECT job_id,result FROM direction_1d_job WHERE task_key=:key AND kind='FORECAST'"), {"key": key}
            )
            .mappings()
            .one()
        )
        if job["result"] != output:
            raise ValueError("MANUAL_RESULT_IDEMPOTENCY_MISMATCH")
    return {
        "job_id": str(job["job_id"]),
        "target_date": payload["target_nav_date"],
        "content_hash": output["content_hash"],
        "review_hash": review_hash,
        "calls": 0,
    }


def ready_result(target: str) -> dict:
    """网页生成仅读取本期已导入判断；未导入时说明缺口，绝不回退调用付费供应商。"""
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text("""SELECT result FROM direction_1d_job WHERE kind='FORECAST' AND state='SUCCEEDED'
            AND payload->>'fund_code'='002112' AND payload->>'target_nav_date'=:target
            AND payload->>'generation_source'=:source ORDER BY created_at DESC,job_id DESC LIMIT 1"""),
                {"target": target, "source": SOURCE},
            )
            .mappings()
            .first()
        )
    if not row:
        raise ValueError("MANUAL_ANALYSIS_REQUIRED")
    import json

    value = json.loads(row["result"]["payload_json"])
    if (
        digest(value) != row["result"]["content_hash"]
        or value.get("generation_source") != SOURCE
        or value.get("target_nav_date") != target
        or value.get("fund_code") != "002112"
        or repo.clock() >= datetime.fromisoformat(value["deadline_at"])
    ):
        raise ValueError("MANUAL_RESULT_INVALID")
    return row["result"]
