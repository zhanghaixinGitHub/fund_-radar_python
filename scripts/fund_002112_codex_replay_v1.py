"""002112 聊天辅助判断的隔离历史回放；先存首次判断，全部完成后才评分。

仅复用历史包的公开时间和原文片段，不复用旧预测、旧方向或供应商回答。
原始包包含评分所需的历史净值，但分析视图仅展示基准日及以前的信息。
所有命令禁用网络；产物只能新增，正式数据库、运行配置及旧实验均不修改。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.deepseek_policy import PrivateDeepSeekSettings
from app.services.direction_1d_protocol import features, label

from scripts import fund_002112_analysis_replay_v1 as legacy
from scripts import fund_002112_event_comparison_data_v1 as data

OUT = data.PY / ".local-runs/codex-backtest-002112-20261014-v1"
ATTENTION = re.compile(r"业绩|季度报告|半年度报告|回购|增持|重组|收购|担保|募集|发行|减持|诉讼|处罚|立案|政策|出口")
ROUTINE = re.compile(r"制度|议事规则|法律意见|核查意见|审计报告|更正公告|董事会决议|监事会决议|会议资料")
POLICY = "CODEX_MANUAL_HISTORY_DIAGNOSTIC_V1"


def no_network(*args, **kwargs):
    """网络被拒绝，包括本机服务；回放绝不能触发供应商、采集或现用数据库。"""
    raise RuntimeError("REPLAY_NETWORK_FORBIDDEN")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """追加不可覆盖文件，保留首次成功、失败及其输入摘要。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ledger():
    """只读私人调用台账；不初始化表，不把旧调用误算成本轮新增。"""
    settings = get_settings()
    private = PrivateDeepSeekSettings()
    if settings.fund_002112_analysis_mode != "CODEX_MANUAL" or private.enabled or private.api_key.get_secret_value():
        raise ValueError("PRIVATE_PROVIDER_MUST_BE_DISABLED")
    path = Path(settings.fund_parse_cache_path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        count, amount = connection.execute(
            "SELECT count(*),coalesce(sum(coalesce(estimated_peak_cny,reserved_cny)),0) FROM private_deepseek_call"
        ).fetchone()
    return {"calls": count, "estimated_cny": amount}


def temporal_guard(value):
    """显式检查时间及答案字段；不能依赖可被关闭的 assert 保护输入。"""
    cutoff = data.moment(value["as_of"])
    base = value["window"]["base_nav_date"]
    target = value["window"]["target_nav_date"]
    if not base < target or value["latest_nav_date"] != base:
        raise ValueError("REPLAY_DATE_MISMATCH")
    if any(row["nav_date"] > base for row in value["nav"]):
        raise ValueError("REPLAY_FUTURE_NAV")
    proofs = [value["report"]["available_at"], *value["time_proof"].values()]
    for stamp in proofs:
        if isinstance(stamp, str) and "T" in stamp and stamp[0].isdigit() and data.moment(stamp) > cutoff:
            raise ValueError("REPLAY_LATE_SOURCE")
    if any(company["quote"] and company["quote"]["date"] > base for company in value["companies"]):
        raise ValueError("REPLAY_FUTURE_QUOTE")
    if any(item["date"] > base for item in value["market"].values()):
        raise ValueError("REPLAY_FUTURE_MARKET")
    for document in value["documents"]:
        if document["date"] > cutoff.date().isoformat() or data.moment(document["available_at"]) > cutoff:
            raise ValueError("REPLAY_FUTURE_DOCUMENT")
        if document.get("body_truncated"):
            raise ValueError("REPLAY_SOURCE_EXCERPT_TRUNCATED")
    forbidden = {"actual", "actual_direction", "label", "target_unit_nav", "predicted", "y", "mature_at"}

    def visit(item):
        if isinstance(item, dict):
            if forbidden.intersection(item):
                raise ValueError("REPLAY_ANSWER_FIELD")
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)


def attention(value):
    """固定阅读清单只按题目、已披露仓位和日期选取，绝不依据目标日涨跌。

    同一天最多展示12份相关原文片段；全文和未选资料仍在输入包留存。
    这一阅读上限以及历史包的缺项必须随成绩披露，不能声称完整复制当前资料量。
    """
    weights = {company["code"]: company["weight_pct"] for company in value["companies"]}
    candidates = [
        doc for doc in value["documents"] if ATTENTION.search(doc["title"]) and not ROUTINE.search(doc["title"])
    ]
    candidates.sort(
        key=lambda doc: (max((weights.get(code, 0) for code in doc["codes"]), default=0), doc["date"], doc["id"]),
        reverse=True,
    )
    return candidates[:12]


def prepare():
    """固定历史包末尾40个交易日，全部候选日期留档；缺资料日期不从计划中删除。"""
    if (OUT / "protocol.json").exists():
        return checked()
    source = data.load_sources()
    targets = [day for day in source["sessions"] if day <= "2026-09-29"][-40:]
    protected = [data.SOURCE, data.MATERIAL, Path(__file__), Path(legacy.__file__), Path(data.__file__)]
    protocol = {
        "method": POLICY,
        "fund": "002112",
        "created_at": datetime.now(data.ZONE).isoformat(),
        "cutoff": "北京时间每个目标交易日10:00，不使用当日收盘行情或净值",
        "selection": "旧历史原始包覆盖末尾40个交易日，仅按日历选择，不按答案或成败挑样本",
        "targets": [],
        "protected_hashes": {str(path): sha(path) for path in protected},
        "private_ledger_before": ledger(),
        "provider_calls_allowed": 0,
        "method_description": (
            "宿主聊天逐日权衡历史净值、旧披露持仓行情、市场和已读原文事件，"
            "保存方向、反证、变化条件及引用；无规则模型拟合和供应商调用"
        ),
        "attention_policy": (
            "固定题目筛选、排除例行制度/法律等文件，按披露仓位、发布日期排序，最多12份；"
            "重复原文首次读后引用，不用公告条数作为方向权重"
        ),
        "limits": [
            "历史公开时间保守重建，并非历史首见存档；无修订记录的股票/指数收盘按基日18点公开假设",
            "历史原文仅为已核验片段，不是全部PDF；经理、部分公司财务及实时仓位缺失",
            "此前历史包已用于其他研究；宿主模型已有知识与本聊天后续事件记忆无法完全排除，匿名化仅降低影响，不称独立前瞻验证",
            "本次是当前综合分析思路在有限历史输入上的诊断，未证明完整当前资料版本的未来准确率",
        ],
    }
    for index, target in enumerate(targets, 1):
        case = f"C{index:02d}"
        try:
            value = legacy.build_input(source, target, target + "T10:00:00+08:00")
            temporal_guard(value)
            save(OUT / "inputs" / (case + ".json"), value)
            selected = attention(value)
            state = {
                "state": "READY",
                "input_hash": sha(OUT / "inputs" / (case + ".json")),
                "documents": len(value["documents"]),
                "attention_ids": [doc["id"] for doc in selected],
            }
        except (ValueError, AssertionError) as exc:
            state = {"state": "UNAVAILABLE", "reason": str(exc)}
            save(OUT / "reviews" / (case + ".json"), {"case": case, "state": "UNAVAILABLE", "reason": str(exc)})
        protocol["targets"].append({"case": case, "target": target, **state})
    save(OUT / "protocol.json", protocol)
    return protocol


def checked():
    protocol = read(OUT / "protocol.json")
    for path, expected in protocol["protected_hashes"].items():
        if sha(path) != expected:
            raise ValueError("REPLAY_PROTECTED_SOURCE_CHANGED")
    for row in protocol["targets"]:
        if row["state"] == "READY" and sha(OUT / "inputs" / (row["case"] + ".json")) != row["input_hash"]:
            raise ValueError("REPLAY_INPUT_CHANGED")
    return protocol


def pending(protocol):
    return [row for row in protocol["targets"] if not (OUT / "reviews" / (row["case"] + ".json")).exists()]


def next_case():
    """只揭示最早未判断日期；实际年月日、公司名称与净值水平不进入分析显示。

    新片段输出完整保存的片段正文；已读片段通过同一ID、摘要复用。原日期及逐字
    原文仍在冻结输入内，盲化只作用于显示层，不篡改来源和审阅记录。
    """
    protocol = checked()
    queue = pending(protocol)
    if not queue:
        return {"state": "ALL_REVIEWS_FROZEN"}
    row = queue[0]
    value = read(OUT / "inputs" / (row["case"] + ".json"))
    temporal_guard(value)
    nav = [float(item["unit_nav"]) for item in value["nav"]]
    feat = features(nav)
    impact = sum(
        company["weight_pct"] * company["quote"]["change_pct"] / 100
        for company in value["companies"]
        if company["quote"]
    )
    seen = set()
    for previous in protocol["targets"]:
        path = OUT / "reviews" / (previous["case"] + ".json")
        if path.exists():
            seen.update(read(path).get("read_documents", []))
    aliases = {company["code"]: f"持仓公司{i:02d}" for i, company in enumerate(value["companies"], 1)}

    def mask(text):
        for company in sorted(value["companies"], key=lambda item: len(item["name"]), reverse=True):
            text = text.replace(company["name"], aliases[company["code"]])
        return re.sub(r"20\d{2}[-年/]\d{1,2}[-月/]\d{1,2}日?", "〔原文所列日期〕", text)

    documents = []
    for doc in attention(value):
        documents.append(
            {
                "id": doc["id"],
                "kind": doc["kind"],
                "title": mask(doc["title"]),
                "companies": [aliases.get(code, "未披露公司") for code in doc["codes"]],
                "already_read": doc["id"] in seen,
                "body": None if doc["id"] in seen else mask(doc["body"]),
            }
        )
    view = {
        "case": row["case"],
        "state": "READY",
        "input_hash": row["input_hash"],
        "facts": {
            "NAV": {
                "last_change_pct": round((nav[-1] / nav[-2] - 1) * 100, 4),
                "five_day_pct": round(feat[0] * 100, 4),
                "twenty_day_pct": round(feat[1] * 100, 4),
                "sixty_day_range_position_pct": round(feat[5] * 100, 4),
            },
            "HOLDINGS": {
                "estimated_base_day_impact_ppt": round(impact, 4),
                "quoted_disclosed_weight_pct": round(sum(c["weight_pct"] for c in value["companies"] if c["quote"]), 4),
                "leading": [
                    {
                        "company": aliases[c["code"]],
                        "weight_pct": c["weight_pct"],
                        "previous_close_change_pct": c["quote"]["change_pct"],
                    }
                    for c in sorted(value["companies"], key=lambda c: c["weight_pct"], reverse=True)[:5]
                    if c["quote"]
                ],
            },
            "MARKET": {code: item["change_pct"] for code, item in value["market"].items()},
        },
        "documents": documents,
        "document_inventory_count": len(value["documents"]),
        "unreviewed_documents": len(value["documents"]) - len(documents),
        "limits": "旧仓位；已核验原文片段；经理与部分财务背景缺失；没有当日盘中行情；方向仅为低把握假设",
    }
    path = OUT / "views" / (row["case"] + ".json")
    if not path.exists():
        save(path, view)
    return view


def record(args):
    """记录聊天作出的首次判断；不能越过当前日期或修改已写答案。"""
    protocol = checked()
    queue = pending(protocol)
    if not queue or queue[0]["case"] != args.case:
        raise ValueError("REPLAY_FIRST_PENDING_ONLY")
    view = read(OUT / "views" / (args.case + ".json"))
    if args.direction not in {"UP", "DOWN"} or not all([args.reason, args.counter, args.conditions]):
        raise ValueError("REPLAY_REVIEW_INCOMPLETE")
    refs = args.refs.split(",")
    valid = {"NAV", "HOLDINGS", "MARKET", *[doc["id"] for doc in view["documents"]]}
    if not set(refs) <= valid or len(set(refs)) < 2:
        raise ValueError("REPLAY_REFERENCES_INVALID")
    save(
        OUT / "reviews" / (args.case + ".json"),
        {
            "case": args.case,
            "state": "REVIEWED",
            "generation_source": "CODEX_ASSISTED_REVIEW",
            "method": POLICY,
            "input_hash": view["input_hash"],
            "direction": args.direction,
            "confidence": "LOW",
            "reason": args.reason,
            "counterevidence": args.counter,
            "change_conditions": args.conditions,
            "references": refs,
            "read_documents": [doc["id"] for doc in view["documents"]],
            "read_scope": "VERIFIED_EXCERPTS_ONLY",
            "unreviewed_documents": view["unreviewed_documents"],
            "saved_at": datetime.now(data.ZONE).isoformat(),
        },
    )
    return {"saved": args.case, "remaining": len(queue) - 1}


def wilson(hits, total):
    if not total:
        return None
    z = 1.959963984540054
    p = hits / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [center - half, center + half]


def score():
    """所有首次记录封存后统一揭晓；基准只在同一可评日期比较，缺资料日单独保留。"""
    protocol = checked()
    if pending(protocol):
        raise ValueError("REPLAY_ALL_FIRST_REVIEWS_REQUIRED")
    if (OUT / "summary.json").exists():
        return read(OUT / "summary.json")
    source = data.load_sources()
    rows = []
    for item in protocol["targets"]:
        review = read(OUT / "reviews" / (item["case"] + ".json"))
        row = {**item, "review": review}
        index = source["sessions"].index(item["target"])
        base = source["sessions"][index - 1]
        if item["target"] in source["nav"] and base in source["nav"]:
            actual = label(source["nav"][base]["unit_nav"], source["nav"][item["target"]]["unit_nav"])
            row["answer"] = actual
            if review["state"] == "REVIEWED":
                inp = read(OUT / "inputs" / (item["case"] + ".json"))
                row["correct"] = actual["actual_direction"] == review["direction"]
                row["previous_direction"] = label(inp["nav"][-2]["unit_nav"], inp["nav"][-1]["unit_nav"])[
                    "actual_direction"
                ]
        rows.append(row)
    assessed = [row for row in rows if "correct" in row]
    hits = sum(row["correct"] for row in assessed)
    baselines = {
        "ALWAYS_UP": sum(row["answer"]["actual_direction"] == "UP" for row in assessed),
        "ALWAYS_DOWN": sum(row["answer"]["actual_direction"] == "DOWN" for row in assessed),
        "PREVIOUS_DIRECTION": sum(row["answer"]["actual_direction"] == row["previous_direction"] for row in assessed),
    }
    before, after = protocol["private_ledger_before"], ledger()
    summary = {
        "method": POLICY,
        "start": protocol["targets"][0]["target"],
        "end": protocol["targets"][-1]["target"],
        "planned": len(rows),
        "reviewed": sum(row["review"]["state"] == "REVIEWED" for row in rows),
        "unavailable": sum(row["review"]["state"] == "UNAVAILABLE" for row in rows),
        "assessed": len(assessed),
        "correct": hits,
        "accuracy": hits / len(assessed) if assessed else None,
        "wilson_95_interval_descriptive": wilson(hits, len(assessed)),
        "coverage": len(assessed) / len(rows),
        "correct_and_output_share": hits / len(rows),
        "prediction_counts": dict(Counter(row["review"]["direction"] for row in assessed)),
        "actual_counts": dict(Counter(row["answer"]["actual_direction"] for row in assessed)),
        "baselines_same_dates": {
            name: {"correct": value, "assessed": len(assessed), "accuracy": value / len(assessed)}
            for name, value in baselines.items()
        },
        "recall": {
            direction: {
                "actual_count": sum(row["answer"]["actual_direction"] == direction for row in assessed),
                "correct": sum(row["correct"] and row["answer"]["actual_direction"] == direction for row in assessed),
            }
            for direction in ("UP", "DOWN", "FLAT")
        },
        "private_ledger_before": before,
        "private_ledger_after": after,
        "new_private_calls": after["calls"] - before["calls"],
        "review_hashes": {row["case"]: sha(OUT / "reviews" / (row["case"] + ".json")) for row in rows},
        "limits": protocol["limits"] + ["Wilson区间仅描述本样本，未处理时间相关性，不能当未来准确率保证"],
    }
    if after != before:
        raise ValueError("REPLAY_PRIVATE_LEDGER_CHANGED")
    save(OUT / "scored-days.json", rows)
    save(OUT / "summary.json", summary)
    return summary


def main():
    socket.socket.connect = no_network
    socket.create_connection = no_network
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "next", "record", "score", "verify"])
    for name in ("case", "direction", "reason", "counter", "conditions", "refs"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.action == "prepare":
        p = prepare()
        result = {"planned": len(p["targets"]), "states": dict(Counter(row["state"] for row in p["targets"]))}
    elif args.action == "next":
        result = next_case()
    elif args.action == "record":
        result = record(args)
    elif args.action == "score":
        result = score()
    else:
        p = checked()
        result = {
            "protected_sources_unchanged": True,
            "private_ledger_unchanged": ledger() == p["private_ledger_before"],
            "pending": len(pending(p)),
            "summary_exists": (OUT / "summary.json").exists(),
        }
        if (OUT / "summary.json").exists():
            summary = read(OUT / "summary.json")
            if any(
                sha(OUT / "reviews" / (case + ".json")) != expected
                for case, expected in summary["review_hashes"].items()
            ):
                raise ValueError("REPLAY_FIRST_REVIEW_CHANGED")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
