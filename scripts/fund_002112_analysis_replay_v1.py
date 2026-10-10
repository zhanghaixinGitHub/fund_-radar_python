"""002112 综合分析的有限历史回放；只写独立文件，不写现用预测或训练注册表。

prepare 固定最近二十个既有早间可比日期；run 不读取目标答案；score 最后核对。
公开时间部分来自保守重建，因此结果属于历史诊断，不能当作提前预测的成绩。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import httpx
from app.core.config import get_settings
from app.integrations.fund_information_analysis import Budget
from app.schemas.fund_information_analysis import validate_events
from app.services import fund_information_analysis as engine
from app.services.direction_1d_protocol import canonical, digest, label, window
from app.services.fund_information_snapshot import facts, inventory, unique_documents

from scripts import fund_002112_event_comparison_data_v1 as data

OUT = data.PY / ".local-runs/direction-1d-information/20261009-history-replay-v1"
MAX_CALLS = 220
GUARD = "\n历史回放：只能使用本次输入的事实，不得使用记忆中的后续价格、事件或结果。"


def build_input(source: dict, target: str, cutoff: str) -> dict:
    """按公开可得时间裁剪；只装入基日及以前数据，不把目标答案或旧方向传给模型。"""
    base, actual_target = data.prediction_target(source["sessions"], cutoff)
    if actual_target != target:
        raise ValueError("TARGET_MISMATCH")
    _, nav = data.strict_nav(source, base, cutoff)
    if not nav.get("strict_base_ready"):
        raise ValueError(nav["reason"])
    report = data.legacy.choose_report(source["reports"], cutoff)
    if not report:
        raise ValueError("REPORT_UNAVAILABLE")
    companies = []
    prices = source["stocks"].get(base, {})
    for h in report["holdings"]:
        price = prices.get(h["stock_code"])
        companies.append(
            {
                "code": h["stock_code"],
                "name": h["stock_name"],
                "weight_pct": float(h["nav_weight_pct"]),
                "quote": {"date": base, "change_pct": price["pct_chg"]} if price else None,
                "financials": [],
                "business": [],
            }
        )
    start = str(data.date.fromisoformat(cutoff[:10]) - timedelta(days=14))
    documents = []
    for event in data.related_events(source, base, cutoff):
        if not start <= (event.get("published_date") or "") <= cutoff[:10]:
            continue
        # 仅使用冻结包内已经核验的正文片段；不拿今天的新正文补过去输入。
        body = "\n".join(event["facts"])
        documents.append(
            {
                "id": event["id"],
                "title": event["title"],
                "kind": event["kind"],
                "date": event["published_date"],
                "available_at": event["available_at"],
                "source_hash": event["source_hash"],
                "url": event["source_url"],
                "codes": sorted({link["code"] for link in event["links"]}),
                "body": body[:18000],
                "body_scope": "VERIFIED_EXCERPT",
                "body_truncated": len(body) > 18000,
            }
        )
    documents = unique_documents(documents)
    documents.sort(key=lambda d: (d["date"], d["id"]), reverse=True)
    value = {
        "fund_code": "002112",
        "fund_name": "德邦鑫星价值灵活配置混合C",
        "as_of": cutoff,
        "window": window(data.moment(cutoff)),
        "latest_nav_date": base,
        "nav": [{"nav_date": day, "unit_nav": v} for day, v in zip(nav["dates"], nav["values"], strict=True)],
        "report": {
            "endDate": report["report_end"],
            "publishedDate": report["published_date"],
            "available_at": report["available_at"],
            "hash": report["raw"]["sha256"],
            "industries": [
                {"name": r["name"], "weightPct": float(r["nav_weight_pct"])} for r in report["reported_industries"]
            ],
        },
        "companies": companies,
        "market": {
            code: {"date": base, "change_pct": items[base]["pct_chg"]}
            for code, items in source["markets"].items()
            if base in items
        },
        "documents": documents[:100],
        "overflow": max(0, len(documents) - 100),
        "dividends": [],
        "shares": [],
        "excluded": [],
        "time_proof": {
            "nav_max_available_at": nav["max_available_at"],
            "quote_available_at": base + "T18:00:00+08:00",
            "basis": "PUBLICATION_TIME_RECONSTRUCTION_NOT_CONTEMPORANEOUS_ARCHIVE",
        },
    }
    value["facts"] = facts(value)
    value["inventory"] = inventory(value)
    for item in value["inventory"]:
        if item["id"] in {"D09", "D10", "D12"}:
            item.update(status="MISSING", detail="本次历史包未提供独立可核验资料，不能当作没有相关事项")
        if item["id"] == "D16":
            item["detail"] = "北京时间；按公开日期及保守可用时间重建，不是当时首次收到的档案"
    validate_input(value)
    return value


def validate_input(value: dict) -> None:
    """在每次调用前再次核对时间；排除持仓、正文、净值和行情中的未来记录。"""
    cutoff = data.moment(value["as_of"])
    base = value["window"]["base_nav_date"]
    assert value["window"]["target_nav_date"] > base
    assert value["latest_nav_date"] == base
    assert all(r["nav_date"] <= base for r in value["nav"])
    assert data.moment(value["report"]["available_at"]) <= cutoff
    assert data.moment(value["time_proof"]["nav_max_available_at"]) <= cutoff
    assert data.moment(value["time_proof"]["quote_available_at"]) <= cutoff
    assert all(c["quote"] is None or c["quote"]["date"] <= base for c in value["companies"])
    assert all(v["date"] <= base for v in value["market"].values())
    assert all(data.moment(d["available_at"]) <= cutoff and d["date"] <= str(cutoff.date()) for d in value["documents"])
    forbidden = {"actual", "actual_direction", "label", "target_unit_nav", "predicted", "return", "mature_at"}

    def visit(item):
        if isinstance(item, dict):
            assert not forbidden.intersection(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)


def prepare() -> dict:
    if (OUT / "protocol.json").exists():
        return checked_protocol()
    # 只按目标日期和输入合格状态选样本，不按答案、原预测是否正确或消息方向筛选。
    previous = data.jsonl(data.ROOT / "dataset.jsonl")
    chosen = sorted((r for r in previous if r["phase"] == "MORNING_0830"), key=lambda r: r["target"])[-20:]
    source = data.load_sources()
    inputs = [build_input(source, r["target"], r["cutoff"]) for r in chosen]
    paths = [
        data.SOURCE,
        data.MATERIAL,
        data.ROOT / "dataset.jsonl",
        Path(engine.__file__),
        data.PY / "app/services/fund_information_snapshot.py",
        data.PY / "app/schemas/fund_information_analysis.py",
        Path(data.__file__),
        Path(__file__),
    ]
    paths += sorted((data.ROOT / "predictions").glob("MORNING_0830-2026Q3-*.jsonl"))
    protocol = {
        "fund": "002112",
        "created_at": datetime.now(data.ZONE).isoformat(),
        "targets": [v["window"]["target_nav_date"] for v in inputs],
        "phase": "MORNING_0830",
        "max_calls": MAX_CALLS,
        "max_fits": 0,
        "model": get_settings().deepseek_model,
        "prompt_guard": GUARD,
        "adoption_allowed": False,
        "selection": "既有同截点合格日期中最近20天，先固定输入，所有判断保存后才读取答案评分",
        "input_hashes": {},
        "protected_hashes": {str(p): data.sha(p) for p in paths},
        "limitations": [
            "历史公开时点重建，不是提前保存的实盘预测；大模型既有知识污染无法完全排除",
            "净值可用时间造成非连续且非随机的样本缺口；只有20天，不能证明未来有效",
            "同一来源与时点比较，旧分类器使用聚合特征和截短文本，新分析阅读两周内最多100份核验片段",
            "未取得完整原文、独立公司财务、基金自身事项的部分按缺失记录",
            "公告解析按正文缓存复用；不以条数或材料长度推动方向，不重试已失败的历史日期",
        ],
    }
    for value in inputs:
        target = value["window"]["target_nav_date"]
        path = OUT / "inputs" / (target + ".json")
        data.save(path, value)
        protocol["input_hashes"][target] = data.sha(path)
    data.save(OUT / "protocol.json", protocol)
    return protocol


def checked_protocol() -> dict:
    p = data.read(OUT / "protocol.json")
    for filename, expected in p["protected_hashes"].items():
        if data.sha(Path(filename)) != expected:
            raise ValueError("PROTECTED_INPUT_CHANGED: " + filename)
    for target, expected in p["input_hashes"].items():
        if data.sha(OUT / "inputs" / (target + ".json")) != expected:
            raise ValueError("REPLAY_INPUT_CHANGED")
    if get_settings().deepseek_model != p["model"]:
        raise ValueError("MODEL_CHANGED")
    return p


class LocalRequests:
    """公共基金资料送到项目既有服务；单独记账缓存，绝不写运行时数据库。

    单篇只有通过逐字引用校验才复用；相同文档不因参与多个日期重复收费解析。
    总调用上限含失败，原请求/响应及用量留存；没有自动重试和模型参数搜索。
    """

    def __init__(self):
        self.settings = get_settings()
        self.calls = len(list((OUT / "calls").glob("*/request.json")))
        self.hits = 0

    def request(self, prompt: str, value: dict, stage: str, budget: Budget) -> dict:
        if stage != "EVENT":
            return self.direct(prompt, value, stage, budget)
        found, missing = {}, []
        for doc in value["documents"]:
            key = digest({"prompt": prompt + GUARD, "doc": doc, "model": self.settings.deepseek_model})
            path = OUT / "documents" / (key + ".json")
            if path.exists():
                item = data.read(path)
                validate_events({"items": [item]}, [doc])
                found[doc["id"]] = item
                self.hits += 1
            else:
                missing.append(doc)
        if missing:
            raw = self.direct(prompt, {"documents": missing}, stage, budget)
            # 不在传输层重试；校验失败交给原分析流程的有界修复。
            try:
                validate_events(raw, missing)
            except (ValueError, TypeError, KeyError):
                return {"items": list(found.values()) + raw.get("items", [])}
            for item in raw["items"]:
                doc = next(d for d in missing if d["id"] == item["id"])
                key = digest({"prompt": prompt + GUARD, "doc": doc, "model": self.settings.deepseek_model})
                data.save(OUT / "documents" / (key + ".json"), item)
                found[item["id"]] = item
        return {"items": [found[d["id"]] for d in value["documents"]]}

    def direct(self, prompt: str, value: dict, stage: str, budget: Budget) -> dict:
        budget.check()
        endpoint = self.settings.deepseek_base_url.rstrip("/")
        u = urlparse(endpoint)
        if u.scheme != "https" or u.hostname != "api.deepseek.com" or u.query or u.username:
            raise ValueError("UNEXPECTED_PROVIDER")
        prompt += GUARD
        key = digest({"prompt": prompt, "value": value, "model": self.settings.deepseek_model})
        folder = OUT / "calls" / key
        if (folder / "response.json").exists():
            self.hits += 1
            return data.read(folder / "response.json")["payload"]
        if (folder / "request.json").exists():
            raise ValueError("PRIOR_CALL_FAILED_NO_RETRY")
        if self.calls >= MAX_CALLS or budget.calls >= 28:
            raise ValueError("REPLAY_CALL_LIMIT")
        payload = {
            "model": self.settings.deepseek_model,
            "stream": False,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 7000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": canonical(value)}],
        }
        data.save(
            folder / "request.json", {"stage": stage, "at": datetime.now(data.ZONE).isoformat(), "payload": payload}
        )
        self.calls += 1
        budget.calls += 1
        start = time.monotonic()
        try:
            with httpx.Client(timeout=httpx.Timeout(90, connect=5), follow_redirects=False) as client:
                response = client.post(
                    endpoint + "/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer " + self.settings.deepseek_api_key.get_secret_value()},
                )
            if response.status_code != 200 or len(response.content) > 250000:
                raise ValueError("PROVIDER_FAILED")
            body = response.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("OUTPUT_INCOMPLETE")
            result = json.loads(choice["message"]["content"])
            data.save(
                folder / "response.json",
                {"payload": result, "usage": body.get("usage"), "seconds": round(time.monotonic() - start, 2)},
            )
            return result
        except Exception as exc:
            # 不打印HTTP请求或异常内容，避免将凭据和原始错误带入日志。
            data.save(folder / "failure.json", {"type": type(exc).__name__})
            raise ValueError("REPLAY_PROVIDER_FAILED") from None


def run() -> None:
    protocol = checked_protocol()
    requests = LocalRequests()
    for target in protocol["targets"]:
        path = OUT / "predictions" / (target + ".json")
        if path.exists():
            continue
        value = data.read(OUT / "inputs" / (target + ".json"))
        validate_input(value)
        before = requests.calls
        print(canonical({"target": target, "state": "RUNNING", "calls": before}), flush=True)
        try:
            # 历史截止时间仅限制资料；实际运行预算使用今天的时钟，不伪造过去生成时间。
            budget = Budget(datetime.now(data.ZONE) + timedelta(minutes=15))
            with patch.object(engine, "request", requests.request):
                analysis, evidence, gaps = engine.analyze(value, budget)
            result = {"state": "READY", "analysis": analysis, "evidence": evidence, "gaps": gaps}
        except (ValueError, TypeError, KeyError) as error:
            result = {"state": "FAILED", "error": str(error)[:500]}
        result.update(
            target=target,
            as_of=value["as_of"],
            generated_at=datetime.now(data.ZONE).isoformat(),
            input_hash=protocol["input_hashes"][target],
            calls=requests.calls - before,
        )
        data.save(path, result)
        print(canonical({"target": target, "state": result["state"], "calls": requests.calls}), flush=True)


def metrics(rows: list[dict], key: str) -> dict:
    valid = [r for r in rows if r.get(key) in {"UP", "DOWN", "FLAT"}]
    correct = sum(r[key] == r["actual"] for r in valid)
    return {
        "planned": len(rows),
        "judged": len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else None,
        "correct_per_planned": correct / len(rows) if rows else None,
        "predicted_counts": dict(Counter(r[key] for r in valid)),
        "by_actual": {
            d: {
                "total": sum(r["actual"] == d for r in valid),
                "correct": sum(r["actual"] == d == r[key] for r in valid),
            }
            for d in ("UP", "DOWN", "FLAT")
        },
    }


def score() -> dict:
    protocol = checked_protocol()
    predictions = [data.read(OUT / "predictions" / (t + ".json")) for t in protocol["targets"]]
    # 只有所有日期已有不可变成功/失败记录后，评分阶段才接触目标日答案。
    source = data.load_sources()
    baseline = {
        group: {
            r["target"]: r for r in data.jsonl(data.ROOT / "predictions" / ("MORNING_0830-2026Q3-" + group + ".jsonl"))
        }
        for group in data.GROUPS
    }
    rows = []
    for prediction in predictions:
        target = prediction["target"]
        value = data.read(OUT / "inputs" / (target + ".json"))
        answer = label(source["nav"][value["window"]["base_nav_date"]]["unit_nav"], source["nav"][target]["unit_nav"])
        row = {
            "target": target,
            "actual": answer["actual_direction"],
            "return": answer["nav_return"],
            "new": prediction.get("analysis", {}).get("direction"),
            "state": prediction["state"],
            "always_up": "UP",
            "always_down": "DOWN",
        }
        for group in data.GROUPS:
            old = baseline[group][target]
            assert old["cutoff"] == value["as_of"] and old["actual"] == row["actual"]
            row[group] = old["predicted"]
        rows.append(row)
    common = [r for r in rows if r["new"] is not None]
    summary = {
        "targets": protocol["targets"],
        "total": len(rows),
        "ready": len(common),
        "actual_counts": dict(Counter(r["actual"] for r in rows)),
        "all_dates": {k: metrics(rows, k) for k in ["new", *data.GROUPS, "always_up", "always_down"]},
        "common_dates": {k: metrics(common, k) for k in ["new", *data.GROUPS, "always_up", "always_down"]},
        "paired": {
            g: {
                "new_right_old_wrong": sum(r["new"] == r["actual"] != r[g] for r in common),
                "old_right_new_wrong": sum(r[g] == r["actual"] != r["new"] for r in common),
            }
            for g in data.GROUPS
        },
        "calls": len(list((OUT / "calls").glob("*/request.json"))),
        "real_fits": 0,
        "failed": [{"target": p["target"], "error": p.get("error")} for p in predictions if p["state"] != "READY"],
        "data_coverage": {},
        "limitations": protocol["limitations"],
        "adoption_allowed": False,
        "protected_files_unchanged": len(protocol["protected_hashes"]),
    }
    for kind in ("ANNOUNCEMENT", "NEWS", "POLICY"):
        docs = [
            d
            for t in protocol["targets"]
            for d in data.read(OUT / "inputs" / (t + ".json"))["documents"]
            if d["kind"] == kind
        ]
        summary["data_coverage"][kind] = {"occurrences": len(docs), "unique": len({d["id"] for d in docs})}
    data.save(OUT / "daily-results.json", rows)
    data.save(OUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "score"])
    command = parser.parse_args().command
    if command == "run":
        run()
    else:
        print(canonical(prepare() if command == "prepare" else score()), flush=True)
