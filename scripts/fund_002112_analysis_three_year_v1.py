"""002112 三年固定规则回放：全窗口登记缺口，首次结果留档，全部保存后评分。

沿用已冻结 V5 分析，不拟合、不改变规则、不写数据库。不同年份可在独立进程
顺序执行，各自拥有固定预算及文件目录；避免共享请求替身或并发修改旧实验。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import httpx
from app.core.config import get_settings
from app.integrations.fund_information_analysis import Budget, ResponseFormatError, decode_response
from app.schemas.fund_information_analysis import quote_catalog
from app.services.direction_1d_protocol import canonical, digest, label
from app.services.fund_information_contracts import time_context

from scripts import fund_002112_analysis_replay_v1 as original
from scripts import fund_002112_analysis_replay_v5 as prior
from scripts import fund_002112_event_comparison_data_v1 as data

OUT = prior.OUT.parent / "20261010-three-year-replay-v1"
START, END = "2023-10-01", "2026-09-30"
LIMITS = {"2024": 650, "2025": 650, "2026": 500}
YEARS = ("2023", "2024", "2025", "2026")
engine = prior.engine


def selected_sessions(sessions: list[str]) -> list[str]:
    """完整登记固定三年内所有交易日，不按涨跌、旧预测对错或材料数量挑日期。"""
    return [t for t in sessions if START <= t <= END]


def code_files() -> list[Path]:
    return list(
        dict.fromkeys(
            [*prior.code_files(), Path(__file__), Path(__file__).with_name("fund_002112_analysis_three_year_verify.py")]
        )
    )


def prepare() -> dict:
    if (OUT / "protocol.json").exists():
        return checked_protocol()
    prior.checked_protocol()
    source = data.load_sources()
    targets = selected_sessions(source["sessions"])
    protected = prior.protected_files()
    old = [p for root in prior.OUT.parent.glob("*-history-replay-v*") for p in root.rglob("*") if p.is_file()]
    workspaces = {}
    for name, directory in {
        "python": data.PY,
        "java": Path("C:/ideaProject/workSpace12"),
        "vue": Path("C:/WebStormProject/workSpace05"),
    }.items():
        status = subprocess.check_output(["git", "status", "--porcelain", "-z"], cwd=directory).decode("utf-8")
        files = [directory / line[3:] for line in status.split("\0") if line]
        workspaces[name] = {
            "root": str(directory),
            "status": status,
            "hashes": {str(p): data.sha(p) for p in files if p.is_file()},
        }
    data.save(OUT / "before/workspaces.json", workspaces)
    protocol = {
        "fund": "002112",
        "version": "THREE_YEAR_REPLAY_V1",
        "created_at": datetime.now(data.ZONE).isoformat(),
        "start": START,
        "end": END,
        "targets": targets,
        "phase": "MORNING_0830",
        "eligible": [],
        "new_targets": [],
        "reused": [],
        "selection": (
            "固定三年完整交易日历；08:30前61条净值、历史已公开持仓及基准日股票和指数行情齐备才生成，"
            "否则登记资料不可用"
        ),
        "reuse_policy": "原样复用V5的60份冻结输入及首次成功或失败；不能只复用成功，也不因已知答案重新生成",
        "model": get_settings().deepseek_model,
        "max_calls": sum(LIMITS.values()),
        "year_limits": LIMITS,
        "per_day_calls": 45,
        "per_day_minutes": 15,
        "max_concurrent_requests": 2,
        "max_fits": 0,
        "adoption_allowed": False,
        "score_gate": "全726个日期均登记结果后封存，再统一评分；成功、失败、资料不可用分开统计",
        "prior_exposure": "旧60天已查看答案；其余日期已进入既有历史数据和旧研究，不能称为独立前瞻验证",
        "comparison": (
            "各年及全体；恒涨恒跌及上一基准日净值方向；有既有08:30旧模型记录时才作同日共同样本比较，"
            "不补训练2024旧模型"
        ),
        "review_policy": "每个新增成功日一次不带答案的单独解释复核；每年预留新日期数的调用，旧60天复核原样复用",
        "limitations": [
            "公开时间重建不等于历史首见档案",
            "保留V5已知语义错误和审校误报风险；只扩大日期，不按本轮答案修规则",
            "旧持仓不等于实时仓位",
            "资料不可用非随机，不能将可用日期准确率称作完整三年每日准确率",
        ],
        "input_hashes": {},
        "coverage": [],
        "protected_hashes": {str(p): data.sha(p) for p in protected},
        "old_artifact_hashes": {str(p): data.sha(p) for p in old},
        "code_hashes": {str(p): data.sha(p) for p in code_files()},
    }
    for target in targets:
        cutoff = data.cutoff_for(target, "MORNING_0830")
        try:
            old_input = prior.OUT / "inputs" / f"{target}.json"
            value = data.read(old_input) if old_input.exists() else original.build_input(source, target, cutoff)
            value["time_context"] = time_context(value, source["sessions"])
            original.validate_input(value)
            path = OUT / "inputs" / f"{target}.json"
            data.save(path, value)
            protocol["input_hashes"][target] = data.sha(path)
            if not value["time_context"]["baseline_complete"]:
                raise ValueError("INCOMPLETE_MARKET")
            protocol["eligible"].append(target)
            coverage = {
                "target": target,
                "year": target[:4],
                "state": "READY_INPUT",
                "documents": len(value["documents"]),
            }
            if old_input.exists():
                if digest(value) != digest(data.read(old_input)):
                    raise ValueError("REUSE_INPUT_CHANGED")
                prediction = data.read(prior.OUT / "predictions" / f"{target}.json")
                data.save(
                    OUT / "predictions" / f"{target}.json",
                    {**prediction, "reused_from": str(prior.OUT), "input_hash": data.sha(path)},
                )
                data.save(
                    OUT / "semantic-review" / f"{target}.json",
                    data.read(prior.OUT / "semantic-review" / f"{target}.json"),
                )
                protocol["reused"].append(target)
            else:
                protocol["new_targets"].append(target)
        except ValueError as error:
            if str(error) not in {"NAV_GAP", "NAV_NOT_YET_AVAILABLE", "REPORT_UNAVAILABLE", "INCOMPLETE_MARKET"}:
                raise
            coverage = {"target": target, "year": target[:4], "state": "DATA_UNAVAILABLE", "reason": str(error)}
            data.save(OUT / "predictions" / f"{target}.json", {**coverage, "as_of": cutoff, "calls": 0})
            data.save(
                OUT / "semantic-review" / f"{target}.json", {"target": target, "verdict": "NO_INPUT", "issues": []}
            )
        protocol["coverage"].append(coverage)
    protocol["review_reserve"] = dict(Counter(t[:4] for t in protocol["new_targets"]))
    for p in code_files():
        destination = OUT / "frozen-code" / p.relative_to(data.PY)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(p.read_bytes())
    data.save(OUT / "coverage.json", protocol["coverage"])
    data.save(OUT / "protocol.json", protocol)
    return compact(protocol)


def compact(protocol):
    return {
        "planned": len(protocol["targets"]),
        "eligible": len(protocol["eligible"]),
        "new": len(protocol["new_targets"]),
        "reused": len(protocol["reused"]),
        "by_year": {
            y: dict(Counter(r.get("reason", r["state"]) for r in protocol["coverage"] if r["year"] == y)) for y in YEARS
        },
        "max_calls": protocol["max_calls"],
    }


def checked_protocol() -> dict:
    p = data.read(OUT / "protocol.json")
    for section in ("code_hashes", "protected_hashes", "old_artifact_hashes"):
        for filename, expected in p[section].items():
            if data.sha(Path(filename)) != expected:
                raise ValueError("THREE_YEAR_PROTECTED_CHANGED: " + filename)
    for target, expected in p["input_hashes"].items():
        if data.sha(OUT / "inputs" / f"{target}.json") != expected:
            raise ValueError("THREE_YEAR_INPUT_CHANGED")
    if get_settings().deepseek_model != p["model"]:
        raise ValueError("THREE_YEAR_MODEL_CHANGED")
    return p


class Requests(prior.LocalRequests):
    """每年份单进程预算；只复用逐字核验通过且文档身份完全一致的事件解析。"""

    def __init__(self, year: str, protocol: dict):
        super().__init__("unused")
        self.year, self.protocol = year, protocol
        self.calls = len(list((OUT / "calls" / year).glob("*/request.json")))
        for path in sorted((OUT / "calls" / year).glob("*/request.json")):
            response = path.with_name("response.json")
            request = data.read(path)
            if response.exists() and request["stage"].startswith("EVENT"):
                value = json.loads(request["payload"]["messages"][1]["content"])
                self.remember(value.get("documents", []), data.read(response)["payload"], str(response))

    def direct(self, prompt, value, stage, budget):
        budget.check()
        reserve = self.protocol["review_reserve"][self.year] if stage != "SEPARATE_REVIEW" else 0
        if self.calls >= LIMITS[self.year] - reserve or budget.calls >= 45:
            raise ValueError("THREE_YEAR_CALL_LIMIT")
        if stage.startswith("EVENT") and "documents" in value:
            value = {**value, "quote_catalogs": {d["id"]: quote_catalog(d["body"]) for d in value["documents"]}}
        endpoint = self.settings.deepseek_base_url.rstrip("/")
        url = urlparse(endpoint)
        if url.scheme != "https" or url.hostname != "api.deepseek.com" or url.query or url.username:
            raise ValueError("THREE_YEAR_PROVIDER_INVALID")
        payload = {
            "model": self.settings.deepseek_model,
            "stream": False,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 7000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt + prior.GUARD},
                {"role": "user", "content": canonical(value)},
            ],
        }
        folder = OUT / "calls" / self.year / digest({"target": self.target, "payload": payload, "stage": stage})
        if (folder / "response.json").exists():
            return data.read(folder / "response.json")["payload"]
        if (folder / "request.json").exists():
            raise ValueError("THREE_YEAR_PREVIOUS_FAILED_CALL_NOT_RETRIED")
        data.save(
            folder / "request.json",
            {
                "target": self.target,
                "year": self.year,
                "stage": stage,
                "at": datetime.now(data.ZONE).isoformat(),
                "payload": payload,
            },
        )
        self.calls += 1
        budget.calls += 1
        started = time.monotonic()
        try:
            with httpx.Client(
                timeout=httpx.Timeout(min(90, max(1, budget.until - started)), connect=5), follow_redirects=False
            ) as client:
                response = client.post(
                    endpoint + "/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer " + self.settings.deepseek_api_key.get_secret_value()},
                )
            with (folder / "raw-http-response.txt").open("x", encoding="utf-8") as stream:
                stream.write(response.text)
            if response.status_code != 200 or len(response.content) > 250000:
                raise ValueError("THREE_YEAR_PROVIDER_FAILED")
            envelope = response.json()
            data.save(folder / "raw-response.json", envelope)
            choice = envelope["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("THREE_YEAR_OUTPUT_INCOMPLETE")
            raw = decode_response(choice["message"]["content"])
            data.save(
                folder / "response.json",
                {"payload": raw, "usage": envelope.get("usage"), "seconds": round(time.monotonic() - started, 2)},
            )
            return raw
        except Exception as error:
            code = (
                str(error) if isinstance(error, ValueError) and str(error).isupper() else "THREE_YEAR_PROVIDER_FAILED"
            )
            data.save(
                folder / "failure.json",
                {"type": type(error).__name__, "code": code, "seconds": round(time.monotonic() - started, 2)},
            )
            if isinstance(error, ResponseFormatError):
                raise
            raise ValueError(code) from None


def run(year: str):
    protocol = checked_protocol()
    requests = Requests(year, protocol)
    for target in [t for t in protocol["new_targets"] if t.startswith(year)]:
        path = OUT / "predictions" / f"{target}.json"
        if path.exists():
            continue
        value = data.read(OUT / "inputs" / f"{target}.json")
        original.validate_input(value)
        requests.target, before = target, requests.calls
        print(canonical({"year": year, "target": target, "state": "RUNNING", "calls": before}), flush=True)
        try:
            with patch.object(engine, "request", requests.request):
                analysis, facts, gaps = engine.analyze(value, Budget(datetime.now(data.ZONE) + timedelta(minutes=15)))
            result = {
                "state": "READY",
                "analysis": analysis,
                "evidence": facts,
                "gaps": gaps,
                "narrative": engine.narrative(analysis, facts, value["inventory"], gaps),
            }
            result["narrative"]["limitations"] = (
                value["time_context"]["statements"] + result["narrative"]["limitations"]
            )
        except (ValueError, TypeError, KeyError) as error:
            result = {"state": "FAILED", "error": str(error)[:3000]}
        result.update(
            target=target,
            year=year,
            as_of=value["as_of"],
            time_context=value["time_context"],
            calls=requests.calls - before,
            generated_at=datetime.now(data.ZONE).isoformat(),
            input_hash=protocol["input_hashes"][target],
        )
        data.save(path, result)
        data.save(
            OUT / "cache-provenance" / f"{target}.json",
            {
                d["id"]: {"document_digest": digest(d), "parse_source": requests.cache_sources[digest(d)]}
                for d in value["documents"]
                if digest(d) in requests.cache_sources
            },
        )
        print(
            canonical(
                {
                    "year": year,
                    "target": target,
                    "state": result["state"],
                    "calls": requests.calls,
                    "error": result.get("error", "")[:100],
                }
            ),
            flush=True,
        )


def require_complete(protocol):
    if any(not (OUT / "predictions" / f"{t}.json").exists() for t in protocol["targets"]):
        raise ValueError("THREE_YEAR_ALL_PREDICTIONS_REQUIRED")


def review(year):
    protocol = checked_protocol()
    require_complete(protocol)
    requests = Requests(year, protocol)
    for target in [t for t in protocol["new_targets"] if t.startswith(year)]:
        path = OUT / "semantic-review" / f"{target}.json"
        if path.exists():
            continue
        prediction = data.read(OUT / "predictions" / f"{target}.json")
        result = {"target": target, "verdict": "NO_ANALYSIS", "issues": []}
        if prediction["state"] == "READY":
            requests.target = target
            a, facts = prediction["analysis"], prediction["evidence"]
            refs = {k for r in a["reasons"] + a["counterpoints"] for k in r["refs"]}
            try:
                raw = requests.direct(
                    prior.REVIEW_PROMPT,
                    {"analysis": a, "facts": {k: facts[k] for k in refs}, "time_context": prediction["time_context"]},
                    "SEPARATE_REVIEW",
                    Budget(datetime.now(data.ZONE) + timedelta(minutes=2)),
                )
                texts = [
                    a["summary"],
                    a["synthesis"],
                    *a["limitations"],
                    *a["change_conditions"],
                    *[r[k] for r in a["reasons"] + a["counterpoints"] for k in ("title", "meaning", "implication")],
                ]
                if (
                    raw.get("verdict") not in {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}
                    or not isinstance(raw.get("issues"), list)
                    or len(raw["issues"]) > 5
                    or (raw["verdict"] == "SUPPORTED") != (raw["issues"] == [])
                ):
                    raise ValueError("REVIEW_SCHEMA_INVALID")
                for issue in raw["issues"]:
                    if (
                        not issue.get("statement")
                        or not any(issue["statement"] in t for t in texts)
                        or not set(issue["refs"]) <= refs
                    ):
                        raise ValueError("REVIEW_CITATION_INVALID")
                result = {"target": target, **raw}
            except (ValueError, TypeError, KeyError) as error:
                result = {"target": target, "verdict": "REVIEW_FAILED", "issues": [], "error": str(error)}
        data.save(path, result)
        print(canonical({"target": target, "review": result["verdict"], "calls": requests.calls}), flush=True)


def metrics(rows, key):
    valid = [r for r in rows if r.get(key) is not None and r.get("actual") is not None]
    correct = sum(r[key] == r["actual"] for r in valid)
    return {
        "planned": len(rows),
        "judged": len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else None,
        "correct_per_planned": correct / len(rows) if rows else None,
        "by_actual": {
            d: {
                "total": sum(r["actual"] == d for r in valid),
                "correct": sum(r["actual"] == d and r[key] == d for r in valid),
            }
            for d in ("UP", "DOWN", "FLAT")
        },
        "predicted_counts": dict(Counter(r[key] for r in valid)),
    }


def score():
    protocol = checked_protocol()
    require_complete(protocol)
    if any(not (OUT / "semantic-review" / f"{t}.json").exists() for t in protocol["targets"]):
        raise ValueError("THREE_YEAR_ALL_REVIEWS_REQUIRED")
    data.save(OUT / "prediction-seal.json", {str(p): data.sha(p) for p in sorted((OUT / "predictions").glob("*.json"))})
    source = data.load_sources()
    baseline = {
        g: {
            r["target"]: r
            for p in sorted((data.ROOT / "predictions").glob(f"MORNING_0830-2026Q*-{g}.jsonl"))
            for r in data.jsonl(p)
        }
        for g in data.GROUPS
    }
    for g in data.GROUPS:
        baseline[g].update(
            {
                r["target"]: r
                for p in sorted((data.ROOT / "predictions").glob(f"MORNING_0830-2025Q*-{g}.jsonl"))
                for r in data.jsonl(p)
            }
        )
    rows = []
    for target in protocol["targets"]:
        pred = data.read(OUT / "predictions" / f"{target}.json")
        index = source["sessions"].index(target)
        base, earlier = source["sessions"][index - 1], source["sessions"][index - 2]
        nav = source["nav"]
        actual = (
            label(nav[base]["unit_nav"], nav[target]["unit_nav"])["actual_direction"]
            if base in nav and target in nav
            else None
        )
        # 上一日方向也要求净值当时可用；不能拿后来公布的基准净值构造便宜基线。
        momentum = (
            label(nav[earlier]["unit_nav"], nav[base]["unit_nav"])["actual_direction"]
            if target in protocol["input_hashes"] and earlier in nav and base in nav
            else None
        )
        row = {
            "target": target,
            "year": target[:4],
            "base": base,
            "state": pred["state"],
            "actual": actual,
            "new": pred.get("analysis", {}).get("direction"),
            "always_up": "UP",
            "always_down": "DOWN",
            "previous_day": momentum,
            "reused": target in protocol["reused"],
            "review": data.read(OUT / "semantic-review" / f"{target}.json")["verdict"],
        }
        for group in data.GROUPS:
            old = baseline[group].get(target)
            if old and (old["cutoff"] != data.cutoff_for(target, "MORNING_0830") or old["actual"] != actual):
                raise ValueError("THREE_YEAR_COMPARISON_MISMATCH")
            row[group] = old["predicted"] if old else None
        rows.append(row)
    keys = ["new", *data.GROUPS, "always_up", "always_down", "previous_day"]
    summary = {
        "start": START,
        "end": END,
        "cohorts": {},
        "calls": len(list((OUT / "calls").glob("*/*/request.json"))),
        "max_calls": protocol["max_calls"],
        "real_fits": 0,
        "reused": len(protocol["reused"]),
        "adoption_allowed": False,
        "coverage": compact(protocol),
        "semantic_review": dict(Counter(r["review"] for r in rows)),
        "prior_exposure": protocol["prior_exposure"],
    }
    for group in [*YEARS, "all", "new_dates", "reused_dates"]:
        subset = [
            r
            for r in rows
            if group == "all"
            or r["year"] == group
            or (group == "new_dates" and r["target"] in protocol["new_targets"])
            or (group == "reused_dates" and r["reused"])
        ]
        valid = [r for r in subset if r["new"] is not None]
        common = [r for r in valid if all(r[k] is not None for k in data.GROUPS)]
        summary["cohorts"][group] = {
            "planned": len(subset),
            "states": dict(Counter(r["state"] for r in subset)),
            "all_dates": {k: metrics(subset, k) for k in keys},
            "successful_dates": {k: metrics(valid, k) for k in keys},
            "common_old_dates": {k: metrics(common, k) for k in keys},
        }
    docs = [d for t in protocol["eligible"] for d in data.read(OUT / "inputs" / f"{t}.json")["documents"]]
    summary["documents"] = {
        k: {"unique": len({d["id"] for d in docs if d["kind"] == k}), "occurrences": sum(d["kind"] == k for d in docs)}
        for k in data.KINDS
    }
    data.save(OUT / "daily-results.json", rows)
    data.save(OUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "review", "score", "status"])
    parser.add_argument("year", nargs="?", choices=list(LIMITS))
    args = parser.parse_args()
    if args.command in {"run", "review"}:
        if not args.year:
            parser.error("run/review requires year")
        {"run": run, "review": review}[args.command](args.year)
    elif args.command == "status":
        print(
            canonical(
                {
                    "states": dict(Counter(data.read(p)["state"] for p in (OUT / "predictions").glob("*.json"))),
                    "calls": len(list((OUT / "calls").glob("*/*/request.json"))),
                }
            )
        )
    else:
        print(canonical({"prepare": prepare, "score": score}[args.command]()), flush=True)
