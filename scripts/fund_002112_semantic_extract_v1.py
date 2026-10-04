"""近年正文事实提取：固定抽样、已有服务、引用校验、内容缓存与请求账本。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from collections import Counter, defaultdict

import httpx

from scripts import fund_002112_holding_impact_v1 as legacy
from scripts import fund_002112_semantic_upgrade_v1 as core

io, ROOT = core.io, core.ROOT
LOCK = threading.Lock()
PROMPT = legacy.PROMPT + """
另外输出quantities数组，最多3项，只提取本次事件的具体数值，不收录法律条文编号或页码：
{"metric":"PROFIT_YOY|REVENUE_YOY|CONTRACT_AMOUNT|BUYBACK_AMOUNT|OTHER",
 "value_text":"原文数值及单位，例如增长20%至30%、1.2亿元",
 "period_text":"原文所属期间，没有则空字符串",
 "quote":"同时含指标与数值的单段连续原文，最多220字"}。
quantities必须保留区间、负号、币种和单位，不换算、不推断、不将不同表格列拼在一起。
只提取当前公告报告期，不把文中旧事当新公告。不确定就留空。
"""


def selected_text(doc):
    """短文完整读取，长文保存确定性的连续窗口；选段不接触行情或目标标签。"""
    body = legacy.clean(doc["body"])
    limit = core.plan()["llm_source_limit_characters"]
    if len(body) <= limit:
        return body, {"partial": False, "source_chars": len(body), "ranges": [[0, len(body)]]}
    spans = [(0, 1800)]
    pattern = (r"公司.{0,16}(?:主营|主要业务|主要产品|从事)|归属于.*?净利润|业绩变动|业绩.{0,10}原因|"
               r"营业收入|合同金额|中标金额|回购金额|交易金额|集成电路|光模块|算力|光通信|医药|实施日期")
    for match in re.finditer(pattern, body):
        left, right = max(0, match.start() - 200), min(len(body), match.end() + 550)
        if left <= spans[-1][1]:
            old = spans[-1]
            candidate = (old[0], max(old[1], right))
            if sum(b - a for a, b in spans[:-1]) + candidate[1] - candidate[0] > limit - 200:
                break
            spans[-1] = candidate
        elif sum(b - a for a, b in spans) + right - left <= limit - 200:
            spans.append((left, right))
        else:
            break
    return "\n【省略】\n".join(body[a:b] for a, b in spans), {
        "partial": True, "source_chars": len(body), "ranges": spans,
        "selected_chars": sum(b - a for a, b in spans),
    }


def prepare():
    path = ROOT / "extraction-documents.jsonl"
    if path.exists():
        return core.previous.read_lines(path)
    assert (ROOT / "public-completion.json").exists()
    docs = core.previous.read_lines(core.OLD / "documents.jsonl")
    reports = io.read(ROOT / "reports.json")
    selected = []
    for d in docs:
        if not d["body"] or d.get("scope_exclusion") or d["body_available_at"] > "2026-09-29T08:00:00+08:00":
            continue
        report = core.previous.choose_report(reports, d["body_available_at"])
        if core.previous.event_weight(d, d["body_available_at"], report, reports) <= 0:
            continue
        selected.append(d)
    for path in sorted((ROOT / "public-documents").glob("*.json")):
        d = io.read(path)
        if d["admitted"]:
            selected.append(d)
    # 仅同字节正文并同发行人复用；不同公司模板相似不能合并成同一事件。
    unique = {}
    for d in sorted(selected, key=lambda d: (d["body_available_at"], d["id"])):
        issuer = [d["stock_code"]] if d.get("stock_code") else []
        key = io.digest([d["body_sha256"], issuer])
        if key in unique:
            unique[key]["aliases"].append(d["id"])
            continue
        text, selection = selected_text(d)
        unique[key] = {**d, "index": len(unique), "event_id": key, "aliases": [d["id"]],
                       "issuer_codes": issuer, "type": "POLICY" if d["kind"] == "policy" else
                       "NEWS" if d["kind"] == "news" else "ANNOUNCEMENT",
                       "text": text, "text_sha256": io.digest(text), "text_level": "BODY",
                       "selection": selection}
    documents = list(unique.values())
    io.save_lines(ROOT / "extraction-documents.jsonl", documents)
    buckets = defaultdict(list)
    for d in documents:
        buckets[(d["published_date"][:4], d["kind"])].append(d)
    for values in buckets.values():
        values.sort(key=lambda d: io.digest(["sample-v1", d["event_id"]]))
    sample = []
    while len(sample) < min(100, len(documents)):
        for key in sorted(buckets):
            if buckets[key] and len(sample) < 100:
                sample.append(buckets[key].pop()["event_id"])
    io.save(ROOT / "sample-ids.json", sample)
    io.save(ROOT / "extraction-protocol.json", {"at": io.now(), "prompt": PROMPT,
             "prompt_sha256": io.digest(PROMPT), "documents": len(documents), "sample": len(sample),
             "strata": dict(Counter(d["published_date"][:4] + "_" + d["kind"] for d in documents)),
             "limits": core.plan(), "code_sha256": io.sha(__file__)})
    return documents


def validate(raw, doc):
    result = legacy.validate(raw, doc)
    text = legacy.quote_text(doc["text"])
    # 抽样发现的主体和入口页错误必须在全量阶段自动拦截，而不是只改100篇样例。
    checked_facts = []
    for fact in result["facts"]:
        normalized = legacy.quote_text(fact)
        if doc["kind"] == "fund" and re.search(r"交通银行.{0,25}(?:资产总额|实现净利润)", normalized):
            result["field_warnings"].append("CUSTODIAN_FINANCIAL_FACT_REMOVED")
            continue
        if doc["kind"] == "policy" and (normalized in legacy.quote_text(doc["title"])
                or re.search(r"文件下载链接|中国医保.*一生守护|举报电话|^链接[：:]", normalized)):
            result["field_warnings"].append("POLICY_TITLE_OR_NAVIGATION_NOT_BODY_FACT")
            continue
        checked_facts.append(fact)
    result["facts"] = checked_facts
    if not checked_facts:
        result["issues"] = sorted(set(result["issues"] + ["NO_MATERIAL_FACT_AFTER_REVIEW"]))
        result["status"] = "QUARANTINED"
    quantities = []
    for q in raw.get("quantities", []) if isinstance(raw.get("quantities", []), list) else []:
        if not isinstance(q, dict):
            continue
        quote = legacy.quote_text(q.get("quote", ""))
        value = legacy.quote_text(q.get("value_text", ""))
        period = legacy.quote_text(q.get("period_text", ""))
        metric = q.get("metric")
        semantic_ok = True
        if metric in ("PROFIT_YOY", "REVENUE_YOY"):
            noun = "利润|盈利|亏损" if metric == "PROFIT_YOY" else "营业收入|营收"
            semantic_ok = bool(doc["issuer_codes"] and re.search(noun, quote)
                               and re.search(r"同比|去年同期|上年同期", quote) and re.search(r"%|％", value))
        if metric == "CONTRACT_AMOUNT":
            semantic_ok = bool(re.search(r"中标|订单|采购合同|销售合同|供货合同|工程合同", quote))
        if metric == "BUYBACK_AMOUNT":
            semantic_ok = bool(re.search(r"回购", quote) and re.search(r"元", value))
        if (semantic_ok and metric in {"PROFIT_YOY", "REVENUE_YOY", "CONTRACT_AMOUNT", "BUYBACK_AMOUNT", "OTHER"}
                and 5 <= len(quote) <= 500 and quote in text and value and value in quote
                and re.search(r"\d", value) and (not period or period in text)):
            quantities.append(q)
        else:
            result["field_warnings"].append("UNSUPPORTED_QUANTITY_REMOVED")
    result["quantities"] = quantities[:3]
    result["prompt_sha256"] = io.digest(PROMPT)
    result["validation_version"] = 5
    return result


def extract(doc):
    output = ROOT / "semantic-results" / (doc["event_id"] + ".json")
    if output.exists():
        saved = io.read(output)
        assert saved["text_sha256"] == doc["text_sha256"] and saved["prompt_sha256"] == io.digest(PROMPT)
        checked = {**saved, **validate(saved["raw"], doc)} if "raw" in saved else saved
        io.save(ROOT / "validated-v5" / output.name, checked)
        return checked
    from app.core.config import get_settings
    settings = get_settings()
    base = settings.deepseek_base_url.rstrip("/")
    key = settings.deepseek_api_key.get_secret_value()
    if base != "https://api.deepseek.com" or not key:
        raise RuntimeError("EXISTING_PROVIDER_UNAVAILABLE")
    request = {"model": settings.deepseek_model, "temperature": 0, "max_tokens": 1800,
               "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": PROMPT},
                            {"role": "user", "content": json.dumps({k: doc[k] for k in
                              ("type", "issuer_codes", "title", "published_date", "text")}, ensure_ascii=False)}]}
    prefix = ROOT / "semantic-attempts"
    attempts = len(list(prefix.glob(doc["event_id"] + "-*.json")))
    last = "NO_COMPLETED_RESPONSE"
    for number in range(attempts, core.plan()["llm_attempts_per_document"]):
        with LOCK:
            if len(list(prefix.glob("*.json"))) >= core.plan()["llm_requests_max"]:
                raise RuntimeError("LLM_REQUEST_BUDGET_REACHED")
            io.save(prefix / (doc["event_id"] + f"-{number + 1}.json"), {
                "at": io.now(), "id": doc["event_id"], "request_sha256": io.digest(request),
                "attempt": number + 1, "text_sha256": doc["text_sha256"]})
        try:
            with httpx.Client(timeout=90, trust_env=False) as client:
                response = client.post(base + "/chat/completions", json=request,
                                       headers={"Authorization": "Bearer " + key})
            if response.status_code in (401, 402, 403):
                raise RuntimeError("PROVIDER_ACCOUNT_UNAVAILABLE")
            response.raise_for_status()
            data = response.json()
            if data["choices"][0]["finish_reason"] != "stop":
                raise ValueError("TRUNCATED_RESPONSE")
            raw = json.loads(data["choices"][0]["message"]["content"])
            result = {"at": io.now(), "event_id": doc["event_id"], "text_sha256": doc["text_sha256"],
                      "provider_model": data.get("model"), "usage": data.get("usage"),
                      "raw": raw, **validate(raw, doc)}
            io.save(output, result)
            io.save(ROOT / "validated-v5" / output.name, result)
            return result
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            last = type(exc).__name__
            io.save(ROOT / "semantic-errors" / (doc["event_id"] + f"-{number + 1}.json"),
                    {"at": io.now(), "id": doc["event_id"], "reason": last})
            time.sleep(1)
    failed = {"event_id": doc["event_id"], "text_sha256": doc["text_sha256"], "prompt_sha256": io.digest(PROMPT),
              "status": "EXTRACTION_FAILED", "reason": last, "facts": [], "quantities": [], "usage": {}}
    io.save(output, failed)
    return failed


def run(sample=False):
    documents = prepare()
    if sample:
        ids = set(io.read(ROOT / "sample-ids.json"))
        documents = [d for d in documents if d["event_id"] in ids]
    else:
        review = io.read(ROOT / "sample-review.json")
        assert review["passed"] and review["prompt_sha256"] == io.digest(PROMPT)
    counts = Counter()
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(extract, d) for d in documents]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            counts[result["status"]] += 1
            if n % 20 == 0 or n == len(documents):
                status = {"at": io.now(), "done": n, "total": len(documents),
                          "counts": dict(counts), "elapsed_seconds": round(time.monotonic() - started)}
                io.replace(ROOT / ("sample-progress.json" if sample else "semantic-progress.json"), status)
                print(io.canonical(status), flush=True)
    io.save(ROOT / ("sample-completion.json" if sample else "semantic-completion.json"), status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    run(args.sample)
