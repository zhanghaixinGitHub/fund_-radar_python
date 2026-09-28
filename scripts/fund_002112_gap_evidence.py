"""002112 缺口补证工具：普通公开资料有界下载，独立存档，不含拟合入口。

此目录与上一轮准入结果隔离。下载回执只证明本次收到的字节，不冒充历史
下载记录；正文一致也不自动证明首次公开时间或旧版净值未被更正。
"""

import hashlib
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from app.integrations.fund_report_sections_v2 import parse_text
from app.services.fund_002112_peer_admission import BUSINESS_FIELDS, exact_label, report_inventory
from app.services.fund_002112_peer_coverage_audit import calendar_days
from app.services.fund_002112_peer_fold_impact import FOLDS, time_status
from app.services.fund_002112_peer_material_repair import FrozenSources, inspect_input
from app.services.fund_002112_round3_data import exposure, nav_window, select_report
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once
from pypdf import PdfReader

from scripts.fund_002112_peer_pdf_probe import validate_pdf_identity

OUTPUT = ROOT / "peer-gap-evidence/20260927-v1"
PREVIOUS = ROOT / "peer-admission/20260927-v1"
HOSTS = {
    "pdf.dfcfw.com",
    "static.cninfo.com.cn",
    "www.cninfo.com.cn",
    "www.chinaamc.com",
    "accountquery.chinaamc.com",
    "www.dfham.com",
    "eid.csrc.gov.cn",
    "web.archive.org",
    "www.cgbchina.com.cn",
    "q.fund.sohu.com",
    "fund.eastmoney.com",
    "www.tushare.pro",
    "tushare.pro",
}


def freeze():
    """锁定补证范围和预算；原始准入决定保持原样，本次最多 40 个公开请求。"""
    target = OUTPUT / "protocol.json"
    if target.exists():
        protocol = read_json(target)
        if protocol.get("current_fit_budget") != 0:
            raise ValueError("ZERO_FIT_BUDGET_CHANGED")
        return protocol
    reports = read_json(PREVIOUS / "report-admission.json")
    missing = read_json(ROOT / "peer-material-gaps/20260927-v1/remaining-report-worklist.json")
    missing = [r for r in missing if r["fund_code"] == "017493" and r["published_date"] < "2024-01-01"]
    protocol = {
        "version": "PUBLIC_GAP_EVIDENCE_ZERO_FIT_V1",
        "maximum_download_requests": 40,
        "current_fit_budget": 0,
        "actual_new_fits": 0,
        "old_actual_fits": 52,
        "report_targets": reports,
        "missing_report_targets": missing,
        "sources": {
            str(PREVIOUS / name): file_hash(PREVIOUS / name)
            for name in ("protocol.json", "report-admission.json", "row-admission.json", "source-review.json")
        },
        "nav_scope": {"codes": ["017493", "160323"], "first": "2021-01-01", "last": "2023-12-31"},
        "unchanged": "Original dates, cohort, classes, thresholds, L20 parameters and family weights",
        "evidence_rule": "Report reprint or current NAV equality alone does not grant historical version eligibility",
        "historical_local_download_required": False,
    }
    save_once(target, protocol)
    return protocol


def fetch(label, url, *, form=None):
    """按登记 URL 下载一次；失败也占预算，不跟随跳转、不绕过登录或拒绝访问。

    原文按 SHA-256 命名，开始及结束回执分开保存。中断和失败都不自动重试；
    每个响应限制 15 MB，防止无界抓取。调用者不得传入账号、密钥或付费接口。
    """
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("INVALID_LABEL")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ValueError("PUBLIC_URL_NOT_ALLOWED")
    if form is not None and (
        url != "https://www.cninfo.com.cn/new/hisAnnouncement/query"
        or form.get("searchkey") != "华夏磐泰"
        or form.get("seDate") not in {"2021-09-01~2023-04-21", "2023-07-20~2023-07-20", "2024-03-28~2024-03-28"}
        or form.get("pageSize") != 30
        or form.get("pageNum") not in {1, 2, 3, 4}
    ):
        raise ValueError("PUBLIC_SEARCH_SCOPE_INVALID")
    protocol = freeze()
    folder = OUTPUT / "downloads"
    folder.mkdir(exist_ok=True)
    reserve, finish = folder / (label + ".json"), OUTPUT / (label + "-receipt.json")
    if reserve.exists():
        result = read_json(finish if finish.exists() else reserve)
        if result["url"] != url or result.get("form") != form:
            raise ValueError("URL_CHANGED")
        if "path" in result and file_hash(result["path"]) != result["sha256"]:
            raise ValueError("RECEIPT_CONTENT_CHANGED")
        return result
    if len(list(folder.glob("*.json"))) >= protocol["maximum_download_requests"]:
        raise ValueError("DOWNLOAD_BUDGET_EXHAUSTED")
    result = {
        "url": url,
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "ATTEMPT_RESERVED",
        "historical_received_at_verified": False,
    }
    if form is not None:
        result["form"] = form
    save_once(reserve, result)
    try:
        with httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False) as client:
            with client.stream("POST" if form is not None else "GET", url, data=form) as response:
                result["http_status"] = response.status_code
                # ETag 和 Last-Modified 只是服务器本次声明，不能当作历史首发证明。
                result["response_metadata"] = {
                    k: response.headers[k] for k in ("content-type", "etag", "last-modified") if k in response.headers
                }
                if response.status_code != 200:
                    raise ValueError("PUBLIC_SOURCE_NON_200")
                raw = bytearray()
                for part in response.iter_bytes():
                    raw.extend(part)
                    if len(raw) > 15_000_000:
                        raise ValueError("PUBLIC_SOURCE_TOO_LARGE")
        raw = bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        target = folder / (sha + (".pdf" if raw.startswith(b"%PDF") else ".html"))
        if target.exists():
            if file_hash(target) != sha:
                raise ValueError("RAW_HASH_CONFLICT")
        else:
            with target.open("xb") as stream:
                stream.write(raw)
        result.update(
            status="RECEIVED",
            path=str(target),
            sha256=sha,
            byte_count=len(raw),
            received_at=datetime.now().astimezone().isoformat(),
        )
    except (httpx.HTTPError, ValueError) as exc:
        result.update(status="UNRESOLVED", reason=type(exc).__name__ if isinstance(exc, httpx.HTTPError) else str(exc))
    save_once(finish, result)
    return result


def parse_received(entry, receipt):
    """核原文字节、份额代码、报告期和持仓勾稽，保留最晚公开日。

    此函数只确认新取得资料是否完整。它不接受外部传来的资格布尔值，也不把
    官网附件日期直接当作历史版本证明。仅处理冻结清单及两份已知缺失报告。
    """
    targets = freeze()["report_targets"] + freeze()["missing_report_targets"]
    identity = tuple(entry[k] for k in ("fund_code", "report_end", "report_type"))
    matched = [r for r in targets if tuple(r[k] for k in ("fund_code", "report_end", "report_type")) == identity]
    if len(matched) != 1:
        raise ValueError("REPORT_OUTSIDE_FIXED_SCOPE")
    public = matched[0].get("conservative_publication_unchanged", matched[0].get("published_date"))
    if entry["published_date"] != public:
        raise ValueError("FROZEN_PUBLICATION_CHANGED")
    if receipt["status"] != "RECEIVED" or file_hash(receipt["path"]) != receipt["sha256"]:
        raise ValueError("RECEIPT_CONTENT_CHANGED")
    if not Path(receipt["path"]).read_bytes().startswith(b"%PDF"):
        raise ValueError("SOURCE_IS_NOT_PDF")
    reader = PdfReader(receipt["path"])
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 180:
        raise ValueError("PDF_ENCRYPTED_OR_PAGE_LIMIT")
    pages = [page.extract_text() or "" for page in reader.pages]
    code = entry["fund_code"]
    validate_pdf_identity(pages, code)
    parsed = parse_text(
        pages,
        entry["title"],
        fund_code=code,
        fund_name="东方红新动力" if code == "017493" else "华夏磐泰",
        master_code="000480" if code == "017493" else code,
    )
    if tuple(parsed[k] for k in ("fund_code", "report_end", "report_type")) != identity:
        raise ValueError("REPORT_PERIOD_OR_IDENTITY_MISMATCH")
    pub = max(entry["published_date"], parsed["published_date"])
    parsed.update(
        raw=receipt,
        source=entry,
        source_publication_date=pub,
        available_at=str(date.fromisoformat(pub) + timedelta(days=1)) + "T08:00:00+08:00",
        historical_version_verified=False,
        training_eligible=False,
        text_sha256=digest(pages),
        page_count=len(pages),
        pdf_metadata={k: str(v) for k, v in (reader.metadata or {}).items() if k in ("/CreationDate", "/ModDate")},
    )
    return parsed


def load_original():
    """核上轮协议后只读原来未封存的快照和清单；不导入或运行训练器。"""
    protocol = read_json(PREVIOUS / "protocol.json")
    for path, sha in protocol["files"].items():
        if file_hash(path) != sha:
            raise ValueError("ORIGINAL_SOURCE_CHANGED:" + Path(path).name)
    snapshot = read_json(protocol["named"]["snapshot"])
    reports = report_inventory(read_json(protocol["named"]["worklist"]), snapshot)
    return protocol, snapshot, reports


def review_downloads():
    """解析官网和巨潮原件，与原转载表格逐字段比对；差异必须显式保留。"""
    _, _, originals = load_original()
    entries = []
    prior = {(r["fund_code"], r["report_end"], r["report_type"]): r for r in originals.values()}
    missing = {(r["fund_code"], r["report_end"], r["report_type"]): r for r in freeze()["missing_report_targets"]}
    for item in read_json(OUTPUT / "df-issuer-browser-catalog.json")["reports"]:
        key = tuple(item[k] for k in ("fund_code", "report_end", "report_type"))
        title = (prior.get(key) or missing[key])["title"]
        label = "df-" + item["report_end"] + "-" + item["report_type"].lower()
        entries.append(({**item, "title": title}, read_json(OUTPUT / (label + "-receipt.json"))))
    # 后两条仍属于原 24 份清单，仅补来源副本；不会打开 2024 考核或封存标签。
    catalog = read_json(OUTPUT / "hx-cninfo-report-links.json")
    extra = OUTPUT / "hx-cninfo-extra-report-links.json"
    if extra.exists():
        catalog += read_json(extra)
    for item in catalog:
        public = (
            datetime.fromtimestamp(item["announcementTime"] / 1000, timezone(timedelta(hours=8))).date().isoformat()
        )
        matches = [r for r in originals.values() if r["fund_code"] == "160323" and r["published_date"] == public]
        if len(matches) != 1:
            raise ValueError("CATALOGUE_REPORT_AMBIGUOUS")
        source = matches[0]
        entry = {k: source[k] for k in ("fund_code", "report_end", "report_type", "published_date")}
        entry.update(title=re.sub(r"<[^>]*>", "", item["announcementTitle"]), notice=item)
        entries.append((entry, read_json(OUTPUT / ("hx-pdf-" + item["announcementId"] + "-receipt.json"))))
    results = []
    for entry, receipt in entries:
        key = tuple(entry[k] for k in ("fund_code", "report_end", "report_type"))
        parsed = parse_received(entry, receipt)
        old = prior.get(key)
        changed = [field for field in BUSINESS_FIELDS if old and old[field] != parsed[field]]
        target = OUTPUT / "reports" / ("-".join(key) + ".json")
        target.parent.mkdir(exist_ok=True)
        save_once(target, parsed)
        results.append(
            {
                "fund_code": key[0],
                "report_end": key[1],
                "report_type": key[2],
                "parsed_file": str(target),
                "sha256": file_hash(target),
                "raw_sha256": receipt["sha256"],
                "source_url": receipt["url"],
                "source_publication_date": parsed["source_publication_date"],
                "old_raw_sha256": old["raw"]["sha256"] if old else None,
                "newly_recovered_report": old is None,
                "table_differences": changed,
                "table_equal": not changed if old else None,
                "holding_count": parsed["holding_count"],
                "page_count": parsed["page_count"],
                "historical_version_verified": False,
                "training_eligible": False,
            }
        )
        print("已核原文", key, "持仓", parsed["holding_count"], "与旧表差异", changed, flush=True)
    save_once(OUTPUT / ("report-supplements-v2.json" if extra.exists() else "report-supplements.json"), results)
    return results


def fill_report_exclusions():
    """只重查因这两份报告缺失的 80 个目标日，旧 750 行不重新计算权重。

    新输入与 2021—2023 原单位净值答案分别落盘；停牌缺行情仍排除，报告公开
    当日不使用报告。所有数值通过也只能说明候选资料完整，不能绕过版本准入。
    """
    protocol, snapshot, original_reports = load_original()
    sources = FrozenSources(read_json(protocol["named"]["repair_protocol"]))
    sources.indices = snapshot["indices"]
    days = calendar_days([read_json(protocol["named"][k]) for k in ("calendar_older", "calendar_recent")])
    new = [r for r in read_json(OUTPUT / "report-supplements.json") if r["newly_recovered_report"]]
    if len(new) != 2:
        raise ValueError("TWO_MISSING_REPORTS_NOT_RECOVERED")
    pool = [r for r in original_reports.values() if r["fund_code"] == "017493"]
    for row in new:
        if file_hash(row["parsed_file"]) != row["sha256"]:
            raise ValueError("NEW_PARSED_REPORT_CHANGED")
        pool.append(read_json(row["parsed_file"]))
    fund = snapshot["funds"]["017493"]
    mapping = {r["date"]: r for r in fund["nav"]["rows"] if r["date"] <= "2023-12-31"}
    candidates = [
        r
        for r in read_json(PREVIOUS / "excluded-dates.json")
        if r["fund_code"] == "017493" and r["after"]["status"] == "LATEST_DISCLOSURE_MISSING_NO_FALLBACK"
    ]
    if len(candidates) != 80 or any(not "2021-01-04" <= r["target"] <= "2023-12-29" for r in candidates):
        raise ValueError("REPORT_EXCLUSION_SCOPE_CHANGED")
    checks, inputs, labels = [], [], []
    for row in candidates:
        u = row["target"]
        after = inspect_input(fund, pool, mapping, days, u, sources)
        checks.append({"fund_code": "017493", "target": u, "before": row["after"], "after": after})
        if after["status"] != "INPUT_COMPLETE":
            continue
        i = days.index(u)
        window = nav_window(mapping, days, u)
        report = select_report(pool, fund["catalog"], u)
        x = window["x"] + exposure(report, sources.quotes, sources.indices, days[i - 21 : i])
        if digest(x) != after["input_vector_sha256"]:
            raise ValueError("NEW_INPUT_REPLAY_MISMATCH")
        label = exact_label(mapping[days[i - 1]], mapping[u], days)
        times = {f: time_status(u, label["label_publication"], start)[0] for f, start in FOLDS.items()}
        for fold, status in times.items():
            if status == "TIME_ELIGIBLE" and label["base_publication"] >= FOLDS[fold]:
                raise ValueError("BASE_NAV_NOT_PUBLIC_BY_CUTOFF")
        inputs.append(
            {
                "fund_code": "017493",
                "target": u,
                "x": x,
                "input_vector_sha256": digest(x),
                "report_sha256": report["raw"]["sha256"],
                "nav_dependency_dates": sorted(set(window["nav_dates"] + [days[i - 1], u])),
                "training_eligible": False,
            }
        )
        labels.append(
            {
                "fund_code": "017493",
                **label,
                "fold_time": times,
                "training_eligible": False,
                "gaps": ["REPORT_HISTORICAL_VERSION_PENDING", "NAV_INPUT_AND_LABEL_HISTORICAL_VERSION_PENDING"],
            }
        )
    save_once(OUTPUT / "report-exclusion-recheck.json", checks)
    save_once(OUTPUT / "additional-inputs-not-admitted.json", inputs)
    save_once(OUTPUT / "additional-labels-not-admitted.json", labels)
    save_once(OUTPUT / "additional-market-sources.json", sources.verified)
    summary = {
        "rechecked_dates": len(checks),
        "after": dict(Counter(r["after"]["status"] for r in checks)),
        "new_input_and_label_numeric_complete": len(inputs),
        "additional_time_eligible": {f: sum(r["fold_time"][f] == "TIME_ELIGIBLE" for r in labels) for f in FOLDS},
        "new_training_eligible": 0,
        "actual_new_fits": 0,
    }
    save_once(OUTPUT / "report-gap-recovery-summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "reports", "coverage"))
    command = parser.parse_args().command
    result = {"freeze": freeze, "reports": review_downloads, "coverage": fill_report_exclusions}[command]()
    print(result if command == "coverage" else "独立证据处理完成；新增拟合 0。")
