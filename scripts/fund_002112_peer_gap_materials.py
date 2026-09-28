"""两只参考基金的限定资料补查；旧实验和正式采集入口均不使用本脚本。

下载按事先登记的 URL 留存逐次回执，失败也计入 40 次上限。默认入口只做本地
解析及覆盖复核；网络动作须由本次材料清单显式调用 fetch，不提供无限抓取入口。
"""

import hashlib
import io
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
from app.integrations.fund_report_sections_v2 import parse_text
from app.services.fund_002112_peer_coverage_audit import CODES, calendar_days
from app.services.fund_002112_peer_material_repair import OUTPUT as PREVIOUS
from app.services.fund_002112_peer_material_repair import FrozenSources, inspect_input
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once
from pypdf import PdfReader

from scripts.fund_002112_peer_pdf_probe import validate_pdf_identity

OUTPUT = ROOT / "peer-material-gaps/20260927-v1"
# 仅普通公开披露站点；禁止重定向、登录及付费服务。
HOSTS = {
    "static.cninfo.com.cn",
    "www.cninfo.com.cn",
    "www.chinaamc.com",
    "accountquery.chinaamc.com",
    "www.lixinger.com",
    "pdf.dfcfw.com",
    "www.dfham.com",
    "www.sse.com.cn",
    "vip.stock.finance.sina.com.cn",
    "www.psbc.com",
    "download.hexun.com",
}


def fetch(label, url, *, form=None):
    """获取单份已登记资料。缓存命中先核摘要；失败回执不自动重试或更换网址。"""
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("INVALID_RECEIPT_LABEL")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ValueError("PUBLIC_URL_NOT_ALLOWED")
    if form is not None and (
        url != "https://www.cninfo.com.cn/new/hisAnnouncement/query"
        or form.get("searchkey") not in {"华夏磐泰", "东方红新动力"}
        or form.get("pageSize") != 30
        or form.get("pageNum") not in {1, 2}
        or form.get("seDate") not in {"2023-08-01~2023-10-31", "2020-10-01~2021-09-01"}
    ):
        raise ValueError("PUBLIC_SEARCH_SCOPE_INVALID")
    folder = OUTPUT / "downloads"
    folder.mkdir(exist_ok=True)
    record = folder / (label + ".json")
    if record.exists():
        finished = OUTPUT / (label + "-receipt.json")
        result = read_json(finished if finished.exists() else record)
        if result["url"] != url or result.get("form") != form:
            raise ValueError("RECEIPT_URL_CHANGED")
        if "path" in result and file_hash(result["path"]) != result["sha256"]:
            raise ValueError("RECEIPT_CONTENT_CHANGED")
        return result
    if len(list(folder.glob("*.json"))) >= read_json(OUTPUT / "protocol.json")["maximum_download_requests"]:
        raise ValueError("DOWNLOAD_BUDGET_EXHAUSTED")
    result = {
        "url": url,
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "ATTEMPT_RESERVED",
        "historical_first_seen_verified": False,
    }
    if form is not None:
        result["form"] = form
    # 开始网络前落一条回执；中断也占用次数，避免重启后隐瞒请求。
    save_once(record, result)
    try:
        with httpx.Client(timeout=httpx.Timeout(30, connect=5), follow_redirects=False) as client:
            with client.stream("POST" if form is not None else "GET", url, data=form) as response:
                result["http_status"] = response.status_code
                if response.status_code != 200:
                    raise ValueError("PUBLIC_SOURCE_NON_200")
                raw = bytearray()
                for part in response.iter_bytes():
                    raw.extend(part)
                    if len(raw) > 15000000:
                        raise ValueError("PUBLIC_SOURCE_TOO_LARGE")
        raw = bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        path = folder / (sha + (".pdf" if raw.startswith(b"%PDF") else ".html"))
        if path.exists():
            if file_hash(path) != sha:
                raise ValueError("RAW_HASH_COLLISION")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
        result.update(
            status="RECEIVED",
            path=str(path),
            sha256=sha,
            received_at=datetime.now().astimezone().isoformat(),
            byte_count=len(raw),
        )
    except (httpx.HTTPError, ValueError) as exc:
        result.update(status="UNRESOLVED", reason=type(exc).__name__ if isinstance(exc, httpx.HTTPError) else str(exc))
    # 同一尝试的完成回执另存；最初预留回执永不覆盖。
    save_once(OUTPUT / (label + "-receipt.json"), result)
    return result


def parse_report(entry, receipt):
    """核基金、份额、报告期和原表格约束；实际收到时间不倒填成历史收到时间。"""
    protocol = read_json(OUTPUT / "protocol.json")
    key = (entry["fund_code"], entry["report_end"], entry["report_type"])
    if key not in {(e["fund_code"], e["report_end"], e["report_type"]) for e in protocol["report_targets"]}:
        raise ValueError("REPORT_OUTSIDE_FIXED_SCOPE")
    if receipt["status"] != "RECEIVED" or file_hash(receipt["path"]) != receipt["sha256"]:
        raise ValueError("PDF_RECEIPT_INVALID")
    raw = Path(receipt["path"]).read_bytes()
    if not raw.startswith(b"%PDF"):
        raise ValueError("SOURCE_NOT_PDF")
    reader = PdfReader(io.BytesIO(raw))
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 180:
        raise ValueError("PDF_ENCRYPTED_OR_PAGE_LIMIT")
    pages = [p.extract_text() or "" for p in reader.pages]
    code = entry["fund_code"]
    validate_pdf_identity(pages, code)
    report = parse_text(
        pages,
        entry["title"],
        fund_code=code,
        fund_name="东方红新动力" if code == "017493" else "华夏磐泰",
        master_code="000480" if code == "017493" else code,
    )
    if (report["report_end"], report["report_type"]) != key[1:]:
        raise ValueError("REPORT_PERIOD_MISMATCH")
    public_day = max(entry["published_date"], report["published_date"])
    report.update(
        raw=receipt,
        source=entry,
        source_publication_date=public_day,
        available_at=str(date.fromisoformat(public_day) + timedelta(days=1)) + "T08:00:00+08:00",
        historical_version_verified=False,
        training_eligible=False,
        original_pdf_saved=True,
        document_format="PUBLIC_ORIGINAL_PDF",
        purpose="HISTORICAL_INPUT_RECONSTRUCTION_ONLY",
        text_sha256=digest(pages),
    )
    folder = OUTPUT / "reports"
    folder.mkdir(exist_ok=True)
    path = folder / ("-".join(key) + ".json")
    save_once(path, report)
    return {
        "fund_code": code,
        "report_end": key[1],
        "report_type": key[2],
        "parsed_file": str(path),
        "sha256": file_hash(path),
        "raw_receipt": receipt,
        "holding_count": report["holding_count"],
        "page_count": len(pages),
    }


def run_coverage():
    """使用原 727 日、原净值公开日期、原行情原文和原 20 项公式逐日核覆盖，不训练。"""
    for path, sha in read_json(OUTPUT / "protocol.json")["sources"].items():
        if file_hash(path) != sha:
            raise ValueError("PREVIOUS_STAGE_SOURCE_CHANGED")
    sources = FrozenSources(read_json(PREVIOUS / "protocol.json"))
    snapshot = sources.load("snapshot")
    sources.indices = snapshot["indices"]
    days = calendar_days(
        [sources.load("calendar_cn_a_share_2015_2020_research_v1"), sources.load("calendar_cn_a_share_2021_2025_v1")]
    )
    reports = {c: list(snapshot["funds"][c]["reports"]) for c in CODES}
    paths = list((PREVIOUS / "reports").glob("*.json"))
    paths += [
        Path(r["parsed_file"])
        for r in read_json(PREVIOUS / "public-pdf-probe/reviewed-results.json")["reviews"]
        if r["status"] == "PARSED_CANDIDATE"
    ]
    paths += [Path(r["parsed_file"]) for r in read_json(OUTPUT / "report-results.json")["parsed"]]
    expected = dict(read_json(OUTPUT / "protection-before.json")["artifacts"])
    expected.update({r["parsed_file"]: r["sha256"] for r in read_json(OUTPUT / "report-results.json")["parsed"]})
    manifest = {}
    for path in paths:
        if expected.get(str(path)) != file_hash(path):
            raise ValueError("PARSED_REPORT_CHANGED")
        report = read_json(path)
        raw = report["raw"]
        raw_path = raw.get("path") or str(ROOT / raw["file"])
        if "path" not in raw:
            sources.raw(raw)
        if file_hash(raw_path) != raw["sha256"]:
            raise ValueError("REPORT_RAW_CHANGED")
        reports[report["fund_code"]].append(report)
        manifest[str(path)] = file_hash(path)
        manifest[raw_path] = raw["sha256"]
    save_once(OUTPUT / "coverage-sources.json", manifest)
    maps = {c: {r["date"]: r for r in snapshot["funds"][c]["nav"]["rows"]} for c in CODES}
    if any(max(mapping) > "2024-12-31" for mapping in maps.values()):
        raise ValueError("NAV_SCOPE_CHANGED")
    before = read_json(PREVIOUS / "public-input-coverage-daily.json")
    rows = []
    for n, item in enumerate(before):
        code, target = item["fund_code"], item["target"]
        after = inspect_input(snapshot["funds"][code], reports[code], maps[code], days, target, sources)
        if item["after"]["status"] == "INPUT_COMPLETE" and after != item["after"]:
            raise ValueError("PREVIOUS_COMPLETE_INPUT_CHANGED")
        rows.append({"fund_code": code, "target": target, "before": item["after"], "after": after})
        if n % 300 == 0:
            print(f"剩余资料补查覆盖 {n}/{len(before)}，新增拟合 0", flush=True)
    summary = {"new_fits": 0, "cumulative_fits": 52, "input_sha256": digest(rows), "funds": {}}
    for code in CODES:
        subset = [r for r in rows if r["fund_code"] == code]
        if len(subset) != 727 or len({r["target"] for r in subset}) != 727:
            raise ValueError("ORIGINAL_DATE_SET_CHANGED")
        summary["funds"][code] = {
            "before": dict(Counter(r["before"]["status"] for r in subset)),
            "after": dict(Counter(r["after"]["status"] for r in subset)),
            "new_complete": sum(
                r["after"]["status"] == "INPUT_COMPLETE" and r["before"]["status"] != "INPUT_COMPLETE" for r in subset
            ),
        }
    save_once(OUTPUT / "input-coverage-daily.json", rows)
    save_once(OUTPUT / "input-coverage-summary.json", summary)
    save_once(OUTPUT / "market-sources-verified.json", sources.verified)
    print(summary["funds"], flush=True)
    return summary


if __name__ == "__main__":
    run_coverage()
