"""新信息研究的早期遗漏资料准入；只新增证据，绝不修改旧资料包。

公开日沿用原来源声明，并拦截已知修订；当前网页用于核数值，不能代替历史日期。
本模块没有模型导入。所有网络请求先登记，失败不自动重复，成功收据可恢复复用。
"""

import json
import re
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.integrations.fund_report_sections_v2 import parse_text
from app.integrations.public_fund_reports import PEERS
from app.services.fund_002112_holdings_compatibility import json_subtree
from app.services.fund_002112_peer_admission import BUSINESS_FIELDS
from app.services.fund_002112_peer_ready import report_date_check
from app.services.fund_002112_round3_data import eligible_at, exposure, gate, nav_window, public_day, weights
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json
from app.services.fund_002112_zero_fit_review import save_once as immutable_save

RUN = ROOT / "information-research/20260928-v1"
PREVIOUS = ROOT / "pre-q2-material-review/20260928-v1"
BASE = ROOT / "peer-training-ready/20260928-v2"
OUT = RUN / "early-admission"


def save_once(path, value):
    """创建独立证据子目录；底层仍拒绝覆盖内容不同的已有结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    immutable_save(path, value)


def source_data():
    """先验证前阶段冻结来源，再限定读取九只基金的 2023 年前记录。"""
    protocol = read_json(PREVIOUS / "protocol.json")
    for spec in protocol["sources"].values():
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("PREVIOUS_SOURCE_CHANGED")
    raw = Path(protocol["sources"]["snapshot"]["path"]).read_text(encoding="utf-8")
    candidates = [r for r in read_json(PREVIOUS / "input-feasibility.json") if r["status"] == "INPUT_AND_TIME_ELIGIBLE"]
    funds = {}
    for code in sorted({r["fund_code"] for r in candidates}):
        funds[code] = {k: json_subtree(raw, ["payload", "funds", code, k]) for k in ("reports", "catalog", "family")}
        funds[code]["nav"] = {
            r["date"]: r
            for r in json_subtree(raw, ["payload", "funds", code, "nav", "rows"])
            if r["date"] < "2023-04-01"
        }
    indices = json_subtree(raw, ["payload", "indices"])
    calendar = sorted(d for d in indices["000300.SH"]["rows"] if d < "2023-04-01")
    required = defaultdict(set)
    for row in candidates:
        window = nav_window(funds[row["fund_code"]]["nav"], calendar, row["target"])
        required[row["fund_code"]].update(window["nav_dates"] + [row["base"], row["target"]])
    return raw, candidates, funds, indices, calendar, required


def publication(row):
    """返回最晚已知版本日；日期未知、未来修订或伪造摘要均拒绝。"""
    announced = row.get("ann_date")
    if not announced or date.fromisoformat(announced).isoformat() != announced or announced < row["date"]:
        raise ValueError("NAV_PUBLICATION_INVALID")
    if not re.fullmatch(r"[0-9a-f]{64}", row.get("source_hash", "")):
        raise ValueError("NAV_SOURCE_HASH_INVALID")
    known = [announced]
    for field in ("revised_at", "version_publication_date"):
        if field in row:
            if not row[field]:
                raise ValueError("KNOWN_REVISION_DATE_UNKNOWN")
            known.append(date.fromisoformat(row[field][:10]).isoformat())
    if row.get("revision_known") and len(known) == 1:
        raise ValueError("KNOWN_REVISION_DATE_UNKNOWN")
    return max(known)


def label(base, target, calendar):
    """仅新增 2019—2022 遗漏记录的答案；两端均按十进制原值和较晚成熟时间核对。"""
    day = target["date"]
    if (
        not "2019-01-01" <= day < "2023-01-01"
        or calendar.index(day) == 0
        or calendar[calendar.index(day) - 1] != base["date"]
    ):
        raise ValueError("LABEL_SCOPE_OR_ADJACENCY")
    a, b = [Decimal(str(r["nav"])) for r in (base, target)]
    if any(not n.is_finite() or n <= 0 for n in (a, b)):
        raise ValueError("INVALID_NAV_VALUE")
    version = max(publication(base), publication(target), day)
    return {
        "base": base["date"],
        "target": day,
        "base_unit_nav": str(a),
        "target_unit_nav": str(b),
        "actual_direction": "UP" if b > a else "DOWN" if b < a else "FLAT",
        "base_publication": base["ann_date"],
        "label_publication": target["ann_date"],
        "base_source_hash": base["source_hash"],
        "target_source_hash": target["source_hash"],
        "mature_at": (date.fromisoformat(version) + timedelta(days=1)).isoformat() + "T08:00:00+08:00",
    }


def report_pages(report):
    """读取绑定原件，校验转载身份；只返回正文、公开日期和已知 PDF 修改日期。"""
    raw = report["raw"]
    path = Path(raw["path"]) if raw.get("path") else ROOT / raw["file"]
    if file_hash(path) != raw["sha256"]:
        raise ValueError("REPORT_RAW_CHANGED")
    if raw.get("revised_after_receipt"):
        raise ValueError("REPORT_KNOWN_REVISION_UNRESOLVED")
    modified = None
    if path.suffix.lower() == ".pdf":
        pdf = PdfReader(path)
        if pdf.is_encrypted or not 1 <= len(pdf.pages) <= 180:
            raise ValueError("REPORT_PDF_LIMIT")
        pages = [p.extract_text() or "" for p in pdf.pages]
        stamp = str((pdf.metadata or {}).get("/ModDate", ""))
        if stamp:
            if not re.match(r"D:\d{8}", stamp):
                raise ValueError("REPORT_MODIFIED_DATE_UNREADABLE")
            modified = datetime.strptime(stamp[2:10], "%Y%m%d").date().isoformat()
        source = report.get("source", {})
        if source.get("catalog_activation_ms"):
            # 德邦目录历史核验已确认 activation 字段代表展示日期；publish 是站点迁移时间。
            public = datetime.fromtimestamp(int(source["catalog_activation_ms"]) / 1000).date().isoformat()
        else:
            public = source.get("published_date")
    elif path.suffix == ".json":
        data = read_json(path)["data"]
        if data["art_code"] != report["source"]["ID"] or report["fund_code"] not in {
            s.get("stock") for s in data["security"]
        }:
            raise ValueError("REPORT_REPRINT_IDENTITY")
        pages = [data["notice_content"]]
        public = data["notice_date"][:10]
    else:
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        pages = [soup.select_one("#sohu_content").get_text("\n", strip=True)]
        title = soup.select_one("h1").get_text(" ", strip=True)
        stamp = soup.select_one(".article_info > .txt .c")
        match = re.search(r"20\d{2}-\d{2}-\d{2}", stamp.text if stamp else "")
        if not match or digest({"title": title, "text": pages[0], "published": match[0]}) != raw["content_hash"]:
            raise ValueError("REPORT_REPRINT_HASH_OR_DATE")
        public = match[0]
    if len("\n".join(pages)) < 1500:
        raise ValueError("REPORT_BODY_INCOMPLETE")
    return pages, public, modified


def review_reports(candidates, funds):
    needed = {r["report_sha256"] for r in candidates}
    reports = {r["raw"]["sha256"]: r for f in funds.values() for r in f["reports"] if r["raw"]["sha256"] in needed}
    reviews = {}
    for sha, report in sorted(reports.items()):
        destination = OUT / "reports" / (sha + ".json")
        if destination.exists():
            reviews[sha] = read_json(destination)
            continue
        item = {"fund_code": report["fund_code"], "sha256": sha, "report_end": report["report_end"]}
        try:
            pages, public, modified = report_pages(report)
            replay = parse_text(
                pages,
                report["title"],
                fund_code=report["fund_code"],
                fund_name=PEERS.get(report["fund_code"], "德邦鑫星价值"),
                master_code=report["fund_master_code"],
                derive_missing_weights=report.get("parser_version") == "DBFUND_EXTENDED_HOLDINGS_V1",
            )
            differences = [k for k in BUSINESS_FIELDS if replay[k] != report[k]]
            if differences:
                raise ValueError("REPORT_REPLAY_MISMATCH:" + ",".join(differences))
            version = report_date_check(public_day(report), public, modified)
            item.update(
                passed=True,
                version_publication=version,
                table_replay_equal=True,
                pdf_modified=modified,
                independent_historical_capture_verified=False,
                public_date_basis="ORIGINAL_SOURCE_DECLARATION",
            )
        except Exception as exc:
            item.update(passed=False, reason=type(exc).__name__ + ":" + str(exc))
        save_once(destination, item)
        reviews[sha] = item
        print(json.dumps(item, ensure_ascii=False), flush=True)
    save_once(OUT / "report-admission.json", list(reviews.values()))
    return reviews, reports


def database_check(required):
    """一次有界只读查询；不输出连接地址、参数、凭据或原始异常。"""
    path = OUT / "database-review.json"
    if path.exists():
        return read_json(path)
    from sqlalchemy import bindparam, text

    from app.db.session import get_nav_preview_engine

    result = {"checked_at": datetime.now().astimezone().isoformat(), "writes": 0}
    try:
        with get_nav_preview_engine().connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION READ ONLY"))
            source = (
                connection.execute(
                    text(
                        "SELECT source_id,source_code,license_scope,enabled,retention_days,"
                        "authorized_api_names,authorization_verified_at FROM source_registry "
                        "WHERE source_code='TUSHARE_PRO_FUND'"
                    )
                )
                .mappings()
                .one()
            )
            result["source"] = {
                k: str(v) if isinstance(v, datetime) or k == "source_id" else v for k, v in source.items()
            }
            result["rows"] = {}
            query = text(
                "SELECT nav_date,unit_nav,ann_date,content_hash,source_published_at,created_at,updated_at "
                "FROM nav_daily WHERE source_id=:source AND fund_code=:code AND nav_date IN :dates"
            ).bindparams(bindparam("dates", expanding=True))
            for code, days in required.items():
                rows = connection.execute(
                    query,
                    {
                        "source": source["source_id"],
                        "code": code,
                        "dates": [date.fromisoformat(d) for d in sorted(days)],
                    },
                ).mappings()
                result["rows"][code] = [{k: str(v) if v is not None else None for k, v in row.items()} for row in rows]
            result["status"] = "VERIFIED_CURRENT_READ_ONLY"
    except Exception as exc:
        result.update(status="UNAVAILABLE", error_type=type(exc).__name__, current_database_verified=False)
    save_once(path, result)
    return result


def public_crosscheck(required):
    """最多 200 页免费净值读取；校验完整分页和重复日，保留当前值与历史时间的区别。"""
    folder = OUT / "public-nav"
    save_once(
        folder / "protocol.json",
        {
            "fund_dates": {c: sorted(d) for c, d in required.items()},
            "maximum_requests": 200,
            "provider": "EASTMONEY_PUBLIC_NAV",
            "purpose": "VALUE_CROSSCHECK_ONLY",
            "automatic_retries": 0,
        },
    )
    result = {}
    with httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False) as client:
        for code, days in required.items():
            collected, total, page = {}, None, 1
            while total is None or len(collected) < total:
                key = f"{code}-{page:03d}"
                receipt_path = folder / (key + "-receipt.json")
                if receipt_path.exists():
                    receipt = read_json(receipt_path)
                else:
                    request_path = folder / (key + "-request.json")
                    if request_path.exists():
                        raise ValueError("INCOMPLETE_REQUEST_NO_AUTORETRY:" + key)
                    if len(list(folder.glob("*-request.json"))) >= 200:
                        raise ValueError("PUBLIC_REQUEST_BUDGET_EXHAUSTED")
                    params = {
                        "fundCode": code,
                        "pageIndex": page,
                        "pageSize": 20,
                        "startDate": min(days),
                        "endDate": max(days),
                    }
                    save_once(
                        request_path,
                        {
                            "url": "https://api.fund.eastmoney.com/f10/lsjz",
                            "params": params,
                            "started_at": datetime.now().astimezone().isoformat(),
                        },
                    )
                    try:
                        response = client.get(
                            "https://api.fund.eastmoney.com/f10/lsjz",
                            params=params,
                            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"},
                        )
                        response.raise_for_status()
                        if len(response.content) > 4_000_000:
                            raise ValueError("NAV_RESPONSE_TOO_LARGE")
                        raw_path = folder / (key + "-response.json")
                        with raw_path.open("xb") as stream:
                            stream.write(response.content)
                        receipt = {"passed": True, "path": str(raw_path), "sha256": file_hash(raw_path)}
                    except Exception as exc:
                        receipt = {"passed": False, "error_type": type(exc).__name__}
                    save_once(receipt_path, receipt)
                    time.sleep(0.2)
                if not receipt["passed"]:
                    raise ValueError("NAV_PUBLIC_SOURCE_FAILED:" + key)
                if file_hash(receipt["path"]) != receipt["sha256"]:
                    raise ValueError("NAV_PUBLIC_RESPONSE_CHANGED")
                data = read_json(receipt["path"])
                if int(data["PageIndex"]) != page or (total is not None and total != int(data["TotalCount"])):
                    raise ValueError("NAV_PAGE_IDENTITY_CHANGED")
                total = int(data["TotalCount"])
                rows = data["Data"]["LSJZList"]
                if not rows or len(rows) > 20 or total > 1500:
                    raise ValueError("NAV_PAGE_EMPTY_OR_LIMIT")
                for item in rows:
                    day = item["FSRQ"]
                    if day in collected or not min(days) <= day <= max(days):
                        raise ValueError("NAV_PUBLIC_DUPLICATE_OR_SCOPE")
                    collected[day] = item["DWJZ"]
                page += 1
            if len(collected) != total or not days <= collected.keys():
                raise ValueError("NAV_PUBLIC_COVERAGE_INCOMPLETE")
            result[code] = {d: collected[d] for d in sorted(days)}
            print(f"净值来源已核：{code}，{len(days)} 条，{page - 1} 页", flush=True)
    save_once(OUT / "public-nav-values.json", result)
    return result


def prepare():
    """完成报告、净值、行情、标签复核后冻结 H1 数据包，绝不直接拟合。"""
    OUT.mkdir(parents=True, exist_ok=True)
    raw, candidates, funds, indices, calendar, required = source_data()
    reviews, reports = review_reports(candidates, funds)
    db = database_check(required)
    public = public_crosscheck(required)
    dbrows = {(c, r["nav_date"]): r for c, rows in db.get("rows", {}).items() for r in rows}
    nav_checks = {}
    for code, days in required.items():
        for day in sorted(days):
            frozen = funds[code]["nav"][day]
            check = {"fund_code": code, "date": day}
            try:
                version = publication(frozen)
                if Decimal(str(frozen["nav"])) != Decimal(str(public[code][day])):
                    raise ValueError("PUBLIC_NAV_VALUE_CONFLICT")
                if db["status"] == "VERIFIED_CURRENT_READ_ONLY":
                    current = dbrows.get((code, day))
                    if (
                        not current
                        or Decimal(current["unit_nav"]) != Decimal(str(frozen["nav"]))
                        or current["ann_date"] != frozen["ann_date"]
                        or current["content_hash"] != frozen["source_hash"]
                    ):
                        raise ValueError("DATABASE_NAV_CONFLICT")
                check.update(passed=True, version_publication=version, source_hash=frozen["source_hash"])
            except ValueError as exc:
                check.update(passed=False, reason=str(exc))
            nav_checks[code, day] = check
    save_once(OUT / "nav-admission.json", list(nav_checks.values()))
    # 行情只从上一轮冻结原文重建，不重复下载或重跑旧权重分析。
    for path, sha in read_json(PREVIOUS / "quote-protocol.json")["sources"].items():
        if file_hash(path) != sha:
            raise ValueError("FROZEN_QUOTE_CHANGED")
    old = json_subtree(raw, ["payload", "older_quotes"])
    cache = {}

    def quotes(day):
        if day not in cache:
            if day < "2021-01-01":
                cache[day] = old["days"][day]
            else:
                wrapper = read_json(ROOT / "stock-days" / (day + ".json"))
                body = read_json(wrapper["receipt"]["raw_path"])["data"]
                rows = [dict(zip(body["fields"], v, strict=True)) for v in body["items"]]
                cache[day] = {"rows": {r["ts_code"]: r for r in rows}}
        return cache[day]

    admitted, row_checks = [], []
    for item in candidates:
        code, target = item["fund_code"], item["target"]
        check = {"fund_code": code, "target": target}
        try:
            review = reviews[item["report_sha256"]]
            if not review["passed"] or review["version_publication"] >= target:
                raise ValueError("REPORT_ADMISSION_FAILED")
            mapping = funds[code]["nav"]
            window = nav_window(mapping, calendar, target)
            dependencies = window["nav_dates"] + [item["base"], target]
            if any(not nav_checks[code, d]["passed"] for d in dependencies):
                raise ValueError("NAV_ADMISSION_FAILED")
            if any(nav_checks[code, d]["version_publication"] >= target for d in window["nav_dates"]):
                raise ValueError("INPUT_NAV_NOT_PUBLIC")
            x = window["x"] + exposure(reports[item["report_sha256"]], quotes, indices, item["quote_days"])
            if digest(x) != item["input_vector_sha256"]:
                raise ValueError("PREVIOUS_INPUT_REPLAY_CHANGED")
            row = {
                "fund_code": code,
                "family": funds[code]["family"],
                **label(mapping[item["base"]], mapping[target], calendar),
                "nav": {k: v for k, v in window.items() if k != "x"},
                "quote_dates": item["quote_days"],
                "report_end": item["report_end"],
                "report_publication": item["publication"],
                "report_sha256": item["report_sha256"],
                "x": x,
            }
            if not eligible_at(row, "2023-04-01"):
                raise ValueError("LABEL_NOT_MATURE_BEFORE_Q2")
            admitted.append(row)
            check.update(passed=True, row_sha256=digest(row))
        except ValueError as exc:
            check.update(passed=False, reason=str(exc))
        row_checks.append(check)
    base = read_json(BASE / "inputs.json")
    pool = sorted(base["train"] + admitted, key=lambda r: (r["target"], r["fund_code"]))
    if len({(r["fund_code"], r["target"]) for r in pool}) != len(pool):
        raise ValueError("NEW_ROW_IDENTITY_DUPLICATE")
    family = {r["fund_code"]: r["family"] for r in pool}
    folds = []
    for original in read_json(BASE / "folds.json"):
        train = [r for r in pool if eligible_at(r, original["start"])]
        folds.append(
            {
                **original,
                "train_ids": [[r["fund_code"], r["target"]] for r in train],
                "train_sha256": digest(train),
                "weights": weights(train),
                "gate": gate(train, family),
            }
        )
    outputs = {
        "row-admission.json": row_checks,
        "added-rows.json": admitted,
        "inputs.json": {"train": pool, "development": base["development"]},
        "folds.json": folds,
        "decision.json": {
            "material_ready": bool(admitted) and all(f["gate"]["passed"] for f in folds),
            "candidates": len(candidates),
            "admitted": len(admitted),
            "blocked": len(candidates) - len(admitted),
            "pool_rows": len(pool),
            "new_fits": 0,
            "database_status": db["status"],
            "by_fund": dict(Counter(r["fund_code"] for r in admitted)),
            "blocked_reasons": dict(Counter(r["reason"] for r in row_checks if not r["passed"])),
        },
    }
    for name, value in outputs.items():
        save_once(OUT / name, value)
    return outputs["decision.json"]
