"""按原作业的基金、字段和所需日期继续取件，逐只保存证据。"""

import re
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from time import monotonic

import httpx

from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_daily_budget import assert_daily_budget
from app.services.fund_daily_prediction import FUND_INPUT_ERRORS, record_fund_failure, supplement_run_id
from app.services.fund_daily_profiles import VERIFIED_SOURCE_RESULTS, save_fund_checklist, save_pending
from app.services.fund_exposure_common import PROJECT, ROOT, blob, now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import source_bytes, source_exists


def task_key(row):
    field = row["field"]
    if row["fund_code"] == "008764" and field == "FX":
        field = "FX:VND/USD"
    return row["fund_code"], field, row.get("required_date")


def verified(source, summary, evidence, *, new=False):
    return {
        "status": "NEW_VERIFIED" if new else "REUSED_VERIFIED",
        "attempted_source": source, "summary": summary, "observations": evidence,
    }


def pending(status, source, summary, evidence=None, next_source=None):
    return {
        "status": status, "attempted_source": source, "summary": summary,
        "observations": evidence, "next_source": next_source,
    }


def item_date(value):
    return str(value or "")[:10]


def record_supplement_summary(directory, state, result):
    """独立补件只更新原作业的补件部分和原工作单。"""
    state["supplements"] = result
    if state.get("supplement_run_id"):
        state["materials_terminal"] = {
            "mode": "FUND_CHECKLIST", "job_id": result["job_id"], "terminal": True, "status": result["status"],
        }
        versioned_save(directory / "materials-status.json", state["materials_terminal"])
    if result.get("unattempted_item_count"):
        state["status"] = "SUPPLEMENT_EXECUTION_INCOMPLETE"
    versioned_save(directory / "run.json", state)
    path = directory / "work-items.json"
    if path.is_file():
        work = read(path)
        if work.get("window") == state["window"] and work.get("owner_user_id") == state["owner_user_id"]:
            work["supplements"] = result
            work["remaining_funds"] = sorted(set(work.get("remaining_funds", [])) | set(result["pending_funds"]))
            versioned_save(path, work)


def funding_evidence(owner, at):
    """按正式账户的最后确认或撤销读回，回执只存摘要及期限。"""
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    from app.commands.holding_advice_inventory import core_engine
    from app.services.direction_1d_protocol import ZONE

    source = "本人已确认的资金安排及持有偏好账本"
    next_source = "从我的持仓读取本人最后确认或撤销记录；安排到期或缺少确认时由本人确认后再核。"
    engine = core_engine()
    try:
        with engine.connect() as connection:
            funding = connection.execute(text(
                "SELECT status,payload,confirmed_at FROM account_funding_preference WHERE user_id=:owner "
                "ORDER BY confirmed_at DESC,revision DESC,scope DESC,preference_id DESC LIMIT 1"
            ), {"owner": owner}).mappings().one_or_none()
            preference = connection.execute(text(
                "SELECT preference FROM user_strategy_preference WHERE user_id=:owner"
            ), {"owner": owner}).scalar_one_or_none()
    except SQLAlchemyError:
        return pending("TEMPORARY_FAILURE", source, "本次未能读回本人资金安排及持有偏好账本。",
                       next_source=next_source)
    finally:
        engine.dispose()
    if not funding or funding["status"] != "ACTIVE" or funding["confirmed_at"] > at:
        return pending("ACCESS_REQUIRED", source, "本次读回没有当前生效的已确认资金安排。",
                       {"status": funding["status"] if funding else "MISSING"}, next_source)
    payload = funding["payload"] or {}
    details = {"confirmed_at": funding["confirmed_at"].isoformat(), "funding_version": digest(payload)}
    try:
        amount = Decimal(str(payload["requiredAmount"]))
        affordable = Decimal(str(payload["affordableLossAmount"]))
        loss = Decimal(str(payload["willingLossPct"]))
        use = payload.get("useDate")
        use_day = date(*use) if isinstance(use, list) else date.fromisoformat(use) if use else None
        interval = payload.get("reviewIntervalDays")
        if interval is not None:
            review_day = funding["confirmed_at"].astimezone(ZONE).date() + timedelta(days=int(interval))
            details["review_due"] = review_day.isoformat()
            if int(interval) < 1 or review_day <= at.astimezone(ZONE).date():
                return pending("ACCESS_REQUIRED", source, "当前资金安排已到复查日或复查周期无效。", details, next_source)
        if (all(n.is_finite() for n in (amount, affordable, loss)) and amount >= 0 and affordable >= 0
                and 0 <= loss <= 100 and (amount == 0 or use_day)
                and preference in {"SHORT", "BALANCED", "LONG"}):
            return verified(source, "已核对最后确认版本、持有偏好、用款及可承受损失字段和复查期限。", details)
    except (InvalidOperation, TypeError, ValueError, KeyError):
        pass
    return pending("ACCESS_REQUIRED", source, "资金安排或持有偏好缺少适用字段，需从正式账本核实有效确认。",
                   details, next_source)


class DailySupplements:
    def __init__(self, directory, state, profiles):
        self.root, self.window, self.profiles = directory, state["window"], profiles
        self.day = item_date(state["checked_at"])
        self.scope = state["scope"]
        self.job = supplement_run_id(state)
        self.queue_path = directory / "supplement-tasks.json"
        self.source_path = directory / "source-rechecks.json"
        self.catalog_path = directory / "supplement-catalog.json"
        old_queue = read(self.queue_path) if self.queue_path.is_file() else {}
        self.tasks = {}
        old_catalog = read(self.catalog_path) if self.catalog_path.is_file() else {}
        if old_catalog.get("window") == self.window and old_catalog.get("job_id") == self.job:
            self.remember(old_catalog["tasks"])
        if old_queue.get("window") == self.window and old_queue.get("job_id", self.job) == self.job:
            self.remember(old_queue["tasks"])
        old_source = read(self.source_path) if self.source_path.is_file() else {}
        self.previous = {}
        if (old_source.get("job_id", self.job) == self.job
                and all(old_source.get(k) == v for k, v in self.window_fields().items())):
            self.previous = {
                task_key(row): row for row in old_source["items"]
                if "required_date" in row and item_date(row.get("attempted_at")) == self.day
            }
        self.read_proof = lru_cache(maxsize=80)(read)
        self.raw_matches = lru_cache(maxsize=3000)(self.check_raw)
        self.requests = {}
        self.results = {}
        self.storage_checked = float("-inf")
        self.started = now()
        self.since = self.started.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    def check_storage(self):
        if monotonic() - self.storage_checked >= 25:
            from app.services.prediction_trial_v2 import storage_check

            storage_check(self.since)
            self.storage_checked = monotonic()

    def window_fields(self):
        return {"day": self.day, **{k: self.window[k] for k in ("base_nav_date", "target_nav_date")}}

    def remember(self, rows):
        for row in rows:
            key = task_key(row)
            if key[0] in self.scope:
                fields = {k: v for k, v in row.items() if k not in {
                    "source_review", "item_alignment_status", "item_alignment_note", "item_alignment_evidence",
                }}
                self.tasks[key] = {**self.tasks.get(key, {}), **fields, "field": key[1]}

    def prepare(self, state):
        self.remember(save_pending(self.root, state, records=self.profiles, inventory=True))
        self.check_storage()
        value = {"window": self.window, "job_id": self.job, "tasks": list(self.tasks.values())}
        if not self.catalog_path.is_file() or read(self.catalog_path) != value:
            versioned_save(self.catalog_path, value)

    def original(self, value):
        path = Path(value).resolve()
        if not path.is_relative_to(PROJECT.parent) or not source_exists(path):
            return None
        return path

    def check_raw(self, path, sha):
        from hashlib import sha256

        original = self.original(path)
        return original is not None and sha256(source_bytes(original)).hexdigest() == sha

    def material_for_check(self, code, item):
        from app.services.fund_workflow_materials import receipt_path

        if item.get("materials_path") and (path := self.original(item["materials_path"])):
            return path, read(path)
        handoff_path = receipt_path(code)
        if not handoff_path.is_file():
            return None, {}
        handoff = read(handoff_path)
        path = self.original(handoff["materials_path"])
        if not path:
            return None, {}
        payload = read(path)
        if handoff.get("fund_code") != code or digest(payload) != handoff.get("materials_hash"):
            raise ValueError("DAILY_SUPPLEMENT_HANDOFF_INVALID")
        return path, payload

    def document(self, doc):
        if not doc or not doc.get("body") or digest(doc["body"]) != doc.get("body_hash"):
            return None
        path, sha = doc.get("raw_path"), doc.get("raw_sha256")
        if not path or not sha or not self.raw_matches(path, sha):
            return None
        return {k: doc.get(k) for k in ("id", "url", "raw_path", "raw_sha256", "published_date", "available_at")}

    def previous_proof(self, key):
        receipt = self.previous.get(key, {})
        if receipt.get("status") not in VERIFIED_SOURCE_RESULTS or not receipt.get("evidence_ref"):
            return None
        tracking = (receipt.get("observations") or {}).get("benchmark_tracking") or {}
        if key[1].startswith("prediction_target_") and tracking.get("difference_pct") is None:
            # 场内折溢价与相对合同基准的跟踪偏离不同。前者齐备也要继续查后者。
            return None
        path = self.original(receipt["evidence_ref"])
        if not path:
            return None
        try:
            data = self.read_proof(path)
        except (ValueError, KeyError, TypeError):
            return None
        if (not isinstance(data, dict) or data.get("fund_code") != key[0] or data.get("window") != self.window
                or not any((r["field"], r.get("required_date")) == key[1:] for r in data.get("items", []))):
            return None
        return receipt

    def request_json(self, url, params):
        key = (url, tuple(sorted(params.items())))
        if key not in self.requests:
            at = now().isoformat()
            try:
                with httpx.Client(timeout=httpx.Timeout(10, connect=4), trust_env=False) as client:
                    response = client.get(url, params=params, headers={
                        "User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/",
                    })
                response.raise_for_status()
                if len(response.content) > 1_000_000:
                    raise ValueError("PUBLIC_DATA_RESPONSE_TOO_LARGE")
                data = response.json()
            except (httpx.HTTPError, ValueError) as error:
                self.requests[key] = (None, {"at": at, "error": type(error).__name__})
            else:
                sha, path = blob(response.content, "json")
                self.requests[key] = (data, {
                    "at": at, "raw_path": str(ROOT / path), "sha256": sha, "params": params,
                    "http_status": response.status_code,
                })
        return self.requests[key]

    def nav(self, code, day, item):
        from app.services.fund_daily_nav import refresh_missing_nav

        attempt = item.get("base_nav_refresh") or {}
        key = ("nav", code, day)
        if key not in self.requests:
            fresh = (attempt.get("attempted_at", "") >= self.started.isoformat()
                     and attempt.get("base_nav_date") == day)
            self.requests[key] = attempt if fresh else refresh_missing_nav(
                code, day, self.root / "supplement-nav" / code / day
            )
        attempt = self.requests[key]
        if attempt.get("status") in {"AVAILABLE", "BASE_NAV_PRESENT"} and attempt.get("nav_input_hash"):
            return verified("正式净值库与 Tushare fund_nav", f"{code} 本份额 {day} 正式净值已读回。",
                            attempt, new=attempt["status"] == "AVAILABLE")
        source = "https://api.fund.eastmoney.com/f10/lsjz"
        payload, proof = self.request_json(source, {
            "fundCode": code, "pageIndex": "1", "pageSize": "5", "startDate": day, "endDate": day,
        })
        proof["formal_attempt"] = attempt
        if not isinstance(payload, dict) or payload.get("ErrCode") != 0:
            return pending("TEMPORARY_FAILURE", source, "正式净值未取得，备用按日查询也未返回有效结果。", proof)
        rows = (payload.get("Data") or {}).get("LSJZList") or []
        try:
            exact = [
                str(Decimal(row["DWJZ"])) for row in rows
                if row.get("FSRQ") == day and Decimal(row["DWJZ"]).is_finite() and Decimal(row["DWJZ"]) > 0
            ]
        except (InvalidOperation, KeyError, TypeError):
            return pending("MISMATCH", source, "备用渠道返回了无法核实的净值格式。", proof)
        proof["exact_rows"] = exact
        detail = ("代销渠道已出现同日数值，需与管理人或正式库核对，未作正式入库。"
                  if exact else "代销渠道当日查询返回零条；尚无本份额正式净值。")
        return pending("REVIEW_REQUIRED" if exact else "TEMPORARY_FAILURE", source, detail, proof,
                       f"下一次查 {code} 的管理人正式披露及 Tushare fund_nav 的 {day} 净值。")

    def trading_conditions(self, code, news, docs):
        source = f"https://fundf10.eastmoney.com/jjfl_{code}.html"
        from bs4 import BeautifulSoup

        from app.services.eastmoney_fee import fetch_fee_profile

        channel = {}

        def observe(response):
            soup = BeautifulSoup(response.text, "html.parser")
            labels = {
                "申购状态", "赎回状态", "申购起点", "日累计申购限额", "持仓上限", "最小赎回份额",
                "部分赎回最低保留份额", "赎回到账时间", "赎回款到账时间",
            }
            values = {}
            for table_row in soup.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in table_row.find_all(["th", "td"], recursive=False)]
                for label, value in zip(cells[::2], cells[1::2]):
                    if label in labels and value:
                        values.setdefault(label, set()).add(value)
            sha, path = blob(response.content, "html")
            channel.update(at=now().isoformat(), raw_path=str(ROOT / path), raw_sha256=sha,
                           values={label: next(iter(items)) for label, items in values.items() if len(items) == 1})

        try:
            profile = fetch_fee_profile(code, observe=observe)
            fees = {
                "purchase_rate": str(profile.purchase_original_rate),
                "redeem_bands": [
                    {"min_days": band.min_days, "max_days": band.max_days, "rate": str(band.rate)}
                    for band in profile.redeem_bands
                ],
            }
        except ValueError as error:
            fees = None
            channel["failure"] = str(error).split(":", 1)[0]
        published = [
            {"id": doc["id"], "title": doc["title"], "date": doc.get("published_date"),
             "evidence": proof}
            for doc in docs.values() if code in doc.get("codes", [])
            and any(word in doc.get("title", "") for word in ("申购", "赎回", "资料概要", "招募说明书"))
            and (proof := self.document(doc))
        ]
        return pending(
            "REVIEW_REQUIRED", source + "；本份额已登记公告、合同及产品原件",
            ("已读回本份额申购和赎回费率分档。" if fees else "费率页未返回可核实的本份额完整分档。")
            + ("已保存代销渠道展示的申赎状态和限额。" if channel.get("values") else "未读取到可核对的渠道状态表。")
            + f"找到 {len(published)} 份真实原件可以继续核对。"
            + "仍需以适用渠道和正式公告核实交易限制及到账日。",
            {"fees": fees, "distributor": channel, "coverage": news.get("coverage"), "originals": published},
            "核对本份额当前生效的管理人申赎公告、产品资料概要及实际办理渠道的限额、收费分档和到账规则。",
        )

    def reference_index(self, symbol, day):
        from app.services.fund_exposure_quotes import Provider
        from app.services.fund_prediction_market_recovery import FIELDS, quote_row

        if not re.fullmatch(r"[A-Z0-9]{6}\.(?:SH|CSI|SI)", symbol):
            raise ValueError("INVALID_REGISTERED_REFERENCE_INDEX")
        source = "已登记权限的 Tushare index_daily"
        next_source = f"向指数发布机构或授权渠道取得 {symbol} 在 {day} 的正式收盘原件，核对是否有补发或修正。"
        key = ("reference-index", symbol, day)
        if key not in self.requests:
            if not hasattr(self, "market_provider"):
                self.market_provider = Provider()
            payload, source_receipt = self.market_provider.query(
                "index_daily", {"ts_code": symbol, "trade_date": day.replace("-", "")}, FIELDS
            )
            proof = {**source_receipt, "raw_path": str(ROOT / source_receipt["file"])}
            if not self.raw_matches(proof["raw_path"], proof["sha256"]):
                self.requests[key] = pending("MISMATCH", source, "指数响应原件摘要不一致。", proof, next_source)
            else:
                rows = (payload.get("data") or {}).get("items") or []
                if not rows:
                    self.requests[key] = pending(
                        "TEMPORARY_FAILURE", source, f"{symbol} {day} 的指定查询返回零条，仍缺正式收盘点位。",
                        proof, next_source,
                    )
                else:
                    row = quote_row(payload, symbol, day, "index_daily")
                    self.requests[key] = verified(source, f"{symbol} {day} 正式收盘 {row['close']}。",
                                                  {"receipt": proof, "row": row})
        return self.requests[key]

    def latest_report(self, code, *, existing=None):
        """公告缺项不能挡住独立的本份额定期报告查询。"""
        from app.core.config import get_settings
        from app.integrations.market_fund_reports import MarketReportClient, read_source

        key = ("latest-report", code)
        if key in self.requests:
            return self.requests[key]
        directory = Path(get_settings().fund_material_directory)
        root = directory / "market-maintenance"
        checkpoint = root / "funds" / (code + ".json")
        old = read(checkpoint) if checkpoint.is_file() else {}
        client = MarketReportClient(root)
        try:
            catalog = [row for row in client.catalog(code) if row["published"] <= self.day]
            if not catalog:
                result = pending("TEMPORARY_FAILURE", "本份额定期报告目录", "本次目录返回零份可识别的定期报告。")
            else:
                latest = max(catalog, key=lambda row: (row["end"], row["published"], row["id"]))
                report, proof = existing or ({}, {})
                if (report.get("sourceArticleId") == latest["id"] and report["endDate"] == latest["end"]
                        and report["publishedDate"] == latest["published"]):
                    result = verified(proof["url"], f"目录最新报告期 {latest['end']}，与已有原件和持仓版本一致。",
                                      {"catalog_entry": latest, "original": proof, "report_id": report["id"]})
                else:
                    candidate = old.get("entries", {}).get(latest["id"], {})
                    old_ref = candidate if candidate.get("fingerprint") == digest(latest) else None
                    source = client.body(code, latest, old_ref)
                    read_source(root, source)
                    evidence = {"report": latest, "source": source, "raw_path": str(root / source["file"])}
                    result = self.confirm_report(code, latest, source, evidence, client)
        finally:
            client.close()
        self.requests[key] = result
        return result

    def official_shibor(self, day):
        from app.services.fund_daily_source_routes import SHIBOR, shibor_publication_time, shibor_rows

        base = self.window["base_nav_date"]
        publication = shibor_publication_time(day)
        next_source = f"从中国货币网历史报价核对 {base} 和 {day} 的隔夜、七天利率。"
        if now() < publication:
            result = pending("NOT_PUBLISHED", SHIBOR, "已核对官方公布时点，当前尚未到北京时间 11:00。",
                             {"publication_time": publication.isoformat()}, next_source)
            result["retry_at"] = publication.isoformat()
            return result
        required = {base, day}
        payload, proof = self.request_json(SHIBOR, {"startDate": min(required), "endDate": max(required)})
        if payload is None:
            return pending("TEMPORARY_FAILURE", SHIBOR, "中国货币网未返回可读取的同业利率响应。", proof, next_source)
        points = shibor_rows(payload, required)
        evidence = {"official_response": proof, "required_dates": sorted(required), "rates_pct": points}
        if set(points) != required:
            return pending("TEMPORARY_FAILURE", SHIBOR, f"官方响应缺少 {', '.join(sorted(required - set(points)))}。",
                           evidence, next_source)
        return verified(SHIBOR, f"已取得 {base} 和 {day} 的官方隔夜与七天利率，已核对日期和期限。",
                        evidence, new=day == self.day)

    def same_time_bonds(self, day):
        from datetime import datetime

        from app.services.direction_1d_protocol import ZONE
        from app.services.fund_daily_source_routes import BOND_TYPES, CFETS_BOND_QUOTES, cfets_bond_rows

        base = self.window["base_nav_date"]
        stamp = "11:20"
        planned = datetime.fromisoformat(f"{day}T{stamp}").replace(tzinfo=ZONE)
        if now() < planned:
            result = pending("NOT_PUBLISHED", CFETS_BOND_QUOTES, "本轮选定的同一时点报价尚未发生。")
            result["retry_at"] = planned.isoformat()
            return result
        groups, gaps, evidence = {}, [], {}
        for family, name in BOND_TYPES.items():
            by_day = {}
            for required in (base, day):
                date_label = date.fromisoformat(required).strftime("%d %b %Y")
                data, raw = self.request_json(
                    CFETS_BOND_QUOTES, {"lang": "EN", "bondType": family, "qttndate": date_label, "qttntime": stamp},
                )
                evidence[f"{family}/{required}"] = raw
                try:
                    by_day[required] = cfets_bond_rows(data, date_label, stamp)
                except (ValueError, KeyError, TypeError, ArithmeticError):
                    by_day[required] = {}
            old, new = by_day[base], by_day[day]
            matched = {code: {"name": new[code]["name"], "base_yield_pct": old[code]["yield_pct"],
                              "target_yield_pct": new[code]["yield_pct"],
                              "difference_bps": str((Decimal(new[code]["yield_pct"]) -
                                                     Decimal(old[code]["yield_pct"])) * 100)}
                       for code in sorted(old.keys() & new.keys()) if old[code]["name"] == new[code]["name"]}
            groups[name] = matched
            if not matched:
                gaps.append(name)
        proof = {"date_pair": [base, day], "time_cst": stamp, "matched": groups, "originals": evidence}
        if gaps:
            return pending("REVIEW_REQUIRED", CFETS_BOND_QUOTES, f"两日同一时点已查询，仍缺 {', '.join(gaps)} 同券。", proof)
        return verified(CFETS_BOND_QUOTES, f"已取得 {base} 与 {day} 同为 {stamp} 的四类现券，同券逐一对照。",
                        proof, new=day == self.day)

    def repo_trades(self, day):
        from app.services.fund_daily_source_routes import CFETS_REPO_TRADES, cfets_repo_rows

        payload, raw = self.request_json(CFETS_REPO_TRADES, {})
        try:
            observed, rates = cfets_repo_rows(payload, day, now())
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            return pending("TEMPORARY_FAILURE", CFETS_REPO_TRADES,
                           f"{day} 的银行间成交加权利率及期限没有通过日期或数据校验。",
                           {"receipt": raw, "validation": str(error)[:110]})
        proof = {"observed_at": observed, "rates": rates, "original": raw}
        if not rates:
            return pending("TEMPORARY_FAILURE", CFETS_REPO_TRADES, "当天页面没有披露可核的实际加权利率和平均期限。", proof)
        n = sum(row["covers_month_start"] for row in rates.values())
        return verified(CFETS_REPO_TRADES,
                        f"已核 {day} 的 {len(rates)} 个成交产品及实际平均期限，其中 {n} 个期限覆盖下月初。",
                        proof, new=day == self.day)

    def open_market_operations(self, code, task, day, docs, sessions):
        from urllib.parse import urljoin, urlsplit

        from bs4 import BeautifulSoup

        source = "中国人民银行公开市场业务交易公告"
        directory = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html"
        publication_path = re.compile(r"/zhengcehuobisi/125207/125213/125431/125475/(\d{8})\d{11}/index\.html")

        def announcement_date(url):
            parsed = urlsplit(url)
            match = publication_path.fullmatch(parsed.path)
            if parsed.scheme != "https" or parsed.netloc != "www.pbc.gov.cn" or not match or parsed.query or parsed.fragment:
                return None
            return date.fromisoformat(match[1])

        day_date = date.fromisoformat(day)
        previous = max((item for item in sessions if item < day_date), default=None)
        week_prior = day_date - timedelta(days=7)
        relevant = {d.isoformat() for d in (day_date, previous, week_prior) if d is not None}
        signed = {}
        for doc in docs.values():
            if "aligned_bond_pbc_operations" not in doc["id"] or doc.get("published_date") != day:
                continue
            if original := self.document(doc):
                actual = read(Path(original["raw_path"]))
                if ((actual.get("fund_code"), actual.get("field"), actual.get("required_date"))
                        == (code, task["field"], day)):
                    signed = actual
                    break
        legs, raw, directory_proof = [], {}, None
        refs = signed.get("references") or []
        try:
            with httpx.Client(timeout=httpx.Timeout(12, connect=5), trust_env=False) as client:
                if not refs:
                    listed = client.get(directory, headers={"User-Agent": "Mozilla/5.0"})
                    listed.raise_for_status()
                    if len(listed.content) > 1_000_000:
                        raise ValueError("PBC_OPERATIONS_DIRECTORY_SIZE")
                    sha, saved = blob(listed.content, "html")
                    directory_proof = {"sha256": sha, "path": str(ROOT / saved), "url": directory}
                    for link in BeautifulSoup(listed.content, "html.parser").find_all("a", href=True):
                        if not re.search(r"公开市场业务交易公告\s*\[\d{4}\]\s*第\d+号",
                                         link.get_text(" ", strip=True)):
                            continue
                        url = urljoin(directory, link["href"])
                        listed_on = announcement_date(url)
                        if listed_on and listed_on.isoformat() in relevant and url not in refs:
                            refs.append(url)
                for url in refs:
                    published = announcement_date(url)
                    if published is None:
                        raise ValueError("PBC_OPERATIONS_REFERENCE_IDENTITY")
                    if published.isoformat() not in relevant or published.isoformat() in raw:
                        raise ValueError("PBC_OPERATIONS_REFERENCE_DATE")
                    response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    response.raise_for_status()
                    if len(response.content) > 1_000_000:
                        raise ValueError("PBC_OPERATIONS_RESPONSE_SIZE")
                    sha, saved = blob(response.content, "html")
                    raw[published.isoformat()] = {"sha256": sha, "path": str(ROOT / saved), "url": url}
                    text = re.sub(r"\s+", "", BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True))
                    if "中国人民银行公开市场业务操作室" not in text or "公开市场业务交易公告" not in text:
                        raise ValueError("PBC_OPERATIONS_TITLE")
                    operations = re.findall(r"开展了([\d,]+)亿元(隔夜|\d+天期)逆回购操作", text)
                    if not operations:
                        raise ValueError("PBC_OPERATIONS_VALUE_MISSING")
                    for amount, term in operations:
                        if term == "隔夜":
                            maturity = next((d for d in sessions if d > published), None)
                        else:
                            maturity = published + timedelta(days=int(term[:-2]))
                        legs.append({"published": published.isoformat(), "term": term,
                                     "amount": int(amount.replace(",", "")), "maturity": maturity.isoformat() if maturity else None})
        except (httpx.HTTPError, ValueError, TypeError) as error:
            return pending("TEMPORARY_FAILURE", source, "已查对应央行原公告，部分条目未能通过原件验收。",
                           {"directory": directory_proof, "originals": raw, "reason": type(error).__name__})
        if raw.keys() != relevant or not any(row["maturity"] == day for row in legs):
            return pending("REVIEW_REQUIRED", source, "原公告尚未覆盖全部指定日期或缺到期对应项。",
                           {"directory": directory_proof, "originals": raw, "legs": legs})
        supplied = sum(row["amount"] for row in legs if row["published"] == day)
        expired = sum(row["amount"] for row in legs if row["maturity"] == day)
        return verified(source, f"逐笔核实 {day} 逆回购投放 {supplied} 亿元、到期 {expired} 亿元，同口径净额 {supplied-expired} 亿元。",
                        {"directory": directory_proof, "originals": raw, "operations": legs, "supplied": supplied, "expired": expired,
                         "net": supplied-expired, "unit": "CNY_100M", "scope": "短期逆回购"})

    def provider_response(self, api, symbol, params, fields):
        from app.services.fund_exposure_quotes import Provider

        key = (api, symbol, tuple(sorted(params.items())), tuple(fields))
        if key not in self.requests:
            if not hasattr(self, "market_provider"):
                self.market_provider = Provider()
            payload, receipt = self.market_provider.query(api, {"ts_code": symbol, **params}, fields)
            proof = {**receipt, "raw_path": str(ROOT / receipt["file"])}
            if not self.raw_matches(proof["raw_path"], proof["sha256"]):
                raise ValueError("ETF_SOURCE_HASH")
            self.requests[key] = (payload, proof)
        return self.requests[key]

    def target_market(self, task, code, day, sessions, published=None):
        from app.services.fund_daily_source_routes import source_values, target_symbol
        from app.services.fund_prediction_market_recovery import FIELDS, quote_row

        symbol = target_symbol(self.profiles[code], task["target_fund"])
        raw_day = day.replace("-", "")
        key = task.get("check_key")
        source = "已授权 Tushare 基金净值、份额及交易所日收盘"
        evidence, gaps = {}, []

        def observe(name, api, params, fields, validate):
            proof = None
            try:
                payload, proof = self.provider_response(api, symbol, params, fields)
                data = validate(payload)
                evidence[name] = {"rows": data, "receipt": proof}
                return data
            except (ValueError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as error:
                evidence[name] = {"receipt": proof, "failure": type(error).__name__}
                return None

        if key != "gold_etf_base_premium":
            previous = max((d for d in sessions if d < date.fromisoformat(day)), default=None)
            if previous is None:
                raise ValueError("ETF_PREVIOUS_SESSION_NOT_VERIFIED")
            prior = previous.strftime("%Y%m%d")
            fields = ["ts_code", "trade_date", "fd_share"]
            shares = observe(
                "shares", "fund_share", {"start_date": prior, "end_date": raw_day}, fields,
                lambda payload: source_values(payload, fields, symbol, "trade_date", "fd_share", {prior, raw_day}),
            ) or {}
            evidence["shares"]["unit"] = "万份"
            if set(shares) == {prior, raw_day}:
                evidence["shares"]["change_ten_thousand"] = str(Decimal(shares[raw_day]) - Decimal(shares[prior]))
            else:
                gaps.append("缺基准日或前一实际交易日的正式份额")
        if key != "gold_etf_base_shares":
            fields = ["ts_code", "ann_date", "nav_date", "unit_nav"]

            def net_values(payload):
                nav = source_values(payload, fields, symbol, "nav_date", "unit_nav", {raw_day})
                posts = [r[1] for r in payload["data"]["items"]]
                return nav if posts and all(stamp and raw_day <= stamp <= self.day.replace("-", "")
                                            for stamp in posts) else {}

            nav = observe("nav", "fund_nav", {"start_date": raw_day, "end_date": raw_day}, fields, net_values) or {}
            quote = observe(
                "exchange", "fund_daily", {"trade_date": raw_day}, FIELDS,
                lambda payload: quote_row(payload, symbol, day, "fund_daily") if payload["data"]["items"] else None,
            )
            if raw_day in nav and quote:
                evidence["market_premium_pct"] = str((Decimal(str(quote["close"])) / Decimal(nav[raw_day]) - 1) * 100)
            else:
                gaps.append("缺同日正式披露净值或场内收盘")
        if key not in {"gold_etf_base_shares", "gold_etf_base_premium"}:
            tracking = self.target_tracking(code, task["target_fund"], symbol, prior, raw_day, published)
            if tracking:
                evidence["benchmark_tracking"] = tracking
                if tracking.get("difference_pct") is None:
                    gaps.append("已核正式基准身份，指定日的指数原始行情或底层复权净值仍缺")
            else:
                gaps.append("本份额原招募书中的标的指数尚未核对，不能确认跟踪偏离")
        if gaps:
            return pending("REVIEW_REQUIRED", source, f"{symbol} {day} 已按本条查询。{'；'.join(gaps)}。",
                           evidence, task.get("next_source"))
        label = {"gold_etf_base_shares": "份额变动", "gold_etf_base_premium": "场内折溢价"}.get(key, "份额、净值、场内行情及当期跟踪偏离")
        return verified(source, f"{symbol} {day} 对应日期的{label}已核实。", evidence)

    def target_tracking(self, code, target, symbol, prior, raw_day, published):
        from app.services.fund_daily_rating_sources import target_contract_index
        from app.services.fund_daily_source_routes import source_values
        from app.services.fund_prediction_market_recovery import FIELDS, quote_row

        try:
            benchmark = target_contract_index(published, code, target)
        except (ValueError, KeyError, TypeError, FileNotFoundError, httpx.HTTPError):
            return None
        if not benchmark:
            return None
        proof = {"contract_benchmark": benchmark, "difference_pct": None}
        fields = ["ts_code", "ann_date", "nav_date", "adj_nav"]
        try:
            payload, nav_receipt = self.provider_response(
                "fund_nav", symbol, {"start_date": prior, "end_date": raw_day}, fields
            )
            proof["nav_receipt"] = nav_receipt
            values = source_values(payload, fields, symbol, "nav_date", "adj_nav", {prior, raw_day})
            if (values.keys() == {prior, raw_day} and all(
                row[1] and row[2] <= row[1] <= self.day.replace("-", "") for row in payload["data"]["items"]
            )):
                proof["adjusted_nav"] = values
        except (ValueError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as error:
            proof["nav_failure"] = type(error).__name__
        try:
            index = benchmark["code"]
            payload, index_receipt = self.provider_response("index_daily", index, {"trade_date": raw_day}, FIELDS)
            proof["index_receipt"] = index_receipt
            if payload["data"]["items"]:
                quote = quote_row(payload, index, date.fromisoformat(raw_day).isoformat(), "index_daily")
                proof["index_quote"] = quote
        except (ValueError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as error:
            proof["index_failure"] = type(error).__name__
        if proof.get("adjusted_nav") and proof.get("index_quote"):
            values = proof["adjusted_nav"]
            actual = (Decimal(values[raw_day]) / Decimal(values[prior]) - 1) * 100
            proof["difference_pct"] = str(actual - Decimal(str(proof["index_quote"]["pct_chg"])))
        return proof

    def gold_close(self, code, day, material):
        from app.services.fund_prediction_gold_recovery import gold_rows, refresh

        if not material.get("facts", {}).get("current_gold"):
            return pending("NOT_ATTEMPTED", "上海黄金交易所",
                           "本份额资料缺少已登记的现货合约关联，未发起来源查询。",
                           next_source="先核对本份额合同关联的 Au99.99 合约，再查上海黄金交易所同日收盘。")
        key = ("sge-gold", code, day)
        if key not in self.requests:
            self.requests[key] = refresh(material, day, self.root / "supplement-gold" / code)
        receipt = self.requests[key]
        if receipt["status"] not in {"ACQUIRED", "CURRENT_SOURCE_RETAINED"}:
            return pending("TEMPORARY_FAILURE", "上海黄金交易所每日行情",
                           f"按 {day} 查询后未取得可核验的 Au99.99 原始收盘。", receipt)
        refs = set(material["facts"]["current_gold"]["source_refs"])
        for doc in material["documents"]:
            observation = doc.get("gold_observation", {})
            if doc["id"] not in refs or observation.get("valuation_date") != day:
                continue
            original = self.document(doc)
            if original and observation.get("contract") == "Au99.99":
                row, _ = gold_rows(source_bytes(Path(original["raw_path"])), day)
                if row == observation.get("source_row"):
                    return verified("上海黄金交易所每日行情", f"{day} 的 Au99.99 收盘、变动及涨跌幅已复核。",
                                    {"original": original, "row": row}, new=receipt["status"] == "ACQUIRED")
        return pending("MISMATCH", "上海黄金交易所每日行情", "现货原件未通过代码、日期和收盘核对。", receipt)

    def gold_real_yield(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from app.services.fund_daily_source_routes import TREASURY_REAL_YIELD, treasury_real_yield_rows

        # 只接受本次估值窗口中、执行时点前已过去的纽约当地日期。
        local = now().astimezone(ZoneInfo("America/New_York"))
        cutoff = min(local.date() - timedelta(days=1), date.fromisoformat(self.window["target_nav_date"]))
        params = {"type": "daily_treasury_real_yield_curve", "field_tdr_date_value": str(cutoff.year)}
        try:
            with httpx.Client(timeout=httpx.Timeout(14, connect=5), trust_env=False) as client:
                response = client.get(TREASURY_REAL_YIELD, params=params, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                raise ValueError("TREASURY_REAL_YIELD_RESPONSE_SIZE")
            sha, path = blob(response.content, "html")
            rates = treasury_real_yield_rows(response.content, cutoff)
        except (httpx.HTTPError, ArithmeticError, ValueError) as error:
            return pending("TEMPORARY_FAILURE", TREASURY_REAL_YIELD, "美国财政部实际收益率原表未取得或无法核验。",
                           {"error": type(error).__name__, "query": params})
        proof = {"raw_path": str(ROOT / path), "sha256": sha, "url": str(response.url),
                 "cutoff": cutoff.isoformat(), "ten_year_rates_pct": rates, "retrieved_at": datetime.isoformat(now())}
        if len(rates) < 5 or (cutoff - date.fromisoformat(max(rates))).days > 5:
            return pending("REVIEW_REQUIRED", TREASURY_REAL_YIELD, "财政部原表日期未覆盖最近的实际交易时段。", proof)
        return verified(TREASURY_REAL_YIELD, f"已核实截至 {max(rates)} 最近五个公开观测日的美国十年期实际利率。", proof)

    def vietnam_culture_day(self, code, day, docs):
        from bs4 import BeautifulSoup

        from app.services.fund_rating_market_review import normalized_text

        source = "天弘基金公告与越南证券登记结算公司公告"
        url = "https://vsdc.vn/en/ad/202680"
        notice_url = "https://fund.eastmoney.com/gonggao/008764,AN202611201831877278.html"
        notice = next((doc for doc in docs.values() if doc.get("url") == notice_url), {})
        fund_original = self.document(notice)
        terms = (
            "暂停申购起始日2026年11月24日", "暂停赎回起始日2026年11月24日",
            "2026年11月25日恢复申购、赎回及定期定额投资业务",
            "008763008764022524交易代码该分级基金是否暂停申购、赎回是是是",
        )
        if (not fund_original or code != "008764" or notice.get("published_date") != "2026-11-20"
                or notice.get("published_date") > day
                or not all(term in normalized_text(notice["body"]) for term in terms)):
            return pending("REVIEW_REQUIRED", source, "本份额公告原件、日期或申赎适用范围还未核对一致。",
                           {"fund_notice": fund_original, "official_market_url": url})
        try:
            with httpx.Client(timeout=httpx.Timeout(12, connect=5), trust_env=False) as client:
                response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            if len(response.content) > 1_000_000:
                raise ValueError("VIETNAM_OFFICIAL_HOLIDAY_RESPONSE_SIZE")
            sha, path = blob(response.content, "html")
            market_original = {"url": url, "sha256": sha, "path": str(ROOT / path), "retrieved_at": now().isoformat()}
            text = " ".join(BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True).split())
        except (httpx.HTTPError, ValueError) as error:
            return pending("TEMPORARY_FAILURE", source, "已核本份额公告，越南结算机构原公告未成功取回。",
                           {"fund_notice": fund_original, "official_market_url": url, "reason": type(error).__name__})
        proof = {"fund_notice": fund_original, "official_market_holiday": market_original,
                 "closed_date": "2026-11-24", "fund_resumption_date": "2026-11-25"}
        if not ("Date update 13/11/2026" in text and "28/2026/QH16" in text and
                "VSDC will be closed on Tuesday, 24/11/2026" in text):
            return pending("REVIEW_REQUIRED", source, "越南结算机构原公告中休市范围或日期未通过核验。", proof)
        return verified(source, "已核对 008764 的正式公告及越南结算机构原文。11 月 24 日暂停申赎和定投，11 月 25 日恢复。",
                        proof)

    def confirm_report(self, code, latest, source, evidence, client):
        from app.services.fund_exposure_runtime import execution_lock
        from app.services.fund_holdings_publication import publish_report
        from app.services.fund_materials import snapshot
        from app.services.market_holdings_parse import parse_report

        present = (snapshot(code) or {}).get("reports", [])
        entries = [row for row in present if row.get("sourceArticleId") == latest["id"]
                   and row.get("endDate") == latest["end"] and row.get("publishedDate") == latest["published"]]
        if len(entries) == 1 and entries[0].get("holdingsPublicationVersion"):
            return verified(source["url"], f"本份额报告期 {latest['end']}，目录、原文和已保存持仓版本一致。",
                            {**evidence, "report_id": entries[0]["id"]})
        with execution_lock() as locked:
            if not locked:
                return pending("REVIEW_REQUIRED", source["url"], "原文已取得，持仓维护正忙，下次继续解析和正式读回。", evidence)
            report, issues = parse_report(
                client, client.root, {"fund_code": code, "fund_name": self.profiles[code]["name"]}, latest, source,
            )
            if issues:
                return pending("REVIEW_REQUIRED", source["url"], "报告原文已核，持仓解析有待确认项。",
                               {**evidence, "issues": issues})
            self.check_storage()
            saved = publish_report(code, report, directory=client.root.parent)
            ids = {row["id"] for row in saved["reports"]}
            if report["id"] not in ids or not {row["id"] for row in present} <= ids:
                raise ValueError("SUPPLEMENT_REPORT_READBACK_MISMATCH")
        return verified(source["url"], f"已取得并读回本份额报告期 {latest['end']} 的报告及披露持仓。",
                        {**evidence, "report_id": report["id"]}, new=report["id"] not in {r["id"] for r in present})

    def valuation_evidence(self, code, item, material, docs, sessions):
        from app.services.comprehensive_calendar import verified_domestic_window, verified_gold_valuation_days
        from app.services.qdii_valuation_contract import verified_qdii_window
        from app.services.vietnam_valuation_contract import POLICY as VN_POLICY
        from app.services.vietnam_valuation_contract import assess_vietnam_valuation

        evidence = material.get("calendar_evidence") or {}
        profile = {**self.profiles[code], "fund_name": self.profiles[code]["name"]}
        dates, version = sessions
        if version != self.window["calendar_version"] or not all(
            date.fromisoformat(self.window[k]) in dates for k in ("base_nav_date", "target_nav_date")
        ):
            return None
        if profile.get("valuation_policy") == "现有沪深官方日历规则":
            return {"profile": profile["valuation_policy"], "calendar_version": version}
        if not material:
            return None
        if evidence.get("policy") == VN_POLICY:
            result = assess_vietnam_valuation(profile, material, docs, self.window, as_of=now())
            if not result["supported"]:
                return None
            refs = [result["contract_ref"], result["calendar_ref"]]
        elif profile.get("template") == "overseas":
            if not verified_qdii_window(profile, material, docs, self.window, as_of=now()):
                return None
            refs = [evidence.get("prospectus_ref"), evidence.get("calendar_ref")]
        else:
            if not verified_domestic_window(profile, material, docs, self.window):
                return None
            if evidence.get("policy") == "CN_GOLD_ETF_NAV_WINDOW_V1" and not verified_gold_valuation_days(material, docs):
                return None
            refs = [evidence.get("prospectus_ref") or evidence.get("valuation_contract_ref"), evidence.get("report_ref")]
        originals = [self.document(docs.get(ref)) for ref in refs if ref]
        return ({"materials": item["materials_path"], "calendar": evidence, "originals": originals}
                if originals and all(originals) else None)

    def fund_notices(self, code):
        from app.services.fund_market_preparation import MarketFundPreparation
        from app.services.fund_prediction_news_recovery import refresh
        from app.services.fund_workflow_materials import workflow_root

        root = workflow_root()
        path = root / "news/funds" / (code + ".json")
        news = read(path) if path.is_file() else {}
        fresh = (news.get("fund_code") == code and news.get("status") == "SUCCEEDED" and news.get("catalog_complete")
                 and news.get("coverage", {}).get("end", "") >= self.day)
        if not fresh:
            preparation = MarketFundPreparation(root=root)
            try:
                result = preparation.sync_news(code)
            finally:
                preparation.close()
            if result["status"] != "SUCCEEDED" or not result["catalog_complete"]:
                return pending("REVIEW_REQUIRED", "本基金公告目录", "公告目录或原文仍有缺项。", result)
            if read(path).get("fund_code") != code:
                raise ValueError("DAILY_SUPPLEMENT_NOTICE_FUND_MISMATCH")
        candidate = {"fund_code": code, "documents": [], "source_manifest": {}, "comprehensive_coverage": {}}
        receipt = refresh(candidate, root=root, checked_at=now())
        if receipt["publication_time_required"]:
            return pending("REVIEW_REQUIRED", "本基金公告目录", "有同日公告的实际披露时间仍需核验。", receipt)
        coverage = candidate["source_manifest"]["current_notice_check"]["coverage"]
        return verified("本基金公告目录及原文", f"核对公告目录及适用原文，覆盖至 {self.day}。",
                        {"ledger": str(path), "coverage": coverage, **receipt}, new=not fresh)

    def acquire(self, task, item, material, docs, quotes, support, *, reuse_only=False):
        code, field, day = task_key(task)
        base = self.window["base_nav_date"]
        manifest = material.get("source_manifest", {})
        notice = manifest.get("current_notice_check", {})
        if field == "NAV" and day:
            return self.nav(code, day, item)
        if field == "outcome_nav" and support.get("direct"):
            historical, daily = support["outcomes"]
            if (historical.get("status") == daily.get("status") == "SUCCEEDED"
                    and historical.get("summary", {}).get("dueBefore") == self.day
                    and daily.get("dueBefore") == self.day and code in daily.get("scope", [])
                    and code in historical.get("scope", [])):
                return verified("正式到期预测核对及本人结果读回", "本轮覆盖的到期日净值及结果已核对。",
                                {"historical": str(self.root / "prediction-outcomes.json"),
                                 "one_day": str(self.root / "one-day-outcomes.json")})
            return pending("REVIEW_REQUIRED", "正式到期预测核对", "等待本轮到期预测核对及结果读回。",
                           {"historical": historical.get("status"), "one_day": daily.get("status")})
        if task.get("collection") == "登记指数日线" and task.get("reference_index") and day:
            return self.reference_index(task["reference_index"], day)
        if task.get("collection") == "中国货币网同业利率" and day:
            return self.official_shibor(day)
        if task.get("collection") == "中国货币网同刻现券报价" and day:
            return self.same_time_bonds(day)
        if task.get("collection") == "中国货币网回购成交" and day:
            return self.repo_trades(day)
        if task.get("collection") == "央行逆回购原公告" and day:
            return self.open_market_operations(code, task, day, docs, support["calendar"][0])
        if task.get("collection") == "越南新增休市与申赎公告" and day:
            return self.vietnam_culture_day(code, day, docs)
        if task.get("collection") == "目标基金指定日行情" and day:
            return self.target_market(task, code, day, support["calendar"][0], support["published"].get(code))
        if task.get("collection") == "上金所现货收盘" and day:
            return self.gold_close(code, day, material)
        if field in {"rating_cost", "rating_management"} and day:
            from app.services.fund_daily_rating_sources import fees_from_originals, fund_manager_originals

            lookup = fees_from_originals if field == "rating_cost" else fund_manager_originals
            try:
                proof = lookup(support["published"].get(code), code, day)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                return pending("REVIEW_REQUIRED", "本基金评分所用原件",
                               f"已读取与所需评分月匹配的原件，专项核验未通过（{type(error).__name__}）。",
                               {"validation": str(error)[:160]})
            if proof:
                label = "产品资料概要、招募说明书及费率" if field == "rating_cost" else "经理任职与管理依据"
                return verified("正式评分绑定的已校验原件", f"{code} 的 {proof['as_of']} {label}已逐项复核。", proof)
        if field in {"rating_return", "rating_risk", "rating_stability"} and day:
            result = self.rating_history(code, day, field, support["published"].get(code))
            if result:
                return result
        if field in {"rating_holdings", "rating_liquidity", "rating_execution", "rating_product_detail"} and day:
            result = self.rating_portfolio(code, day, field, support["published"].get(code))
            if result:
                return result
        check_key = task.get("check_key")
        if check_key == "gold_real_yield_latest":
            return self.gold_real_yield()
        if check_key in {"advice_ledger", "advice_funding", "advice_quality", "advice_current_predictions"}:
            field = {"advice_quality": "advice_rating", "advice_current_predictions": "advice_predictions"}.get(
                check_key, check_key
            )
        if field.startswith("CLOSE:"):
            symbol = field.split(":", 1)[1]
            quote = quotes.get(symbol, {})
            proof = self.document(docs.get(quote.get("source_ref")))
            if proof and quote.get("date") == day and quote.get("close") is not None and quote.get("change_pct") is not None:
                return verified(proof["url"], f"{symbol} {day} 收盘值 {quote['close']}，原始行情和适用日期一致。",
                                {"quote": quote, "source": proof})
            if reuse_only:
                return None
            attempts = [
                row for kind in ("market_attempts", "company_attempts", "hk_attempts", "overseas_attempts", "bond_attempts")
                for row in item.get(kind, []) if isinstance(row, dict) and
                (row.get("code") or row.get("security")) == symbol
            ]
            return pending("REVIEW_REQUIRED", "本轮行情资料及正式取数回执",
                           f"{symbol} 缺 {day} 可核对原件的有效收盘价；同日行情、换股或停牌规则仍待核实。",
                           {"attempts": attempts, "quote_date": quote.get("date")})
        if field == "MATERIALS" and day == base:
            entry = manifest.get("latest_partial_report") or {}
            report = entry.get("report") or material.get("report", {})
            proof = self.document(docs.get(entry.get("source_ref") or "report:" + code))
            if (proof and report.get("id") and report["id"] == item.get("latest_report", {}).get("id")
                    and report.get("publishedDate") == proof["published_date"]):
                return verified(proof["url"], f"本份额最新披露报告期 {report['endDate']}，已核报告原件和持仓版本。", proof)
            return None if reuse_only else self.latest_report(code)
        if field.startswith("TARGET:") and day == base:
            target = field.split(":", 1)[1]
            entry = (manifest.get("target_partial_reports") or {}).get(target, {})
            doc = docs.get(entry.get("source_ref"), {})
            proof = self.document(doc)
            report = entry.get("report") or {}
            if proof and target in doc.get("codes", []) and report.get("publishedDate") == proof["published_date"]:
                return verified(proof["url"], f"底层 {target} 已核对 {report['endDate']} 披露及持仓原件。", proof)
            if not reuse_only:
                for row in item.get("target_reports", []):
                    if (row.get("target_fund_code") != target or row.get("owner_fund_code") != code
                            or row.get("status") not in {"ORIGINAL_REUSED", "PUBLISHED"}):
                        continue
                    from app.core.config import get_settings
                    from app.integrations.market_fund_reports import read_source

                    report, entry, source = row["report"], row["entry"], row["source"]
                    if (report.get("id") != row.get("report_id") or report.get("endDate") != entry["end"]
                            or report.get("publishedDate") != entry["published"] or entry["published"] > self.day):
                        continue
                    root = Path(get_settings().fund_material_directory) / "prediction-lookthrough"
                    read_source(root, source)
                    return verified(
                        source["url"], f"底层 {target} 的 {entry['end']} 报告已核对原文和正式持仓。",
                        {"report": report["id"], "raw_path": str(root / source["file"]), "source": source},
                        new=row["status"] == "PUBLISHED",
                    )
        if field == "NOTICES" and day == base:
            refs = notice.get("source_refs", [])
            originals = [self.document(docs.get(ref)) for ref in refs]
            if (notice.get("status") == "SUCCEEDED" and notice.get("coverage", {}).get("end", "") >= self.day
                    and not notice.get("publication_time_required") and refs and all(originals)):
                return verified("本份额公告目录与原文", f"目录覆盖至 {self.day}，逐份原文和适用范围已核验。",
                                {"coverage": notice["coverage"], "originals": originals})
            if not reuse_only and support.get("direct"):
                return self.fund_notices(code)
        if field == "VALUATION_CALENDAR" and day == base:
            proof = self.valuation_evidence(code, item, material, docs, support["calendar"])
            if proof:
                return verified("本份额的适用估值条款及已核官方日历",
                                f"已验证 {base} 至 {self.window['target_nav_date']} 的实际估值安排。", proof)
        if field == "FX:VND/USD" and day == base:
            from app.services.vietnam_valuation_contract import assess_vietnam_valuation

            status = assess_vietnam_valuation({"fund_name": self.profiles[code]["name"]}, material, docs,
                                               self.window, as_of=now()) if material else {}
            if status.get("fx_ready"):
                return verified("本份额合同指定的越南盾折算资料", f"{base} 合同汇率核验通过。",
                                {"materials": item["materials_path"], "check": status})
            if reuse_only:
                return None
            old = self.previous.get(task_key(task))
            if old and old.get("status") == "TEMPORARY_FAILURE":
                return old
            return pending("ACCESS_REQUIRED", "合同、估值资料包与已有取件回执",
                           f"{base} 缺伦敦 16:00 合同指定折算价的合格原件。", status,
                           "按本份额合同向授权定盘渠道或正式托管人核取当日实际折算率，复核日期和定盘时点。")
        if field == "FX" and day == base:
            valuation = material.get("calendar_evidence") or {}
            doc = docs.get(valuation.get("fx_ref"), {})
            proof = self.document(doc)
            if proof and base in doc["body"] and doc.get("kind") == "FX_REFERENCE_DATA":
                # 参考中间价本身不足以证明合同指定折算。以已有产品验证器核实全窗口。
                from app.services.qdii_valuation_contract import verified_qdii_window

                if verified_qdii_window({"fund_name": self.profiles[code]["name"]}, material, docs,
                                        self.window, as_of=now()):
                    return verified(proof["url"], f"已核 {base} 与本份额估值合同匹配的汇率资料。", proof)
        if reuse_only:
            return None
        if field == "advice_conditions" and day == self.day:
            key = ("conditions", code, day)
            if key not in self.requests:
                self.requests[key] = self.trading_conditions(code, notice, docs)
            return self.requests[key]
        if field == "advice_funding" and day == self.day and support["funding"] is not None:
            return support["funding"]
        if field in {"advice_ledger", "advice_rating", "advice_funding", "advice_predictions"} and day == self.day:
            advice = support["advice"].get(code, {})
            if (advice.get("read_back") and item_date(advice.get("checked_at")) == self.day
                    and (field in advice.get("checklist_verified", [])
                         or (field == "advice_predictions" and advice.get("checklist_current_predictions")))):
                return verified("本份额正式建议及对应账本、评分或预测读回",
                                f"{code} 的本日 {task.get('item') or field} 已核实。",
                                {"report_id": advice["report_id"], "readback": str(self.root / "advice-status.json")})
            return pending("REVIEW_REQUIRED", "本轮正式建议及依赖读回",
                           "本日正式读回未验证此项依赖；所需期次、持有期限或资金状态仍需核查。",
                           {"report_id": advice.get("report_id"), "dependencies": advice.get("pending_dependencies", []),
                            "missing": advice.get("missing", [])})
        # 无安全自动取件路由时明确保留人工取证，不能将一次本地核对写成外部查询失败。
        old = self.previous.get(task_key(task), {})
        if old.get("status") in {"MISMATCH", "TEMPORARY_FAILURE", "NOT_PUBLISHED", "ADDRESS_CHANGED", "ACCESS_REQUIRED"}:
            return old
        relevant = {
            "materials": item.get("materials_path"),
            "monthly": support["monthly"].get(code),
            "source_checks": {
                "report": item.get("latest_report", {}).get("id"),
                "notice": notice.get("status"),
                "market_dates": sorted({value["date"] for value in quotes.values() if value.get("date")}),
            },
            "failure": (item.get("failure") or {}).get("type") if isinstance(item.get("failure"), dict) else None,
        }
        wanted = (task.get("verification") or "按本条要求核对原文、日期、份额及口径").rstrip("。")
        return pending("NOT_ATTEMPTED", "本基金同窗口的资料、正式回执和清单",
                       f"“{task.get('item') or field}”没有可用的专项取数或验收入口，本项取数尚未执行。",
                       relevant, f"{task.get('next_source') or '查询基金管理人或托管人正式披露'}。{wanted}。")

    def rating_history(self, code, day, field, record):
        from app.services.fund_daily_rating_sources import performance_originals

        key = ("rating-history", code, day)
        if key not in self.requests:
            try:
                self.requests[key] = performance_originals(record, code, day)
            except (ValueError, KeyError, TypeError, ArithmeticError, FileNotFoundError) as error:
                self.requests[key] = {"validation_failure": str(error)[:160]}
        evidence = self.requests[key]
        if not evidence:
            return None
        source = "本份额正式评分绑定的授权原始净值和合同基准资料"
        if evidence.get("validation_failure"):
            return pending("REVIEW_REQUIRED", source, "已回查正式原件，净值或基准没有通过所需期次校验。", evidence)
        if evidence["official_comparison"] and evidence.get("peer_count") == 0 and field == "rating_stability":
            return verified(source, "已复核实际评分月末与前十二个月的本份额复权净值、合同基准及逐月比较。"
                            "本份额已采用合同基准比较口径。", evidence)
        missing = {
            "rating_return": "仍需核适用比较组及收益评估口径",
            "rating_risk": "仍需核适用的风险暴露与披露原件",
            "rating_stability": "仍需核正式比较组或合同基准的完整月序列",
        }[field]
        return pending("REVIEW_REQUIRED", source, f"已核实际日净值及月末观测。{missing}。", evidence)

    def rating_portfolio(self, code, day, field, record):
        from app.services.fund_daily_rating_sources import portfolio_originals

        key = ("rating-portfolio", code, day)
        if key not in self.requests:
            try:
                self.requests[key] = portfolio_originals(record, code, day)
            except (ValueError, KeyError, TypeError, ArithmeticError, FileNotFoundError) as error:
                self.requests[key] = {"validation_failure": str(error)[:160]}
        evidence = self.requests[key]
        if not evidence:
            return None
        source = "本份额正式评分绑定的定期报告原件"
        if evidence.get("validation_failure"):
            return pending("REVIEW_REQUIRED", source, "已回查评分绑定报告，报告身份、期间或原件校验未通过。", evidence)
        observations = [
            label for item, label in (("asset_balances", "账面资产和净资产"), ("share_flows", "本份额申购、赎回和期末份额"),
                                      ("stock_holdings", "本基金股票明细"), ("stock_risk_budget", "目标基金穿透持仓"),
                                      ("execution_snapshot", "该报告期的实际合同配置核对值"))
            if evidence.get(item) is not None
        ]
        gaps = {
            "rating_holdings": "仍需核对适用的最新披露、非股票资产清单及集中度",
            "rating_liquidity": "仍需核对近期成交、受限资产、集中持有人和最新赎回条件",
            "rating_execution": "仍需按本份额生效合同核对实际比例门槛、跟踪误差或策略执行",
            "rating_product_detail": "仍需核对本基金清单要求的专项数据、合同口径和覆盖期",
        }
        found = "已提取该期报告中的" + "、".join(observations) if observations else "已核到该期原报告"
        return pending("REVIEW_REQUIRED", source, f"{found}。{gaps[field]}。", evidence)

    def recover_candidate(self, state, code, item, tasks, material, docs, quotes, support, recover, publish_prepared):
        """验证已有证据，再针对缺项调用能补齐该项的入口。"""
        fields = {"MATERIALS", "NOTICES", "VALUATION_CALENDAR", "FX", "FX:VND/USD"}
        missing = set()
        for key, task in tasks:
            field = key[1]
            if task.get("collection") in {"登记指数日线", "目标基金指定日行情"}:
                continue
            if field not in fields | {"NAV"} and not field.startswith(("TARGET:", "CLOSE:")):
                continue
            try:
                result = self.previous_proof(key)
                if result is None:
                    result = self.acquire(
                        task, item, material, docs, quotes, support, reuse_only=field != "NAV",
                    )
                    if field == "MATERIALS" and result:
                        report = item["latest_report"]
                        result = self.latest_report(code, existing=(report, result["observations"]))
                        if (result.get("observations") or {}).get("report_id") != report["id"]:
                            missing.add(field)
            except FUND_INPUT_ERRORS as error:
                result = pending("REVIEW_REQUIRED", "本基金资料来源", f"取件失败，需复核来源（{type(error).__name__}）。")
                if field not in {"NAV", "MATERIALS", "NOTICES"}:
                    result = None
            if result:
                self.results[key] = result
                continue
            if field in {"MATERIALS", "NOTICES"}:
                try:
                    self.results[key] = self.acquire(task, item, material, docs, quotes, support)
                except FUND_INPUT_ERRORS as error:
                    self.results[key] = pending("REVIEW_REQUIRED", "本基金资料来源",
                                                f"取件失败，需复核来源（{type(error).__name__}）。")
            missing.add(field)
        if not missing or not material or publish_prepared:
            return
        try:
            item.update(recover(code, self.root / "funds" / code, self.window, missing_fields=missing))
        except FUND_INPUT_ERRORS as error:
            # 综合候选失败时，其余独立条目仍然取件并留回执。
            record_fund_failure(state, item, code, error)

    def process(self, state, *, refresh_dependents=False, recover_materials=None, publish_prepared=False):
        from app.services.fund_daily_rating_sources import published_inputs

        direct = bool(state.get("supplement_run_id"))
        self.prepare(state)
        support = {
            "calendar": calendar(),
            "published": published_inputs(self.scope) if state.get("owner_user_id") else {},
            "funding": funding_evidence(state["owner_user_id"], now()) if state.get("owner_user_id") else None,
            "advice": {row["fund_code"]: row for row in (state.get("advice") or {}).get("items", [])},
            "monthly": {row["fund_code"]: row for row in (state.get("monthly_compilation") or {}).get("items", [])},
            "direct": direct,
            "outcomes": (state.get("prediction_outcomes") or {}, state.get("one_day_outcomes") or {}),
        }
        completed, waiting, counts = [], [], {}
        folder = self.root / "supplement-funds"
        tasks_by_fund = {code: [] for code in self.scope}
        for key, task in self.tasks.items():
            tasks_by_fund[key[0]].append((key, task))
        for code in self.scope:
            assert_daily_budget(self.root, now())
            self.check_storage()
            item = state["funds"].setdefault(code, {"fund_code": code})
            if recover_materials:
                from app.services.fund_materials import snapshot

                item.pop("failure", None)
                reports = (snapshot(code) or {}).get("reports", [])
                item["latest_report"] = max(
                    (row for row in reports if row["publishedDate"] <= self.day),
                    key=lambda row: (row["endDate"], row["publishedDate"]), default={},
                )
            material, material_failure = {}, None
            try:
                path, material = self.material_for_check(code, item)
                if path:
                    if material.get("fund_code") != code:
                        raise ValueError("DAILY_SUPPLEMENT_MATERIALS_FUND_MISMATCH")
                    item["materials_path"] = str(path)
            except (ValueError, KeyError, TypeError) as error:
                material, material_failure = {}, str(error)[:150]
                item.pop("materials_path", None)
            docs = {row["id"]: row for row in material.get("documents", [])}
            quotes = {row["code"]: row.get("quote") or {} for row in material.get("companies", [])}
            quotes.update(material.get("market", {}))
            proof_path = folder / (code + ".json")
            fund_rows, verified_rows, remaining = [], {}, []
            if proof_path.is_file():
                saved = read(proof_path)
                if saved.get("day") == self.day and saved.get("job_id") == self.job and saved.get("window") == self.window:
                    for row in saved["items"]:
                        key = task_key(row)
                        old = self.previous.get(key, {})
                        if row["attempted_at"] >= old.get("attempted_at", ""):
                            self.previous[key] = row
            if recover_materials:
                if material:
                    item["materials_checklist_day"] = self.day
                    self.remember(save_pending(self.root, state, records=self.profiles, fund_code=code, inventory=True))
                    tasks_by_fund[code] = [(k, t) for k, t in self.tasks.items() if k[0] == code]
                self.recover_candidate(state, code, item, tasks_by_fund[code], material, docs, quotes, support,
                                       recover_materials, publish_prepared)
                if item.get("materials_path"):
                    material = read(Path(item["materials_path"]))
                    if material.get("fund_code") != code:
                        raise ValueError("DAILY_SUPPLEMENT_MATERIALS_FUND_MISMATCH")
                    docs = {row["id"]: row for row in material.get("documents", [])}
                    quotes = {row["code"]: row.get("quote") or {} for row in material.get("companies", [])}
                    quotes.update(material.get("market", {}))
                    item["materials_checklist_day"] = self.day
            for key, task in tasks_by_fund[code]:
                assert_daily_budget(self.root, now())
                previous = self.previous_proof(key)
                recheck = refresh_dependents and key[1].startswith(("advice_", "rating_", "outcome_"))
                if direct and key in self.results and not recheck:
                    result = self.results[key]
                elif previous and not recheck:
                    result = previous
                else:
                    try:
                        result = self.acquire(task, item, material, docs, quotes, support)
                    except (ValueError, KeyError, TypeError, RuntimeError, FileNotFoundError, httpx.HTTPError) as error:
                        result = pending("REVIEW_REQUIRED", "本条已登记取数入口及原件",
                                         f"本条处理未通过：{type(error).__name__}。"
                                         "重新核对该来源的响应、份额及口径后继续取件。")
                    if material_failure and result["status"] == "REVIEW_REQUIRED":
                        result = {**result, "summary": result["summary"] + f" 当前原件核对失败：{material_failure}。"}
                self.results[key] = result
                row = {
                    "attempted_at": now().isoformat(), **result,
                    "fund_code": code, "field": key[1], "required_date": key[2],
                    "next_source": result.get("next_source") or task.get("next_source")
                    or f"按 {code} 的基金正式档案取得本条所需原件，再核份额、日期和口径。",
                }
                row.setdefault("evidence_ref", str(proof_path))
                fund_rows.append(row)
                if row["status"] in VERIFIED_SOURCE_RESULTS:
                    verified_rows[row["field"]] = row
                else:
                    remaining.append({**task, "source_review": row, "next_source": row["next_source"],
                                      "item_alignment_status": row["status"], "item_alignment_note": row["summary"]})
            self.check_storage()
            versioned_save(proof_path, {
                "fund_code": code, "day": self.day, "window": self.window, "job_id": self.job, "items": fund_rows,
            })
            save_fund_checklist(self.root.parent / "fund-supplements", code, self.profiles[code], self.window,
                                self.day, remaining, set(verified_rows), verified=verified_rows)
            completed.extend(fund_rows)
            waiting.extend(remaining)
            counts[code] = dict(Counter(row["status"] for row in fund_rows))
        self.check_storage()
        versioned_save(self.catalog_path, {"window": self.window, "job_id": self.job, "tasks": list(self.tasks.values())})
        combined = {**self.previous, **{task_key(row): row for row in completed}}
        versioned_save(self.source_path, {**self.window_fields(), "job_id": self.job, "items": list(combined.values())})
        versioned_save(self.queue_path, {
            "checked_at": now().isoformat(), "window": self.window, "job_id": self.job,
            "tasks": waiting, "item_alignment": str(self.root / "supplement-execution.json"),
        })
        result = {
            **self.window_fields(), "job_id": self.job, "checked_at": now().isoformat(),
            "status": "ACTION_REQUIRED" if waiting else "SUCCEEDED",
            "item_count": len(completed), "pending_item_count": len(waiting), "funds": counts,
            "pending_funds": sorted({row["fund_code"] for row in waiting}),
            "unattempted_item_count": sum(row["status"] == "NOT_ATTEMPTED" for row in completed),
            "counts": dict(Counter(row["status"] for row in completed)),
        }
        if result["unattempted_item_count"]:
            result["status"] = "EXECUTION_INCOMPLETE"
        versioned_save(self.root / "supplement-execution.json", result)
        self.previous.update({task_key(row): row for row in completed})
        return result
