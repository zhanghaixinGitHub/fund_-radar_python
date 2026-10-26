"""经理目录维护：本基金锚点、跨基金身份证据与公开快照分开处理。"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_engine
from app.integrations.fund_manager_source import ManagerSource, digest, official_dbfund_profile
from app.models.fund import FundManagerAssignment, FundProfile, FundShareClass, SourceRegistry, SourceSyncRun
from app.models.fund_manager_directory import (
    FundManager,
    FundManagerCoverageSnapshot,
    FundManagerIdentityEvidence,
    FundManagerMetricSnapshot,
    FundManagerProfileVersion,
    FundManagerRelatedFund,
    FundManagerTenureVersion,
)
from app.repositories.fund_manager_performance import read_manager_performance_inputs
from app.repositories.fund_read import get_fund_watchlist_detail
from app.schemas.fund_manager_directory import ManagerDetail, ManagerPerformance
from app.services.fund_manager_performance import calculate_tenure_return, manager_calendar
from app.services.tushare_fund_sync import _MARKET_NAV_INCREMENTAL_LOCK_KEY, _normalize_market_dividend_records

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2] / "data/fund-managers"
SOURCE_URL = "https://tushare.pro/document/2?doc_id=208"


def day(value) -> date | None:
    """供应商空格与空字符串均为缺失；不以公告日猜测任职日。"""
    value = str(value or "").strip()
    if not value:
        return None
    return datetime.strptime(value, "%Y%m%d" if len(value) == 8 else "%Y-%m-%d").date()


def stable_id(key: str) -> UUID:
    return uuid5(NAMESPACE_URL, "fund-radar/manager/" + key)


def save_original(value: dict, root: Path = ROOT) -> dict:
    """来源原件内容寻址，文件名由哈希生成，绝不接受浏览器路径。"""
    fingerprint = digest(value)
    target = root / "sources" / (fingerprint + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")
    return {"sha256": fingerprint, "reference": "sources/" + target.name}


def normalize(records: list[dict]) -> list[dict]:
    """同一来源同一开始日的冲突不择优猜测；完整响应中的唯一离任修订才生效。"""
    grouped = defaultdict(list)
    for row in records:
        begin, end = day(row.get("begin_date")), day(row.get("end_date"))
        if not re.fullmatch(r"\d{6}\.(OF|SH|SZ)", str(row.get("ts_code", ""))):
            raise ValueError("经理资料基金代码不合法")
        grouped[(row["ts_code"], row["name"].strip(), begin)].append({**row, "begin": begin, "end": end})
    result = []
    for values in grouped.values():
        ends = {value["end"] for value in values}
        row = values[-1]
        row["valid"] = (
            len(ends) == 1 and row["begin"] is not None and (row["end"] is None or row["end"] >= row["begin"])
        )
        result.append(row)
    return result


def _state(row: dict, today: date) -> str:
    if not row["valid"]:
        return "PENDING"
    if row["begin"] > today:
        return "FUTURE"
    return "HISTORY" if row["end"] and row["end"] < today else "CURRENT"


def _team(row: dict, roster: list[dict] | None) -> str:
    if roster is None or not row["valid"]:
        return "UNKNOWN"
    if any(not person["valid"] for person in roster):
        return "UNKNOWN"
    end = row["end"] or date.max
    overlap = any(
        person["name"] != row["name"] and person["begin"] <= end and (person["end"] or date.max) >= row["begin"]
        for person in roster
    )
    return "CO_MANAGED" if overlap else "SOLO"


def _solo_row(row: dict, roster: list[dict] | None, cutoff: date) -> dict | None:
    """只取任期末最近一段连续独立管理阶段；团队不完整或截止时仍共管则不推算。"""
    if roster is None or not row["valid"] or any(not person["valid"] for person in roster):
        return None
    if row["begin"] > cutoff:
        return None
    others = [
        person
        for person in roster
        if person["name"] != row["name"] and person["begin"] <= cutoff and (person["end"] or date.max) >= row["begin"]
    ]
    if any(person["end"] is None or person["end"] >= cutoff for person in others):
        return None
    begin = max([row["begin"], *[person["end"] + timedelta(days=1) for person in others]])
    return {**row, "begin": begin} if begin < cutoff else None


def _metric(session, related, row, roster, dividend_verified, today):
    """只重用已有现金再投方法；本基金分红检查不完整时不借全局成功记录放行。"""
    missing = ManagerPerformance(reason="该份额的完整净值、分红或估值日历尚未核实").model_dump(mode="json")
    if not dividend_verified or not related.mapped_fund_code or not row["valid"] or roster is None:
        return missing
    snapshot = get_fund_watchlist_detail(session, related.mapped_fund_code)
    if (
        snapshot is None
        or snapshot.detail.nav_date is None
        or snapshot.detail.fund.fund_type not in {"BOND", "STOCK", "MIXED", "INDEX"}
        or (snapshot.profile and "QDII" in (snapshot.profile.profile.source_fund_type or "").upper())
    ):
        return missing
    end = min(row["end"] or today, today, snapshot.detail.nav_date)
    if end <= row["begin"]:
        return missing
    navs, dividends = read_manager_performance_inputs(
        session, fund_code=related.mapped_fund_code, source_id=related.source_id, start=row["begin"], end=end
    )
    if len(navs) > 5000 or len(dividends) > 100:
        return missing
    performance = calculate_tenure_return(
        begin=row["begin"],
        end=row["end"],
        latest_nav_date=snapshot.detail.nav_date,
        today=today,
        navs=navs,
        dividends=dividends,
        sessions=manager_calendar(),
        co_managed=_team(row, roster) == "CO_MANAGED",
    )
    value = {
        "status": performance.status,
        "returnRate": performance.return_rate,
        "startDate": performance.start_date,
        "endDate": performance.end_date,
        "reason": performance.reason,
        "includesCoManagement": performance.includes_co_management,
    }
    if performance.status == "AVAILABLE":
        # 已通过原计算器的覆盖与分红核对后，使用同一再投指数计算回撤，不能改用另一份净值。
        selected = [point for point in navs if performance.start_date <= point.nav_date <= performance.end_date]
        cash = {event.nav_ex_date or event.ex_date: event.cash_dividend for event in dividends}
        growth = peak = Decimal(1)
        drawdown = Decimal(0)
        for previous, point in zip(selected[:-1], selected[1:], strict=True):
            growth *= (point.unit_nav + cash.get(point.nav_date, Decimal(0))) / previous.unit_nav
            peak = max(peak, growth)
            drawdown = max(drawdown, 1 - growth / peak)
        value["maxDrawdownRate"] = drawdown.quantize(Decimal("0.000000000001"))
        span = (performance.end_date - performance.start_date).days
        if span >= 365:
            value["annualizedRate"] = Decimal(str(float(growth) ** (365 / span) - 1)).quantize(
                Decimal("0.000000000001")
            )
    result = ManagerPerformance.model_validate(value).model_dump(mode="json")
    result["_inputHash"] = digest(
        {
            "navs": [asdict(point) for point in navs],
            "dividends": [asdict(event) for event in dividends],
            "begin": row["begin"],
            "end": row["end"],
            "sourceId": str(related.source_id),
            "effectiveEnd": end,
        }
    )
    return result


class ManagerDirectorySync:
    """每个经理短事务发布；外部关系没有官方锚点或明确身份证据时只留待核对。"""

    def __init__(self, engine=None, source=None, *, root: Path = ROOT, official_reader=official_dbfund_profile):
        self.engine = engine or get_engine()
        self.source = source
        self.root = root
        self.official_reader = official_reader
        self.name_cache, self.roster_cache, self.basic_cache, self.official_cache, self.dividend_cache = (
            {},
            {},
            {},
            {},
            {},
        )
        self.name_evidence, self.roster_evidence = {}, {}
        self.dividend_evidence = {}
        self.disclosure_cache = {}

    def disclosures(self, seed, today):
        """复用已有基金披露目录；只展示经理相关基金公告，不猜测本人离任原因。"""
        if seed["code"] not in self.disclosure_cache:
            path = Path(get_settings().fund_material_directory) / (seed["code"] + ".json")
            try:
                if not path.exists() or path.stat().st_size > 64_000_000:
                    value = {}
                else:
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    value = saved.get("payload", saved)
                self.disclosure_cache[seed["code"]] = value.get("documents", [])
            except (OSError, ValueError):
                logger.warning("fund_manager_directory_sync.disclosures >>> local disclosure index unavailable")
                self.disclosure_cache[seed["code"]] = []
        begins = [day(row.get("begin_date")) for row in seed["rows"] if day(row.get("begin_date"))]
        ends = [day(row.get("end_date")) for row in seed["rows"]]
        earliest = min(begins) if begins else today
        latest = today if None in ends else min(today, max(ends) + timedelta(days=14))
        result = []
        for item in self.disclosure_cache[seed["code"]]:
            url, title = item.get("sourceUrl", ""), item.get("title", "")
            published = item.get("publishedDate")
            if item.get("kind") != "fund" or "基金经理" not in title or not url.startswith("https://"):
                continue
            try:
                if not published or not earliest <= date.fromisoformat(published) <= latest:
                    continue
            except ValueError:
                continue
            result.append({"title": title, "url": url, "publishedDate": published})
        return sorted(result, key=lambda item: item["publishedDate"], reverse=True)[:50]

    def seeds(self):
        """只从原市场来源范围读锚点，不因新发现份额递归扩展目标集合。"""
        with Session(self.engine) as session:
            rows = session.execute(
                select(FundShareClass, FundManagerAssignment, FundProfile)
                .join(SourceRegistry, SourceRegistry.source_code == FundShareClass.source_code)
                .join(
                    FundManagerAssignment,
                    (FundManagerAssignment.fund_code == FundShareClass.fund_code)
                    & (FundManagerAssignment.source_id == SourceRegistry.source_id),
                )
                .outerjoin(
                    FundProfile,
                    (FundProfile.fund_code == FundShareClass.fund_code)
                    & (FundProfile.source_id == SourceRegistry.source_id),
                )
                .where(FundShareClass.status == "ACTIVE", SourceRegistry.enabled.is_(True))
                .order_by(FundShareClass.fund_code, FundManagerAssignment.manager_name)
            ).all()
            grouped = {}
            for fund, assignment, profile in rows:
                key = f"{assignment.source_id}:{fund.fund_code}:{assignment.manager_name}"
                item = grouped.setdefault(
                    key,
                    {
                        "key": key,
                        "id": stable_id(key),
                        "sourceId": assignment.source_id,
                        "name": assignment.manager_name,
                        "code": fund.fund_code,
                        "sourceCode": fund.source_fund_code or fund.fund_code + ".OF",
                        "fundName": fund.fund_name,
                        "fundType": fund.fund_type,
                        "company": profile.management_company_name if profile else None,
                        "rows": [],
                    },
                )
                item["rows"].append(
                    {
                        "ts_code": item["sourceCode"],
                        "name": assignment.manager_name,
                        "begin_date": str(assignment.begin_date or ""),
                        "end_date": str(assignment.end_date or ""),
                        "ann_date": str(assignment.ann_date or ""),
                        "edu": assignment.education,
                    }
                )
            return list(grouped.values())

    def _official(self, code, name):
        key = (code, name)
        if key not in self.official_cache:
            self.official_cache[key] = self.official_reader(code, name)
        return self.official_cache[key]

    def _basic(self, code, seed):
        if code == seed["sourceCode"]:
            return {
                "name": seed["fundName"],
                "company": seed["company"],
                "type": seed["fundType"],
                "mapped": seed["code"],
            }
        if code not in self.basic_cache:
            basic = self.source.basics((code,))[0]
            kind = next(
                (
                    value
                    for word, value in [
                        ("QDII", "QDII"),
                        ("货币", "MONEY"),
                        ("指数", "INDEX"),
                        ("债券", "BOND"),
                        ("股票", "STOCK"),
                        ("混合", "MIXED"),
                    ]
                    if word in ((basic.fund_type or "") + basic.name)
                ),
                "OTHER",
            )
            with Session(self.engine) as session:
                mapped = session.scalar(
                    select(FundShareClass.fund_code).where(
                        FundShareClass.source_fund_code == code, FundShareClass.source_code == "TUSHARE_PRO_FUND"
                    )
                )
            self.basic_cache[code] = {"name": basic.name, "company": basic.management, "type": kind, "mapped": mapped}
        return self.basic_cache[code]

    def _dividends_verified(self, code, mapped, source_id):
        """本基金远端完整事件与已存输入对照；不写旧分红表，也不凭全局任务成功放行。"""
        if code in self.dividend_cache:
            return self.dividend_cache[code]
        verified = False
        if self.source and mapped:
            original_remote = self.source.client.list_fund_dividends(code)
            self.dividend_evidence[code] = save_original(
                {
                    "api": "fund_div",
                    "parameters": {"ts_code": code},
                    "receivedAt": datetime.now(UTC).isoformat(),
                    "rows": [asdict(item) for item in original_remote],
                },
                self.root,
            )
            # 复用原同步已经核验的空字段补全规则；互相冲突的非空字段仍会拒绝。
            remote, invalid = _normalize_market_dividend_records(original_remote, ts_codes=(code,))
            with Session(self.engine) as session:
                _, local = read_manager_performance_inputs(
                    session, fund_code=mapped, source_id=source_id, start=date(2100, 1, 1), end=date(2100, 1, 1)
                )

            def signature(event):
                return (
                    event.ex_date,
                    event.nav_ex_date,
                    event.cash_dividend,
                    event.process_status,
                    event.ann_date,
                    event.implementation_ann_date,
                )

            remote_keys, local_keys = [signature(row) for row in remote], [signature(row) for row in local]
            verified = not invalid and (
                len(original_remote) < 5000
                and len(remote) < 100
                and len(local) < 100
                and len(remote_keys) == len(set(remote_keys))
                and (set(remote_keys) == set(local_keys))
            )
        self.dividend_cache[code] = verified
        return verified

    def prepare(self, seed, now):
        rows, roster, candidates = seed["rows"], None, []
        references, checked = [], None
        profile = None
        accepted_codes = {seed["sourceCode"]}
        official_anchor = None
        if self.source:
            name = seed["name"]
            if name not in self.name_cache:
                start = len(self.source.pages)
                self.name_cache[name] = self.source.managers(name=name)
                self.name_evidence[name] = [save_original(page, self.root) for page in self.source.pages[start:]]
            all_rows = self.name_cache[name]
            if seed["sourceCode"] not in self.roster_cache:
                start = len(self.source.pages)
                self.roster_cache[seed["sourceCode"]] = normalize(self.source.managers(code=seed["sourceCode"]))
                self.roster_evidence[seed["sourceCode"]] = [
                    save_original(page, self.root) for page in self.source.pages[start:]
                ]
            references.extend(self.name_evidence[name])
            references.extend(self.roster_evidence[seed["sourceCode"]])
            roster = self.roster_cache[seed["sourceCode"]]
            fresh = [row for row in roster if row["name"] == name]
            rows = fresh or [{**row, "missingOnRecheck": True} for row in seed["rows"]]
            matches = [
                row
                for row in all_rows
                if row["ts_code"] == seed["sourceCode"]
                and day(row.get("begin_date")) in {day(old.get("begin_date")) for old in seed["rows"]}
            ]
            resumes = {str(row.get("resume") or "").strip() for row in matches} - {""}
            profile = next(iter(resumes)) if len(resumes) == 1 else None
            checked = now.isoformat() if fresh else None
            if "德邦基金" in (seed["company"] or "") and fresh:
                official_anchor = self._official(seed["code"], name)
            # 跨基金关系需要每只官方产品页明确经理名单，并匹配独立职业履历锚点；仅同名不会进入。
            if official_anchor:
                references.append(save_original(official_anchor, self.root))
                anchor_bio = re.sub(r"\s+", "", official_anchor["resume"])
                for code in sorted({row["ts_code"] for row in all_rows} - accepted_codes):
                    basic = self._basic(code, seed)
                    if basic["company"] != seed["company"]:
                        continue
                    official = self._official(code.split(".")[0], name)
                    if official and re.sub(r"\s+", "", official["resume"]) == anchor_bio:
                        accepted_codes.add(code)
                        references.append(save_original(official, self.root))
            candidates = [row for row in all_rows if row["ts_code"] not in accepted_codes]
            extra = [
                row for row in all_rows if row["ts_code"] in accepted_codes and row["ts_code"] != seed["sourceCode"]
            ]
            rows = [*rows, *extra]
        normalized = normalize(rows)
        for row in normalized:
            if row.get("missingOnRecheck"):
                row["valid"] = False
        return {
            "rows": normalized,
            "roster": roster,
            "profile": profile,
            "checked": checked,
            "references": references,
            "pending": len({row["ts_code"] for row in candidates}),
            "officialAnchor": official_anchor,
        }

    def publish(self, seed, prepared, now):
        today = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
        # 网络请求和来源对照全部在写事务之前，避免长时间持有数据库行锁。
        basics = {row["ts_code"]: self._basic(row["ts_code"], seed) for row in prepared["rows"]}
        verified_dividends = {}
        for code, basic in basics.items():
            verified_dividends[code] = self._dividends_verified(code, basic["mapped"], seed["sourceId"])
        with Session(self.engine) as session, session.begin():
            manager = session.get(FundManager, seed["id"])
            created = manager is None
            if manager is None:
                manager = FundManager(manager_id=seed["id"], anchor_key=seed["key"], display_name=seed["name"])
                session.add(manager)
                session.flush()
            snapshot_id = uuid4()
            funds, education = [], None
            for row in prepared["rows"]:
                code, basic = row["ts_code"], basics[row["ts_code"]]
                related_id = stable_id("fund:" + str(seed["sourceId"]) + ":" + code)
                related = session.get(FundManagerRelatedFund, related_id)
                if related is None:
                    related = FundManagerRelatedFund(
                        related_fund_id=related_id,
                        source_id=seed["sourceId"],
                        source_fund_code=code,
                        fund_name=basic["name"],
                        company_name=basic["company"],
                        fund_type=basic["type"],
                        mapped_fund_code=basic["mapped"],
                    )
                    session.add(related)
                    session.flush()
                tenure_key = stable_id(f"{manager.manager_id}:{code}:{row['begin']}")
                hash_value = digest(
                    {key: row.get(key) for key in ["ts_code", "name", "begin_date", "end_date", "valid"]}
                )
                current = session.scalar(
                    select(FundManagerTenureVersion).where(
                        FundManagerTenureVersion.tenure_key == tenure_key, FundManagerTenureVersion.is_current.is_(True)
                    )
                )
                if current and current.content_hash != hash_value:
                    current.is_current = False
                    session.flush()
                    current = None
                if current is None:
                    current = session.scalar(
                        select(FundManagerTenureVersion).where(
                            FundManagerTenureVersion.tenure_key == tenure_key,
                            FundManagerTenureVersion.content_hash == hash_value,
                        )
                    )
                    if current:
                        current.is_current = True
                    else:
                        current = FundManagerTenureVersion(
                            tenure_key=tenure_key,
                            manager_id=manager.manager_id,
                            related_fund_id=related_id,
                            begin_date=row["begin"],
                            end_date=row["end"],
                            verification_status="CONFIRMED" if row["valid"] else "PENDING",
                            team_status=_team(row, prepared["roster"] if code == seed["sourceCode"] else None),
                            content_hash=hash_value,
                            evidence_refs=prepared["references"],
                            observed_at=now,
                        )
                        session.add(current)
                    session.flush()
                metric = _metric(
                    session,
                    related,
                    row,
                    prepared["roster"] if code == seed["sourceCode"] else None,
                    verified_dividends[code],
                    today,
                )
                roster = prepared["roster"] if code == seed["sourceCode"] else None
                cutoff = day(metric["endDate"]) if metric["endDate"] else min(row["end"] or today, today)
                solo_row = _solo_row(row, roster, cutoff)
                solo = (
                    _metric(session, related, solo_row, roster, verified_dividends[code], today) if solo_row else None
                )
                metric_hash = digest(
                    {
                        "tenure": metric.pop("_inputHash", digest(metric)),
                        "solo": solo.pop("_inputHash", digest(solo)) if solo else None,
                    }
                )
                previous_metric = session.scalar(
                    select(FundManagerMetricSnapshot.metric_id).where(
                        FundManagerMetricSnapshot.tenure_version_id == current.tenure_version_id,
                        FundManagerMetricSnapshot.input_hash == metric_hash,
                        FundManagerMetricSnapshot.method_version == metric["calculationBasis"],
                    )
                )
                if previous_metric is None:
                    session.add(
                        FundManagerMetricSnapshot(
                            manager_id=manager.manager_id,
                            related_fund_id=related_id,
                            tenure_version_id=current.tenure_version_id,
                            return_rate=metric["returnRate"],
                            max_drawdown_rate=metric["maxDrawdownRate"],
                            annualized_rate=metric["annualizedRate"],
                            input_hash=metric_hash,
                            method_version=metric["calculationBasis"],
                            payload={**metric, "soloPerformance": solo},
                            calculated_at=now,
                        )
                    )
                education = education or row.get("edu")
                funds.append(
                    {
                        "relatedFundId": str(related_id),
                        "tenureVersionId": str(current.tenure_version_id),
                        "fundCode": code.split(".")[0],
                        "fundName": related.fund_name,
                        "fundType": related.fund_type,
                        "companyName": related.company_name,
                        "mappedFundCode": related.mapped_fund_code,
                        "state": _state(row, today),
                        "beginDate": str(row["begin"]) if row["begin"] else None,
                        "endDate": str(row["end"]) if row["end"] else None,
                        "teamStatus": current.team_status,
                        "performance": metric,
                        "soloPerformance": solo,
                        "sourceUrl": (self.official_cache.get((code.split(".")[0], seed["name"])) or {}).get(
                            "url", SOURCE_URL
                        ),
                    }
                )
            if prepared["profile"]:
                profile_hash = digest({"resume": prepared["profile"], "education": education})
                if not session.scalar(
                    select(FundManagerProfileVersion.profile_version_id).where(
                        FundManagerProfileVersion.manager_id == manager.manager_id,
                        FundManagerProfileVersion.content_hash == profile_hash,
                    )
                ):
                    session.add(
                        FundManagerProfileVersion(
                            manager_id=manager.manager_id,
                            education=education,
                            professional_resume=prepared["profile"],
                            evidence_ref={"url": SOURCE_URL, "anchor": seed["sourceCode"]},
                            content_hash=profile_hash,
                            received_at=now,
                        )
                    )
            identity = {
                "anchorFundCode": seed["code"],
                "acceptedCodes": sorted(basics),
                "references": prepared["references"],
                "proof": "OFFICIAL_PRODUCT_ROSTER_AND_CAREER"
                if prepared["officialAnchor"]
                else "LOCAL_FUND_ASSIGNMENT",
            }
            identity_hash = digest(identity)
            if not session.scalar(
                select(FundManagerIdentityEvidence.evidence_id).where(
                    FundManagerIdentityEvidence.manager_id == manager.manager_id,
                    FundManagerIdentityEvidence.content_hash == identity_hash,
                )
            ):
                session.add(
                    FundManagerIdentityEvidence(
                        manager_id=manager.manager_id,
                        source_id=seed["sourceId"],
                        decision="ACCEPTED",
                        evidence_ref=identity,
                        content_hash=identity_hash,
                        reviewed_at=now,
                    )
                )
            sources = [{"title": "基金经理任职资料", "url": SOURCE_URL}]
            if prepared["officialAnchor"]:
                sources.append({"title": "基金公司官方产品资料", "url": prepared["officialAnchor"]["url"]})
            payload = {
                "managerId": str(manager.manager_id),
                "managerName": manager.display_name,
                "snapshotId": str(snapshot_id),
                "companies": sorted({item["companyName"] for item in funds if item["companyName"]}),
                "education": education,
                "shareCount": len({item["relatedFundId"] for item in funds if item["state"] == "CURRENT"}),
                "productCount": None,
                "historyCount": len({item["relatedFundId"] for item in funds if item["state"] == "HISTORY"}),
                "anchorFundCode": seed["code"],
                "anchorFundName": seed["fundName"],
                "checkedAt": prepared["checked"],
                "professionalResume": prepared["profile"],
                "disclosures": self.disclosures(seed, today),
                "sources": sources,
                "funds": funds,
                "coverage": {
                    "sourceScanComplete": bool(self.source and prepared["checked"]),
                    "identityComplete": prepared["pending"] == 0 and bool(prepared["checked"]),
                    "productMappingComplete": False,
                    "pendingShareCount": prepared["pending"],
                    "lastSuccessfulCheckAt": prepared["checked"],
                    "notice": "仅展示已核实关联，其他同名任职尚待核对。"
                    if prepared["pending"]
                    else "已收录范围，A/C等份额分别展示。",
                },
            }
            payload = ManagerDetail.model_validate(payload).model_dump(mode="json")
            if payload["shareCount"] == 0 and (
                prepared["pending"] or any(item["state"] == "PENDING" for item in funds)
            ):
                payload["shareCount"] = None
            session.add(
                FundManagerCoverageSnapshot(
                    snapshot_id=snapshot_id,
                    manager_id=manager.manager_id,
                    scope_kind="MANAGER",
                    payload=payload,
                    content_hash=digest(payload),
                    published_at=now,
                )
            )
            manager.current_snapshot_id = snapshot_id
            if prepared["officialAnchor"]:
                manager.identity_status = "VERIFIED"
        # 只把首次建档计为新建；新的资料快照不等于新增一个基金经理。
        return created

    def publish_directory(self):
        with Session(self.engine) as session, session.begin():
            rows = session.execute(
                select(FundManagerCoverageSnapshot)
                .join(FundManager, FundManager.current_snapshot_id == FundManagerCoverageSnapshot.snapshot_id)
                .where(FundManager.redirect_to.is_(None))
            ).scalars()
            items = []
            for snapshot in rows:
                item = {
                    key: value
                    for key, value in snapshot.payload.items()
                    if key not in {"funds", "sources", "disclosures", "professionalResume", "coverage", "stale"}
                }
                item["fundTypes"] = sorted({row["fundType"] for row in snapshot.payload["funds"]})
                items.append(item)
            if len(items) > 5000:
                raise ValueError("经理目录超过当前保护范围")
            items.sort(key=lambda row: (row["managerName"], row["managerId"]))
            payload = {"items": items}
            session.add(
                FundManagerCoverageSnapshot(
                    scope_kind="DIRECTORY",
                    payload=payload,
                    content_hash=digest(payload),
                    published_at=datetime.now(UTC),
                )
            )

    def run(self, manager_ids=(), *, dry_run=False, progress=None, frozen_targets=None):
        targets = self.seeds() if frozen_targets is None else frozen_targets
        if manager_ids:
            targets = [seed for seed in targets if seed["id"] in set(manager_ids)]
            if {seed["id"] for seed in targets} != set(manager_ids):
                raise LookupError("部分经理不在当前已登记范围内。")
        if dry_run:
            return {
                "status": "DRY_RUN",
                "managerCount": len(targets),
                "externalCalls": False,
                "targets": [
                    {"managerId": str(seed["id"]), "name": seed["name"], "anchorFundCode": seed["code"]}
                    for seed in targets
                ],
            }
        result = {
            "status": "SUCCEEDED",
            "managerCount": len(targets),
            "completed": 0,
            "failed": 0,
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "items": [],
        }
        with self.engine.connect() as lock:
            if not lock.scalar(select(func.pg_try_advisory_lock(_MARKET_NAV_INCREMENTAL_LOCK_KEY))):
                raise RuntimeError("已有来源任务正在执行，请稍后重试。")
            try:
                for index, seed in enumerate(targets, 1):
                    try:
                        if not self.source:
                            with Session(self.engine) as session:
                                existing = session.get(FundManager, seed["id"])
                                if existing and existing.current_snapshot_id:
                                    result["completed"] += 1
                                    result["skipped"] += 1
                                    continue
                        prepared = self.prepare(seed, datetime.now(UTC))
                        created = self.publish(seed, prepared, datetime.now(UTC))
                        result["created" if created else "updated"] += 1
                        result["completed"] += 1
                        result["items"].append(
                            {
                                "managerId": str(seed["id"]),
                                "status": "SUCCEEDED",
                                "pendingShareCount": prepared["pending"],
                            }
                        )
                    except Exception as error:
                        logger.exception(
                            "fund_manager_directory_sync.run   >>> manager failed, manager_id=%s", seed["id"]
                        )
                        result["failed"] += 1
                        result["items"].append(
                            {"managerId": str(seed["id"]), "status": "FAILED", "errorType": type(error).__name__}
                        )
                    if progress:
                        progress(index, len(targets), seed["code"], "基金经理资料核对中")
                if self.source:
                    for original in self.source.pages:
                        save_original(original, self.root)
                self.publish_directory()
            finally:
                lock.scalar(select(func.pg_advisory_unlock(_MARKET_NAV_INCREMENTAL_LOCK_KEY)))
        result["pendingIdentityCount"] = sum(bool(item.get("pendingShareCount")) for item in result["items"])
        result["status"] = "PARTIAL_SUCCESS" if result["failed"] or result["pendingIdentityCount"] else "SUCCEEDED"
        if targets and result["failed"] == len(targets):
            result["status"] = "FAILED"
        return result


def synchronize(manager_ids=(), *, dry_run=False, fetch=False, progress=None):
    """HTTP与CLI共用入口；真实读取先检查本机来源授权，不改变原市场目录。"""
    source, run_id = None, None
    maintenance = ManagerDirectorySync()
    frozen_targets = maintenance.seeds()
    if manager_ids:
        frozen_targets = [seed for seed in frozen_targets if seed["id"] in set(manager_ids)]
        if {seed["id"] for seed in frozen_targets} != set(manager_ids):
            raise LookupError("部分经理不在当前已登记范围内。")
    if fetch:
        with Session(get_engine()) as session:
            registry = session.scalar(select(SourceRegistry).where(SourceRegistry.source_code == "TUSHARE_PRO_FUND"))
            if (
                registry is None
                or not registry.enabled
                or not {"fund_manager", "fund_basic", "fund_div"} <= set(registry.authorized_api_names)
            ):
                raise ValueError("基金经理资料所需来源尚未授权。")
        if not dry_run:
            source = ManagerSource()
            with Session(get_engine()) as session, session.begin():
                run = SourceSyncRun(
                    source_id=registry.source_id,
                    sync_type="FUND_MANAGERS",
                    status="RUNNING",
                    scope_snapshot={
                        "managerIds": [str(seed["id"]) for seed in frozen_targets],
                        "anchorFundCodes": sorted({seed["code"] for seed in frozen_targets}),
                        "scope": "REGISTERED_ANCHORS",
                    },
                )
                session.add(run)
                session.flush()
                run_id = run.sync_run_id
    try:
        maintenance.source = source
        result = maintenance.run(manager_ids, dry_run=dry_run, progress=progress, frozen_targets=frozen_targets)
        if run_id:
            result["fetchedRows"] = sum(len(page["rows"]) for page in source.pages)
            result["syncRunId"] = str(run_id)
            with Session(get_engine()) as session, session.begin():
                run = session.get(SourceSyncRun, run_id)
                run.status, run.finished_at = result["status"], datetime.now(UTC)
                run.fetched_count = result["fetchedRows"]
                run.created_count, run.updated_count = result["created"], result["updated"]
                run.skipped_count = result["skipped"]
                run.result_summary = result
        return result
    except Exception:
        if run_id:
            with Session(get_engine()) as session, session.begin():
                run = session.get(SourceSyncRun, run_id)
                run.status, run.finished_at = "FAILED", datetime.now(UTC)
                run.error_summary = "经理资料核对未完成，详见同次服务日志。"
        raise
    finally:
        if source:
            source.close()
