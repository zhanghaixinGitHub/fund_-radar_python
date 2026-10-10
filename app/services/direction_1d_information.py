"""002112 用户明确选择的综合信息路径；原研究只读，运行时不拟合、不采集。

复用冻结输入层的纯特征函数，避免另写一套近似公式。登记包携带该层代码摘要；
更新特征配方必须重新导出核验，不能静默使训练与实际推理脱节。
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit
from scripts import fund_002112_event_comparison_data_v1 as recipe
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import text

from app.services import direction_1d_events as event_model
from app.services.direction_1d_protocol import ZONE, canonical, digest, window
from app.services.fund_exposure_common import ROOT, read
from app.services.fund_materials_store import source_path

FEATURE_VERSION = "002112_FULL_INFORMATION_V1"
SUPPORTED_VERSIONS = (FEATURE_VERSION, event_model.VERSION)
ACTIVE_FILE = ROOT.parent / "direction-1d-information/active.json"
GROUPS = {"MORNING_0830": "CN_MIXED_002112_INFO_AM", "EVENING_2300": "CN_MIXED_002112_INFO_PM"}
FACTOR_NAMES = {
    "nav_history": "基金净值走势",
    "holding_market": "已披露持仓与大盘表现",
    "company_announcements": "公司公告事实",
    "policy_information": "政策相关资料",
    "news_information": "新闻相关资料",
    "document_content": "公告和消息正文",
}


def selection(code: str, now: datetime) -> dict | None:
    """只对显式登记的002112启用；目标日当天取晨间包，之前取晚间包。"""
    if code != "002112" or not ACTIVE_FILE.exists():
        return None
    active = read(ACTIVE_FILE)
    if active.get("fund_code") != code or active.get("feature_version") not in SUPPORTED_VERSIONS:
        raise ValueError("INFORMATION_REGISTRATION_INVALID")
    if not active.get("enabled"):
        return None
    phase = "MORNING_0830" if str(now.astimezone(ZONE).date()) == window(now)["target_nav_date"] else "EVENING_2300"
    groups = event_model.GROUPS if active["feature_version"] == event_model.VERSION else GROUPS
    return {**active["models"][phase], "phase": phase, "group_id": groups[phase],
            "feature_version": active["feature_version"]}


def apply_mapping(mapping: dict, selected: dict | None) -> dict:
    if not selected:
        return mapping
    return {
        **mapping,
        "group_id": selected["group_id"],
        "group_evidence": {
            **mapping["group_evidence"],
            "asset_group": mapping["group_id"],
            "input_policy": selected.get("feature_version", FEATURE_VERSION),
        },
    }


def validate_model(model: dict) -> None:
    """扩展包单独核验，不降低普通基金原三态训练门槛。持平无训练样本明确记为不支持。"""
    if model.get("feature_version") == event_model.VERSION:
        event_model.validate_model(model)
        return
    if (
        model.get("feature_version") != FEATURE_VERSION
        or model.get("fund_code") != "002112"
        or model.get("protocol") != "DIRECTION_1D_V2"
        or model.get("features") != recipe.NAMES["C_EVENTS"]
        or model.get("classes") != ["DOWN", "UP"]
        or model.get("unsupported_classes") != ["FLAT"]
        or model.get("adoption_basis") != "EXPLICIT_USER_SELECTION"
        or model.get("group_id") not in GROUPS.values()
    ):
        raise ValueError("INFORMATION_MODEL_INVALID")
    n = len(model["features"])
    dimensions = 2 * n - 7 + len(model["tfidf"]["vocabulary"])
    for key, size in (("medians", n), ("mean", 2 * n - 7), ("scale", 2 * n - 7), ("coef", dimensions)):
        if len(model.get(key, [])) != size or any(not math.isfinite(v) for v in model[key]):
            raise ValueError("INFORMATION_MODEL_INVALID")
    if any(v <= 0 for v in model["scale"]) or not math.isfinite(model["intercept"]):
        raise ValueError("INFORMATION_MODEL_INVALID")
    vocabulary, idf = model["tfidf"]["vocabulary"], model["tfidf"]["idf"]
    if (
        not 1 <= len(vocabulary) <= 256
        or sorted(vocabulary.values()) != list(range(len(vocabulary)))
        or len(idf) != len(vocabulary)
        or any(not math.isfinite(v) or v <= 0 for v in idf)
    ):
        raise ValueError("INFORMATION_MODEL_INVALID")
    for filename, expected in model["recipe_hashes"].items():
        path = recipe.PY / filename
        if not path.resolve().is_relative_to(recipe.PY) or recipe.sha(path) != expected:
            raise ValueError("INFORMATION_RECIPE_CHANGED")


@lru_cache(maxsize=8)
def text_transformer(spec: str):
    value = json.loads(spec)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), vocabulary=value["vocabulary"])
    vectorizer.idf_ = np.asarray(value["idf"])
    return vectorizer


def transform(model: dict, information: dict) -> np.ndarray:
    """严格使用原训练包的中位数、缺失标志、标准化与词表；任何步骤均不fit。"""
    raw = information.get("numeric")
    if (
        not isinstance(raw, list)
        or len(raw) != len(model["features"])
        or any(
            v is not None and (isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v))
            for v in raw
        )
        or any(v is None for v in raw[:7])
    ):
        raise ValueError("INFORMATION_INPUT_INVALID")
    content = information.get("text")
    if not isinstance(content, str) or len(content) > 25000:
        raise ValueError("INFORMATION_TEXT_INVALID")
    values = np.asarray([np.nan if v is None else v for v in raw])
    filled = np.where(np.isnan(values), model["medians"], values)
    numeric = np.r_[filled, np.isnan(values[7:]).astype(float)]
    numeric = (numeric - np.asarray(model["mean"])) / np.asarray(model["scale"])
    words = text_transformer(canonical(model["tfidf"])).transform([content]).toarray()[0]
    return np.r_[numeric, words]


def predict(model: dict, information: dict) -> dict:
    if model.get("feature_version") == event_model.VERSION:
        result = event_model.predict(model, information)
        # 历史少数出手仅50%正确不能证明可用；数据接入和实际输出方向分别把关。
        # 研究复算仍保留原始打分，正式读回仅采用登记包明确通过独立验证的版本。
        if model.get("validation_status") != "INDEPENDENT_VALIDATION_PASSED":
            result.update(direction=None, runner=None, score=None, class_scores=None, status="ABSTAINED")
            result["decision"]["reason_codes"].append("VALIDATION_INSUFFICIENT")
        return result
    x = transform(model, information)
    up = float(expit(float(x @ np.asarray(model["coef"]) + model["intercept"])))
    winner = "UP" if up >= 0.5 else "DOWN"
    return {
        "direction": winner,
        "runner": "DOWN" if winner == "UP" else "UP",
        "score": max(up, 1 - up),
        "class_scores": {"DOWN": 1 - up, "FLAT": 0.0, "UP": up},
    }


def receipt_available(receipt: dict, now: datetime) -> str:
    """核对本地原件、接收时间与保留期；不由历史公开日期伪造首次接收。"""
    received = recipe.moment(receipt["received_at"])
    if received > now or (receipt.get("expires_at") and recipe.moment(receipt["expires_at"]) <= now):
        raise ValueError("INFORMATION_SOURCE_NOT_AVAILABLE")
    path = ROOT / receipt["file"] if "file" in receipt else Path(receipt["raw_path"])
    if not path.resolve().is_relative_to(ROOT.parent.resolve()) or recipe.sha(path) != receipt["sha256"]:
        raise ValueError("INFORMATION_SOURCE_HASH_MISMATCH")
    return received.isoformat()


def current_sources(base: str, now: datetime, connection) -> tuple[dict, dict]:
    """融合冻结事实与手动同步已保存的新原件；不调用采集服务、不读取目标日答案。"""
    source = recipe.load_sources()
    # 快照格式不同的冻结原件只有明确登记摘要吻合才可使用。
    active = read(ACTIVE_FILE)
    for filename, expected in active["source_hashes"].items():
        if recipe.sha(recipe.PY / filename) != expected:
            raise ValueError("INFORMATION_FROZEN_SOURCE_CHANGED")
    cutoff = now.astimezone(ZONE).isoformat()
    versions = {"frozen": active["source_hashes"], "quotes": {}, "reports": {}, "documents": {}}
    from app.services.fund_exposure_quotes import reports

    current_reports = reports()
    for report in current_reports:
        raw = report["raw"]
        try:
            received = receipt_available(raw, now)
        except ValueError as error:
            if str(error) == "INFORMATION_SOURCE_NOT_AVAILABLE":
                continue
            raise
        # 接收时间用于证明现在可见；报告新旧仍按来源公开时间，避免晚下载的旧季报挤掉半年报。
        versions["reports"][raw["sha256"]] = {
            "available_at": report["available_at"],
            "received_at": received,
        }
    # 当前成功报告清单具有撤回语义；不混入已从清单消失的历史版本。
    source["reports"] = [r for r in current_reports if r["raw"]["sha256"] in versions["reports"]]
    report = recipe.legacy.choose_report(source["reports"], cutoff)
    if not report or (now.date() - datetime.fromisoformat(report["report_end"]).date()).days > 210:
        raise ValueError("INFORMATION_HOLDINGS_NOT_READY")
    weights = recipe.legacy.holding_weights(report)
    i = source["sessions"].index(base)
    days = source["sessions"][i - 20 : i + 1]
    # 运行时价格完全取当前留存原件，缺口仍为空；不能沿用冻结包中的过期来源兜底。
    source["stocks"] = {}
    for day in days:
        path = ROOT / "stock-days" / (day + ".json")
        if not path.exists():
            continue
        value = read(path)
        try:
            available = receipt_available(value["receipt"], now)
        except ValueError as error:
            if str(error) == "INFORMATION_SOURCE_NOT_AVAILABLE":
                continue
            raise
        source["stocks"][day] = {
            code: {**row, "available_at": available} for code, row in value["rows"].items() if code in weights
        }
        versions["quotes"][day] = value["receipt"]["sha256"]
    source["markets"] = {"000300.SH": {}, "000905.SH": {}}
    for code in source["markets"]:
        value = read(ROOT / "indices" / (code + ".json"))
        # 每个行情值必须能在已验证、未过期的原响应中找到；不凭合并文件日期猜来源。
        for receipt in value["receipts"]:
            start = receipt["params"].get("start_date", "")
            end = receipt["params"].get("end_date", "")
            if end < days[0].replace("-", "") or start > base.replace("-", ""):
                continue
            try:
                available = receipt_available(receipt, now)
            except ValueError as error:
                if str(error) == "INFORMATION_SOURCE_NOT_AVAILABLE":
                    continue
                raise
            raw = json.loads((ROOT / receipt["file"]).read_bytes())["data"]
            for values in raw["items"]:
                row = dict(zip(raw["fields"], values, strict=True))
                day = datetime.strptime(row["trade_date"], "%Y%m%d").date().isoformat()
                if day in days and row["ts_code"] == code:
                    source["markets"][code][day] = {**row, "available_at": available}
                    versions["quotes"][code + ":" + day] = receipt["sha256"]
    add_current_documents(source, report, now, versions)
    add_database_news(source, connection, now, versions)
    source["documents"].sort(key=lambda item: (item["available_at"], item["id"]))
    return source, versions


def add_current_documents(source: dict, report: dict, now: datetime, versions: dict) -> None:
    """新公告只使用已核对正文片段；不凭标题制造经营数字、已实施政策或利好方向。"""
    from app.services.company_news_facts import company_identity_evidence, inspect_company, legacy_catalog_evidence

    cutoff_day = str(now.date() - timedelta(days=45))
    existing = {event.get("source_hash") for event in source["documents"]}
    # 旧核验事实的摘要与PDF原件摘要不是同一口径；同一公告必须再按来源编号去重，
    # 否则一份公告会同时作为历史事实和新下载正文被重复计入。已有核验事实优先保留。
    known_documents = {event.get("document_id") for event in source["documents"]}
    considered = 0
    for holding in report["holdings"]:
        code = holding["stock_code"]
        catalog_path = source_path(ROOT, "supplement/company-announcements/" + code + ".json")
        if not catalog_path.exists():
            continue
        catalog = read(catalog_path)
        for item in catalog["rows"]:
            if "company-" + item["announcementId"] in known_documents:
                continue
            if not cutoff_day <= item.get("announced_at_source", "")[:10] <= str(now.date()):
                continue
            considered += 1
            if considered > 500:
                raise ValueError("INFORMATION_DOCUMENT_LIMIT")
            path = source_path(ROOT, "supplement/company-documents/" + digest(item["announcementId"]) + ".json")
            if not path.exists():
                continue
            document = read(path)
            receipt = document["receipt"]
            if receipt["sha256"] in existing:
                continue
            try:
                available = receipt_available(receipt, now)
            except ValueError as error:
                if str(error) == "INFORMATION_SOURCE_NOT_AVAILABLE":
                    continue
                raise
            try:
                inspect_company(
                    item,
                    document,
                    {"stockCode": code, "stockName": holding["stock_name"], "weightPct": holding["nav_weight_pct"]},
                    {
                        "id": report["raw"]["sha256"],
                        "endDate": report["report_end"],
                        "publishedDate": report["published_date"],
                    },
                    now,
                    legacy_catalog=legacy_catalog_evidence(item, document, code),
                    identity_evidence=company_identity_evidence(code),
                )
            except ValueError as error:
                if not str(error).startswith("COMPANY_PUBLICATION_NOT_VERIFIED:"):
                    raise
                # 未核验公告留存排除原因；不能把其文本当成已可用的事实输入。
                versions.setdefault("excluded_documents", {})[item["announcementId"]] = str(error)
                continue
            body = re.sub(r"\s+", " ", " ".join(document.get("pages", [])))
            if not body:
                continue
            title = document["title"]
            uncertain = bool(recipe.UNCERTAIN_TERMS.search(title + body[:500]))
            source["documents"].append(
                {
                    "id": "LIVE_CNINFO:" + item["announcementId"],
                    "document_id": "company-" + item["announcementId"],
                    "event_type": "ANNOUNCEMENT",
                    "source_kind": "company",
                    "available_at": max(available, item["announced_at_source"]),
                    "published_date": item["announced_at_source"][:10],
                    "title": title,
                    "facts": [body[:1200]],
                    "source_url": receipt["url"],
                    "source_hash": receipt["sha256"],
                    "stage": "UNKNOWN",
                    "direction": "UNKNOWN",
                    "uncertain_claim": uncertain,
                    "links": [{"code": code, "at_event_weight": float(holding["nav_weight_pct"]) / 100}],
                }
            )
            existing.add(receipt["sha256"])
            known_documents.add("company-" + item["announcementId"])
            versions["documents"][item["announcementId"]] = receipt["sha256"]


def add_database_news(source: dict, connection, now: datetime, versions: dict) -> None:
    """只读取未过期且有已审核事件的现存消息，使用真实接收/批准时间；无需外部调用。"""
    rows = (
        connection.execute(
            text("""
        SELECT n.news_id,n.title,n.url,n.summary,n.content_hash,n.published_at,n.fetched_at,
               e.event_type,e.approved_at
        FROM news_item n JOIN market_event e ON e.news_id=n.news_id
        WHERE n.active AND n.retention_until>:now AND n.published_at BETWEEN :start AND :now
          AND n.fetched_at<=:now AND e.approval_status='APPROVED' AND e.approved_at<=:now
        ORDER BY n.published_at,n.news_id LIMIT 501
    """),
            {"now": now, "start": now - timedelta(days=45)},
        )
        .mappings()
        .all()
    )
    if len(rows) > 500:
        raise ValueError("INFORMATION_NEWS_LIMIT")
    existing = {event.get("source_hash") for event in source["documents"]}
    for row in rows:
        if not row["summary"] or row["content_hash"] in existing:
            continue
        content = row["title"] + row["summary"]
        terms = recipe.links.topic_terms(content)
        kind = "POLICY" if row["event_type"] == "INDUSTRY_POLICY" else "NEWS"
        source["documents"].append(
            {
                "id": "LIVE_NEWS:" + str(row["news_id"]),
                "event_type": kind,
                "source_kind": kind.lower(),
                "title": row["title"],
                "facts": [row["summary"][:1200]],
                "source_url": row["url"],
                "source_hash": row["content_hash"],
                "published_date": str(row["published_at"].astimezone(ZONE).date()),
                "available_at": max(row["published_at"], row["fetched_at"], row["approved_at"])
                .astimezone(ZONE)
                .isoformat(),
                "stage": "UNKNOWN",
                "direction": "UNKNOWN",
                "uncertain_claim": bool(recipe.UNCERTAIN_TERMS.search(content)),
                "targets": [{"term": term, "quote": row["summary"][:1200]} for term in terms.values()],
            }
        )
        existing.add(row["content_hash"])
        versions["documents"][str(row["news_id"])] = row["content_hash"]


def build_input(nav_features: list, base: str, now: datetime, connection, *, version=FEATURE_VERSION) -> dict:
    source, versions = current_sources(base, now, connection)
    cutoff = now.astimezone(ZONE).isoformat()
    market, market_proof = recipe.legacy.market_features(base, cutoff, source)
    if market_proof["coverage_weight"] <= 0 or any(
        source["markets"][code].get(base) is None for code in source["markets"]
    ):
        raise ValueError("INFORMATION_MARKET_NOT_READY")
    events, content, proof = recipe.event_features(source, base, cutoff)
    counts = {kind: proof["counts"].get(kind, 0) for kind in recipe.KINDS}
    sources = [
        {
            "id": e["id"],
            "kind": e["kind"],
            "title": e["title"],
            "published_date": e["published_date"],
            "available_at": e["available_at"],
            "source_hash": e["source_hash"],
            "source_url": e["source_url"],
        }
        for e in proof["events"]
        if e["id"] in proof["text_event_ids"]
    ]
    limitations = [
        "历史验证尚未证明判断稳定有效，结果仅供参考。",
        "持仓来自已披露报告，可能不同于当日真实持仓。",
        "新闻和政策资料覆盖不完整；未找到相关消息不代表没有风险。",
        "持平样本不足，当前判断主要区分上涨与下跌。",
    ]
    result = {
        "version": version,
        "numeric": nav_features + market + events,
        "text": content,
        "source_identity": digest(versions),
        "source_versions": versions,
        "market_date": base,
        "holding_report_date": recipe.legacy.choose_report(source["reports"], cutoff)["report_end"],
        "holding_coverage": market_proof["coverage_weight"],
        "counts": counts,
        "sources": sources,
        "text_source_count": len(sources),
        "collection_complete": False,
        "public_material_through": max(
            (e["published_date"] or "" for e in sources if e["kind"] != "ANNOUNCEMENT"), default=None
        ),
        "limitations": limitations,
    }
    if version == event_model.VERSION:
        result["event_evidence"] = event_model.build_evidence(proof, base, cutoff, source["sessions"])
    return result


def explain(model: dict, information: dict) -> dict:
    """把全部非零作用归入业务信息类别；和原模型逐项加总一致，不虚构市场因果。"""
    if model.get("feature_version") == event_model.VERSION:
        restored = event_model.explain(model, information)
        decision = predict(model, information)
        restored.update(direction=decision["direction"], referenceDirection=decision["runner"],
                        status=decision["status"], decision=decision["decision"])
        return restored
    result = predict(model, information)
    orientation = 1 if result["direction"] == "UP" else -1
    impacts = transform(model, information) * np.asarray(model["coef"]) * orientation
    n = len(model["features"])
    totals = dict.fromkeys(FACTOR_NAMES, 0.0)
    for i, value in enumerate(impacts):
        if i >= 2 * n - 7:
            group = "document_content"
        else:
            original = i if i < n else i - n + 7
            name = model["features"][original]
            group = (
                "nav_history"
                if original < 7
                else "holding_market"
                if original < 29
                else "policy_information"
                if name.startswith("POLICY_")
                else "news_information"
                if name.startswith("NEWS_")
                else "company_announcements"
            )
        totals[group] += float(value)
    quantities = {
        "nav_history": 61.0,
        "holding_market": information["holding_coverage"],
        "company_announcements": float(information["counts"]["ANNOUNCEMENT"]),
        "policy_information": float(information["counts"]["POLICY"]),
        "news_information": float(information["counts"]["NEWS"]),
        "document_content": float(len(information["text"])),
    }
    return {
        "direction": result["direction"],
        "referenceDirection": result["runner"],
        "intercept": model["intercept"] * orientation,
        "factors": [
            {"feature": name, "value": quantities[name], "contribution": value} for name, value in totals.items()
        ],
        "information": {
            "marketDate": information["market_date"],
            "holdingReportDate": information["holding_report_date"],
            "holdingCoverage": information["holding_coverage"],
            "counts": information["counts"],
            "publicMaterialThrough": information["public_material_through"],
            "limitations": information["limitations"],
        },
    }


def narrative(restored: dict) -> dict:
    """确定性组织保存的综合输入，避免旧净值文案覆盖新模型；不调用外部文字模型。"""
    branches = list({b["modelHash"]: b for b in restored["branches"]}.values())
    if len({b["direction"] for b in branches}) != 1:
        raise ValueError("DIRECTION_DISAGREEMENT")
    branch = branches[0]
    facts = branch["information"]
    names = {"ANNOUNCEMENT": "公告", "POLICY": "政策", "NEWS": "新闻"}
    available = [name for key, name in names.items() if facts["counts"][key] > 0]
    missing = [name for key, name in names.items() if facts["counts"][key] == 0]
    label = "上涨" if branch["direction"] == "UP" else "下跌"
    summary = f"综合当时可用的净值、持仓行情{'、' + '、'.join(available) if available else ''}，判断偏向{label}。"
    context = (
        f"持仓依据截至 {facts['holdingReportDate']} 的披露报告，股票及大盘行情截至 {facts['marketDate']}。"
        f"可取得行情的披露股票合计占报告净资产的 {facts['holdingCoverage'] * 100:.2f}%。"
    )
    if available:
        context += (
            "本次使用的相关资料包括"
            + "、".join(f"{name}{facts['counts'][key]}条" for key, name in names.items() if facts["counts"][key] > 0)
            + "，同一事项可能有多份文件。"
        )
    if missing:
        context += "此次未找到可用的相关" + "、".join(missing) + "，不代表没有相关事件或风险。"
    return {
        "styleVersion": "PREDICTION_INFORMATION_ZH_V1",
        "summary": summary,
        "context": context,
        "supporting": "",
        "opposing": "",
        "limitations": facts["limitations"],
    }
