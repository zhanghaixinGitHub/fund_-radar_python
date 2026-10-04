"""既有新闻公告与披露持仓纠偏：逐项解释时间证据，先冻结协议后监督训练。"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

from pypdf import PdfReader

from scripts import fund_002112_existing_data_inputs_v1 as inputs
from scripts import fund_002112_holding_impact_v1 as impact
from scripts import fund_002112_morning_coverage_revision_v1 as morning

io = impact.io
ROOT = io.RESEARCH / "news-holdings-correction/20261001-v1"
U = io.RESEARCH / "update-frequency-development/20261001-v1"
P = io.RESEARCH / "price-volume-development/20261001-v1"
V = io.RESEARCH / "event-time-combination-review/20261001-v1"

# 固定字面语义词典。方向仅表示标题明确声称的经营变化，不产生股价方向标签。
LEXICON = {
    "earnings": "业绩|净利润|营业收入|盈利|亏损",
    "order": "中标|订单|采购合同|销售合同",
    "buyback": "回购",
    "financing": "融资|发行股票|增发|募集资金|可转债|债券发行",
    "dividend": "分红|权益分派|利润分配|派息",
    "governance": "董事|监事|股东大会|高管|会计师|基金经理",
    "pricing_payment": "价格|支付|结算|收费|集采|集中采购",
    "insurance_access": "医保|医疗保障|目录|准入|医药",
    "regulation": "监管|处罚|违法|违规|自查|合规|审核",
    "innovation": "创新|新药|研发|临床|新技术|新产品",
    "action_issue": "发布|印发|出台|公告|通知",
    "action_change": "调整|变更|修订|更正|补充",
    "action_expand": "扩大|扩围|新增|纳入|增加",
    "action_reduce": "减少|取消|终止|撤销|暂停",
    "action_implement": "实施|落实|执行|开展|推进|落地",
    "stage_proposal": "拟|草案|征求意见|建议",
    "stage_forecast": "预告|预计|预测|预期",
    "stage_pilot": "试行|试点|探索",
    "stage_realized": "完成|已获|获批|取得|决议|报告|快报",
    "stage_correction": "更正|修订|补充|澄清",
    "uncertainty": "风险|可能|不确定|尚未|预计|拟",
    "operating_increase": "(?:净利润|营业收入|需求|盈利).{0,10}(?:增长|增加|上升|改善)",
    "operating_decrease": "(?:净利润|营业收入|需求|盈利).{0,10}(?:下降|减少|下滑)|亏损",
    "cost_increase": "成本.{0,8}(?:增加|上涨|上升)",
    "cost_decrease": "成本.{0,8}(?:下降|减少|降低)",
}


def log(message):
    """实时追加进展；原阶段文件保持不可变。"""
    with (ROOT / "execution-report.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n{io.now()}：{message}\n")


class Reader:
    """只读取既有注册摘要匹配的字节；每个实际入口另记读取账本。"""

    def __init__(self):
        self.registry = io.read(ROOT / "source-registry.json")["files"]
        supplement = ROOT / "source-registry-supplement.json"
        if supplement.exists():
            self.registry.update(io.read(supplement)["files"])
        self.seen = set()

    def path(self, path):
        path = Path(path).resolve()
        key = str(path)
        expected = self.registry[key]
        if key not in self.seen:
            assert io.sha(path) == expected, key
            io.append(ROOT / "read-ledger.jsonl", {"at": io.now(), "path": key, "sha256": expected})
            self.seen.add(key)
        return path

    def read(self, path):
        return io.read(self.path(path))

    def lines(self, path):
        return io.lines(self.path(path))


def pdf_day(value):
    """PDF时刻只提供生成/修改证据，不单独提供公众可见时间。"""
    matched = re.match(r"D:(\d{4})(\d{2})(\d{2})", value or "")
    return "-".join(matched.groups()) if matched else None


def prepare():
    """锁定六组与时间重建规则；不加载或生成涨跌标签。"""
    if (ROOT / "protocol.json").exists():
        return io.read(ROOT / "protocol.json")
    supplement = {}
    inc_path = inputs.DEFAULT_ROOT / "handoff/data/revisions/v2/source-increment.json"
    inc = io.read(inc_path)
    supplement[str(inc_path)] = io.sha(inc_path)
    for record in inc["new_sources"]:
        supplement[str(inc_path.parent / record["snapshot_path"])] = record["sha256"]
    for name in ("coverage-source-manifest.json", "event-time-boundaries.jsonl"):
        supplement[str(V / name)] = io.sha(V / name)
    supplement.update(io.read(V / "coverage-source-manifest.json")["files"])
    io.save(ROOT / "source-registry-supplement.json", {"at": io.now(), "files": supplement})
    rd = Reader()
    same_site = []
    for record in inc["new_sources"]:
        if not record["source_path"].endswith(".json"):
            continue
        doc = io.payload(rd.read(inc_path.parent / record["snapshot_path"]))
        native = doc.get("catalog", {})
        if native:
            same_site.append({
                "source": str(inc_path.parent / record["snapshot_path"]), "sha256": record["sha256"],
                "title": native.get("title"), "content_id": native.get("contentId"),
                "times": {k: datetime.fromtimestamp(int(native[k]) / 1000, io.ZONE).isoformat()
                          if native.get(k) else None for k in
                          ("activationDate", "creationDate", "modificationDate", "publishDate")},
                "raw_fields": {k: native.get(k) for k in
                               ("activationDate", "creationDate", "modificationDate", "publishDate")},
            })
    report_evidence = rd.read(io.RESEARCH / "development-only-optimization/20260930-v1/report-time-evidence.json")
    saved_reports = rd.read(impact.ROOT / "reports.json")
    reports_by_hash = {r["raw"]["sha256"]: r for r in saved_reports}
    audits, admitted = [], []
    for evidence in report_evidence:
        pdf = PdfReader(rd.path(evidence["raw_path"]))
        cover = pdf.pages[0].extract_text() or ""
        compact = re.sub(r"\s+", "", cover)
        match = re.search(r"送出日期[：:]?(\d{4})年(\d{1,2})月(\d{1,2})日", compact)
        declared = date(*(int(v) for v in match.groups())).isoformat() if match else None
        src = evidence["source"]
        activation = datetime.fromtimestamp(int(src["catalog_activation_ms"]) / 1000, io.ZONE)
        catalog_publish = datetime.fromtimestamp(int(src["catalog_publish_ms"]) / 1000, io.ZONE)
        created = pdf_day(pdf.metadata.get("/CreationDate"))
        modified = pdf_day(pdf.metadata.get("/ModDate"))
        url_dates = re.findall(r"20\d{2}-\d{2}-\d{2}", unquote(src["url"]))
        reasons = []
        if declared != evidence["declared_publication"] or activation.date().isoformat() != declared:
            reasons.append("COVER_AND_ACTIVATION_PUBLICATION_CONFLICT")
        if created is None or modified is None:
            reasons.append("PDF_METADATA_MISSING")
        elif declared and max(created, modified) > declared:
            reasons.append("PDF_CREATED_OR_MODIFIED_AFTER_DECLARED_DAY")
        if url_dates and declared not in url_dates:
            reasons.append("URL_DAY_CONFLICT")
        available = (date.fromisoformat(declared) + timedelta(days=1)).isoformat() + "T00:00:00+08:00"
        # 晚CMS发布与同站旧creation/modification相互独立；不把字段名当正文修订事实。
        audit = {
            "raw_sha256": evidence["raw_sha256"], "raw_path": evidence["raw_path"],
            "report_end": evidence["report_end"], "cover_publication": declared,
            "cover_quote": cover[-240:], "activation": activation.isoformat(),
            "cms_publish": catalog_publish.isoformat(), "pdf_created_day": created,
            "pdf_modified_day": modified, "url": src["url"], "url_dates": url_dates,
            "local_received_at": src["catalog_received_at"], "known_body_revision_at": None,
            "admitted_for_reconstructed_research": not reasons,
            "available_at": available if not reasons else None, "exclusions": reasons,
            "cms_interpretation": "CATALOG_RECORD_PUBLISH_FIELD_NOT_PROOF_OF_BODY_REVISION_OR_FIRST_PUBLICATION",
            "limitation": "No contemporaneous byte archive; reconstruction supported by cover/activation/PDF metadata",
        }
        audits.append(audit)
        if not reasons:
            report = dict(reports_by_hash[evidence["raw_sha256"]])
            report["available_at"] = available
            report["time_reconstruction_evidence"] = audit
            admitted.append(report)
    io.save(ROOT / "report-time-readmission.json", audits)
    io.save(ROOT / "same-site-cms-evidence.json", same_site)
    io.save(ROOT / "admitted-reports.json", admitted)
    docs = rd.lines(impact.ROOT / "documents.jsonl")
    body = [d for d in docs if d["text_level"] != "TITLE_ONLY"]
    statuses = Counter()
    quote_errors = []
    for doc in body:
        item = rd.read(impact.ROOT / "validated-v4" / (doc["event_id"] + ".json"))
        statuses[item["status"]] += 1
        assert item["text_sha256"] == doc["text_sha256"]
        if item["status"] != "SOURCE_GROUNDED":
            continue
        for quote in item.get("facts", []) + [item.get("impact_quote", ""), item.get("period_quote", "")]:
            if quote and impact.quote_text(quote) not in impact.quote_text(doc["text"]):
                quote_errors.append({"event_id": doc["event_id"], "quote": quote})
    io.save(ROOT / "body-reuse-audit.json", {
        "at": io.now(), "body_documents": len(body), "statuses": dict(statuses),
        "body_years": dict(Counter(d["published_date"][:4] for d in body)),
        "2025_body_documents": sum(d["published_date"].startswith("2025") for d in body),
        "2025_zero_reason": "ACTUAL_FROZEN_BODY_CORPUS_ENDS_IN2023_NOT_NEW_TIMING_FILTER",
        "quote_errors": quote_errors, "new_llm_requests": 0,
    })
    io.save(ROOT / "title-lexicon.json", LEXICON)
    plan = {
        "at": io.now(), "objective": "Improve002112next-trading-day prediction with actual existing news/announcements/holdings",
        "horizon": "D0800_TO_NEXT_U_NAV_U_VS_NAV_D", "last_label_date": "2025-12-31",
        "development": "SAME_PREVIOUSLY_SEEN243_DATES_NOT_BLIND", "max_actual_supervised_fits": 80,
        "candidates": [
            {"id": "NNE", "groups": ["N", "NE"], "reuse": True},
            {"id": "NNE_H", "groups": ["N", "NE", "H"]},
            {"id": "NNE_T", "groups": ["N", "NE", "T"]},
            {"id": "NNE_HT", "groups": ["N", "NE", "H", "T"]},
            {"id": "NNE_HTB", "groups": ["N", "NE", "H", "T", "B"]},
            {"id": "NNE_HTBO", "groups": ["N", "NE", "H", "T", "B", "O"]},
        ],
        "learner": {"recipe": "RF_D4", "n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10},
        "update_calendar": str(U / "update-calendar.json"), "cadence": 20,
        "maturity": "SAME_FROZEN_CALENDAR_STRICTLY_BEFORE_UPDATE_CUTOFF",
        "preprocessing": "FIT_MEDIAN_MISSING_INDICATOR_STANDARDIZATION_ON_TRAINING_ROWS_ONLY",
        "maximum_replay_fits": 1, "replay": "BEST_NEW_UNIQUE_CANDIDATE_LAST_UPDATE_ABS_TOL1e-13_DIRECTION_IDENTICAL",
        "selection": "CORRECT_DESC_MIN_FOLD_ACCURACY_DESC_ID_ASC",
        "equivalence": "EMPTY_GROUP_OR_IDENTICAL_ENTIRE_MATRIX_ALIAS_NO_DUPLICATE_FIT",
        "H": "Latest admitted disclosed report atD08; original9holding-price features on latest21known sessions; missing remainsNone",
        "T": "Literal title semantic fields plus counts; independent104/46/complete14 and issuer announcements matched to holdings already disclosed atpublication andD08",
        "title_semantics": "Fixed lexicon multi-hot topics/actions/stages/uncertainty/explicit operating trends plus quoted numbers; unknown price impact; no LLM memory",
        "title_windows": [1, 5, 20], "title_sources": ["INDEPENDENT_NEWS_POLICY", "TIME_MATCHED_COMPANY_ANNOUNCEMENT"],
        "title_aggregation": "count, mean semantic indicators, mean signed operating claim, numeric-quote share, mean capped log number and quoted percentage",
        "B": "Reuse validated quoted body semantics: recent20sessions by two source groups, plus current disclosed holding weighted latest admissible company body within730calendar days with age/coverage; old text not a2025new article",
        "O": "Already frozen4market plus10market_extra features; no new numeric financial admission",
        "event_time": "Reuse R007 explicit publication/revision bounds, date-only nextcalendar00 thenfirsttrading08; do not changeNAVrules",
        "report_time": "Cover+activation day agree; PDFcreation/modification<=declared day; URLdate if present agrees. ConflictingPDF excluded. Date-only publication+1calendar00. CMS publish alone is not body revision",
        "same_site_inference": "CMS publish clustered2026 while creation/modification remainhistorical; exact CMS business meaning and migration event unproven",
        "source_scope": "Existing frozen registry plus expected-hash direct references only; no current unbound bytes",
        "company_scope": "Must match a positive holding in report available at event publication as well asD08; no latestHeld flag",
        "body_unknown": "Missing/body quarantine not zero meaning; retain coverage and age; zero recent2025body explicitly reported",
        "diagnostic_dates": "10evenly spaced common development indices0..242 chosen before scoring",
        "baselines": {"N": 124, "FIXED_NNE": 129, "RF20_NNE": 130, "ALWAYS_UP": 128},
        "uncertainty": "Existing5-session paired block bootstrap2000replicates seed0; descriptive post-selection only",
        "stop": "Complete6registered slots with aliases and atmost1replay or80actualattempts; no parameter/weight expansion",
        "external_requests": 0, "new_llm_requests": 0, "new2026answers_or_scores": False,
        "adoption": False, "strict_contemporaneous_version_archive_proven": False,
        "prepared_without_labels": True,
    }
    io.save(ROOT / "protocol.json", plan)
    frozen = {str(ROOT / name): io.sha(ROOT / name) for name in (
        "protocol.json", "source-registry.json", "source-registry-supplement.json", "title-lexicon.json",
        "report-time-readmission.json", "same-site-cms-evidence.json", "admitted-reports.json", "body-reuse-audit.json",
    )}
    io.save(ROOT / "pretraining-protocol-freeze.json", {"at": io.now(), "files": frozen})
    log(f"来源复审完成：15报告中{len(admitted)}份按交叉证据重建可用，{15-len(admitted)}份隔离；"
        f"{len(body)}份正文、{statuses['SOURCE_GROUNDED']}份已提取可复用；2025原正文0，明确区分标题与历史正文。"
        "六候选协议和词典已冻结，尚未加载监督标签。")
    return {"at": plan["at"], "reports_admitted": len(admitted), "body_documents": len(body),
            "grounded_extractions": statuses["SOURCE_GROUNDED"], "quote_errors": len(quote_errors), "candidates": 6}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
