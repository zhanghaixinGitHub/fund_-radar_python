"""010737扩围47产品的原件重放候选：原R5执行/收益，风险不在此组件范围。

依赖同批已验证的efunds基础组件；不读取研究缓存、不联网、不写数据库。
输入中的准入审阅、净值/分红原件绑定均锁定摘要，计算值不取调用方候选分。
"""

import json
import re
from copy import deepcopy
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services import fund_rating_efunds_components as base
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import content_digest, source_bytes, source_exists, source_size

VERSION = "R5_EFUNDS_010737_MECHANISM47_RETURN_ORIGINALS_20261030_1"
BASE_SHA = "c214a325e8c4d58a48c13d4d47ffde31fd365c34f71ff6f43f536c0a600faf33"
# 保留旧数值基线，精确允许已验证的资料库读取适配版本，不接受任意新模块指纹。
MATERIAL_ADAPTER_BASE_SHA = "2189751a098c6d235ca15459295d826c52aec9bef358e95e0311a1781b171a7a"
COHORT_SHA = "535647c553ea5c00a03d685339d4095c3c0c2bbbc7894ca08f5f13c505d1e26b"
HISTORY_BINDING_SHA = "94d656236b09c7a571f9d92749a91b2863332d5a2881ae97cbdfee0d2aedeeeb"
DIVIDEND_BINDING_SHA = "399482d4dd49c9a8227f2e5fec0f2b1604510e76982c553b69444f640eeb6a22"
CATALOGUE_BINDING_SHA = "7747d357e99eb1e7fc7f82a2178ad4480ee769a8e034a5212b4673d39c38e81f"
PEERS = [
    "000172",
    "000176",
    "000311",
    "000312",
    "000512",
    "001015",
    "002310",
    "002670",
    "003015",
    "003957",
    "004190",
    "004513",
    "004716",
    "004788",
    "005113",
    "005137",
    "005248",
    "005530",
    "005850",
    "005870",
    "006020",
    "007044",
    "007143",
    "008238",
    "008592",
    "009059",
    "010854",
    "010872",
    "010908",
    "011545",
    "012206",
    "012911",
    "015061",
    "015387",
    "016204",
    "018257",
    "100038",
    "110030",
    "162213",
    "163407",
    "165310",
    "166802",
    "200002",
    "310318",
    "320014",
    "450008",
    "519116",
]
NEW = [
    "000172",
    "000311",
    "001015",
    "002310",
    "002670",
    "003957",
    "004716",
    "005870",
    "012911",
    "015061",
    "100038",
    "162213",
    "163407",
    "166802",
    "200002",
    "310318",
    "320014",
    "450008",
    "519116",
]


def require(condition, code):
    if not condition:
        raise ValueError("MECHANISM47_RETURN_" + code)


def canonical_hash(value):
    return content_digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


class Archive:
    """仅使用本次登记原件；按基金释放PDF缓存，避免同时保存两整套原件。"""

    def __init__(self, inputs, root):
        self.root = Path(root).resolve()
        self.sources = inputs["sources"]
        self.raw = {}

    def read(self, digest):
        if digest not in self.raw:
            ref = self.sources[digest]
            path = (self.root / ref["file"]).resolve()
            require(path.is_relative_to(self.root / "sources") and source_exists(path), "SOURCE_PATH")
            require(source_size(path) <= 50000000, "SOURCE_SIZE")
            data = source_bytes(path)
            require(content_digest(data) == digest == ref["sha256"], "SOURCE_HASH")
            self.raw[digest] = data
        return self.raw[digest]

    def text(self, digest, kind):
        self.read(digest)
        return base.compact(
            "".join(base.load_source(self, self.sources[digest], first_pages=500 if kind == "quarter" else 20))
        )


def api(archive, digest, market_code):
    ref = archive.sources[digest]
    require(ref["params"]["ts_code"] == market_code, "API_REQUEST_IDENTITY")
    data = json.loads(archive.read(digest), parse_float=D)
    require(data.get("code") == 0 and not data["data"].get("has_more"), "API_RESPONSE")
    rows = [dict(zip(data["data"]["fields"], r, strict=True)) for r in data["data"]["items"]]
    require(all(r["ts_code"] == market_code for r in rows), "API_RESPONSE_IDENTITY")
    return rows


def no_distribution_scope(archive, code, spec):
    """只接受实际全窗临时公告完整分页，绝不把失败/空正文当作无分红。"""
    require(spec["code"] == code and spec["window"] == ["2023-09-28", "2026-09-30"], "NO_DIVIDEND_SCOPE")
    rows = []
    total = None
    for i, page in enumerate(spec["pages"]):
        data = json.loads(archive.read(page["source_hash"]))
        require(data.get("success") is not False and isinstance(data.get("aaData"), list), "CATALOGUE_RESPONSE")
        count = data["iTotalDisplayRecords"]
        if total is None:
            total = count
        require(count == total and page["offset"] == i * 20, "CATALOGUE_PAGINATION")
        require(len(data["aaData"]) == page["rows"], "CATALOGUE_ROW_COUNT")
        actual = {r["name"]: r["value"] for r in json.loads(page["params"]["aoData"])}
        require(
            actual["fundCode"] == code
            and actual["reportType"] == "FC"
            and actual["startUploadDate"] == "2023-09-28"
            and actual["endUploadDate"] == "2026-09-30",
            "CATALOGUE_REQUEST",
        )
        rows += data["aaData"]
    require(len(rows) == total and all(r["fundCode"] == code for r in rows), "CATALOGUE_COMPLETE")
    require(not any("分红" in r["reportName"] or "收益分配" in r["reportName"] for r in rows), "UNKNOWN_CASH_EVENT")
    return {"code": code, "actual_catalogue_rows": len(rows), "full_declared_window_checked": True}


def notice_binding(archive, code, events, specs):
    checked = []
    for ex, cash in events.items():
        matches = [r for r in specs if r["code"] == code and r["ex_date"] == ex]
        require(len(matches) == 1, "DIVIDEND_NOTICE_REQUIRED")
        item = matches[0]
        ref = archive.sources[item["source_hash"]]
        require(ref["published_on"] <= "2026-09-30", "DIVIDEND_NOTICE_AFTER_CUTOFF")
        text = archive.text(item["source_hash"], "full")
        codes = "000172010234" if code == "000172" else "162213003548"
        require("下属分级基金的交易代码" + codes in text, "DIVIDEND_SHARE_ORDER")
        marker = "本次下属分级基金分红方案(单位:元/10份基金份额)"
        require(marker in text, "DIVIDEND_ORIGINAL_UNIT")
        scheme = text.split(marker, 1)[1]
        digits = 2 if code == "000172" else 4
        match = re.match(r"(\d+\.\d{" + str(digits) + r"})(\d+\.\d{" + str(digits) + r"})", scheme)
        require(match and D(match[1]) / 10 == cash == D(item["per_share_cash"]), "DIVIDEND_A_SCHEME")
        y, m, d = int(ex[:4]), int(ex[4:6]), int(ex[6:])
        require(re.search(rf"除息日{y}年0?{m}月0?{d}日", text), "DIVIDEND_EX_DATE")
        checked.append(
            {
                "ex_date": ex,
                "per_share_cash": str(cash),
                "A_per10_cash": match[1],
                "C_per10_cash_separate": match[2],
                "source_hash": item["source_hash"],
            }
        )
    return checked


def growth_checked(archive, code, nav, div, days, no_dividend_catalogues):
    if code not in ["320014", "450008"]:
        n, g, e = base.growth(nav, div, days)
        return n, g, e, None
    proof = no_distribution_scope(archive, code, no_dividend_catalogues[code])
    require(
        not any(
            (r.get("net_ex_date") or r.get("ex_date"))
            and days[0] < (r.get("net_ex_date") or r.get("ex_date")) <= days[-1]
            for r in div
        ),
        "NO_DIVIDEND_API_CONFLICT",
    )
    n = {r["nav_date"]: r for r in nav}
    require(len(n) == len(nav) and set(days) <= n.keys(), "NAV_CALENDAR")
    require(all(days[0] <= d <= days[-1] and D(r["unit_nav"]) > 0 for d, r in n.items()), "NAV_VALUE")
    ordered = sorted(n)
    first = D(n[ordered[0]]["unit_nav"])
    g = {d: D(n[d]["unit_nav"]) / first for d in ordered}
    differences = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        delta = (D(n[b]["accum_nav"]) - D(n[b]["unit_nav"])) - (D(n[a]["accum_nav"]) - D(n[a]["unit_nav"]))
        if delta:
            differences.append({"date": b, "change": str(delta)})
    proof.update(
        cumulative_source_differences=differences,
        formula_uses_original_unit_NAV_and_proven_cash=True,
        auxiliary_cumulative_column_not_rewritten=True,
    )
    return n, g, {}, proof


def new_peer(archive, code, spec, own_budget, days, dividend_specs, no_dividend_catalogues, research_on):
    nav = api(archive, spec["nav_hash"], spec["market_code"])
    div = api(archive, spec["div_hash"], spec["market_code"])
    for row in nav:
        require(
            row.get("ann_date") and row["nav_date"] <= row["ann_date"] <= research_on.replace("-", ""),
            "NAV_PUBLICATION_CLOCK",
        )
    n, g, events, diagnostic = growth_checked(archive, code, nav, div, days, no_dividend_catalogues)
    require(
        sorted(spec["reports"]) == sorted([p + "-quarter" for p in base.EXPECTED] + [p + "-full" for p in base.FULL]),
        "REPORT_WINDOW",
    )
    budgets = []
    checks = []
    for key, item in sorted(spec["reports"].items()):
        require(key == item["period"] + "-" + item["kind"], "REPORT_KEY_PERIOD")
        ref = archive.sources[item["source_hash"]]
        require(ref["published_on"] <= "2026-09-30", "FUNDAMENTAL_AFTER_CUTOFF")
        text = archive.text(item["source_hash"], item["kind"])
        require(code in text, "REPORT_SHARE")
        y, m, d = map(int, item["period"].split("-"))
        require(re.search(rf"{y}(?:年|[-/.])0?{m}(?:月|[-/.])0?{d}(?:日)?", text[:2500]), "ACTUAL_REPORT_PERIOD")
        at = text.find("投资目标")
        require(at >= 0, "PRODUCT_OBJECTIVE")
        role = text[at : at + 1100]
        require("指数" in role and "增强" in role, "ACTUAL_ENHANCED_MECHANISM")
        require(
            any("沪深300" in text[m.start() : m.end() + 450] for m in re.finditer("业绩比较基准", text)),
            "ACTUAL_TARGET_INDEX",
        )
        if item["kind"] == "quarter":
            if code == "162213" and item["period"] == "2025-06-30":
                require(
                    item["source_hash"] == "4e1d3f2d73f084816eddaae5a662e439e8b8908fbb0aa1caf8fcd60df1a05c4b",
                    "EXACT_NAV_QUARTER_SOURCE",
                )
                full = spec["reports"]["2025-06-30-full"]
                require(
                    full["source_hash"] == "1b5753509ffee39c1a60a64b3380e35d739290c8458fc5ada4296fd67764ad73",
                    "EXACT_NAV_FULL_SOURCE",
                )
                ft = archive.text(full["source_hash"], "full")
                pattern = r"期末基金资产净值([\d,]+\.\d{2})([\d,]+\.\d{2})"
                qnav = re.findall(pattern, text)
                fnav = re.findall(pattern, ft)
                require(qnav == fnav == [("778,359,413.08", "148,419,420.40")], "EXACT_NAV_TWO_ORIGINALS")
                require(
                    "其中:股票859,903,186.68" in text and "其中:股票投资859,903,186.68" in ft,
                    "EXACT_STOCK_TWO_ORIGINALS",
                )
                lo = hi = D("859903186.68") / D("926778833.48") * 100
            else:
                lo, hi = base.quarter(text)
            target = own_budget[item["period"]]
            budgets.append({"period": item["period"], "high_gap_pp": str(max(abs(target - lo), abs(target - hi)))})
        elif item["period"] != base.FULL[0]:
            marker = "基金份额净值增长率及其与同期业绩比较基准收益率的比较"
            start = text.find(marker)
            require(start >= 0, "RETURN_REPORT_HEADER")
            section = text[start:]
            match = re.search(
                r"过去(?:六|6)个月([-+]?\d+(?:\.\d+)?)%([-+]?\d+(?:\.\d+)?)%([-+]?\d+(?:\.\d+)?)%", section
            )
            require(match, "RETURN_REPORT_ROW")
            header = section[: match.start()]
            has_A = bool(re.search(r"A(?:/B)?[^ABC]*阶段", header))
            solo = (
                "下属分级基金" not in text[:start]
                and "下属各类别基金" not in text[:start]
                and ("基金主代码" + code in text[:start] or "交易代码" + code in text[:start])
            )
            require(has_A or solo, "REPRESENTATIVE_RETURN_TABLE")
            period = item["period"]
            boundary = period[:4] + ("0101" if period[5:7] == "06" else "0701")
            a = max(d for d in g if d < boundary)
            b = period.replace("-", "")
            actual = (g[b] / g[a] - 1) * 100
            require(abs(actual - D(match[1])) <= D(".005"), "RETURN_REPORT_CONFLICT")
            checks.append(
                {
                    "period": period,
                    "reported_percent": match[1],
                    "actual_percent": str(actual),
                    "source_hash": item["source_hash"],
                }
            )
    require(
        len(budgets) == 12
        and sum(D(r["high_gap_pp"]) for r in budgets) / 12 <= 5
        and max(D(r["high_gap_pp"]) for r in budgets) <= 10,
        "ACTUAL_BUDGET_MEAN5_MAX10",
    )
    require(len(checks) == 5, "FIVE_HALF_YEAR_CROSSCHECKS")
    notices = notice_binding(archive, code, events, dividend_specs) if events else []
    return g, {
        "budgets": budgets,
        "half_year_checks": checks,
        "dividend_notices": notices,
        "cumulative_field_diagnostic": diagnostic,
        "actual_events": {k: str(v) for k, v in events.items()},
    }


def check_binding(bundle, inputs):
    require(inputs.get("version") == VERSION and inputs.get("fund_code") == "010737", "VERSION_OR_IDENTITY")
    base.check_binding(bundle, inputs["base_inputs"])


def calculate(inputs, *, root):
    """重新读原件编译观察；依赖的旧28组也由同批基础组件实际重放。"""
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "010737"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("research_on") == "2026-10-30",
        "IDENTITY",
    )
    selected = inputs.get("selected_dimensions")
    require(
        isinstance(selected, list)
        and bool(selected)
        and set(selected) <= {"execution", "return"}
        and selected == [k for k in ["execution", "return"] if k in selected],
        "SELECTED_DIMENSIONS",
    )
    require(
        content_digest(source_bytes(Path(base.__file__))) in {BASE_SHA, MATERIAL_ADAPTER_BASE_SHA},
        "BASE_MODULE_VERSION",
    )
    require(content_digest(inputs["cohort_review_text"].encode()) == COHORT_SHA, "COHORT_REVIEW_DIGEST")
    require(canonical_hash(inputs["history_bindings"]) == HISTORY_BINDING_SHA, "HISTORY_BINDING_DIGEST")
    require(canonical_hash(inputs["dividend_notices"]) == DIVIDEND_BINDING_SHA, "DIVIDEND_BINDING_DIGEST")
    require(canonical_hash(inputs["no_distribution_catalogues"]) == CATALOGUE_BINDING_SHA, "CATALOGUE_BINDING_DIGEST")
    frame = json.loads(inputs["cohort_review_text"])
    require(
        frame["admitted_performance_codes"] == PEERS
        and frame["new_admitted_codes"] == NEW
        and sorted(inputs["new_peers"]) == NEW,
        "COHORT_MEMBERS",
    )
    for code in NEW:
        reviewed = next(r for r in frame["all_new_discovery_code_decisions"] if r["code"] == code)
        declared = {r["key"]: r["source"]["sha256"] for r in reviewed["actual_originals"]}
        require(
            {k: r["source_hash"] for k, r in inputs["new_peers"][code]["reports"].items()} == declared,
            "ADMITTED_ORIGINAL_BINDING",
        )
    archive = Archive(inputs, root)
    # 基础组件使用原冻结2026-10-29研究输入，保留旧取得时间；新采集输入按10-30，
    # 不把原件重用时间冒充首次取得时间，也不把新NAV公告倒写为9-30已知。
    old = base.calculate(inputs["base_inputs"], root=root)
    sessions = inputs["base_inputs"]["sources"][inputs["base_inputs"]["calendar_ref"]]
    cal = json.loads(archive.read(sessions["sha256"]))
    calendar_hash = cal["calendar_hash"]
    dates = [d for d in cal["sessions"] if "2023-09-28" <= d <= "2026-09-30"]
    days = [d.replace("-", "") for d in dates]
    require(len(days) == 727, "COMMON_CALENDAR")
    own_budget = {
        r["period"]: D(r["stock_amount"]) / D(r["net_assets"]) * 100
        for r in old["audit"]["execution"]["reports"]
        if r["key"].endswith("quarter")
    }
    metrics = deepcopy(old["audit"]["return"]["metrics"])
    audits = {}
    levels = {}
    with localcontext() as ctx:
        ctx.prec = 70
        # 场内LOF代码来自独立已归档目录及同一A份额的原报告，不借另一份额。
        market_meta = inputs["market_alias_metadata_ref"]
        data = json.loads(archive.read(market_meta))
        require(data["code"] == 0, "MARKET_ALIAS_METADATA_RESPONSE")
        aliases = [dict(zip(data["data"]["fields"], r, strict=True)) for r in data["data"]["items"]]
        require(any(r["ts_code"] == "163407.SZ" and r["market"] == "E" for r in aliases), "LOF_ACTUAL_MARKET_ALIAS")
        for code in NEW:
            spec = inputs["new_peers"][code]
            binding = inputs["history_bindings"][code]
            require({k: spec[k] for k in ["nav_hash", "div_hash", "market_code"]} == binding, "NEW_HISTORY_BINDING")
            g, audit = new_peer(
                archive,
                code,
                spec,
                own_budget,
                days,
                inputs["dividend_notices"],
                inputs["no_distribution_catalogues"],
                inputs["research_on"],
            )
            levels[code] = g
            audits[code] = audit
            span = D((date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days)
            metrics[code] = {
                "return_1y": str(g[days[-1]] / g["20250930"] - 1),
                "return_3y": str((g[days[-1]] / g[days[0]]) ** (D("365.25") / span) - 1),
            }
            archive.raw.clear()
    refs = {
        h: {"path": str(archive.root / r["file"]), "published_on": r.get("published_on")}
        for h, r in archive.sources.items()
    }
    # 旧组由基础组件已验；新组本次逐原件已验。所有额外档案仍核字节，
    # 防止在正式重放时用未核旧path或同名文件替换原始回包。
    for h in archive.sources:
        archive.read(h)
        archive.raw.clear()
    nodes = {
        name: {
            "kind": "rank",
            "target": metrics["010737"][name],
            "peers": [{"product_id": c, "value": metrics[c][name]} for c in PEERS],
            "expected_products": PEERS,
            "direction": "higher" if name.startswith("return_") else "lower",
            "source_refs": sorted(refs),
        }
        for name in ["return_1y", "return_3y"]
    }
    packet = {
        "fund_code": "010737",
        "basis": ACTIVE_BASIS,
        "as_of_date": "2026-09-30",
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "observations": nodes,
            "target_product_aliases": ["010736", "010737", "022914"],
            "material_event_pending": old["audit"]["material_event_review"]["material_event_pending"],
        },
        "quality_public": {"dataDates": {"nav": "2026-09-30", "holdings": "2026-06-30"}},
        "r4_evidence_registry": refs,
        "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in refs],
    }
    result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"return"})
    require(all(result[k] is None for k in ["score", "raw_score", "grade"]), "NO_COMPLETE_SCORE_LEAK")
    points = {"execution": old["dimensions"]["execution"], **result["dimensions"]}
    observations = {}
    if "execution" in selected:
        observations.update(old["audit"]["execution"]["observations"])
    if "return" in selected:
        observations.update({k: v for k, v in nodes.items() if k.startswith("return_")})
    return {
        "dimensions": {k: points[k] for k in selected},
        "observations": observations,
        "audit": {
            "new_peer_originals": audits,
            "base_original_replay": old["audit"],
            "metrics": metrics,
            "calendar_hash": calendar_hash,
            "source_count": len(refs),
        },
        "complete_score": None,
        "grade": None,
        "formal_writes": 0,
        "cohort_scope_review_required": "return" in selected,
        "risk_dimension_not_supported": True,
    }


def prepare(session, request, *, root):
    """只读current生成可审追加包；接入、正式登记与读回由统筹执行。

    请求可只增加收益或风险。已有本系列组件承担的维度并入同一重放范围，
    原正式分值和说明保持原样；不能为了腾出来源字段丢掉既有执行校验。
    """
    from datetime import UTC, datetime

    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    # 检查实际已加载函数的常量，不以磁盘补丁存在冒充运行时已启用。
    require(
        "efunds_mechanism_return_components_calculation" in snapshot_calculate.__code__.co_consts,
        "SNAPSHOT_DISPATCH_NOT_INSTALLED",
    )
    code = request["fund_code"]
    raw = deepcopy(request["inputs"])
    requested = request["selected_dimensions"]
    require(raw["selected_dimensions"] == requested, "REQUEST_SELECTED_DIMENSIONS")
    previous = current_inputs(session, [code]).get(code)
    require(previous and str(previous["as_of_date"]) == request["as_of_date"], "CURRENT_PERIOD_REQUIRED")
    require(request.get("expected_current_ref") == previous["batch_id"], "STALE_CURRENT_REF")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    check_binding(bundle, raw)
    require(
        bundle.get("basis") == BASIS
        and not any(
            bundle.get(k)
            for k in [
                "r5_calculation",
                "passive_components_calculation",
                "passive_execution_calculation",
                "passive_performance_calculation",
                "gold_components_calculation",
                "enhanced_performance_calculation",
                "active_execution_calculation",
                "active_broad_components_calculation",
            ]
        ),
        "CONFLICTING_COMPONENTS",
    )
    inherited = set()
    for field in ["efunds_enhanced_components_calculation", "efunds_mechanism_return_components_calculation"]:
        prior_component = bundle.get(field)
        if prior_component:
            inherited.update(prior_component["inputs"]["selected_dimensions"])
    require(inherited <= {"execution", "return"}, "PRIOR_COMPONENT_SCOPE")
    raw["selected_dimensions"] = [k for k in ["execution", "return"] if k in inherited | set(requested)]
    raw["sources"] = {h: canonical_source(root, ref) for h, ref in raw["sources"].items()}
    raw["base_inputs"]["sources"] = {h: raw["sources"][h] for h in raw["base_inputs"]["sources"]}
    result = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    old = deepcopy(doc["dimensions"])
    for key in raw["selected_dimensions"]:
        point = old[key]
        score = result["dimensions"][key]
        already_point = (
            point is not None and point.get("candidate") is not None and point.get("lower") == point.get("upper")
        )
        if already_point:
            require(abs(D(str(point["candidate"])) - D(score)) <= D("1e-55"), "PUBLISHED_POINT_CHANGED:" + key)
            continue  # 已有正式点完全保留，其原解释也不被改写。
        require(key in requested, "INHERITED_POINT_NOT_FORMAL:" + key)
        reason = request["explanations"][key].strip()
        require(len(reason) >= 40, "BUSINESS_EXPLANATION_REQUIRED")
        doc["dimensions"][key] = {"lower": score, "upper": score, "candidate": score, "reason": reason}
        bundle.setdefault("public_explanations", {})[key] = reason
        bundle.get("public_metrics", {}).pop(key, None)
    require(all(old[k] == doc["dimensions"][k] for k in set(old) - set(requested)), "UNREQUESTED_DIMENSION_CHANGED")
    bundle.pop("efunds_enhanced_components_calculation", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        efunds_mechanism_return_components_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(r for h, r in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=Path(root))
    checked = snapshot_calculate(bundle, root=Path(root))
    require(checked["status"] == "PARTIAL" and checked["score"] is None, "CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
