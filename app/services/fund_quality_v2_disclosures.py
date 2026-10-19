"""从已有正式披露提取V2可复算事实，并将真实缺口定位到具体原文。

本模块只读文件。解析出一个报告期的事实，不等于获得三年治理记录、
历史合同或策略容量。合同的现金加短债下限也不冒充现金目标区间。
"""

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from app.services.fund_rating_public_market import (
    ATTACHMENT_HOST,
    HOSTS,
    attachment_identity,
    compact,
    public_document,
    section,
)
from app.services.fund_rating_rules import decimal


def normalize_number_lines(value: str) -> str:
    """只恢复表格金额的小数末位、千分位断行；不把相邻完整金额粘起来。"""
    value = re.sub(r"(\d[\d,]*,\d{0,2})\s*\n\s*(\d{1,3}\.\d{2})(?!\d)", r"\1\2", value)
    value = re.sub(r"(\d[\d,]*\.)\s*\n\s*(\d{2})(?![\d.])", r"\1\2", value)
    return re.sub(r"(\d[\d,]*\.\d)\s*\n\s*(\d)(?![\d.])", r"\1\2", value)


def holder_pressure(text: str) -> tuple[Decimal, str, list[dict]]:
    """只读当期单一持有人表；以份额收支对账识别净赎回，不复用旧风险档。

    期末低于20%不等于全期没有大持有人。横杠不默认补零；余额可唯一
    求解时才补余额。仅总申购/赎回已披露时可唯一求净变动，未知余额保留。
    """
    matches = list(re.finditer(r"\d+\.\d+\s*报告期内单一投资者持有", text))
    # 部分发行人PDF未保留自动章号。只接纳独占行完整阈值标题，并须有
    # 紧邻的“影响投资者决策”父节，防止从一般风险提示或目录猜持有人状态。
    for match in re.finditer(
        r"(?m)^[ \t]*报告期内单一投资者持有基金份额比例达到(?:或|或者)超过\s*20\s*[%％]的情况[ \t]*$",
        text,
    ):
        if re.search(r"(?m)^[ \t]*影响投资者决策的其他重要信息[ \t]*$",
                     text[max(0, match.start() - 300):match.start()]):
            matches.append(match)
    matches.sort(key=lambda match: match.start())
    if not matches:
        raise ValueError("HOLDER_PERIOD_SECTION_MISSING")
    raw = text[matches[-1].start() :][:8000]
    raw = re.split(r"产品特有风险|\d+\.2\s*影响投资者|§\s*12|12\.1\s*备查|"
                   r"(?m:^[ \t]*(?:影响投资者决策的其他重要信息|备查文件目录)[ \t]*$)", raw)[0]
    normalized = compact(raw).replace("％", "%")
    statement = re.sub(r"^(?:\d+\.\d+)?报告期内单一投资者持有基金份额比例达到(?:或|或者)超过20%的情况[：:]?",
                       "", normalized)
    first = statement.split("。", 1)[0]
    # “无”只在该项正文中有效，不读取产品特有风险中的假设描述。
    negative_claim = re.match(
        r"(?:注[:：])?(?:本基金)?(?:本报告期|报告期)(?:内)?[，,]?(?:本基金)?"
        r"(?:未出现|未有|不存在|无(?:需要披露的)?|未发生)(?:单一投资者持有(?:基金)?份额(?:比例)?|持有基金份额比例)",
        first,
    )
    reverse_claim = re.match(r"(?:注[:：])?(?:本基金)?报告期内单一投资者持有基金份额比例不存在达到", first)
    if re.fullmatch(r"(?:注[:：])?无[。.]?", first) or (
        "单一投资者" in first
        and "20%" in first
        and (negative_claim or reverse_claim)
        and not re.search(r"除外|部分|但|仅", first)
    ):
        return Decimal(0), raw, []
    absence = re.search(
        r"(?:本基金)?本报告期(?:内)?(?:未出现|不存在|无|未发生)(?:单一投资者)?"
        r"(?:持有基金份额(?:比例)?|单一投资者持有基金份额(?:比例)?)?(?:达到或超过|达到或者超过)20%的情况",
        normalized,
    )
    if absence:
        return Decimal(0), raw, []
    if "期初" not in raw or "申购" not in raw or "赎回" not in raw or "拆分" in raw:
        raise ValueError("HOLDER_FLOW_COLUMNS_OR_SPLIT_UNVERIFIED")
    # 日期可能在年份中间断行，合并后仍须是两个有效的八位自然日。
    table = re.sub(
        r"(20\d{0,5})\s*\n\s*(\d{1,6})(?=\s|$)", lambda m: m[1] + m[2] if len(m[1] + m[2]) == 8 else m[0], raw
    )
    table = re.sub(
        r"(20\d{2})\s*[-/年]\s*(\d{1,2}(?:\s*\n\s*\d)?)\s*[-/月]\s*(\d{1,2}(?:\s*\n\s*\d)?)\s*日?",
        lambda m: f"{int(m[1]):04}{int(compact(m[2])):02}{int(compact(m[3])):02}",
        table,
    )
    ranges = list(re.finditer(r"(20\d{6})\s*[-—－至~～]\s*(20\d{6})", table))
    if not ranges:
        raise ValueError("HOLDER_INTERVAL_ROWS_UNPARSED")
    rows, pending_intervals = [], []
    for n, interval in enumerate(ranges):
        start, end = (datetime.strptime(s, "%Y%m%d").date() for s in interval.groups())
        if end < start:
            raise ValueError("HOLDER_INTERVAL_INVALID")
        block = normalize_number_lines(
            table[interval.end() : ranges[n + 1].start() if n + 1 < len(ranges) else len(table)]
        )
        tokens = re.findall(r"(?<![\d,.])(?:\d[\d,]*\.\d{2}(?!\d)|[-—－](?!\d))", block)
        # 同一持有人可列多个达到20%的区间，申赎份额只披露一次。只有区间之间
        # 确实仅有分隔符时合并，不能把另一持有人的残缺行并入后一行。
        pending_intervals.append({"from": str(start), "to": str(end)})
        if not tokens and not compact(block).strip(";；,，") and n + 1 < len(ranges):
            continue
        if len(tokens) != 5:
            raise ValueError("HOLDER_FLOW_ROW_AMBIGUOUS")
        begin, buy, sell, final, percent = [
            None if s in {"-", "—", "－"} else decimal(s.replace(",", "")) for s in tokens
        ]
        values = [begin, buy, sell, final]
        if values.count(None) == 1:
            coefficients = [1, 1, -1, -1]
            unknown = values.index(None)
            values[unknown] = (
                -sum(v * c for v, c in zip(values, coefficients, strict=True) if v is not None) / coefficients[unknown]
            )
            if values[unknown] == 0:
                values[unknown] = abs(values[unknown])
            begin, buy, sell, final = values
        balances_known = begin is not None and final is not None
        gross_known = buy is not None and sell is not None
        if (not balances_known and not gross_known
                or any(value is not None and value < 0 for value in (begin, final))
                or percent is not None and not 0 <= percent <= 100):
            raise ValueError("HOLDER_FLOW_VALUE_MISSING")
        if balances_known and buy is None and sell is not None:
            buy = final - begin + sell
        if balances_known and sell is None and buy is not None:
            sell = begin + buy - final
        net_only = buy is None and sell is None
        # 期初和期末明确时，净申赎=期末-期初已唯一确定；两项总发生额仍保留未知。
        # 这不会把“－”补成0，也不能据此声称总申购、总赎回已经逐项对账。
        if not net_only and (
            buy is None or sell is None or min(buy, sell) < 0
            or balances_known and abs(begin + buy - sell - final) > Decimal("0.01")
        ):
            raise ValueError("HOLDER_FLOW_NOT_RECONCILED")
        # 已有余额完整时维持原有“期末-期初”及一分钱对账口径；两端均未披露
        # 但总申购、总赎回明确时，净变动=申购-赎回仍唯一，不需要猜绝对余额。
        net_change = final - begin if balances_known else buy - sell
        rows.append(
            {
                "from": pending_intervals[0]["from"],
                "to": pending_intervals[-1]["to"],
                "intervals": pending_intervals,
                "begin": None if begin is None else str(begin),
                "purchase": None if buy is None else str(buy),
                "redemption": None if sell is None else str(sell),
                "end": None if final is None else str(final),
                "end_percent": None if percent is None else str(percent),
                "net_redemption": net_change < 0,
                "net_flow": str(net_change),
                "reconciliation": "NET_FLOW_ONLY_GROSS_FLOWS_NOT_DISCLOSED"
                if net_only
                else "NET_CHANGE_FROM_GROSS_FLOWS_BALANCES_NOT_DISCLOSED"
                if not balances_known
                else "ALL_SHARE_FLOWS_RECONCILED",
                "raw": block[:1200],
            }
        )
        pending_intervals = []
    return Decimal(2 if any(r["net_redemption"] for r in rows) else 1), raw, rows


def report_period(text: str) -> tuple[date, date]:
    """期末与覆盖期取报告正文，不把下载日、送出日当资产观察日。"""
    normalized = compact(text)
    match = re.search(
        r"本报告期(?:自|从|为|指)?(20\d{2})年(\d{1,2})月(\d{1,2})日(?:起)?至"
        r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日",
        normalized,
    )
    if match:
        start = date(*map(int, match.groups()[:3]))
        # 年报常写“自2025年1月1日起至12月31日止”，终点省略年份时只沿用已写年份。
        # 不跨年猜测，终点早于起点即拒绝；历史记录另须与登记报告期末一致。
        end = date(int(match[4] or match[1]), int(match[5]), int(match[6]))
        if end < start:
            raise ValueError("REPORT_ECONOMIC_PERIOD_REVERSED")
        return start, end
    # 中期报告标题与财务报表日期须同时出现，不凭文件名推断。
    title = re.search(r"(20\d{2})年中期报告", normalized[:2500])
    if title and str(title[1]) + "年6月30日" in normalized:
        return date(int(title[1]), 1, 1), date(int(title[1]), 6, 30)
    raise ValueError("REPORT_ECONOMIC_PERIOD_UNPARSED")


def source_documents(code: str, refs: list[dict], root: Path, as_of: date | None = None) -> dict:
    """读取原件并选择当时最新的完整年度/中期报告；季度前十持仓不能替代。

    先按正文经济期末排序，再核登记期末与公开/送出日期。年度与中期保留
    各自类型，统一使用report入口；half仅保留既有中期调用兼容性。
    """
    selected = {}
    periodic = []
    historical_prospectuses = []
    periodic_labels = {"half", "link-half", "annual", "historical-half", "historical-annual"}
    for ref in refs:
        label = ref.get("label", "")
        if as_of and ref.get("published_on") and date.fromisoformat(str(ref["published_on"])[:10]) > as_of:
            continue
        if label in periodic_labels:
            periodic.append(ref)
        elif label == "historical-prospectus" and ref.get("published_on"):
            historical_prospectuses.append(ref)
        elif label in {"summary", "link-summary", "prospectus", "link-prospectus"}:
            selected[label.split("-")[-1]] = ref
    # 旧完整包没有招募书时，补读当时已公开的登记版本；不覆盖既有主原件。
    # 最新登记版解析/身份/日期证明失败即保留缺口，不能退回更早书补出覆盖值。
    fallback = "prospectus" not in selected and bool(historical_prospectuses) and as_of is not None
    fallback_conflict = False
    if fallback:
        selected["prospectus"] = max(historical_prospectuses, key=lambda ref: (ref["published_on"], ref["sha256"]))
        newest = selected["prospectus"]["published_on"]
        fallback_conflict = len({ref["sha256"] for ref in historical_prospectuses if ref["published_on"] == newest}) > 1
    for number, ref in enumerate(periodic):
        selected["periodic_" + str(number)] = ref
    result = {}
    by_label = {r.get("label"): r for r in refs}
    for kind, ref in selected.items():
        try:
            if kind == "prospectus" and fallback_conflict:
                raise ValueError("PROSPECTUS_LATEST_PUBLICATION_CONFLICT")
            address = urlsplit(ref["url"])
            host = address.hostname
            cninfo = host == "static.cninfo.com.cn"
            if cninfo:
                if not kind.startswith("periodic_") or as_of is None:
                    raise ValueError("QUALITY_CNINFO_PERIODIC_ROLE_AND_CUTOFF_REQUIRED")
            elif host == "www.cjhxfund.com":
                # 创金合信仅接纳实际已核的定期报告下载路径，并重读同批保存的
                # 官网目录原响应。域名、布尔标志或可打开PDF都不能单独证明
                # 公告归属/公开日期；不扩展第一版正式评级的来源名单。
                if (ref.get("label") not in periodic_labels or address.scheme != "https"
                        or address.username or address.password or address.port not in {None, 443}
                        or not address.path.startswith("/plat_files/uploadProdAd/importFile/")):
                    raise ValueError("CJHX_PERIODIC_ORIGINAL_ROUTE_UNVERIFIED")
                catalogs = [r for r in refs if r.get("sha256") == ref.get("issuer_catalog_sha256")
                            and r.get("label") == "issuer-periodic-catalog"]
                if len(catalogs) != 1:
                    raise ValueError("CJHX_PERIODIC_CATALOG_ORIGINAL_REQUIRED")
                catalog_ref = catalogs[0]
                catalog_address = urlsplit(catalog_ref["url"])
                if (catalog_address.scheme != "https" or catalog_address.hostname != host
                        or catalog_address.path != "/servlet/json" or catalog_address.query
                        or catalog_address.username or catalog_address.password
                        or catalog_address.port not in {None, 443}
                        or catalog_ref.get("params", {}).get("funcNo") != "742006"):
                    raise ValueError("CJHX_PERIODIC_CATALOG_ROUTE_UNVERIFIED")
                catalog_path = (root / catalog_ref["file"]).resolve()
                if (not catalog_path.is_relative_to(root.resolve() / "sources")
                        or catalog_path.stat().st_size > 2_000_000):
                    raise ValueError("CJHX_PERIODIC_CATALOG_PATH_OR_SIZE")
                catalog_raw = catalog_path.read_bytes()
                if hashlib.sha256(catalog_raw).hexdigest() != catalog_ref["sha256"]:
                    raise ValueError("CJHX_PERIODIC_CATALOG_HASH_CHANGED")
                catalog = json.loads(catalog_raw)
                if (not isinstance(catalog, dict) or catalog.get("error_no") != "0"
                        or not isinstance(catalog.get("results"), list) or len(catalog["results"]) != 1
                        or not isinstance(catalog["results"][0], dict)
                        or not isinstance(catalog["results"][0].get("data"), list)
                        or not all(isinstance(entry, dict) for entry in catalog["results"][0]["data"])):
                    raise ValueError("CJHX_PERIODIC_CATALOG_BUSINESS_OR_SCHEMA")
                entries = [entry for entry in catalog["results"][0]["data"]
                           if entry.get("attachment_url") and urljoin(catalog_ref["url"], entry["attachment_url"])
                           == ref["url"] and code in {entry.get("main_product_code"), entry.get("product_code")}
                           and entry.get("pub_date") == ref.get("published_on")
                           and entry.get("title") == ref.get("title")]
                if len(entries) != 1:
                    raise ValueError("CJHX_PERIODIC_CATALOG_IDENTITY_DATE_OR_ATTACHMENT_UNBOUND")
            elif host not in HOSTS | {"www.dbfund.com.cn", "dbfund.com.cn", "www.nffund.com", "www.bosera.com"}:
                raise ValueError("SOURCE_HOST_NOT_REGISTERED:" + str(host))
            if host == ATTACHMENT_HOST:
                attachment_identity(by_label, root, ref["label"], code)
            is_periodic = kind.startswith("periodic_")
            report_kind = "annual" if "annual" in ref.get("label", "") else "half"
            pages, metadata = public_document(
                (root / ref["file"]).read_bytes(),
                ref,
                code,
                ("年度报告" if report_kind == "annual" else "中期报告")
                if is_periodic
                else {"summary": "产品资料概要", "prospectus": "招募说明书"}[kind],
            )
            text = "\n".join(pages)
            if code not in compact(text) and metadata is None and kind != "prospectus":
                raise ValueError("DOCUMENT_FUND_CODE_NOT_PRESENT")
            result[kind] = {
                "pages": pages,
                "text": text,
                "ref": ref,
                "code_present": code in compact(text),
                "source_format": "ISSUER_DOCUMENT_PUBLIC_REPRINT" if metadata is not None else "ISSUER_PDF_OR_DOCX",
            }
            if cninfo:
                from app.services.fund_quality_v2_periodic_sources import verify_cninfo_periodic

                original = (root/ref["file"]).read_bytes()
                proof = verify_cninfo_periodic(original, pages, ref, refs=refs, root=root, code=code, as_of=as_of)
                result[kind].update(regulatory_source_evidence=proof, source_authority="REGULATORY_DISCLOSURE_PDF",
                                    source_format="REGULATORY_DISCLOSURE_PDF")
            else:
                result[kind]["source_authority"] = ("FULL_PUBLIC_REPRINT" if metadata is not None
                    else "BOUND_PROVIDER_PDF" if host == ATTACHMENT_HOST else "ISSUER_ORIGINAL")
            if kind == "prospectus" and (fallback or "publication_evidence" in ref):
                from app.services.fund_quality_v2_publication_sources import verify_prospectus_publication

                if as_of is None:
                    raise ValueError("PROSPECTUS_PUBLICATION_AS_OF_REQUIRED")
                proof = verify_prospectus_publication(ref, root, code, as_of)
                result[kind].update(publication_evidence=proof, publication_source_hashes=proof["source_hashes"])
            if is_periodic:
                result[kind]["report_kind"] = report_kind
                # 无考察日的旧调用保留只读身份绑定行为。历史试算必须传入as_of，
                # 且正文期末、登记期末、原文送出日和公开收据共同满足时点边界。
                if as_of:
                    from app.services.fund_rating_public_market import publication

                    period = report_period(text)
                    title_word = "年度报告" if report_kind == "annual" else "中期报告"
                    if title_word not in compact(text[:4000]):
                        raise ValueError("PRIMARY_REPORT_ORIGINAL_KIND_CONFLICT")
                    if period[1] > as_of:
                        raise ValueError("PRIMARY_REPORT_OBSERVATION_AFTER_AS_OF")
                    if ref.get("report_end") and ref["report_end"] != str(period[1]):
                        raise ValueError("PRIMARY_REPORT_REGISTERED_PERIOD_CONFLICT")
                    sent_on = publication(pages)
                    if sent_on > str(as_of):
                        raise ValueError("PRIMARY_REPORT_ORIGINAL_SENT_AFTER_AS_OF")
                    if report_kind == "annual" and not ref.get("published_on"):
                        raise ValueError("PRIMARY_ANNUAL_PUBLICATION_RECEIPT_REQUIRED")
                    result[kind].update(economic_period=list(map(str, period)), original_sent_on=sent_on)
        except (ValueError, KeyError, OSError) as exc:
            result[kind] = {"ref": ref, "error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}
    candidates = [result.pop("periodic_" + str(i)) for i in range(len(periodic))]
    valid = [document for document in candidates if document.get("text")]
    if valid:
        from app.services.fund_quality_v2_periodic_sources import select_periodic_document

        report, selection = select_periodic_document(valid)
        if report is None:
            conflict_ref = next(d["ref"] for d in valid if d["ref"]["sha256"] in selection["conflicting_sha256"])
            result["report"] = {"ref": conflict_ref, "error": selection["error"], "selection_evidence": selection}
        else:
            report["selection_evidence"] = selection
            result["report"] = report
            result[report["report_kind"]] = report
    elif candidates:
        result["report"] = candidates[-1]
    # 被拒原件仍保留精确错误，不把另一个可用报告掩盖成“所有原件均通过”。
    for i, document in enumerate(candidates):
        if "error" in document:
            result["rejected_periodic_" + str(i)] = document
    prospectus, report = result.get("prospectus", {}), result.get("report", {})
    if prospectus.get("text") and not prospectus.get("code_present"):
        # 招募书覆盖整个产品，可能不印份额代码。仅从已含该份额代码的正式报告
        # 取得法定全称，再同时核对封面全称、管理人、托管人；概要仍必须核对份额。
        body = compact(unicodedata.normalize("NFKC", report.get("text", "")))
        name = re.search(r"基金名称(.{6,90}?)基金简称", body)
        manager = re.search(r"基金管理人[:：]?([^:：。]{4,45}?(?:有限责任公司|有限公司))", body)
        custodian = re.search(r"基金托管人[:：]?([^:：。]{4,45}?(?:有限责任公司|有限公司))", body)
        title = compact(unicodedata.normalize("NFKC", prospectus["text"][:2500]))
        # 页码、管理人抬头和“招募说明书(更新)”页眉不是基金名称。只接纳这些
        # 明确的封面前缀，仍要求紧接法定全称以及管理人、托管人双重一致。
        cover = (
            re.match(
                r"^(?:\d{1,3}(?:/\d{1,3})?|招募说明书(?:\(更新\))?|"
                + re.escape(manager[1])
                + r")*"
                + re.escape(name[1]),
                title,
            )
            if name and manager
            else None
        )
        if not (
            report.get("code_present")
            and name
            and manager
            and custodian
            and cover
            and manager[1] in title
            and custodian[1] in title
        ):
            result["prospectus"] = {"ref": prospectus["ref"], "error": "DOCUMENT_PRODUCT_NAME_BINDING_UNVERIFIED"}
        else:
            prospectus["identity_source_hashes"] = [report["ref"]["sha256"]]
    return result


def locate(document: dict, quote: str) -> dict:
    """提供真正包含摘录的页号，JSON完整转载无分页时保留正文定位。"""
    needle = compact(quote)
    pages = [i + 1 for i, page in enumerate(document["pages"]) if needle and needle[:80] in compact(page)]
    return {
        "source_hashes": [document["ref"]["sha256"], *document.get("identity_source_hashes", []),
                          *document.get("publication_source_hashes", [])],
        "pages": pages,
        "excerpt": quote[:2500],
        "source_url": document["ref"]["url"],
    }


def cost_from_sources(code: str, refs: list[dict], root: Path, as_of: date) -> dict:
    """从本份额费用原文核验一年情景，不以经理字段为费用门槛。

    C类核验零申购费，A类核验场外前端一万元费率和无销售服务费；两类均须
    明示365天赎回费为0、综合年运作费唯一。ETF联接还要核验费用豁免。
    横杠本身不作为零费率证明，不将综合费和基本费重复相加。
    """
    from app.services.fund_rating_public_market import publication

    docs = source_documents(code, refs, root, as_of)
    document = docs.get("summary", {})
    if "text" not in document:
        raise ValueError(document.get("error", "SUMMARY_ORIGINAL_REQUIRED"))
    text = compact(document["text"])
    header = text.split("编制日期")[0].split("送出日期")[0][:180]
    if "FOF" in header or "基金中基金" in header:
        raise ValueError("FOF_UNDERLYING_COSTS_AND_EXEMPTIONS_REQUIRED")
    dedicated_c = bool(re.search(r"C(?:类(?:基金)?份额|人民币[）)]|[）)]|基金产品资料)", header))
    explicit_c = bool(re.search(r"基金代码C" + code, text[:1500]))
    explicit_c = explicit_c or bool(
        re.search(r"下属基金简称(?:(?!下属基金代码).){1,120}C人民币下属基金代码" + code, text[:1700])
    )
    dedicated_a = bool(re.search(r"A类(?:基金)?份额", header))
    # 有的专属概要用“（华夏亚债中国指数A）”标识份额，不写“A类份额”。
    # 须由概要内同一份额简称、准确代码及封面括号三者交叉绑定，不能仅凭名称后缀。
    alias_a = re.search(r"下属基金简称([^:：。]{2,90}?A)下属基金代码" + re.escape(code), text[:1500])
    if alias_a and any(left + alias_a[1] + right in header for left, right in (("（", "）"), ("(", ")"))):
        dedicated_a = True
    if not dedicated_c and not explicit_c and not dedicated_a:
        raise ValueError("C_SHARE_DOCUMENT_IDENTITY_OR_NONZERO_FEE_SCENARIO_REQUIRED")
    if "交易币种人民币" not in text:
        raise ValueError("COST_CURRENCY_UNVERIFIED")
    sent = publication(document["pages"])
    if sent > str(as_of):
        raise ValueError("COST_DOCUMENT_AFTER_ECONOMIC_CUTOFF")
    begin = text.find("三、投资本基金涉及的费用")
    if begin < 0:
        begin = text.find("（一）基金销售相关费用")
    end = text.find("四、风险", begin)
    if begin < 0 or end < 0:
        raise ValueError("COST_FEE_SECTION_UNPARSED")
    fees = text[begin:end]
    transaction_fees, composite_fees, selected_share_label = fees, fees, None
    if explicit_c and not dedicated_c:
        # 合并概要同时披露A/C费用时，按原文独立行标题绑定C类交易和综合费。
        # 不能拿A类的365天赎回档、较低综合费或任意一个零费率套给C类。
        labels = {compact(m[1]) for m in re.finditer(r"(?m)^[ \t]*([^\n:：]{1,80}C)\s*[:：]", document["text"])}
        if len(labels) != 1:
            raise ValueError("COST_COMBINED_SHARE_LABEL_UNRESOLVED")
        selected_share_label = labels.pop()
        positions = list(re.finditer(re.escape(selected_share_label) + r"[:：]", fees))
        operating_begin = fees.find("（二）基金运作相关费用")
        if len(positions) != 2 or not positions[0].end() < operating_begin < positions[1].start():
            raise ValueError("COST_COMBINED_SHARE_FEE_SCOPE_UNRESOLVED")
        transaction_fees = fees[positions[0].end() : operating_begin]
        composite_fees = fees[positions[1].end() :]
    supporting = []
    combined = fees + compact(docs.get("prospectus", {}).get("text", ""))
    subscription = Decimal(0)
    buy_zero = re.search(
        r"不收取(?:认购[、/]申购|认购费/申购|申购|认购费和申购)费(?:用)?|C类(?:基金)?份额无申购费|申购费[（(]前收费[）)][-－—]?0(?:\.0+)?%",
        transaction_fees,
    )
    if buy_zero is None and selected_share_label:
        # 表格备注列可能与“前收费”列交错提取；C类身份已由原文标题单独绑定，
        # 这里只读取该C类表内明示的“份额无申购费”，横杠仍不作为零费率证明。
        buy_zero = re.search(r"份额无申购费", transaction_fees)
    if dedicated_a:
        # 明确场外前端一万元情景；后端费率和交易所佣金不混入这个情景。
        tiers = re.findall(r"(?<![<≤≦=＜〈])M[<〈＜](\d[\d,]*)万(?:元)?([\d.]+)%", fees)
        sales_zero = re.search(r"A类(?:基金)?份额[^。]{0,120}(?:不从本类别基金资产中计提|不收取)销售服务费", combined)
        if not sales_zero:
            # A/H共同明示不收费可用于A；不由C类费率、横杠或未列收费行推断为0。
            sales_zero = re.search(r"A类(?:[、/](?:H|I)类)+(?:基金)?(?:份额)?不收取销售服务费", combined)
        if len(tiers) != 1 or decimal(tiers[0][0].replace(",", "")) < 1 or "前收费" not in fees or not sales_zero:
            raise ValueError("A_SHARE_FRONTEND_AND_SALES_FEE_PROOF_REQUIRED")
        subscription = decimal(tiers[0][1]) / 100
        buy_zero = re.search(r"M[<〈＜]\d[\d,]*万(?:元)?[\d.]+%", fees)
        supporting.append(locate(docs["prospectus"] if sales_zero[0] not in fees else document, sales_zero[0]))
    elif not buy_zero:
        combined = compact(docs.get("prospectus", {}).get("text", ""))
        buy_zero = re.search(
            r"C类(?:基金)?份额(?:(?!A类|。).){0,130}不收取(?:申购|认购[、/]申购|认购费和申购)费(?:用)?", combined
        )
        if not buy_zero:
            raise ValueError("COST_SUBSCRIPTION_ZERO_NOT_EXPLICIT")
        supporting.append(locate(docs["prospectus"], buy_zero[0]))
    sell_zero = re.search(
        r"(?:[NYT](?:[≥≧]|>=)(?:7|30|90|180|365)[日天]|(?:7|30|90|180|365)[日天](?:≤|<=)[NYT])[-－—]?0(?:\.0+)?%",
        transaction_fees,
    )
    if not sell_zero:
        raise ValueError("COST_365D_REDEMPTION_ZERO_NOT_EXPLICIT")
    composite = re.findall(
        r"基金运作综合费率[（(]年化[）)](?:持有期间|持有期|基金运作综合费率)?[-－—]?([\d.]+)%", composite_fees
    )
    if len(composite) != 1:
        raise ValueError("COST_OPERATING_RATE_AMBIGUOUS")
    if "联接" in header:
        waiver_pattern = (
            r"投资于目标ETF的部分不收取(?:基金)?管理费[和、](?:基金)?托管费|净值扣除基金(?:财产|资产)中目标ETF份额所对应资产净值后",
            r"投资于目标ETF的部分不收取(?:基金)?管理费。[^。]{0,50}投资于目标ETF的部分不收取(?:基金)?托管费",
        )
        waiver, waiver_doc = None, None
        for candidate in (document, docs.get("prospectus", {})):
            if not candidate.get("text"):
                continue
            waiver = re.search("|".join(waiver_pattern), compact(candidate["text"]))
            if waiver:
                waiver_doc = candidate
                break
        if waiver_doc is None:
            raise ValueError("ETF_MANAGEMENT_CUSTODY_WAIVER_NOT_EXPLICIT")
        supporting.append(locate(waiver_doc, waiver[0]))
    rate = decimal(composite[0]) / 100
    if not 0 <= rate <= 1:
        raise ValueError("COST_RATE_RANGE")
    evidence = locate(document, fees[:2500])
    evidence["source_hashes"] = sorted(
        set(evidence["source_hashes"] + [h for proof in supporting for h in proof["source_hashes"]])
    )
    return {
        "value": str(subscription / (1 + subscription) + rate),
        "observed_on": sent,
        **evidence,
        "supporting_originals": supporting,
        "fee_proof": {
            "subscription": buy_zero[0],
            "redemption_365d": sell_zero[0],
            "operating_rate": composite[0] + "%",
            "subscription_rate": str(subscription),
            "scenario": "CNY_10000_FRONTEND_365D_EXTERNAL",
            "selected_share_label": selected_share_label,
        },
    }


def manager_history(records: list[dict], start: date, end: date) -> tuple[Decimal, Decimal, list[dict]]:
    """逐日重建正式经理责任覆盖与当前团队任期；任职空档或冲突不填充。

    责任覆盖还须由调用方取得原件职责条文和姓名交叉核验，本函数只做区间运算。
    """
    normalized = []
    for row in records:
        a = row.get("begin_date") or row.get("from")
        b = row.get("end_date") or row.get("to")
        name = row.get("manager_name") or row.get("name")
        if not a or not name or b and b < a:
            raise ValueError("MANAGER_RECORD_DATE_OR_NAME")
        normalized.append({"name": name, "from": a, "to": b})
    if not normalized or min(r["from"] for r in normalized) > str(start):
        raise ValueError("MANAGER_HISTORY_START_MISSING")
    teams = []
    for point in sorted({r["from"] for r in normalized} | {r["to"] for r in normalized if r["to"]}):
        if point > str(end):
            continue
        names = sorted({r["name"] for r in normalized if r["from"] <= point and (not r["to"] or r["to"] > point)})
        if not names and point >= str(start):
            raise ValueError("MANAGER_RESPONSIBILITY_GAP")
        if names and (not teams or teams[-1]["members"] != names):
            teams.append({"effective_date": point, "members": names})
    if not teams:
        raise ValueError("MANAGER_CURRENT_TEAM_MISSING")
    last = date.fromisoformat(teams[-1]["effective_date"])
    months = min(Decimal(36), Decimal((end - last).days) / Decimal("30.4375"))
    # 每一自然日检验有效责任人，边界日按正式生效日半开区间处理。
    covered = sum(
        any(
            r["from"] <= str(start + timedelta(days=i)) and (not r["to"] or r["to"] > str(start + timedelta(days=i)))
            for r in normalized
        )
        for i in range((end - start).days + 1)
    )
    return months, Decimal(covered) / ((end - start).days + 1), teams


def quarter_ends(start: date, end: date) -> list[str]:
    return [
        str(date(y, m, d))
        for y in range(start.year, end.year + 1)
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))
        if start < date(y, m, d) <= end
    ]


def historical_report_facts(code: str, refs: list[dict], root: Path, start: date, end: date) -> dict:
    """对已有历史报告重新取资产组合表，不用研究快照中的解析成功标志代替。

    这里只生成可核对的资产事实，尚未匹配当时生效合同和全部投资范围。
    因此即使资产表对账，也不自动生成配置/合规两个完整三年指标。
    """
    records = []
    for ref in refs:
        observed = ref.get("report_end")
        if not observed or not str(start) <= observed <= str(end):
            continue
        if ref.get("label") not in {"half", "annual", "historical-half", "historical-quarter", "historical-annual"}:
            continue
        record = {"observed_on": observed, "source_hashes": [ref["sha256"]], "published_on": ref.get("published_on")}
        try:
            # 经济观察日早于考察日，并不表示报告当时已经披露；保留错误记录，
            # 避免独立回放直接调用本函数时绕过来源复用层的发布日期过滤。
            if not ref.get("published_on"):
                raise ValueError("HISTORICAL_REPORT_PUBLICATION_DATE_UNVERIFIED")
            if date.fromisoformat(str(ref["published_on"])[:10]) > end:
                raise ValueError("HISTORICAL_REPORT_PUBLISHED_AFTER_AS_OF")
            pages, _ = public_document((root / ref["file"]).read_bytes(), ref, code, "报告")
            text = "\n".join(pages)
            normalized = compact(text)
            sent = re.search(r"(?:报告)?送出日期[:：]?(20\d{2})年(\d{1,2})月(\d{1,2})日", normalized)
            if sent:
                sent_on = date(*map(int, sent.groups()))
                record["original_declared_sent_on"] = str(sent_on)
                if sent_on > end:
                    raise ValueError("HISTORICAL_ORIGINAL_SENT_AFTER_AS_OF")
            if code not in normalized:
                raise ValueError("HISTORICAL_FUND_CODE_MISSING")
            year, month, day = map(int, observed.split("-"))
            if not re.search(rf"{year}年0?{month}月0?{day}日", normalized):
                raise ValueError("HISTORICAL_OBSERVATION_DATE_NOT_IN_ORIGINAL")
            if ref.get("label") in {"half", "annual", "historical-half", "historical-annual"}:
                from app.services.fund_quality_v2_governance import report_governance_sections

                try:
                    coverage = report_period(text)
                    if str(coverage[1]) != observed:
                        raise ValueError("GOVERNANCE_REPORT_PERIOD_MISMATCH")
                    record["governance_declarations"] = report_governance_sections(
                        pages, coverage=coverage, source_hash=ref["sha256"], source_url=ref["url"]
                    )
                    if not record["governance_declarations"]:
                        record["governance_gap"] = "GOVERNANCE_REPORT_SECTIONS_NOT_PARSED"
                except ValueError as exc:
                    record["governance_gap"] = str(exc)
            # 部分年度原文/完整转载没有Word自动编号；只建立有成对完整标题的
            # 内部解析视图。治理摘录和原文页号仍使用未改写的原页。
            asset_text = text
            if ref.get("label") in {"annual", "historical-annual"}:
                from app.services.fund_quality_v2_stock_book import annual_parser_view

                asset_pages, chapter_edits = annual_parser_view(pages)
                asset_text = "\n".join(asset_pages)
                if chapter_edits:
                    record["asset_chapter_view_edits"] = chapter_edits
            starts = list(re.finditer(r"(?:[5678]\.1\s*)?(?:报告期末|期末)基金资产组合情况", asset_text))
            if not starts:
                raise ValueError("HISTORICAL_ASSET_SECTION_UNPARSED")
            block = asset_text[starts[-1].end() :]
            stop = re.search(r"(?:[5678]\.2\s*)?(?:报告期末|期末)?按行业", block)
            if not stop:
                raise ValueError("HISTORICAL_ASSET_SECTION_END_UNPARSED")
            block = normalize_number_lines(block[: stop.start()])
            names = (
                "权益投资",
                "基金投资",
                "固定收益投资",
                "贵金属投资",
                "金融衍生品投资",
                "买入返售金融资产",
                "银行存款和结算备付金合计",
                "其他各项资产",
                "合计",
            )
            amounts = {}
            for label in names:
                label_pattern = r"\s*".join(map(re.escape, label))
                if label == "其他各项资产":
                    label_pattern = r"其他(?:各项)?资产"
                if label == "合计":
                    label_pattern = r"(?m)^\s*(?:\d{1,2}\s+)?合\s*计"
                match = re.search(label_pattern + r"\s+([\d,]+\.\d{2}|[-—－])\s+([\d.]+|[-—－])", block)
                if match:
                    amounts[label] = None if match[1] in {"-", "—", "－"} else str(decimal(match[1].replace(",", "")))
            if not amounts.get("合计"):
                raise ValueError("HISTORICAL_TOTAL_ASSETS_UNPARSED")
            total = decimal(amounts["合计"])
            known = sum(decimal(v) for k, v in amounts.items() if k != "合计" and v is not None)
            if total <= 0 or known > total + Decimal("0.02"):
                raise ValueError("HISTORICAL_ASSETS_NOT_RECONCILED")
            record.update(
                amounts_cny=amounts,
                known_amount_sum=str(known),
                total_assets=str(total),
                remaining_amount=str(total - known),
                excerpt=block[:2500],
                source_url=ref["url"],
            )
            from app.services.fund_quality_v2_allocation import report_allocation_point, report_product_net_assets

            try:
                record["allocation_asset_point"] = report_allocation_point(
                    block, amounts, industry_text=normalize_number_lines(asset_text[starts[-1].end():]))
                try:
                    record["allocation_asset_point"]["net_asset_evidence"] = report_product_net_assets(
                        pages, observed_on=observed, total_assets=amounts["合计"])
                except ValueError as exc:
                    # 净资产尚未识别不抹掉已核总资产点；也不把总资产坐标当
                    # 合同分母。按具名缺口阻止净资产条件演算。
                    record["allocation_asset_point"]["net_assets_gap"] = str(exc)
            except (ValueError, KeyError) as exc:
                # 此处只定位股票及港股的点金额，未读取的现金/其他资产横杠
                # 仍保留原值；子集点账簿不能充作完整允许集合或合规结论。
                record["allocation_point_gap"] = str(exc)
            if abs(total - known) > Decimal("0.02"):
                record["gap"] = "HISTORICAL_ASSET_ROW_OR_PRECISION_UNRESOLVED"
            try:
                pressure, quote, holder_rows = holder_pressure(text)
                record["holder_pressure"] = {"value": str(pressure), "rows": holder_rows, "excerpt": quote[:2500]}
            except ValueError as exc:
                record["holder_gap"] = str(exc)
        except (ValueError, KeyError, OSError) as exc:
            record["error"] = str(exc)
        records.append(record)
    return {
        "reports": records,
        "contract_sources": [
            {k: r.get(k) for k in ("sha256", "url", "published_on", "title")}
            for r in refs
            if r.get("label") in {"historical-contract", "prospectus", "historical-prospectus"}
        ],
    }


def extract_disclosures(
    code: str, refs: list[dict], root: Path, as_of: date, *, bundle: dict | None = None, frozen: datetime | None = None
) -> dict:
    """补齐原件可直接证明的指标；其余项输出字段、原文、缺口及可取得来源。

    bundle仅传已完成既有原文重建校验的事实。费用、治理和持仓分项分数均不读取。
    """
    documents = source_documents(code, refs, root, as_of)
    metrics, findings, facts = {}, {}, {}
    start = as_of.replace(year=as_of.year - 3)
    report = documents.get("report", documents.get("half", {}))
    report_pages = report.get("pages", [])
    if report.get("report_kind") == "annual" and report_pages:
        from app.services.fund_quality_v2_stock_book import periodic_financial_view

        report_pages, view_evidence = periodic_financial_view(report_pages)
        facts["primary_report_parser_view"] = view_evidence | {"source_hashes": [report["ref"]["sha256"]]}
    if report.get("text"):
        facts["primary_report_selection"] = {
            "report_kind": report.get("report_kind", "half"),
            "economic_period": report.get("economic_period"),
            "published_on": report["ref"].get("published_on"),
            "original_sent_on": report.get("original_sent_on"),
            "as_of_date": str(as_of),
            "source_hashes": [report["ref"]["sha256"]],
            "scope": "LATEST_AVAILABLE_COMPLETE_PERIODIC_REPORT_NOT_AS_OF_PORTFOLIO",
            "source_selection": report.get("selection_evidence"),
            "regulatory_source_evidence": report.get("regulatory_source_evidence"),
        }
    summary = documents.get("summary", {})
    prospectus = documents.get("prospectus", {})

    def missing(name, state, reason, required, document=None, quote=""):
        findings[name] = {
            "state": state,
            "reason": reason,
            "required_data": required,
            "searched_source_hashes": [r["sha256"] for r in refs],
        }
        if document and quote:
            findings[name].update(locate(document, quote))

    def record(name, value, observed, document, quote, extra=None):
        metrics[name] = {"value": str(value), "observed_on": str(observed), **locate(document, quote), **(extra or {})}

    period = None
    class_doc = summary if summary.get("text") else report
    if class_doc.get("text"):
        from app.services.fund_quality_v2_classification import classify_original

        normalized = compact(class_doc["text"])
        title = normalized.split("编制日期")[0].split("送出日期")[0][:180]
        classification = classify_original(title, report.get("text", ""))
        # 无QDII封面字样的境外目标证据实际在报告产品说明，不能绑到概要标题。
        scope_document = report if classification["rule"] == "FORMAL_PRODUCT_OBJECTIVE_AND_FOREIGN_SCOPE" else class_doc
        facts["classification"] = {**classification, **locate(scope_document, classification["excerpt"])}
    if report.get("text"):
        master = re.search(r"基金主代码[：:]?([0-9]{6})(?!\d)", compact(report["text"])[:16000])
        if master:
            facts["product_identity"] = {"product_id": "CN_PUBLIC_FUND:" + master[1], **locate(report, master[0])}
    for document in (summary, report):
        if not document.get("text"):
            continue
        effective = re.search(
            r"基金合同生效日(?:期)?[:：]?(20[0-9]{2})[-年]([0-9]{1,2})[-月]([0-9]{1,2})日?",
            compact(document["text"])[:16000],
        )
        if effective:
            facts["contract_effective_date"] = {
                "date": str(date(*map(int, effective.groups()))),
                "scope": "ORIGINAL_CURRENT_CONTRACT_DATE_NOT_PROOF_OF_NO_PRIOR_TRANSFORMED_PRODUCT",
                **locate(document, effective[0]),
            }
            break
    if report.get("text"):
        text = report["text"]
        try:
            candidate_period = report_period(text)
            if candidate_period[1] > as_of or (as_of - candidate_period[1]).days > 210:
                raise ValueError("REPORT_PERIOD_FUTURE_OR_EXPIRED")
            period = candidate_period
            facts["report_period"] = list(map(str, period))
            value, quote, rows = holder_pressure(text)
            record("holder_pressure_band", value, period[1], report, quote, {"holder_rows": rows})
        except ValueError as exc:
            reason = str(exc)
            state = "SOURCE_VALUE_UNRESOLVED"
            required = "完整当期20%持有人及期初/申购/赎回/期末份额表"
            if reason == "HOLDER_PERIOD_SECTION_MISSING" and "单一投资者" not in text:
                # 有完整报告正文且连相关声明也未出现，与表格已有但解析失败分开。
                # 条件填报模板允许省略一些项目，仍不能由栏目缺席推定未触发。
                is_reprint = (
                    report.get("source_format") == "ISSUER_DOCUMENT_PUBLIC_REPRINT"
                    or report["ref"].get("format") == "ISSUER_DOCUMENT_PUBLIC_REPRINT"
                )
                reason = (
                    "REPRINT_THRESHOLD_STATEMENT_NOT_PRESENT_DIRECT_REPORT_CONFIRMATION_REQUIRED"
                    if is_reprint
                    else "ORIGINAL_THRESHOLD_STATEMENT_NOT_PRESENT_NOT_PROOF_OF_ZERO"
                )
                state = "SOURCE_STATEMENT_UNAVAILABLE"
                required = "当期明确未触发20%阈值的原文，或完整触发区间及份额表；现有正文无该声明，不能推定为零"
                facts["holder_statement_presence"] = {
                    "state": reason,
                    "source_hashes": [report["ref"]["sha256"]],
                    "source_url": report["ref"]["url"],
                    "report_back_matter_present": "备查文件" in text,
                    "scope": "原文检索事实；不作未触发阈值的证明，不等于资料渠道权限不足",
                }
            missing(
                "holder_pressure_band",
                state,
                reason,
                required,
                report,
                text[-2500:],
            )
        if period:
            # 财务基础事实直接读取原报告，不要求国内股票表章号，也不使用
            # 内部改写页。它们只补充后续资料链，不产生现金/容量的质量分值。
            try:
                from app.services.fund_quality_v2_accounting import accounting_observations

                accounting = accounting_observations(report.get("pages", []), period[1])
                facts["accounting_observations"] = {
                    **accounting, "source_hashes": [report["ref"]["sha256"]],
                    "source_url": report["ref"]["url"], "published_on": report["ref"].get("published_on"),
                    "source_format": report.get("source_format"),
                }
            except ValueError as exc:
                facts["accounting_observation_gap"] = {
                    "reason": str(exc), "source_hashes": [report["ref"]["sha256"]],
                    "scope": "ORIGINAL_AVAILABLE_ACCOUNTING_LAYOUT_UNRESOLVED_NOT_MISSING_REPORT",
                }
            # 治理原文先提取并保留；半年度未受处罚不等于三年完整核查或独立评审。
            try:
                quote = section(text, r"(?:9|10)\.[67]\.1\s*管理人受调查或处罚", r"(?:9|10)\.[67]\.2")
                facts["governance_current_period"] = locate(report, quote) | {"coverage": list(map(str, period))}
                missing(
                    "governance_state",
                    "SOURCE_WINDOW_INCOMPLETE",
                    "GOVERNANCE_THREE_YEAR_AND_INDEPENDENT_REVIEW_REQUIRED",
                    {
                        "required_from": str(start),
                        "required_to": str(as_of),
                        "available": list(map(str, period)),
                        "source": "管理人历年年度/中期报告、监管机关原文、整改与适用对象独立复核",
                    },
                    report,
                    quote,
                )
            except ValueError as exc:
                missing("governance_state", "SOURCE_VALUE_UNRESOLVED", str(exc), "三年治理原文及独立复核")
    if not report.get("text"):
        facts["report_error"] = report.get("error", "PRIMARY_PERIODIC_REPORT_NOT_REGISTERED")

    # 先把合同里的实际约束解析清楚，避免把现金+短债的5%误当现金目标。
    contracts = [d for d in (prospectus, summary, report) if d.get("text")]
    from app.services.fund_quality_v2_materials import benchmark_expression

    for document in (summary, prospectus, report):
        if document.get("text") and (expression := benchmark_expression(compact(document["text"]))):
            facts["contract_benchmark_expression"] = {"expression": expression, **locate(document, expression)}
            break
    # 基准目录可能比公告正文落款早一天，当前公式也可能已在窗口中途变更。
    # 保留两种日期及具名旧/新列，不能据最新招募书把三年历史统一回算。
    from app.services.fund_quality_v2_benchmark_contracts import benchmark_transitions, contract_calculation

    report_name = re.search(r"基金名称[:：]?([^:：。]{4,100}?证券投资基金)", compact(report.get("text", ""))[:16000])
    if report_name:
        facts["contract_benchmark_transitions"] = benchmark_transitions(
            refs=refs, root=root, legal_name=report_name[1], as_of=as_of, code=code,
        )
    literal = facts.get("contract_benchmark_expression", {}).get("expression")
    if literal and urlsplit(prospectus.get("ref", {}).get("url", "")).hostname == "static.cmfchina.com":
        try:
            calculation = contract_calculation(prospectus["pages"], literal)
            facts["contract_benchmark_calculation"] = calculation | locate(prospectus, calculation["method_excerpt"])
        except ValueError as exc:
            facts["contract_benchmark_calculation_gap"] = {
                "reason": str(exc), "source_hashes": [prospectus["ref"]["sha256"]],
                "scope": "EXISTING_CONTRACT_ORIGINAL_METHOD_UNRESOLVED_NOT_MISSING_DOCUMENT",
            }
    for document in contracts:
        normalized = compact(document["text"])
        basket = re.search(
            r"[^。]{0,140}(?:现金(?:或者|或)|现金、?银行存款)[^。]{0,240}(?:5[%％]|百分之五)[^。]{0,100}", normalized
        )
        if basket and ("一年" in basket[0] or "政府债券" in basket[0]):
            facts["cash_contract_constraint"] = locate(document, basket[0]) | {
                "meaning": "CASH_PLUS_SHORT_GOVERNMENT_BONDS_MINIMUM_NOT_CASH_TARGET"
            }
            missing(
                "cash_target_deviation",
                "METHOD_REFERENCE_REQUIRED",
                "CONTRACT_BASKET_IS_NOT_CASH_TARGET_INTERVAL",
                "预先冻结的适用现金目标区间；合同现金加一年内政府债券下限不能代替现金上下界",
                document,
                basket[0],
            )
            break
    if "cash_target_deviation" not in findings:
        missing(
            "cash_target_deviation",
            "METHOD_REFERENCE_REQUIRED",
            "CASH_TARGET_INTERVAL_NOT_DEFINED_BY_AVAILABLE_CONTRACT",
            "预先冻结并经验证的本类现金目标上下界及对应合同版本",
        )

    # 历史团队可直接从已有来源重建，职责覆盖另须正式职责原文及名单交叉核验。
    records = (bundle or {}).get("manager", {}).get("records", [])
    from app.services.fund_quality_v2_manager_sources import formal_manager_duty

    duty = formal_manager_duty(prospectus.get("text", ""))
    if duty:
        facts["formal_manager_duty_clause"] = {
            **duty,
            **locate(prospectus, duty["excerpt"]),
            "document_published_on": prospectus["ref"].get("published_on"),
            "scope_note": "职位责任条文；任职覆盖另由生效日期逐日核验，不表示职责履行质量或治理清白",
        }
    manager_hashes = []
    provider = next((r for r in refs if r.get("label") == "provider"), None)
    if provider:
        snapshot = json.loads((root / provider["file"]).read_bytes())
        if snapshot.get("fund_code") != code:
            raise ValueError("MANAGER_PROVIDER_PRODUCT_MISMATCH")
        records = snapshot.get("manager_records", [])
        manager_hashes = [provider["sha256"]]
        from app.services.fund_quality_v2_nav import rebuild_nav_coverage

        try:
            facts["own_nav_coverage"] = rebuild_nav_coverage(snapshot, fund_code=code, as_of=as_of) | {
                "source_hashes": [provider["sha256"]]
            }
            if facts.get("classification", {}).get("kind") == "GLOBAL_ACTIVE_EQUITY" or "QDII" in compact(
                class_doc.get("text", "")[:500]
            ):
                facts["own_nav_coverage"].pop("curve", None)
                facts["own_nav_coverage"]["status"] = "GLOBAL_VALUATION_CALENDAR_REVIEW_REQUIRED"
        except (ValueError, KeyError, TypeError) as exc:
            facts["own_nav_coverage"] = {"source_hashes": [provider["sha256"]], "error": str(exc)}
    database_snapshot = None
    database_nav_refs = sorted(
        [r for r in refs if r.get("label") == "nav-database-snapshot"], key=lambda r: r["acquired_at"]
    )
    if database_nav_refs:
        from app.services.fund_quality_v2_database_nav import database_nav_coverage

        nav_ref = database_nav_refs[-1]
        snapshot = json.loads((root / nav_ref["file"]).read_bytes())
        database_snapshot = snapshot
        facts["own_nav_coverage"] = database_nav_coverage(
            snapshot, fund_code=code, as_of=as_of, frozen=frozen or datetime.fromisoformat(nav_ref["acquired_at"])
        ) | {"source_hashes": [nav_ref["sha256"]]}
        if "QDII" in compact(class_doc.get("text", "")[:500]):
            facts["own_nav_coverage"].pop("curve", None)
            facts["own_nav_coverage"]["status"] = "GLOBAL_VALUATION_CALENDAR_REVIEW_REQUIRED"
    # 发布方的成分价格先保留独立日期，不把HSCI点位当人民币合同基准。
    # 缺少的港股收盘日仍待休市/估值规则核验，不能自动填前值或直接补零。
    if any(r.get("label") == "component-price-history" for r in refs):
        from app.services.fund_quality_v2_component_prices import component_histories

        own_curve = facts.get("own_nav_coverage", {}).get("curve", {})
        required_days = own_curve.get("dates", [])
        component_result = component_histories(
            refs=refs, root=root, required_from=date.fromisoformat(required_days[0]) if required_days else start,
            as_of=as_of, required_fund_dates=required_days,
            frozen=frozen,
        )
        bound_contract = facts.get("contract_benchmark_expression", {})
        if "恒生综合指数收益率" not in bound_contract.get("expression", ""):
            component_result = {"error": "COMPONENT_NOT_IN_BOUND_CONTRACT_EXPRESSION", "scoring_metric": None}
        facts["contract_component_price_history"] = component_result | {
            "contract_source_hashes": bound_contract.get("source_hashes", [])}
    # 成对历史不再被旧版八维整包准入阻断。这里只复用真实本份额净值和
    # 发行人日合同基准，旧包已有成对曲线时保持原路径及来源绑定不变。
    benchmark_refs = sorted(
        [r for r in refs if r.get("label") == "benchmark"], key=lambda r: r.get("acquired_at", "")
    )
    own = facts.get("own_nav_coverage", {})
    contract = facts.get("contract_benchmark_expression", {})
    if not bundle and own.get("curve") and contract.get("expression") and benchmark_refs:
        from app.services.fund_quality_v2_materials import relative_history
        from app.services.fund_quality_v2_paired_history import paired_history

        benchmark_ref = benchmark_refs[-1]
        try:
            transitions = facts.get("contract_benchmark_transitions", {})
            if transitions.get("events") or transitions.get("unresolved_originals"):
                raise ValueError("QUALITY_PAIRED_KNOWN_BENCHMARK_CHANGE_EFFECTIVE_HISTORY_UNVERIFIED")
            benchmark_raw = (root / benchmark_ref["file"]).read_bytes()
            reconciliation, cash_ref = None, None
            try:
                paired = paired_history(own["curve"], benchmark_raw, benchmark_ref, fund_code=code)
            except ValueError as conflict:
                cash_refs = sorted([r for r in refs if r.get("label") == "cash-event-snapshot"],
                                   key=lambda r: r["acquired_at"])
                notice_refs = [r for r in refs if r.get("label") == "cash-dividend-notice"]
                if (str(conflict) != "QUALITY_PAIRED_PROVIDER_PUBLISHER_NAV_CONFLICT"
                        or database_snapshot is None or not cash_refs or not notice_refs):
                    raise
                from app.services.fund_quality_v2_cash_reconciliation import reconcile_cash_history

                cash_ref = cash_refs[-1]
                cash_snapshot = json.loads((root / cash_ref["file"]).read_bytes())
                reconciliation = reconcile_cash_history(
                    own["curve"], database_snapshot, benchmark_raw, benchmark_ref, cash_snapshot,
                    notice_refs, root, fund_code=code, as_of=as_of,
                )
                # 派生序列单独保留，不能把它写回own_nav_coverage或供应商原件。
                paired = paired_history(reconciliation["curve"], benchmark_raw, benchmark_ref, fund_code=code)
            values = relative_history(paired)
            hashes = sorted(set(own["source_hashes"] + contract["source_hashes"] + [benchmark_ref["sha256"]]))
            if reconciliation:
                hashes = sorted(set(hashes + reconciliation["event_source_hashes"] + [cash_ref["sha256"]]))
            facts["paired_contract_history"] = {
                "status": "PAIRED_OWN_NAV_ISSUER_CONTRACT_BENCHMARK_REBUILT",
                "history": paired, "source_hashes": hashes,
                "scope": "RETROSPECTIVE_ORIGINAL_RECONSTRUCTION_NOT_PRIOR_PUBLICATION_OR_REVISION_PROOF",
                "valuation_end": paired["dates"][-1],
                "availability_cutoff": str(as_of),
            }
            if reconciliation:
                facts["paired_contract_history"]["cash_reinvestment_reconciliation"] = reconciliation
                facts["paired_contract_history"]["original_provider_curve_conflict"] = {
                    "error": "QUALITY_PAIRED_PROVIDER_PUBLISHER_NAV_CONFLICT",
                    "maximum_normalized_difference": reconciliation["maximum_original_provider_difference"],
                    "source_hashes": own["source_hashes"],
                    "scope": "ORIGINAL_PROVIDER_RETAINED_DERIVED_HISTORY_SEPARATE",
                }
            for name, value in values.items():
                metrics[name] = {
                    **contract, "value": str(value), "observed_on": paired["dates"][-1], "source_hashes": hashes,
                    "publisher_cross_check": paired["publisher_cross_check"],
                    **({"return_source_basis": reconciliation["derived_source"]} if reconciliation else {}),
                }
        except (ValueError, KeyError, TypeError, OSError) as exc:
            facts["paired_contract_history"] = {
                "error": str(exc), "source_hashes": [benchmark_ref["sha256"], *own.get("source_hashes", [])],
            }
    manager_refs = sorted([r for r in refs if r.get("label") == "manager-api"], key=lambda r: r["acquired_at"])
    if manager_refs:
        from app.services.fund_quality_v2_manager_sources import manager_api_records

        latest_manager = manager_refs[-1]
        records = manager_api_records(
            json.loads((root / latest_manager["file"]).read_bytes()), fund_code=code, ts_code=latest_manager["ts_code"]
        )
        manager_hashes = [latest_manager["sha256"]]
        facts["public_manager_history"] = {
            "records": records,
            "source_hashes": manager_hashes,
            "acquired_at": latest_manager["acquired_at"],
        }
    master_refs = sorted([r for r in refs if r.get("label") == "manager-master-api"], key=lambda r: r["acquired_at"])
    if master_refs:
        from app.services.fund_quality_v2_manager_sources import manager_api_records, validate_product_manager_scope

        master_ref = master_refs[-1]
        try:
            product = facts.get("product_identity", {})
            master = product.get("product_id", "").rsplit(":", 1)[-1]
            if not master or master != master_ref.get("master_code") or master_ref["ts_code"].split(".")[0] != master:
                raise ValueError("MANAGER_MASTER_NOT_BOUND_BY_ORIGINAL_PRODUCT")
            product_records = manager_api_records(
                json.loads((root / master_ref["file"]).read_bytes()), fund_code=master, ts_code=master_ref["ts_code"]
            )
            validate_product_manager_scope(product_records, records)
            records = product_records
            manager_hashes += [master_ref["sha256"], *product["source_hashes"]]
            facts["product_manager_history"] = {
                "records": records,
                "product_id": product["product_id"],
                "source_hashes": sorted(set(manager_hashes)),
                "scope": "PRODUCT_MANAGERS_ONLY_NOT_SHARE_NAV_FEES_OR_NEW_CALIBRATION_PRODUCT",
            }
        except ValueError as exc:
            facts["product_manager_history"] = {"error": str(exc), "source_hashes": [master_ref["sha256"]]}
    appointments = []
    for ref in [r for r in refs if r.get("label") == "appointment"]:
        from app.services.fund_quality_v2_manager_sources import new_team_from_appointment

        try:
            if urlsplit(ref["url"]).hostname not in HOSTS | {
                "www.nffund.com",
                "www.bosera.com",
                "www.dbfund.com.cn",
                "dbfund.com.cn",
            }:
                raise ValueError("APPOINTMENT_SOURCE_HOST_NOT_REGISTERED")
            master = facts.get("product_identity", {}).get("product_id", "").rsplit(":", 1)[-1] or code
            pages, _ = public_document((root / ref["file"]).read_bytes(), ref, master, "基金经理")
            notice = new_team_from_appointment("\n".join(pages), fund_code=master, as_of=as_of)
            if ref.get("published_on") and ref["published_on"] != notice["published_on"]:
                raise ValueError("APPOINTMENT_NOTICE_PUBLICATION_MISMATCH")
            notice_document = {"pages": pages, "text": "\n".join(pages), "ref": ref}
            if master != code:
                notice_document["identity_source_hashes"] = facts["product_identity"]["source_hashes"]
            appointments.append((notice, notice_document))
            facts.setdefault("appointment_notices", []).append(notice | locate(notice_document, notice["excerpt"]))
        except (ValueError, KeyError, OSError) as exc:
            facts.setdefault("appointment_notices", []).append({"source_hashes": [ref["sha256"]], "error": str(exc)})
    if records:
        try:
            months, coverage, teams = manager_history(records, start, as_of)
            team_doc = summary if summary.get("text") else report
            # 用正式原文中的确切姓名交叉核对当前责任人，不能只信来源自填标志。
            latest = teams[-1]["members"]
            matching_notices = [
                (notice, document)
                for notice, document in appointments
                if notice["members"] == latest and notice["effective_date"] == teams[-1]["effective_date"]
            ]
            if matching_notices:
                notice, team_doc = matching_notices[-1]
                quote = notice["excerpt"]
            else:
                if not team_doc.get("text") or not all(name in compact(team_doc["text"]) for name in latest):
                    raise ValueError("CURRENT_TEAM_ORIGINAL_NAME_CROSSCHECK_MISSING")
                normalized = compact(team_doc["text"])
                points = [normalized.index(name) for name in latest]
                quote = normalized[max(0, min(points) - 100) : max(points) + 120]
                if "基金经理" not in quote:
                    raise ValueError("CURRENT_TEAM_ROLE_CLAUSE_MISSING")
            hashes = manager_hashes
            record("team_months", months, as_of, team_doc, quote, {"teams": teams})
            metrics["team_months"]["source_hashes"] += hashes
            if duty and coverage == 1:
                record(
                    "key_person_coverage",
                    coverage,
                    as_of,
                    prospectus,
                    duty["excerpt"],
                    {
                        "teams": teams,
                        "duty_clause_kind": duty["kind"],
                        "coverage_scope": "APPOINTMENT_DAYS_WITH_FORMAL_ROLE_NOT_DUTY_EXECUTION_OR_GOVERNANCE",
                    },
                )
                metrics["key_person_coverage"]["source_hashes"] += hashes + [team_doc["ref"]["sha256"]]
            else:
                missing(
                    "key_person_coverage",
                    "SOURCE_VALUE_UNRESOLVED",
                    "FORMAL_RESPONSIBILITY_CLAUSE_OR_FULL_COVERAGE_MISSING",
                    "正式基金经理职责条文、全部任职/代理生效记录和连续自然日覆盖证明",
                )
        except ValueError as exc:
            if str(exc) == "MANAGER_HISTORY_START_MISSING" and facts.get("contract_effective_date", {}).get(
                "date", ""
            ) > str(start):
                missing(
                    "team_months",
                    "SOURCE_WINDOW_INCOMPLETE",
                    "AVAILABLE_PRODUCT_HISTORY_SHORTER_THAN_36_MONTHS",
                    {
                        "required_from": str(start),
                        "available_manager_from": min(r.get("begin_date") or "9999-12-31" for r in records),
                        "original_current_contract_date": facts["contract_effective_date"]["date"],
                        "source": "现有原文与任职记录起点晚于候选窗口；转型前历史须另核，不能补入成立前记录",
                    },
                )
            else:
                missing("team_months", "SOURCE_VALUE_UNRESOLVED", str(exc), "与当前正式原文名单一致的完整团队任职历史")
    else:
        missing(
            "team_months",
            "COLLECTION_REQUIRED",
            "MANAGER_ORIGINAL_RECORDS_NOT_REGISTERED",
            "正式经理任免及历史完整来源快照",
        )

    # 当前报告和当前招募书不能冒充过去每季度的完整配置与历史约束。
    expected = quarter_ends(start, as_of)
    history = historical_report_facts(code, refs, root, start, as_of)
    facts["historical_asset_reports"] = history
    from app.services.fund_quality_v2_report_availability import disclosure_window

    published_dates = {r["observed_on"] for r in history["reports"] if "error" not in r}
    availability = disclosure_window(expected, as_of=as_of, prospectus=prospectus, published_dates=published_dates)
    facts["historical_report_availability"] = availability
    expected = availability["required_report_dates"]
    from app.services.fund_quality_v2_governance import declaration_coverage, report_governance_sections

    declarations = [item for r in history["reports"] for item in r.get("governance_declarations", [])]
    if period:
        try:
            declarations += report_governance_sections(
                report["pages"], coverage=period, source_hash=report["ref"]["sha256"], source_url=report["ref"]["url"]
            )
        except ValueError as exc:
            facts["governance_current_parse_gap"] = str(exc)
    unique = {(item["scope"], tuple(item["coverage"]), tuple(item["source_hashes"])): item for item in declarations}
    facts["governance_report_declarations"] = list(unique.values())
    facts["governance_report_coverage"] = declaration_coverage(list(unique.values()), start=start, end=as_of)
    from app.services.fund_quality_v2_governance import report_manager_name, safe_decision

    current_manager = report_manager_name(report.get("pages", []))
    regulator_decisions = []
    for ref in [r for r in refs if r.get("label") == "governance-regulator"]:
        origin = urlsplit(ref["url"])
        if (
            ref.get("source_kind") != "SAFE_PUBLIC_DECISION"
            or origin.scheme != "https"
            or origin.hostname != "www.safe.gov.cn"
            or origin.path != "/www/illegal/listQuery"
        ):
            raise ValueError("GOVERNANCE_UNTRUSTED_REGULATOR_SOURCE")
        decision = safe_decision((root / ref["file"]).read_bytes())
        if not current_manager or decision["legal_entity"] != current_manager:
            facts.setdefault("governance_source_scope_gaps", []).append(
                {
                    "source_hashes": [ref["sha256"]],
                    "reason": "REGULATOR_SUBJECT_NOT_BOUND_TO_CURRENT_REPORT_MANAGER",
                    "reported_manager": current_manager,
                    "decision_subject": decision["legal_entity"],
                }
            )
            continue
        if decision["event_date"] > str(as_of):
            raise ValueError("GOVERNANCE_REGULATOR_EVENT_AFTER_AS_OF")
        regulator_decisions.append(
            decision
            | {
                "source_hashes": [ref["sha256"], report["ref"]["sha256"]],
                "source_url": ref["url"],
                "source_received_at": ref["acquired_at"],
                "published_on": None,
                "publication_date_note": "页面未列实际公示日；2020更新时间不是2025处罚决定日期",
                "applicability": "MANAGER_COMPANY_EVENT_NOT_PROOF_OF_SPECIFIC_FUND_HARM",
            }
        )
    facts["official_governance_decisions"] = regulator_decisions
    known_dates = sorted(
        {
            day
            for item in unique.values()
            for event in item.get("reported_events", [])
            for day in event["event_dates"]
            if str(start) <= day <= str(as_of)
        }
    )
    if "governance_state" in findings:
        required = findings["governance_state"].get("required_data")
        if not isinstance(required, dict):
            required = {"required": required}
        required.update(
            report_declaration_coverage=facts["governance_report_coverage"],
            known_declared_event_dates=known_dates,
            official_decisions_obtained=[
                {
                    "document_number": d["document_number"],
                    "event_date": d["event_date"],
                    "source_hashes": d["source_hashes"],
                }
                for d in regulator_decisions
            ],
        )
        findings["governance_state"]["required_data"] = required
        if known_dates:
            findings["governance_state"]["reason"] = (
                "KNOWN_ADVERSE_DECLARATIONS_SCOPE_SEVERITY_AND_REMEDIATION_REVIEW_REQUIRED"
            )
    actual = sorted(
        set(
            ([str(period[1])] if period else [])
            + [r["observed_on"] for r in history["reports"] if "error" not in r and "gap" not in r]
        )
    )
    from app.services.fund_quality_v2_allocation import allocation_history

    allocation = allocation_history(
        refs=refs, root=root, reports=history["reports"], required_dates=expected,
        legal_name=report_name[1] if report_name else "", as_of=as_of, code=code,
    ) if report_name else {"observations": [], "reason": "REPORT_LEGAL_NAME_UNPARSED", "scoring_metric": None}
    facts["historical_allocation_constraints"] = allocation
    for name in ("allocation_deviation", "mandate_deviation"):
        required = {
                "expected_report_dates": expected,
                "available_report_dates": actual,
                "missing_report_dates": sorted(set(expected) - set(actual)),
                "disclosure_wait_or_unverified_dates": [
                    r for r in availability["checkpoints"] if r["observed_on"] not in expected
                ],
                "source": "各期正式季度/中期/年度完整资产配置、同期生效合同、投资范围及穿透合规事实",
                "contract_versions_available": len(history["contract_sources"]),
                "remaining_contract_gap": "各版本生效日与历史披露逐期匹配；当前合同不能倒用",
                "allocation_constraint_evidence": allocation,
                "remaining_asset_gap": "配置需完整合同允许集合；投资范围偏离另需完整穿透资产及逐项范围核验",
            }
        missing(
            name,
            "SOURCE_WINDOW_INCOMPLETE" if required["missing_report_dates"] else "SOURCE_VALUE_UNRESOLVED",
            "HISTORICAL_REPORTS_MISSING" if required["missing_report_dates"]
            else "COMPLETE_ALLOWED_SET_EFFECTIVE_CONTRACT_AND_DENOMINATOR_BINDING_REQUIRED"
            if name == "allocation_deviation"
            else "FULL_ASSET_SCOPE_CLASSIFICATION_AND_EFFECTIVE_CONTRACT_BINDING_REQUIRED",
            required,
        )
    missing(
        "capacity_pressure",
        "METHOD_REFERENCE_REQUIRED",
        "STRATEGY_CAPACITY_BOUNDS_NOT_EVIDENCED",
        "经证据支持并预先冻结的策略容量上界与持续运作下界；规模值或募集上限不等于策略容量",
    )

    from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope

    def strict_report_balance():
        """从本次实际原件取得点账簿；保留横杠原值及唯一推导，不走V1金额默认值。"""
        from app.services.fund_quality_v2_balance_proofs import strict_balance_proof

        proof = strict_balance_proof(
            report_pages, observed_on=str(period[1]), source_hash=report["ref"]["sha256"]
        )
        facts["strict_balance_proof"] = proof
        if proof["point_book"] is None:
            raise ValueError("QUALITY_RESTRICTED_EXACT_BALANCE_POINT_NOT_IDENTIFIED")
        return proof["point_book"]

    # 全资产受限范围必须由原文说明，才能使用已经逐项重建的受限金额。
    if bundle and period and report.get("text"):
        normalized = compact(report["text"])
        scope = full_asset_restriction_scope(normalized)
        holding = bundle["holdings"]
        is_link = "link_balance" in holding
        if scope and not is_link and bundle.get("disclosure", {}).get("restricted_stock_amount") is not None:
            amount = decimal(bundle["disclosure"]["restricted_stock_amount"])
            net = decimal(holding["net_assets"])
            try:
                b = strict_report_balance()
                if decimal(b["net_assets"]) != net:
                    raise ValueError("V2_RESTRICTED_ORIGINAL_NET_ASSET_CONFLICT_WITH_LEGACY_COMPONENT")
                rows = []
                if amount > 0:
                    from app.services.fund_quality_v2_restrictions import restricted_securities
                    from app.services.fund_rating_public_stock_tables import parse

                    positions = parse(report_pages, b["stock_amount"], b["net_assets"], require_industry=False)[
                        "holdings"
                    ]
                    rebuilt = restricted_securities(report_pages, positions, b["bond_amount"])
                    facts["restricted_original_rebuild"] = {
                        "source_hashes": [report["ref"]["sha256"]],
                        "parser_header_view": rebuilt.get("parser_header_view"),
                        "stock_layout_evidence": rebuilt.get("stock_layout_evidence"),
                        "empty_bond_table_evidence": rebuilt.get("empty_bond_table_evidence"),
                        "parsed_tables_amount_cny": rebuilt["restricted_amount"],
                        "parsed_row_count": len(rebuilt["rows"]),
                        "subtotal_scope": "PARSED_DISCLOSURE_TABLES_NOT_PROOF_OF_ALL_ASSET_SCOPE",
                        "pledged_bond_evidence": rebuilt.get("pledged_bond_evidence"),
                    }
                    if decimal(rebuilt["restricted_amount"]) != amount or decimal(b["net_assets"]) != net:
                        raise ValueError("V2_RESTRICTED_ORIGINAL_AMOUNT_CONFLICT_WITH_LEGACY_COMPONENT")
                    rows = rebuilt["rows"]
                    legacy = bundle["disclosure"].get("restricted_rows", [])
                    changed = []
                    for item in rows:
                        previous = [
                            r
                            for r in legacy
                            if (r["code"], r["cost"], r["amount"]) == (item["code"], item["cost"], item["amount"])
                        ]
                        if len(previous) == 1 and decimal(previous[0]["quantity"]) != decimal(item["quantity"]):
                            changed.append(
                                {
                                    "code": item["code"],
                                    "legacy_quantity": previous[0]["quantity"],
                                    "original_quantity": item["quantity"],
                                    "cost": item["cost"],
                                    "amount": item["amount"],
                                }
                            )
                    if changed:
                        facts["legacy_restriction_quantity_differences"] = {
                            "rows": changed,
                            "source_hashes": [report["ref"]["sha256"]],
                            "scope": "V2股数从原文重建；旧组件只作差异对照，不改旧来源包或正式评分",
                        }
                else:
                    # 旧原件组件的0也必须逐项重建受限章节，不把已有0当核验标志。
                    # 全部章节明确无受限时不需要股票股数；出现正受限行会因缺
                    # 持仓绑定而拒收，不能退回旧0。下一轮应补全表再重建该正值。
                    from app.services.fund_quality_v2_restrictions import restricted_securities

                    rebuilt = restricted_securities(report_pages, [], b["bond_amount"])
                    if decimal(rebuilt["restricted_amount"]) != amount:
                        raise ValueError("V2_RESTRICTED_ORIGINAL_AMOUNT_CONFLICT_WITH_LEGACY_COMPONENT")
                    rows = rebuilt["rows"]
                    facts["restricted_original_rebuild"] = {
                        "source_hashes": [report["ref"]["sha256"]],
                        "parser_header_view": rebuilt.get("parser_header_view"),
                        "parsed_tables_amount_cny": rebuilt["restricted_amount"], "parsed_row_count": len(rows),
                        "subtotal_scope": "PARSED_DISCLOSURE_TABLES_NOT_PROOF_OF_ALL_ASSET_SCOPE",
                        "pledged_bond_evidence": rebuilt.get("pledged_bond_evidence"),
                    }
                ratio = amount / net
                if not 0 <= ratio <= 1:
                    raise ValueError("RESTRICTED_ASSET_RATIO_RANGE")
                record(
                    "restricted_ratio",
                    ratio,
                    period[1],
                    report,
                    scope[0],
                    {
                        "restricted_amount": str(amount),
                        "net_assets": str(net),
                        "restricted_rows": rows,
                        "scope": "DISCLOSED_FULL_ASSET_RESTRICTIONS_NOT_REALTIME_LIQUIDITY",
                    },
                )
            except (ValueError, KeyError, TypeError) as exc:
                missing(
                    "restricted_ratio",
                    "SOURCE_VALUE_UNRESOLVED",
                    str(exc),
                    "旧组件金额之外还须逐原件重建受限明细；不得回退为未核股数或把分项分数当原始指标",
                    report,
                    scope[0],
                )
        else:
            missing(
                "restricted_ratio",
                "SOURCE_VALUE_UNRESOLVED",
                "LINK_AND_TARGET_FULL_RESTRICTIONS_REQUIRED"
                if is_link
                else "FULL_ASSET_RESTRICTION_SCOPE_NOT_EXPLICIT",
                "全部资产受限声明与金额；联接还需自身及目标ETF可赎回/受限情况分别核验",
                report,
                scope[0] if scope else normalized[-1500:],
            )
    elif period and report.get("text"):
        try:
            from app.services.fund_quality_v2_restrictions import restricted_securities
            from app.services.fund_rating_public_stock_tables import parse

            if "联接" in compact(report["text"][:700]):
                raise ValueError("LINK_AND_TARGET_FULL_RESTRICTIONS_REQUIRED")
            b = strict_report_balance()
            positions = (
                parse(report_pages, b["stock_amount"], b["net_assets"], require_industry=False)["holdings"]
                if decimal(b["stock_amount"]) > 0
                else []
            )
            restricted = restricted_securities(report_pages, positions, b["bond_amount"])
            facts["restricted_original_rebuild"] = {
                "source_hashes": [report["ref"]["sha256"]],
                "parser_header_view": restricted.get("parser_header_view"),
                "stock_layout_evidence": restricted.get("stock_layout_evidence"),
                "empty_bond_table_evidence": restricted.get("empty_bond_table_evidence"),
                "parsed_tables_amount_cny": restricted["restricted_amount"],
                "parsed_row_count": len(restricted["rows"]),
                "subtotal_scope": "PARSED_DISCLOSURE_TABLES_NOT_PROOF_OF_ALL_ASSET_SCOPE",
                "pledged_bond_evidence": restricted.get("pledged_bond_evidence"),
            }
            normalized = compact(report["text"])
            scope = full_asset_restriction_scope(normalized)
            if not scope:
                raise ValueError("FULL_ASSET_RESTRICTION_SCOPE_NOT_EXPLICIT")
            ratio = decimal(restricted["restricted_amount"]) / decimal(b["net_assets"])
            if not 0 <= ratio <= 1:
                raise ValueError("RESTRICTED_ASSET_RATIO_RANGE")
            record(
                "restricted_ratio",
                ratio,
                period[1],
                report,
                scope[0],
                {
                    "restricted_amount": restricted["restricted_amount"],
                    "net_assets": b["net_assets"],
                    "restricted_rows": restricted["rows"],
                },
            )
        except (ValueError, KeyError, TypeError) as exc:
            if partial_pledge := getattr(exc, "evidence", None):
                facts["restricted_partial_pledge_evidence"] = {
                    "source_hashes": [report["ref"]["sha256"]],
                    "evidence": partial_pledge,
                }
            missing(
                "restricted_ratio",
                "SOURCE_VALUE_UNRESOLVED",
                str(exc),
                "报告完整受限表、全部资产范围声明和资产负债表逐元对账；不从旧分项或风险档换算",
            )
    elif "restricted_ratio" not in findings:
        missing(
            "restricted_ratio",
            "COLLECTION_REQUIRED",
            "PRIMARY_REPORT_REQUIRED_FOR_RESTRICTIONS",
            "正式完整报告、受限表、资产负债表及全资产受限范围",
        )
    if period and report.get("pages"):
        # 合计现金义务、运作预警都是独立原件事实，不等于纯现金理想目标或
        # 策略有效容量。仅进入parsed_facts；不补RAW4缺项、不改变权重或评级。
        from app.services.fund_quality_v2_observable_liquidity import observable_liquidity_facts

        facts.update(observable_liquidity_facts(
            report["pages"], observed_on=str(period[1]), source_ref=report["ref"],
            restricted_metric=metrics.get("restricted_ratio"),
        ))
    return {
        "metrics": metrics,
        "findings": findings,
        "facts": facts,
        "document_errors": {k: d["error"] for k, d in documents.items() if "error" in d},
    }
