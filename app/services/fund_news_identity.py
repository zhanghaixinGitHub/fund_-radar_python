"""公告证券列表未覆盖本份额时的人工身份复核，绑定目录、正文及管理人原件。

转载接口的证券列表不是完整适用名单。这里不自动扩大范围；本机审阅须给出
连续原文及明确的适用层级，任何正文、基金身份或来源变化均重新核验。
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import text

from app.db.session import get_engine
from app.integrations.market_fund_reports import read_source
from app.integrations.public_fund_reports import BODY_URL, CATALOG_URL
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save

SCOPES = {
    "MANAGER_SERVICE",
    "MANAGER_GOVERNANCE",
    "EXPLICIT_FUND_CODE",
    "EXPLICIT_PRODUCT_NAME",
    "MANAGER_CONDITIONAL_VALUATION",
    "MANAGER_CONDITIONAL_CONVERSION",
}


def _compact(value):
    # 工商名称的全角括号和数据库半角括号只属排版差异；不删除名称文字、
    # 不折叠品牌简称，避免把另一家管理人也认作相同法律主体。
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _profile(code):
    """管理人以当前库内本份额档案为准，不能从待核验公告反推。"""
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text("""SELECT management_company_name, content_hash FROM fund_profile
                    WHERE fund_code=:code ORDER BY updated_at DESC,fund_profile_id LIMIT 1"""),
                {"code": code},
            )
            .mappings()
            .first()
        )
    if not row or not row["management_company_name"]:
        raise ValueError("MARKET_NEWS_MANAGER_PROFILE_REQUIRED")
    return dict(row)


def _entry(entry):
    return {key: entry[key] for key in ("id", "title", "published")}


def _catalog(root, code, entry, source):
    """目录必须是该份额的真实归档响应，标题、编号及公开日期逐项一致。"""
    if source.get("url") != CATALOG_URL or source.get("params", {}).get("fundcode") != code:
        raise ValueError("MARKET_NEWS_IDENTITY_CATALOG")
    value = json.loads(read_source(root, source))
    rows = value.get("Data")
    if value.get("ErrCode") != 0 or not isinstance(rows, list):
        raise ValueError("MARKET_NEWS_IDENTITY_CATALOG")
    matches = [row for row in rows if row.get("ID") == entry["id"]]
    if len(matches) != 1 or (
        matches[0].get("FUNDCODE") != code
        or matches[0].get("TITLE") != entry["title"]
        or str(matches[0].get("PUBLISHDATEDesc", ""))[:10] != entry["published"]
    ):
        raise ValueError("MARKET_NEWS_IDENTITY_CATALOG")


def _product_table_matches(code, product, proof, document, body, title, manager):
    """有限适配改聘审计机构名单的跨行表格，不从主代码猜测其他份额身份。

    公告的基金名称可能被同一行的简称、主代码隔开。仅在本基金原报告同时证明
    完整名称、简称、主代码及两类份额代码时，接受原文中严格对应的一行；不改写
    正文，不把另一行的代码或其他产品拼入，也不推广为全部旗下基金适用。
    """
    table = (proof or {}).get("product_table") or {}
    parts = table.get("name_parts", [])
    master, short = table.get("master_code", ""), table.get("short_name", "")
    row = table.get("row_quote", "")
    master_quote, share_quote = table.get("master_quote", ""), table.get("share_quote", "")
    proof_body = (document or {}).get("body", "")
    quoted = (proof or {}).get("quote", "")
    if (
        (document or {}).get("kind") != "REPORT"
        or not re.fullmatch(r"\d{6}", master)
        or master == code
        or not isinstance(parts, list)
        or len(parts) != 2
        or any(not isinstance(p, str) or len(p.strip()) < 2 for p in parts)
        or _compact("".join(parts)) != _compact(product)
        or len(short) < 4
        or not row
        or row not in body
        or _compact(row) != _compact(parts[0]) + _compact(short) + master + _compact(parts[1])
        or "基金名称基金简称基金主代码" not in _compact(body)
        or _compact(title) != manager + "关于旗下基金改聘会计师事务所公告"
        or not master_quote
        or master_quote not in proof_body
        or master_quote not in quoted
        or _compact(master_quote) != "基金主代码" + master
        or not share_quote
        or share_quote not in proof_body
        or share_quote not in quoted
        or _compact(share_quote)
        not in {
            "下属分级基金的交易代码" + master + code,
            "下属分级基金的交" + master + code + "易代码",
        }
        or "基金简称" + _compact(short) not in _compact(quoted)
    ):
        return False
    return True


def _report_conditional_valuation(code, product, proof, document, body, title, manager):
    """本份额报告可证明条件式估值通知的管理人背景，不证明当前持股或调整金额。

    只接两种已实际核对的管理人全文格式；独立报告须同时证明法律产品、
    六位份额代码和管理人。直接ETF不适用，ETF联接仍是独立基金产品。
    复牌恢复条件及指数收益法都必须存在，不能将普通估值标题作为通行证。
    """
    text = _compact(body)
    quoted = _compact((proof or {}).get("quote", ""))
    return bool(
        document
        and document.get("kind") == "REPORT"
        and len(product) >= 12
        and not (re.search(r"ETF|交易型开放式", product, re.I) and "联接" not in product)
        and _compact(product) in quoted
        # 代码边界必须在原排版里核验；压掉空白会把同一行A/C两列拼成12位。
        and re.search(r"(?<!\d)" + code + r"(?!\d)", (proof or {}).get("quote", ""))
        and manager in _compact(document["body"])
        and re.fullmatch(
            re.escape(manager)
            + r"关于旗下基金(?:调整长期停牌股票估值方法的提示性公告|持有停牌股票估值调整的公告)",
            _compact(title),
        )
        and any(term in text for term in ("对本公司旗下基金所持有的", "对本公司旗下基金(ETF除外)所持有的"))
        and all(term in text for term in ("基金托管人协商一致", "指数收益法", "当日收盘价"))
        and any(term in text for term in ("复牌", "待该股票的交易体现活跃市场交易特征后"))
        and re.search(r"(?:股票|证券)代码[:：]\d{6}", text)
    )


def _trade_state_table_matches(code, product, proof, document, body, title, manager):
    """有限核验交易状态表同一产品的A/C跨行代码，只证明公告适用身份。

    本基金原报告须同时证明全名、简称、主代码和两个份额代码。转载文本中的
    名称列及代码列分别重建，数字只能来自同一行块的代码列；邻行名字、状态
    栏数字或另一基金代码一律不能借用。不会据此推断当前开放状态或投资方向。
    """
    table = (proof or {}).get("trade_state_table") or {}
    own = (document or {}).get("body", "")
    quoted = (proof or {}).get("quote", "")
    short, master = table.get("short_name", ""), table.get("master_code", "")
    row = table.get("row_quote", "")
    master_quote, share_quote = table.get("master_quote", ""), table.get("share_quote", "")
    if (
        (document or {}).get("kind") != "REPORT"
        or len(product) < 12
        or _compact(product) not in _compact(quoted)
        or not re.fullmatch(r"\d{6}", master)
        or master == code
        or len(short) < 6
        or "基金简称" + _compact(short) not in _compact(quoted)
        or not master_quote
        or master_quote not in own
        or master_quote not in quoted
        or _compact(master_quote) != "基金主代码" + master
        or not share_quote
        or share_quote not in own
        or share_quote not in quoted
        or _compact(share_quote)
        not in {
            "下属分级基金的交易代码" + master + code,
            "下属分级基金的交" + master + code + "易代码",
        }
        or not (
            _compact(title) == manager + "旗下基金交易状态表"
            or (_compact(title) == "旗下基金交易状态表" and manager in _compact(own))
        )
        or not all(term in _compact(body) for term in ("基金简称", "开放申购", "开放赎回", "交易状态表"))
        or not row
        or len(row) > 1500
        or body.count(row) != 1
    ):
        return False
    lines = row.splitlines()
    first = next((line for line in lines if line.strip()), "")
    anchor = re.match(r"^( *)A\s*类[:：]", first)
    if not anchor:
        return False
    column = len(anchor[1])
    if not 8 <= column <= 30:
        return False
    names, classes, pieces = [], [], {"A": [], "C": []}
    current = None

    def column_text(line, begin, end):
        # 转载表格保留的是等宽显示列：汉字占两列，不能用Python字符下标把
        # 同行状态栏或另一列数字误切进代码栏。仍只从原行取字，不重排原文。
        position, selected = 0, []
        for char in line:
            width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
            if begin <= position and position + width <= end:
                selected.append(char)
            position += width
        return "".join(selected)

    for line in lines:
        if not line.strip():
            continue
        left = column_text(line, 0, column).strip()
        field = column_text(line, column, column + 7).strip()
        if left:
            names.append(left)
        label = re.fullmatch(r"([AC])\s*类[:：]", field)
        if label:
            current = label[1]
            classes.append(current)
        elif field:
            if current is None or not re.fullmatch(r"\d{1,6}", field):
                return False
            pieces[current].append(field)
    return (
        classes == ["A", "C"]
        and _compact("".join(names)) == _compact(short)
        and "".join(pieces["A"]) == master
        and "".join(pieces["C"]) == code
    )


def _allocation_product_table_matches(product, proof, document, body, title, manager):
    """只核验可转债关联获配表同一行的完整产品名，不把获配当当前持仓。"""
    row = ((proof or {}).get("allocation_table") or {}).get("row_quote", "")
    if (
        (document or {}).get("kind") != "REPORT"
        or _compact(title) != manager + "关于旗下基金投资关联方承销可转换公司债券的关联交易公告"
        or not all(term in _compact(body) for term in ("基金名称", "获配数量", "获配金额", "投资类型"))
        or not row
        or len(row) > 500
        or body.count(row) != 1
    ):
        return False
    lines = [line.strip() for line in row.splitlines() if line.strip()]
    if not 2 <= len(lines) <= 3:
        return False
    first = re.fullmatch(r"(.+?)[ \t]{2,}[\d,]+[ \t]{2,}[\d,]+(?:\.\d+)?[ \t]{2,}优先配售", lines[0])
    if not first or any(re.search(r"\d|优先配售|获配", line) for line in lines[1:]):
        return False
    return _compact(first[1] + "".join(lines[1:])) == _compact(product)


def _benchmark_product_table_matches(product, proof, document, body, title, manager):
    """核验基准修订公告同一表格行的产品名称，不解析或采用新基准数值。

    部分公告把名称分多行排在左列，右列基准说明夹在名称片段之间。人工须给出
    连续原行、显示列范围及该行序号；仅拼接同一左列，不能借右列、邻行或简称。
    外层仍要求本份额独立原报告证明完整产品名及本代码，并核验目录和原件摘要。
    """
    table = (proof or {}).get("benchmark_table") or {}
    row = table.get("row_quote", "")
    begin, end = table.get("column_start"), table.get("column_end")
    ordinal = table.get("ordinal", "")
    if (
        (document or {}).get("kind") != "REPORT"
        or re.fullmatch(
            re.escape(manager) + r"关于旗下部分基金调整业绩比较基准并修订基金合同(?:等法律文件)?的公告",
            _compact(title),
        ) is None
        or not all(term in _compact(body) for term in ("序号", "基金全称", "业绩比较基准"))
        or not row or len(row) > 1400 or body.count(row) != 1
        or type(begin) is not int or type(end) is not int
        or not 0 <= begin < end <= 40 or end - begin < 12
        or not isinstance(ordinal, str) or not re.fullmatch(r"[1-9]\d{0,2}", ordinal)
    ):
        return False
    names, ordinals = [], []
    for line in row.splitlines():
        match = re.match(r"^\s*(\d{1,3})(?:\s+|$)", line)
        if match:
            ordinals.append(match[1])
        position, selected = 0, []
        for char in line:
            width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
            if begin <= position and position + width <= end:
                selected.append(char)
            position += width
        part = "".join(selected).strip()
        if part:
            names.append(part)
    return ordinals == [ordinal] and 2 <= len(names) <= 5 and _compact("".join(names)) == _compact(product)


def _validate(root, code, entry, source, value):
    current = now()
    profile = _profile(code)
    raw = json.loads(read_source(root, source))
    data = raw.get("data") or {}
    body = data.get("notice_content", "")
    reviewed = datetime.fromisoformat(value["reviewed_at"])
    if (
        not re.fullmatch(r"[0-9]{6}", code)
        or source.get("url") != BODY_URL
        or source.get("params", {}).get("art_code") != entry["id"]
        or raw.get("success") != 1
        or data.get("art_code") != entry["id"]
        or not isinstance(body, str)
        or len(body.strip()) < 50
        or value.get("fund_code") != code
        or value.get("entry") != _entry(entry)
        or value.get("source_hash") != source["sha256"]
        or value.get("manager") != profile["management_company_name"]
        or value.get("profile_hash") != profile["content_hash"]
        or value.get("scope") not in SCOPES
        or reviewed.tzinfo is None
        or reviewed > current
        or entry["published"] > current.date().isoformat()
        or len(value.get("note", "").strip()) < 80
        or not 1 <= len(value.get("quotes", [])) <= 10
        or any(not isinstance(q, str) or len(q.strip()) < 15 or q not in body for q in value["quotes"])
    ):
        raise ValueError("MARKET_NEWS_IDENTITY_REVIEW_INVALID")
    for ref in (source, value["catalog_source"]):
        received = datetime.fromisoformat(ref["received_at"])
        if received.tzinfo is None or received > current or received > reviewed:
            raise ValueError("MARKET_NEWS_IDENTITY_SOURCE_TIME")
    _catalog(root, code, entry, value["catalog_source"])
    manager = _compact(profile["management_company_name"])
    proof = value.get("identity_proof")
    document = None
    if proof:
        material = read(Path(proof["materials_path"]))
        document = next((d for d in material["documents"] if d["id"] == proof["document_id"]), None)
        if (
            material.get("fund_code") != code
            or digest(material) != proof["materials_hash"]
            or not document
            or code not in document.get("codes", [])
            or digest(document["body"]) != document["body_hash"]
            or proof["quote"] not in document["body"]
            or len(proof["quote"].strip()) < 20
        ):
            raise ValueError("MARKET_NEWS_IDENTITY_PRODUCT_PROOF")
        available = datetime.fromisoformat(document["available_at"])
        if available.tzinfo is None or available > reviewed or document["published_date"] > str(current.date()):
            raise ValueError("MARKET_NEWS_IDENTITY_SOURCE_TIME")
    title_body = _compact(data.get("notice_title", "") + body)
    if manager not in title_body:
        anchor = value.get("manager_anchor", "")
        if (
            not document
            or len(anchor) < 5
            or _compact(anchor) not in _compact(body)
            or _compact(anchor) not in _compact(proof["quote"])
            or manager not in _compact(document["body"])
        ):
            raise ValueError("MARKET_NEWS_MANAGER_IDENTITY_UNPROVEN")
    if value["scope"] == "EXPLICIT_FUND_CODE" and not any(
        re.search(r"(?<!\d)" + code + r"(?!\d)", q) for q in value["quotes"]
    ):
        raise ValueError("MARKET_NEWS_EXPLICIT_FUND_REQUIRED")
    if value["scope"] == "EXPLICIT_PRODUCT_NAME":
        product = value.get("product_name", "")
        compact_product = _compact(product)
        # “南方原油证券投资基金”等真实完整法律名称只有十字；仍须独立原件
        # 同时证明全名和本份额代码。短名称仅接受完整法律后缀，不接受品牌简称。
        complete_name = len(compact_product) >= 12 or (
            len(compact_product) >= 10 and compact_product.endswith("证券投资基金")
        )
        if (
            not document
            or not complete_name
            or not re.search(r"(?<!\d)" + code + r"(?!\d)", document["body"])
            or _compact(product) not in _compact(proof["quote"])
            or not (
                any(_compact(product) in _compact(q) for q in value["quotes"])
                or _product_table_matches(code, product, proof, document, body, entry["title"], manager)
                or _trade_state_table_matches(code, product, proof, document, body, entry["title"], manager)
                or _allocation_product_table_matches(product, proof, document, body, entry["title"], manager)
                or _benchmark_product_table_matches(product, proof, document, body, entry["title"], manager)
            )
        ):
            raise ValueError("MARKET_NEWS_EXPLICIT_PRODUCT_REQUIRED")
    if value["scope"] == "MANAGER_CONDITIONAL_VALUATION":
        # 管理人的条件式估值政策可以是本基金目录中的真实背景，但不能变成
        # “本基金当前持有该股票/已按此调整估值”的事实。当前仅接已核实的
        # 国泰官网份额档案格式；另以本份额独立报告核两个已取全文格式，
        # 两条路径均只证明条件式背景，不确认本基金当前组合或具体估值。
        product = value.get("product_name", "")
        uri = urlparse(document.get("url", "")) if document else None
        report_proven = _report_conditional_valuation(
            code, product, proof, document, body, entry["title"], manager
        ) and any("指数收益法" in _compact(q) and "旗下基金" in _compact(q) for q in value["quotes"])
        if value.get("does_not_confirm_current_portfolio_application") is not True or not report_proven and (
            not document
            or document.get("kind") != "FUND_PROFILE"
            or uri.scheme != "https"
            or uri.hostname != "e.gtfund.com"
            or uri.path != "/etrade/Jijin/view/id/" + code
            or len(product) < 12
            or re.search(r"ETF|交易型开放式", product, re.I)
            or _compact(product) not in _compact(proof["quote"])
            or not re.search(r"(?<!\d)" + code + r"(?!\d)", document["body"])
            or manager not in _compact(document["body"])
            or not re.fullmatch(re.escape(manager) + r"调整长期停牌股票估值方法的公告", _compact(entry["title"]))
            or "本公司旗下基金(ETF基金除外)持有的" not in _compact(body)
            or not re.search(r"证券代码[:：]\d{6}", _compact(body))
            or not any("本公司旗下基金(ETF基金除外)持有的" in _compact(q) for q in value["quotes"])
            or value.get("does_not_confirm_current_portfolio_application") is not True
        ):
            raise ValueError("MARKET_NEWS_CONDITIONAL_VALUATION_UNPROVEN")
    if value["scope"] == "MANAGER_CONDITIONAL_CONVERSION":
        # “除所列基金及ETF外、处于正常开放期”的转换公告只是条件式业务背景。
        # 独立报告须证明本份额及管理人；并不由目录归类推断当前开放状态、收费或可转换对象。
        product = value.get("product_name", "")
        compact_body = _compact(body)
        if (
            not document
            or document.get("kind") != "REPORT"
            or len(product) < 12
            or _compact(product) not in _compact(proof["quote"])
            or code not in _compact(proof["quote"])
            or manager not in _compact(proof["quote"])
            or _compact(entry["title"]) != manager + "关于旗下部分基金开通基金转换业务的公告"
            or not all(
                term in compact_body
                for term in (
                    "不开通基金转换业务的基金",
                    "不开通同一基金不同类别基金份额之间转换业务的基金",
                    "除上述(1)、(2)项所列基金及ETF外",
                    "处于正常开放期的其他基金之间开通基金转换业务",
                )
            )
            or value.get("does_not_confirm_current_portfolio_application") is not True
        ):
            raise ValueError("MARKET_NEWS_CONDITIONAL_CONVERSION_UNPROVEN")
    if value["scope"] in {"MANAGER_SERVICE", "MANAGER_GOVERNANCE"}:
        # 仅服务和机构治理类别允许管理人层级；部分基金清单、投资及估值事项不能借此放行。
        title = _compact(entry["title"])
        terms = (
            # 反诈声明、客服时段和官网升级属于管理人服务背景，也须逐篇人工复核。
            # 这里只补足类别词，不代替上方的管理人、目录、原件及引用校验；
            # 估值、部分基金交易、经理等经济事项仍不能借服务类型扩大适用范围。
            # “系统停机维护”与系统升级同属渠道服务，仍须匹配本管理人和逐篇原文；
            # 不因此放开下方对部分基金、估值、经理及关联交易的限制。
            r"系统(?:升级|停机)?维护|服务器维护|服务提示|直销业务迁移|基金销售业务|第三方互联网平台|办公地址变更"
            r"|系统升级暂停网上交易系统服务"
            r"|谨防金融诈骗|防范金融诈骗|客服.*人工服务|官网.*(?:升级|维护)"
            r"|直销平台.*(?:完善|更新).*身份资料"
            r"|提醒客户进一步完善身份信息"
            if value["scope"] == "MANAGER_SERVICE"
            else r"设立.*分公司|高级管理人员|高管|参与上市公司治理.*(?:指引|规则)"
        )
        # 管理人董事会变动属于机构治理，并不等于本基金经理变更。仅接纳法律主体
        # 完全匹配的独立董事变更公告，且正文同时说明12个月累计变更、董事会成员
        # 和监管备案；仍需上方的本份额目录、原件摘要及逐篇人工阅读凭据。
        # 不把普通“董事”关键词扩展为部分基金、投资证券或审计机构公告的通行证。
        board_change = (
            value["scope"] == "MANAGER_GOVERNANCE"
            and re.fullmatch(re.escape(manager) + r"关于公司董事变更的公告", title) is not None
            and all(
                term in _compact(body)
                for term in ("最近12个月内董事变更超过百分之五十", "公司董事会成员", "向监管机构备案")
            )
        )
        # 部分反诈原文以“警惕假冒管理人开展非法证券活动”为标题，不能因转载
        # 证券列表未列出本份额就丢失已取得公告。仅接法律主体逐字一致的该类标题，
        # 并要求官方渠道、非正规信息与拒绝保本承诺等正文共同成立；不扩展到基金
        # 经理、合同、费率或估值变化。目录、原文摘要和人工逐字引用仍先通过上方校验。
        impersonation_warning = (
            value["scope"] == "MANAGER_SERVICE"
            and re.fullmatch(
                r"关于警惕不法分子假冒" + re.escape(manager) + r"进行非法证券活动的重要提示", title
            ) is not None
            and all(term in _compact(body) for term in ("官方信息", "官方渠道", "谨防诈骗", "不会承诺"))
            and not re.search(r"基金经理|基金合同|调整.{0,8}费率|本基金.{0,8}估值", _compact(body))
        )
        # 已核全文的管理人隐私文件更新仅属客户服务背景。精确标题、文件清单
        # 和生效表述必须同时成立；不替客户同意条款，也不放行基金经济事项。
        privacy_update = (
            value["scope"] == "MANAGER_SERVICE"
            and title == manager + "隐私政策更新提示"
            and all(
                term in _compact(body)
                for term in ("客户隐私政策", "第三方信息共享清单", "个人信息收集清单", "自发布之日起生效")
            )
            and not re.search(r"基金经理|基金合同|调整.{0,8}费率|本基金.{0,8}估值", _compact(body))
        )
        if (
            re.search(r"旗下部分基金|关联交易|估值方法|基金经理|获配|承销证券", title)
            or not re.search(terms, title + _compact(body[:400]))
            and not board_change
            and not impersonation_warning
            and not privacy_update
        ):
            raise ValueError("MARKET_NEWS_MANAGER_SCOPE_UNPROVEN")
    return value


def record_applicability_review(
    code,
    entry,
    source,
    catalog_source,
    scope,
    note,
    quotes,
    *,
    root,
    identity_proof=None,
    manager_anchor=None,
    product_name=None,
):
    """本机人工复核入口；管理人服务/治理仅作背景，不等于本基金经济事项。"""
    root = Path(root)
    profile = _profile(code)
    value = {
        "fund_code": code,
        "entry": _entry(entry),
        "source_hash": source["sha256"],
        "catalog_source": catalog_source,
        "manager": profile["management_company_name"],
        "profile_hash": profile["content_hash"],
        "scope": scope,
        "note": note,
        "quotes": quotes,
        "identity_proof": identity_proof,
        "manager_anchor": manager_anchor,
        "product_name": product_name,
        "reviewed_at": now().isoformat(),
        "does_not_determine_direction": True,
    }
    if scope in {"MANAGER_CONDITIONAL_VALUATION", "MANAGER_CONDITIONAL_CONVERSION"}:
        value["does_not_confirm_current_portfolio_application"] = True
    _validate(root, code, entry, source, value)
    versioned_save(root / "identity-reviews" / code / (source["sha256"] + ".json"), value)
    return value


def checked_applicability(root, code, entry, source):
    """采集器只消费核验通过且仍匹配本正文的复核，缺回执仍按原身份规则拒绝。"""
    path = Path(root) / "identity-reviews" / code / (source["sha256"] + ".json")
    if not path.exists():
        return None
    value = _validate(Path(root), code, entry, source, read(path))
    if entry.get("catalog_source"):
        _catalog(Path(root), code, entry, entry["catalog_source"])
    return value
