"""从已核验公告正文摘取阅读要点；不联网、不调用模型、不推断涨跌或事件完成。

摘要沿用 news-facts.summary 字符串契约，每行一个完整要点。数字、条件和否定词
随正文一起保留；原句与物理页码仅存后台证据，不把标题或套话充当正文摘要。
"""

import re
from collections import Counter

RULE = "ANNOUNCEMENT_READING_POINTS_V1"
UNAVAILABLE = "暂未提取到可靠要点。"
MAX_PAGES = 200
MAX_CHARACTERS = 300_000


def _reading_text(quote, label):
    """把已匹配的完整原句压成阅读用语；只改固定句式，数值从同一句的捕获组原样带出。"""
    if label == "拟审议：":
        subject = quote.removeprefix("关于").removesuffix("的议案")
        return label + subject.replace("<", "《").replace(">", "》") + "。"
    fee = re.fullmatch(r"公司(\d{4})年度财务报告和内控审计费用为([\d,.]+万元)，[^。]*?"
                       r"(\d{4})年度审计费用[^。]*?预计增长不超过([\d.]+%)", quote)
    if fee:
        return f"审计费用：{fee[1]}年{fee[2]}；{fee[3]}年预计增长不超过{fee[4]}。"
    if quote.startswith("本次续聘") and "并自公司股东会审议通过之日起生效" in quote:
        return "生效条件：仍需股东会审议，通过后生效。"
    dividend = re.search(r"未来三年以现金方式累计分配的利润不应少于该三年实现的年均可分配利润的([^，。]{2,15})", quote)
    if dividend and quote.startswith("在符合") and "提交股东会审议决定" in quote:
        return (f"分红下限：符合现金分红条件时，三年累计现金分红不低于三年年均可分配利润的{dividend[1]}；"
                "具体比例仍需股东会审议。")
    credit = re.search(r"新增[^。]*?不超过人民币([\d,.]+亿元)[^。]*?将由折合不超过人民币([\d,.]+亿元)"
                       r"调整至折合不超过人民币([\d,.]+亿元)", quote)
    if credit and quote.startswith("同意公司及子公司") and "综合授信额度" in quote:
        return f"综合授信：拟新增不超过人民币{credit[1]}，总额度上限由{credit[2]}增至{credit[3]}。"
    guarantee = re.search(r"公司拟为([^、，]{2,20})、([^，]{2,20})分别新增担保额度折合人民币"
                          r"([\d,.]+亿元)、([\d,.]+亿元)[^。]*?将由折合不超过人民币([\d,.]+亿元)"
                          r"调整至折合不超过人民币([\d,.]+亿元)", quote)
    if guarantee:
        return (f"担保额度：拟为{guarantee[1]}新增{guarantee[3]}、{guarantee[2]}新增{guarantee[4]}；"
                f"总额度上限由人民币{guarantee[5]}增至{guarantee[6]}。")
    repurchase = re.search(r"^截至(\d{4}年\d{1,2}月\d{1,2}日)，公司本次回购[^。]*?已经实施完成[^。]*?"
                           r"累计回购公司股份([\d,]+股)，占公司总股本的([\d.]+%)[^。]*?"
                           r"支付的总金额为([\d,.]+万元)（不含交易费用）", quote)
    if repurchase:
        return (f"回购已完成（截至{repurchase[1]}）：累计回购{repurchase[2]}，占总股本{repurchase[3]}，"
                f"支付{repurchase[4]}（不含交易费用）。")
    return label + quote + "。"


def _compact(value):
    return re.sub(r"\s+", "", value)


def _body(pages):
    """剔除重复页眉和页边页码，再连接跨页文字；不删除表格中的独立数字。"""
    lines = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]
    headers = Counter(row[0] for row in lines if row and len(row[0]) > 12)
    result = []
    for row in lines:
        row = list(row)
        if row and headers[row[0]] > 1:
            row.pop(0)
        while row and re.fullmatch(r"(?:第)?\d+(?:页)?", row[0]):
            row.pop(0)
        while row and re.fullmatch(r"(?:第)?\d+(?:页)?", row[-1]):
            row.pop()
        result.append(_compact("".join(row)))
    text = "".join(result)
    # 部分 PDF 将保证声明排在页脚，恰好打断金额单位或句子，先删除这类固定声明再接页。
    return re.sub(r"本公司(?:及董事会|董事会)[^。]{0,180}(?:法律责任|重大遗漏)[。]", "", text)


def extract_key_points(title, pages):
    """仅在完整可读正文中选择三至五条要点，未命中时返回空列表。

    常见公告按明确标签取数；其他公告摘录完整句子，避免截断金额、分母或
    生效条件。标题只用于选择阅读规则，输出的实质内容必须能在正文中定位。
    """
    if not pages or len(pages) > MAX_PAGES or any(not isinstance(p, str) for p in pages):
        return []
    if sum(map(len, pages)) > MAX_CHARACTERS:
        return []
    text = _body(pages)
    points = []

    def add(quote, label=""):
        quote = quote.strip("。；;，, ")
        if not 8 <= len(quote) <= 280 or any(quote in p["quote"] or p["quote"] in quote for p in points):
            return
        # 跨页句子使用清理后的连续正文定位；页码使用原始页的片段定位，保留所有涉及页。
        if quote not in text:
            return
        page_numbers = [i + 1 for i, page in enumerate(pages) if quote[:16] in _compact(page)]
        if not page_numbers:
            return
        start = page_numbers[0] - 1
        evidence_pages = [start + 1]
        if quote not in _compact(pages[start]):
            end = next((i for i in range(start, min(start + 4, len(pages)))
                        if quote in _body(pages[start:i + 1])), None)
            if end is None:
                return
            evidence_pages = list(range(start + 1, end + 2))
        points.append({"text": _reading_text(quote, label), "quote": quote, "pages": evidence_pages})

    def match(pattern, label="", group=0):
        found = re.search(pattern, text)
        if found:
            add(found.group(group), label)

    date = r"\d{4}年\d{1,2}月\d{1,2}日"
    if re.search(r"召开.*股东.*通知", title):
        match(r"现场会议(?:召开)?(?:日期和)?时间[：:]?" + date
              + r"(?:[（(][^）)]{1,8}[）)])?(?:[上下]午)?\d{1,2}[：:]\d{2}")
        match(r"(?:本次会议的)?股权登记日[：:]?" + date)
        # 保留正文明确列出的议题，不从公告标题猜测本次会议将审议什么。
        agenda_section = re.search(r"(?:二、|二[、.]?)\s*(?:会议审议|本次股东会审议)[\s\S]+?(?:三、|三[、.])", text)
        agenda_text = agenda_section.group() if agenda_section else ""
        agendas = list(dict.fromkeys(re.findall(r"《(关于[^《》。]{4,100}议案)》", agenda_text)))
        for agenda in agendas[:3]:
            add(agenda, "拟审议：")
        if len(points) >= 2:
            return points[:5]

    if "续聘" in title and "会计师" in title:
        match(r"(?:同意|拟)续聘[^。；]{5,130}审计机构[^。；]{0,90}[。；]")
        match(r"公司\d{4}年度[^。；]{0,35}审计费用[^。；]{1,140}[。；]")
        match(r"本次续聘[^。；]{5,160}生效[。；]")
        if points:
            return points

    if "分红回报规划" in title:
        match(r"在符合[^。；]{5,180}至少实施[^。；]{0,30}中期分红[。；]")
        match(r"在符合[^。；]{5,240}年均可分配利润[^。；]{0,100}[。；]")
        match(r"公司实施现金分红的具体条件为[：:][\s\S]{1,220}?（3）[^。；]{1,90}[。；]")
        match(r"(?:本规划|本股东回报规划)[^。；]{0,100}(?:生效|实施)[。；]")
        if points:
            return points[:4]

    if "新增综合授信" in title:
        match(r"同意公司及子公司新增[^。；]{10,240}[。；]")
        match(r"同意在有效期限内，公司拟[^。；]{10,240}[。；]")
        match(r"上述事项[^。；]{5,130}生效[。；]")
        match(r"上述综合授信额度并非[^，。；]{5,60}")
        if points:
            return points[:4]

    if "现金管理" in title:
        match(r"公司拟使用[^。；]{10,150}进行现金管理[。；]")
        match(r"该额度自[^。；]{5,150}[。；]")
        match(r"投资种类[：:][^。；]{10,150}[。；]")
        match(r"该事项[^。；]{5,90}(?:无需|尚需)[^。；]{0,40}[。；]")
        if points:
            return points[:4]

    if "权益分派实施" in title:
        match(r"根据公司审议通过[^。；]{10,220}合计派发[^。；]{0,60}[。；]")
        if not points:
            match(r"本次利润分配以[^。；]{10,180}[。；]")
        match(r"(?:本次A股权益分派)?股权登记日为[：:]?" + date + r"[；;]除权除息日为[：:]?" + date)
        # 只接受完整、顺序明确的三列表头和三个日期，不推测错位表格。
        table = re.search(r"股权登记日除权（息）日现金红利发放日(\d{4}/\d{1,2}/\d{1,2})"
                          r"(\d{4}/\d{1,2}/\d{1,2})(\d{4}/\d{1,2}/\d{1,2})", text)
        if table:
            add(table.group())
            if points and points[-1]["quote"] == table.group():
                points[-1]["text"] = (f"股权登记日：{table[1]}；除权除息日：{table[2]}；"
                                      f"现金红利发放日：{table[3]}。")
        if points:
            return points[:3]

    if "回购结果" in title:
        match(r"截至" + date + r"，公司本次回购[^。]{5,260}支付的总金额[^。]{5,80}[。]")
        match(r"公司如未能[^。]{5,160}注销[。]")
        if points:
            return points[:3]

    if "被动稀释" in title:
        match(r"本次发行上市完成后，公司总股本[^。]{10,220}[。]")
        match(r"本次权益变动性质[^。]{10,160}[。]")
        if points:
            return points[:3]

    if "挂牌并上市交易" in title:
        match(r"经香港联交所批准[^。]{10,180}[。]")
        match(r"根据每股H股发售价[^。]{10,180}[。]")
        if points:
            return points[:3]

    if "激励计划" in title and "条件成就" in title:
        match(r"本次可行权的股票期权数量[^。]{5,90}[。]")
        match(r"本次股票期权行权[^。]{5,90}[。]")
        match(r"本次行权事宜需[^。]{5,130}[。]")
        return points

    if "激励计划" in title and "草案" in title and "法律意见" not in title:
        match(r"本激励计划拟向激励对象授予[^。]{5,180}[。]")
        if not points:
            match(r"本激励计划拟授予[^。]{5,200}[。]")
        match(r"本激励计划[^。]{0,55}授予价格为[^。]{5,120}[。]")
        match(r"本激励计划拟授予的激励对象[^。]{5,200}[。]")
        match(r"本激励计划(?:须|需|尚需)[^。]{5,130}[。]")
        return points[:4]

    if "会议决议" in title or "股东会决议" in title:
        for decision in re.finditer(r"审议(?:并一致|并|一致)?通过(?:了)?[：:]?《[^。]{4,160}?议案》", text):
            add(decision.group())
            if len(points) == 4:
                break
        if points:
            return points

    # 其余只取业务含义明确的正文句式。未覆盖的表格、自查清单及长篇重组报告保持暂缺，
    # 不按“含数字越多越重要”排序，以免把历史计划、机构简介和表格散数当成本次进展。
    if "置换" in title and "募集资金" in title:
        match(r"同意公司使用募集资金置换[^。]{10,180}共计[^。]{5,45}[。]")
        match(r"截至" + date + r"，公司以自筹资金预先投入[^。]{10,180}[。]")
        match(r"截至" + date + r"，公司以自筹资金预先支付[^。]{10,180}[。]")
    elif "增资" in title and "募集资金" in title:
        match(r"同意公司使用部分募集资金[^。]{10,220}[。]")
        match(r"本次增资优先完成[^。]{10,180}[。]")
        match(r"增资完成后[^。]{10,180}[。]")
        match(r"本次增资尚未[^。]{5,110}[。]")
    elif "土地使用权" in title and "进展" in title:
        match(r"近日，公司[^。]{10,240}竞得[^。]{10,100}[。]")
        match(r"本次竞得土地使用权后[^。]{10,220}[。]")
        match(r"本次项目实施尚需[^。]{10,220}[。]")
    elif "共同投资" in title:
        match(r"(?:合伙企业|[^。；]{2,12}基金)(?:目标)?认缴出资总额[^。；]{10,220}[。；]")
        match(r"合伙企业专项投资于[^。]{10,120}[。]")
        match(r"本次投资事项目前尚[^。]{10,180}[。]")
    elif "转股价格" in title:
        match(r"调整前转股价格[：:][\d.]+元/股")
        match(r"调整后转股价格[：:][\d.]+元/股")
        match(r"转股价格调整生效日期[：:]" + date + r"(?:（[^）]{1,30}）)?")
    elif "发行价格" in title:
        match(r"公司已确定本次[^。]{10,220}[。]")
    elif "备案" in title and "发行" in title:
        match(r"公司拟发行不超过[^。]{10,180}[。]")
        match(r"公司本次境外发行上市尚需[^。]{10,220}[。]")
    elif "完成工商变更" in title:
        match(r"近日，公司[^。：]{5,150}《营业执照》")
    return points[:4]


def with_reading_summary(item):
    """仅为已核验、当前可读的事实生成摘要；保留原事件身份、阶段和训练资格。"""
    if not item.get("business_eligible") or item.get("stage") != "已披露文件":
        return {**item, "summary": "此前披露的事项本次尚未核对完整，暂不展示要点。",
                "summary_rule": RULE, "summary_evidence": []}
    evidence = item.get("evidence", {})
    document = evidence.get("document", {})
    pages = document.get("pages", evidence.get("pages", []))
    if document and document.get("text_status") != "TEXT_EXTRACTED":
        pages = []
    points = extract_key_points(item["title"], pages)
    return {**item, "summary": "\n".join(point["text"] for point in points) or UNAVAILABLE,
            "summary_rule": RULE, "summary_evidence": points}
