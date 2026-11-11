"""从同期资料与确权原表核退市证券身份；不按名称相似或旧行业猜测。"""

import re

from bs4 import BeautifulSoup

from app.services.fund_rating_market_review import normalized_text


def verified_securities(source, inputs: dict) -> dict:
    """返回可用于单元格和发行人归并的身份，行业直接取本期公司明示两位码。

    当前仅适配两网及退市板块：本期公司半年报须给代码、简称、全名与
    管理型细行业；券商原确权表须在同一行列出原上市代码和确权代码。
    原上市代码仍从巨潮原登记返回实际orgId，不制造或改写证券登记行。
    """
    result = {}
    if not inputs.get("security_disclosures"):
        return result
    registry = {r["code"]: r["orgId"] for r in source.json(inputs["issuer_registry"])["stockList"]}
    for code, descriptor in inputs.get("security_disclosures", {}).items():
        if not re.fullmatch(r"400\d{3}", code):
            raise ValueError("R5_SECURITY_DELISTED_SCOPE")
        report, transfer = descriptor["report"], descriptor["transfer"]
        for digest in (report, transfer):
            ref = source.sources[digest]
            if (
                ref.get("archive_kind") == "DERIVED"
                or not ref.get("published_on")
                or ref["published_on"] > inputs["as_of_date"]
            ):
                raise ValueError("R5_SECURITY_ORIGINAL_OR_CUTOFF")
        text = normalized_text("\n".join(source.pages(report)))
        period = inputs["holding_date"]
        year, month, day = map(int, period.split("-"))
        if month != 6 or day != 30 or f"报告期指{year}年1月1日-{year}年6月30日" not in text:
            raise ValueError("R5_SECURITY_SAME_PERIOD_REQUIRED")
        header = text[:9000]
        legal = re.search(r"公司中文全称(.{2,100}?)英文名称", header)
        identity = re.search(r"证券简称(.{1,25}?)证券代码(\d{6})", header)
        industry = re.search(r"行业\(挂牌公司管理型行业分类\)(.{1,180}?)主要产品", header)
        if not legal or not identity or identity[2] != code or not industry or "两网公司及退市公司板块" not in header:
            raise ValueError("R5_SECURITY_REPORT_IDENTITY_OR_TAXONOMY")
        fine = re.findall(r"([\u4e00-\u9fff]+)\(([A-S]\d{2})\)", industry[1])
        if len(fine) != 1:
            raise ValueError("R5_SECURITY_FINE_INDUSTRY_REQUIRED")
        # 原文表格的同一行绑定旧新代码及法律全名；分散检索命中不能证明关系。
        ref = source.sources[transfer]
        html = source.load(transfer).decode(ref.get("encoding", "utf-8"))
        rows = [
            [normalized_text(cell.get_text()) for cell in row.find_all(["td", "th"])]
            for row in BeautifulSoup(html, "html.parser").find_all("tr")
        ]
        old = descriptor["original_code"]
        matched = [row for row in rows if legal[1] in row and old in row and code in row]
        if len(matched) != 1 or old not in registry or not registry[old]:
            raise ValueError("R5_SECURITY_TRANSFER_IDENTITY_REQUIRED")
        result[code] = {
            "name": identity[1],
            "market": ".NQ",
            "legal_name": legal[1],
            "issuer": registry[old],
            "original_code": old,
            "industry_code": fine[0][1],
            "industry_name": fine[0][0],
            "source_refs": [report, transfer],
            "identity_row": matched[0],
            "industry_excerpt": industry[0],
        }
    return result
