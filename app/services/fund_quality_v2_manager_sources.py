"""重读基金经理公开任职API原件，不接纳简历、私有表单或来源自填核验标志。"""

import re
from datetime import date, datetime


def manager_api_records(original: dict, *, fund_code: str, ts_code: str) -> list[dict]:
    """逐行验证份额身份、列宽、公开姓名和生效日期，返回可重建团队的任职事实。

    取得这份响应不等于职责已核清。当前人员仍须与正式披露交叉核对，历史
    连续性由自然日时间线检查；任职开始日期缺失保留未知，不倒填成立日。
    """
    if not re.fullmatch(r"[0-9]{6}\.(?:OF|SZ|SH)", ts_code) or ts_code.split(".")[0] != fund_code:
        raise ValueError("MANAGER_API_REQUEST_IDENTITY")
    data = original.get("data") or {}
    fields, items = data.get("fields"), data.get("items")
    required = {"ts_code", "ann_date", "name", "begin_date", "end_date"}
    if original.get("code") != 0 or not isinstance(fields, list) or not isinstance(items, list):
        raise ValueError("MANAGER_API_RESPONSE_SCHEMA")
    if (
        len(fields) != len(set(fields))
        or not required <= set(fields)
        or not 0 < len(items) < 1000
        or data.get("has_more") is True
    ):
        raise ValueError("MANAGER_API_FIELDS_OR_COUNT")

    def iso(value):
        if isinstance(value, str):
            value = value.strip()
        # 供应商空白日期沿用既有客户端语义，保留None；任职开始未知仍由时间线拒绝。
        if value is None or value == "":
            return None
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{8}", value):
            raise ValueError("MANAGER_API_DATE_FORMAT")
        return str(datetime.strptime(value, "%Y%m%d").date())

    records, identities = [], set()
    for item in items:
        if not isinstance(item, list) or len(item) != len(fields):
            raise ValueError("MANAGER_API_ROW_WIDTH")
        row = dict(zip(fields, item, strict=True))
        if row["ts_code"] != ts_code or not isinstance(row["name"], str) or not row["name"].strip():
            raise ValueError("MANAGER_API_ROW_IDENTITY")
        begin, end, announced = iso(row["begin_date"]), iso(row["end_date"]), iso(row["ann_date"])
        if begin and end and end < begin:
            raise ValueError("MANAGER_API_DATE_ORDER")
        identity = (row["name"], begin, end)
        if identity in identities:
            raise ValueError("MANAGER_API_DUPLICATE_APPOINTMENT")
        identities.add(identity)
        records.append(
            {"manager_name": row["name"].strip(), "begin_date": begin, "end_date": end, "ann_date": announced}
        )
    return records


def new_team_from_appointment(text: str, *, fund_code: str, as_of: date) -> dict:
    """读取正式增聘公告基本信息和任职日，不从简历中其他基金经历抽取日期。

    当前适配只覆盖明确列出一位新任及共同管理名单的表格；离任、多人同时
    增聘、代任或名单不全继续留待专门核对，不能用“已登记”标志代替原文。
    """
    body = re.sub(r"\s+", "", text)
    announced = re.search(r"公告送出日期[:：]?(20[0-9]{2})年([0-9]{1,2})月([0-9]{1,2})日", body[:1200])
    if not announced or date(*map(int, announced.groups())) > as_of:
        raise ValueError("APPOINTMENT_NOTICE_PUBLICATION_MISSING_OR_FUTURE")
    basic = re.search(r"1[.、]公告基本信息(.+?)2[.、]新任基金经理的相关信息", body)
    if not basic or not re.search(r"基金主代码" + re.escape(fund_code) + r"(?![0-9])", basic[1]):
        raise ValueError("APPOINTMENT_NOTICE_PRODUCT_OR_BASIC_SECTION")
    names = re.search(
        r"基金经理变更类型增聘基金经理新任基金经理姓名(.+?)共同管理本基金的其他基金经理姓名(.+)$", basic[1]
    )
    if not names or not re.fullmatch(r"[\u4e00-\u9fff·]{2,15}", names[1]):
        raise ValueError("APPOINTMENT_NOTICE_CHANGE_OR_NAMES_UNSUPPORTED")
    others = [] if names[2] == "无" else re.split(r"[、,，]", names[2])
    if any(not re.fullmatch(r"[\u4e00-\u9fff·]{2,15}", name) for name in others):
        raise ValueError("APPOINTMENT_NOTICE_OTHER_MEMBERS_UNRESOLVED")
    detail = body[basic.end() :]
    effective = re.match(
        r"新任基金经理姓名" + re.escape(names[1]) + r"任职日期(20[0-9]{2})[-年]([0-9]{1,2})[-月]([0-9]{1,2})日?", detail
    )
    if not effective:
        raise ValueError("APPOINTMENT_NOTICE_EFFECTIVE_DATE_UNRESOLVED")
    day = date(*map(int, effective.groups()))
    if day > as_of or names[1] in others or len(set(others)) != len(others):
        raise ValueError("APPOINTMENT_NOTICE_FUTURE_OR_DUPLICATE")
    return {
        "effective_date": str(day),
        "published_on": str(date(*map(int, announced.groups()))),
        "members": sorted([names[1], *others]),
        "excerpt": basic[0] + effective[0],
        "scope": "FORMAL_APPOINTMENT_NAMES_AND_DATE_NOT_DUTY_OR_REGISTRATION_VERIFICATION",
    }


def validate_product_manager_scope(product_records: list[dict], share_records: list[dict]) -> None:
    """产品任职史必须覆盖已有本份额同名任期，冲突时不把较长历史套给份额。

    调用方另须从正式报告证明份额与主代码关系。本函数只比较公开任职区间，
    绝不扩展份额净值、费用或独立校准产品数。
    """
    for share in share_records:
        if not share.get("begin_date") or not any(
            product["manager_name"] == share["manager_name"]
            and product.get("begin_date")
            and product["begin_date"] <= share["begin_date"]
            and (
                product.get("end_date") is None
                or share.get("end_date") is not None
                and product["end_date"] >= share["end_date"]
            )
            for product in product_records
        ):
            raise ValueError("MANAGER_PRODUCT_AND_SHARE_APPOINTMENTS_CONFLICT")


def formal_manager_duty(text: str) -> dict | None:
    """识别正式招募书中的基金经理责任条文，不把履历、受罚或笼统公司职责当依据。

    此处只证明职位的授权/受托责任定义。任职覆盖天数仍由完整生效记录计算，
    也不证明责任实际履行、治理清白、任职能力或历版条款均已逐条核对。
    产品身份及原件字节由调用方先核验；返回原文片段，不接受外部核验标志。
    """
    body = re.sub(r"\s+", "", text)
    patterns = (
        (
            "INVESTMENT_AUTHORITY_AND_EXECUTION",
            r"基金经理(?:在|根据)(?:(?!。|；|基金经理).){0,130}"
            r"(?:制定并实施具体的投资策略|制定具体的投资组合方案并执行|拟定投资计划[^。]{0,80}投资组合构建)",
        ),
        (
            "EXPLICIT_RESPONSIBILITY_SECTION",
            r"基金经理(?:的)?(?:主要)?职责[:：]?[^。]{0,180}(?:负责投资组合|制定投资策略|执行投资计划)",
        ),
        (
            "FIDUCIARY_DUTY_UNDER_LAW_AND_CONTRACT",
            # 已有正式招募书存在“基金经理承诺1、”及“承诺（1）”两种排版。
            # 冒号可省略、编号句点兼容全角；“谋取利益”同样是受托职责，
            # 并不因为缺少“最大”二字就代表未定义责任。编号、法律/合同依据
            # 和份额持有人受托责任仍须同时出现；
            # 不能仅凭“承诺”字样、任职履历或管理公司职责补齐岗位责任证据。
            r"基金经理(?:的)?承诺[:：]?(?:[（(]1[）)]|1[、.．])"
            r"[^。；]{0,100}依照[^。；]{0,90}基金合同[^。；]{0,90}为基金份额持有人谋(?:取|求)(?:最大)?利益",
        ),
    )
    for kind, pattern in patterns:
        for match in re.finditer(pattern, body):
            quote = match[0]
            if not re.search(r"未(?:明确|披露|履行|制定|依照)|不存在|不负责|不再负责|不依照|不为基金份额持有人", quote):
                return {"kind": kind, "excerpt": quote, "scope": "FORMAL_ROLE_NOT_EXECUTION_QUALITY_OR_GOVERNANCE"}
    return None
