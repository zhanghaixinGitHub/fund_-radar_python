"""公开任职API反例；虚构测试行不作为基金原件、校准产品或独立验证。"""

import copy
from datetime import date

import pytest
from app.services.fund_quality_v2_manager_sources import (
    manager_api_records,
    new_team_from_appointment,
    validate_product_manager_scope,
)


def original():
    return {
        "code": 0,
        "data": {
            "fields": ["ts_code", "ann_date", "name", "begin_date", "end_date"],
            "items": [["123456.OF", "20230101", "测试甲", "20230101", None]],
        },
    }


def test_appointment_keeps_missing_begin_unknown_and_does_not_create_responsibility_proof():
    data = original()
    data["data"]["items"][0][3] = None
    data["data"]["items"][0][4] = " "
    record = manager_api_records(data, fund_code="123456", ts_code="123456.OF")[0]
    assert record["begin_date"] is None and record["ann_date"] == "2023-01-01"
    assert record["end_date"] is None
    assert "responsibility_verified" not in record


@pytest.mark.parametrize("problem", ["other_share", "date", "order", "duplicate", "row_width", "empty"])
def test_success_flag_is_not_appointment_verification(problem):
    data = copy.deepcopy(original())
    if problem == "other_share":
        data["data"]["items"][0][0] = "654321.OF"
    elif problem == "date":
        data["data"]["items"][0][3] = "20230230"
    elif problem == "order":
        data["data"]["items"][0][4] = "20220101"
    elif problem == "duplicate":
        data["data"]["items"].append(data["data"]["items"][0])
    elif problem == "row_width":
        data["data"]["items"][0].pop()
    else:
        data["data"]["items"].clear()
    with pytest.raises(ValueError):
        manager_api_records(data, fund_code="123456", ts_code="123456.OF")


def appointment_text():
    return (
        "公告送出日期2026年7月25日\n1.公告基本信息\n基金主代码123456\n"
        "基金经理变更类型增聘基金经理 新任基金经理姓名测试甲\n共同管理本基金的其他基金经理姓名测试乙\n"
        "2.新任基金经理的相关信息\n新任基金经理姓名测试甲 任职日期2026-7-24\n"
        "过往从业经历 其他基金654321任职日期2020-1-1 是否已注册/登记是"
    )


def test_formal_appointment_reads_own_effective_date_and_explicit_other_members():
    result = new_team_from_appointment(appointment_text(), fund_code="123456", as_of=date(2026, 9, 30))
    assert result["effective_date"] == "2026-07-24"
    assert result["members"] == ["测试乙", "测试甲"]
    assert result["published_on"] == "2026-07-25"
    assert "登记" not in result["excerpt"]


@pytest.mark.parametrize("problem", ["other_fund", "missing_other", "future", "departure"])
def test_formal_notice_does_not_guess_incomplete_or_different_role(problem):
    body = appointment_text()
    if problem == "other_fund":
        body = body.replace("基金主代码123456", "基金主代码654321")
    elif problem == "missing_other":
        body = body.replace("姓名测试乙", "姓名-")
    elif problem == "future":
        body = body.replace("2026年7月25日", "2026年10月25日")
    else:
        body = body.replace("变更类型增聘基金经理", "变更类型解聘基金经理")
    with pytest.raises(ValueError):
        new_team_from_appointment(body, fund_code="123456", as_of=date(2026, 9, 30))


def test_share_appointment_must_fit_same_person_product_history():
    parent = [{"manager_name": "测试甲", "begin_date": "2020-01-01", "end_date": None}]
    share = [{"manager_name": "测试甲", "begin_date": "2024-06-19", "end_date": None}]
    validate_product_manager_scope(parent, share)
    parent[0]["end_date"] = "2025-01-01"
    with pytest.raises(ValueError, match="CONFLICT"):
        validate_product_manager_scope(parent, share)


@pytest.mark.parametrize(
    "body,expected",
    [
        (
            "本基金实行基金经理负责制。基金经理在投资决策委员会确定的投资范围内制定并实施具体的投资策略，向集中交易室下达投资指令。",
            "INVESTMENT_AUTHORITY_AND_EXECUTION",
        ),
        (
            "基金经理根据投资决策委员会授权和会议决议，拟定投资计划，并进行投资组合构建或调整。",
            "INVESTMENT_AUTHORITY_AND_EXECUTION",
        ),
        (
            "基金经理承诺：（1）依照有关法律、法规和基金合同的规定，本着谨慎的原则为基金份额持有人谋取最大利益；（2）不利用职务便利。",
            "FIDUCIARY_DUTY_UNDER_LAW_AND_CONTRACT",
        ),
        (
            "六、基金经理承诺\n1、依照有关法律法规和基金合同的规定，本着谨慎的原则为基金份额持有人谋取最大利益；2、不利用职务之便。",
            "FIDUCIARY_DUTY_UNDER_LAW_AND_CONTRACT",
        ),
        (
            "4、基金经理承诺（1）依照有关法律法规和基金合同的规定，本着谨慎的原则为基金份额持有人谋取最大利益；（2）不利用职务之便。",
            "FIDUCIARY_DUTY_UNDER_LAW_AND_CONTRACT",
        ),
        ("基金经理承诺本着谨慎的原则为基金份额持有人谋取最大利益。", None),
        ("基金经理承诺1、未履行依照有关法律法规和基金合同为基金份额持有人谋取最大利益的责任。", None),
        (
            "基金经理的承诺1．依照有关法律法规和基金合同的规定，本着谨慎的原则为基金份额持有人谋取最大利益；2．不利用职务之便。",
            "FIDUCIARY_DUTY_UNDER_LAW_AND_CONTRACT",
        ),
        (
            "基金经理承诺（1）依照有关法律、法规和基金合同的规定，本着谨慎的原则为基金份额持有人谋取利益。（2）不利用职务之便。",
            "FIDUCIARY_DUTY_UNDER_LAW_AND_CONTRACT",
        ),
        ("基金经理承诺1．不依照有关法律法规和基金合同为基金份额持有人谋取利益。", None),
        ("基金经理承诺（1）为基金份额持有人谋取利益。", None),
        ("基金经理承诺（1）依照有关法律和基金合同为公司股东谋取最大利益。", None),
        ("基金经理因基金管理业务相关行为受到重大行政处罚。", None),
        ("某人曾任其他基金经理，负责其他公司的投资业务。", None),
        ("基金管理人应履行分别管理基金资产、进行投资的职责。", None),
        ("基金经理职责未明确，不负责投资组合构建。", None),
    ],
)
def test_formal_duty_is_not_biography_company_duty_or_adverse_keyword(body, expected):
    from app.services.fund_quality_v2_manager_sources import formal_manager_duty

    result = formal_manager_duty(body)
    assert (result["kind"] if result else None) == expected
    if result:
        assert result["scope"] == "FORMAL_ROLE_NOT_EXECUTION_QUALITY_OR_GOVERNANCE"
