from copy import deepcopy

import pytest
from app.services.direction_1d_protocol import digest
from app.services.fund_news_identity import _conditional_market_access


@pytest.mark.parametrize("state", ["UNRESOLVED", "ALLOWED", "EXCLUDED"])
def test_conditional_bse_access_requires_actual_contract_state(state):
    args = fixture(state)
    assert _conditional_market_access(**args)


def fixture(state="UNRESOLVED"):
    product = "博道沪深300指数增强型证券投资基金"
    manager = "博道基金管理有限公司"
    rule = {
        "UNRESOLVED": "国内依法发行上市的其他股票",
        "ALLOWED": "可投资北交所股票",
        "EXCLUDED": "不得投资北交所股票",
    }[state]
    clause = "本基金的投资范围包括" + rule + "。" + "遵循基金合同约定及比例限制，不能把投资范围当作实际持仓。" * 5
    contract = {
        "id": "contract",
        "kind": "ANNOUNCEMENT",
        "codes": ["007045"],
        "title": product + "基金合同更新",
        "body": product + manager + clause,
    }
    contract["body_hash"] = digest(contract["body"])
    proof = {
        "quote": product + "007045管理人" + manager,
        "market_access_review": {
            "contract_document_id": "contract",
            "contract_quote": clause,
            "venue": "BSE",
            "state": state,
            "interpretation": (
                "本份额当前合同已逐字核对；本次只登记管理人条件式市场风险提示，"
                "不确定具体北交所准入细节时保留不确定，不能解释为实际已经投资北交所股票或保证增强收益。"
            ),
        },
    }
    return {
        "code": "007045",
        "product": product,
        "proof": proof,
        "document": {"kind": "REPORT"},
        "material": {"documents": [contract]},
        "body": (
            "按照各自基金合同的约定，选择不将基金资产投资于北交所股票，"
            "基金资产并非必然投资北交所股票；流动性风险、退市风险。"
        ),
        "title": manager + "关于旗下部分基金投资北交所股票及相关风险提示的公告",
        "manager": manager,
    }


@pytest.mark.parametrize(
    "failure", ["manager", "title", "scope", "code", "changed_bytes", "quote", "false_allow", "false_excluded"]
)
def test_conditional_market_access_fail_closed(failure):
    args = deepcopy(fixture())
    if failure == "manager":
        args["manager"] = "其他基金管理有限公司"
    if failure == "title":
        args["title"] = "关于旗下部分基金调整投资范围的公告"
    if failure == "scope":
        args["body"] = args["body"].replace("基金资产并非必然投资北交所股票", "基金已经投资北交所股票")
    if failure == "code":
        args["material"]["documents"][0]["codes"] = ["007832"]
    if failure == "changed_bytes":
        args["material"]["documents"][0]["body"] += "已改变"
    if failure == "quote":
        args["proof"]["market_access_review"]["contract_quote"] = "未读合同" * 30
    if failure == "false_allow":
        args["proof"]["market_access_review"]["state"] = "ALLOWED"
    if failure == "false_excluded":
        args["proof"]["market_access_review"]["state"] = "EXCLUDED"
    assert not _conditional_market_access(**args)
