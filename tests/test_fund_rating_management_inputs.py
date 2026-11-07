"""正常履职必须有跨期明示原件；缺一项、对象错误或无实际托管正文均拒绝。"""

import hashlib
import json
from copy import deepcopy

import pytest
from app.services.fund_rating_management_inputs import calculate


@pytest.fixture
def inputs(tmp_path):
    sources, statements = {}, []
    company = "测试基金管理有限公司"

    def original(text):
        content = json.dumps({"body": text}, ensure_ascii=False).encode()
        key = hashlib.sha256(content).hexdigest()
        path = tmp_path / "sources" / (key + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        sources[key] = {"file": "sources/" + path.name, "sha256": key, "kind": "json"}
        return key

    for period in ["2023-12-31", "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31", "2026-06-30"]:
        quote = "基金管理人无受调查或处罚等情况。"
        key = original(company + period + quote)
        statements.append(
            {
                "period_end": period,
                "published_on": "2026-08-31",
                "source_hash": key,
                "manager_name": company,
                "scope": "MANAGER",
                "quote": quote,
            }
        )
    role = "本报告期基金经理实际履职，托管人进行了投资监督及净值复核。"
    key = original(company + role)
    raw = {
        "manager_name": company,
        "as_of_date": "2026-09-30",
        "sources": sources,
        "claims": [{"source_hash": key, "quote": role}],
        "experience": {"state": "BASIC_ROLE_EVIDENCED", "source_refs": [key]},
        "continuity": None,
        "governance": {
            "state": "NORMAL_REVIEWED",
            "negative_event_review_complete": True,
            "known_applicable_events": [],
            "statements": statements,
            "trustee_execution": {"source_hash": key, "quote": role},
        },
    }
    return raw, tmp_path, company


def test_known_governance_does_not_impute_unknown_continuity(inputs):
    raw, root, company = inputs
    result = calculate(raw, root=root, manager_name=company, as_of="2026-09-30")
    assert result["score"] is None and result["lower"] == "50.0" and result["upper"] == "70.0"


@pytest.mark.parametrize("quote,accepted", [
    ("4.9.1 管理人受调查或处罚等情况 无。4.9.2 管理人相关从业人员受调查或处罚等情况", True),
    ("4.9.1 管理人受调查或处罚等情况 无。", True),
    ("4.9.1 管理人受调查或处罚等情况 本报告期内，本基金管理人不存在受调查或处罚的情况。", True),
    ("本报告期内无基金管理人及其高级管理人员受稽查或处罚的情况。", True),
    ("本报告期内，未发生管理人及其高级管理人员受调查或处罚等情况。", True),
    ("本报告期内，未发生管理人及其高级管理人员受稽查或处罚等情形。", True),
    ("报告期内，本基金无涉及管理人及其高级管理人员受调查或处罚等情况。", True),
    ("无。", False),
    ("4.9.1 托管人受调查或处罚等情况 无。4.9.2 管理人相关从业人员受调查或处罚等情况", False),
    ("4.9.1 管理人受调查或处罚等情况 无。4.10.2 管理人相关从业人员受调查或处罚等情况", False),
    ("4.9.1 管理人受调查或处罚等情况 无。……18 4.9.2 管理人相关从业人员受调查或处罚等情况", False),
    ("4.9.1 管理人受调查或处罚等情况 尚无明确结论。4.9.2 管理人相关从业人员受调查或处罚等情况", False),
    ("本报告期内无托管人及其高级管理人员受稽查或处罚的情况。", False),
    ("本报告期内无基金管理人相关从业人员受稽查或处罚的情况。", False),
])
def test_split_manager_declaration_requires_same_section_explicit_body(inputs, quote, accepted):
    """管理人小节正文可以明确为无，但不能借托管人、目录或跨章节拼接。"""
    raw, root, company = inputs
    content = json.dumps({"body": company + quote}, ensure_ascii=False).encode()
    key = hashlib.sha256(content).hexdigest()
    (root / "sources" / (key + ".json")).write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + key + ".json", "sha256": key, "kind": "json"}
    raw["governance"]["statements"][0].update(quote=quote, source_hash=key)
    if accepted:
        assert calculate(raw, root=root, manager_name=company, as_of="2026-09-30")["governance"] == "75"
    else:
        with pytest.raises(ValueError, match="NO_EXPLICIT_MANAGER_DECLARATION"):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30")


@pytest.mark.parametrize("damage", [None, "missing_price_review", "missing_harm_statement", "future_intention"])
def test_trustee_equivalent_wording_keeps_actual_price_and_holder_protection(inputs, damage):
    """等价术语可适配，但申赎计价和实际未损害声明缺一仍不能正常赋分。"""
    raw, root, company = inputs
    quote = "报告期内，本基金投资运作、基金资产净值的计算及基金份额申购赎回价格的计算均经托管核查；托管人未发现损害基金持有人利益的行为。"
    if damage == "missing_price_review":
        quote = quote.replace("及基金份额申购赎回价格的计算", "")
    elif damage == "missing_harm_statement":
        quote = quote.replace("托管人未发现损害基金持有人利益的行为", "托管人提供一般托管服务")
    elif damage == "future_intention":
        quote = quote.replace("未发现损害", "拟核查是否损害")
    content = json.dumps({"body": company + quote}, ensure_ascii=False).encode()
    key = hashlib.sha256(content).hexdigest()
    (root / "sources" / (key + ".json")).write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + key + ".json", "sha256": key, "kind": "json"}
    raw["governance"]["trustee_execution"] = {"source_hash": key, "quote": quote}
    if damage:
        with pytest.raises(ValueError, match="ACTUAL_TRUSTEE_EXECUTION_MISSING"):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30")
    else:
        assert calculate(raw, root=root, manager_name=company, as_of="2026-09-30")["governance"] == "75"


@pytest.mark.parametrize("damage", [None, "missing_event", "other_new_name", "future_date", "invented_quote"])
def test_manager_rename_requires_original_effective_date_and_same_current_name(inputs, damage):
    """同一法定管理人更名可接续历史；未经原件核准的别名不能借入治理记录。"""
    raw, root, company = inputs

    def original(body):
        content = json.dumps({"body": body}, ensure_ascii=False).encode()
        key = hashlib.sha256(content).hexdigest()
        (root / "sources" / (key + ".json")).write_bytes(content)
        raw["sources"][key] = {"file": "sources/" + key + ".json", "sha256": key, "kind": "json"}
        return key

    old_name = "测试管理人原名称有限公司"
    statement = raw["governance"]["statements"][0]
    statement["source_hash"] = original(old_name + statement["quote"])
    statement["manager_name"] = old_name
    quote = "2025年3月14日，" + old_name + "变更名称为" + company + "。"
    event = {"old_name": old_name, "new_name": company, "effective_on": "2025-03-14",
             "source_hash": original(quote), "quote": quote}
    raw["manager_identity_events"] = [event]
    if damage == "missing_event":
        raw.pop("manager_identity_events")
    elif damage == "other_new_name":
        event["new_name"] = "另一管理人有限公司"
    elif damage == "future_date":
        event["effective_on"] = "2026-10-01"
    elif damage == "invented_quote":
        event["quote"] += "未在原件出现的附加内容"
    if damage:
        with pytest.raises(ValueError):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30")
    else:
        assert calculate(raw, root=root, manager_name=company, as_of="2026-09-30")["governance"] == "75"


@pytest.mark.parametrize("subject", ["本基金管理人及其高级管理人员", "本基金托管人"])
def test_explicit_no_event_wording_cannot_change_the_reported_subject(inputs, subject):
    raw, root, company = inputs
    quote = "本报告期内，" + subject + "未发生受稽查或处罚的情况。"
    content = json.dumps({"body": company + quote}, ensure_ascii=False).encode()
    key = hashlib.sha256(content).hexdigest()
    path = root / "sources" / (key + ".json")
    path.write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + path.name, "sha256": key, "kind": "json"}
    raw["governance"]["statements"][0].update(quote=quote, source_hash=key)
    if "托管人" in subject:
        with pytest.raises(ValueError, match="NO_EXPLICIT_MANAGER_DECLARATION"):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30")
    else:
        assert calculate(raw, root=root, manager_name=company, as_of="2026-09-30")["governance"] == "75"


def test_actual_remediation_keeps_50_state_and_never_becomes_clean_governance(inputs):
    raw, root, company = inputs
    quote = "基金管理人收到监管机构警示函，管理人已完成整改。"
    content = json.dumps({"body": company + quote}, ensure_ascii=False).encode()
    key = hashlib.sha256(content).hexdigest()
    path = root / "sources" / (key + ".json")
    path.write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + path.name, "sha256": key, "kind": "json"}
    raw["governance"]["state"] = "REMEDIATION_DISCLOSED"
    raw["governance"]["known_applicable_events"] = [{"quote": quote, "source_hash": key, "date": "2023-11-01"}]
    raw["governance"]["statements"][0].update(quote=quote, source_hash=key)
    result = calculate(raw, root=root, manager_name=company, as_of="2026-09-30")
    assert result["governance"] == "50" and result["score"] is None
    assert result["lower"] == "40.0" and result["upper"] == "60.0"
    raw["governance"]["state"] = "NORMAL_REVIEWED"
    with pytest.raises(ValueError, match="EVENT_REQUIRES_OWN_STATE"):
        calculate(raw, root=root, manager_name=company, as_of="2026-09-30")


@pytest.mark.parametrize("wording,accepted", [
    ("公司已从制度建设、流程机制等方面整改完毕。", True),
    ("公司已从制度建设方面计划整改完毕。", False),
    ("公司拟从制度建设、流程机制等方面整改完毕。", False),
    ("管理人已整改完成。", True),
    ("管理人拟整改完成。", False),
    ("管理人已采取完善制度、优化流程等措施完见）成整改，并通过监管机构验收。", True),
    ("管理人拟采取完善制度、优化流程等措施完见）成整改，并通过监管机构验收。", False),
])
def test_explicit_remediation_finished_wording_does_not_accept_future_intention(inputs, wording, accepted):
    raw, root, company = inputs
    quote = "基金管理人收到监管机构警示函。" + wording
    content = json.dumps({"body": company + quote}, ensure_ascii=False).encode()
    key = hashlib.sha256(content).hexdigest()
    (root / "sources" / (key + ".json")).write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + key + ".json", "sha256": key, "kind": "json"}
    raw["governance"].update(state="REMEDIATION_DISCLOSED",
                            known_applicable_events=[{"quote": quote, "source_hash": key, "date": "2026-02-12"}])
    raw["governance"]["statements"][-1].update(quote=quote, source_hash=key)
    if accepted:
        assert calculate(raw, root=root, manager_name=company, as_of="2026-09-30")["governance"] == "50"
    else:
        with pytest.raises(ValueError, match="REMEDIATION_NOT_DISCLOSED"):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30")


@pytest.mark.parametrize("completed", [True, False])
@pytest.mark.parametrize("wrapped_label", [False, True])
def test_issuer_reported_remediation_acceptance_is_not_mistaken_for_future_intention(inputs, completed, wrapped_label):
    raw, root, company = inputs
    accepted = ("公司已通过优化流程完成提出整改意见)整改，整改成果已经监管局验收通过。"
                if wrapped_label else "公司整改成果已经监管局验收通过，相关措施已被解除。")
    quote = "基金管理人收到监管机构责令改正。" + (
        accepted if completed else "公司整改成果已申请监管局验收通过。"
    )
    content = json.dumps({"body": company + quote}, ensure_ascii=False).encode()
    key = hashlib.sha256(content).hexdigest()
    path = root / "sources" / (key + ".json")
    path.write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + path.name, "sha256": key, "kind": "json"}
    raw["governance"]["state"] = "REMEDIATION_DISCLOSED"
    raw["governance"]["known_applicable_events"] = [{"quote": quote, "source_hash": key, "date": "2025-11-11"}]
    raw["governance"]["statements"][0].update(quote=quote, source_hash=key)
    if completed:
        assert calculate(raw, root=root, manager_name=company, as_of="2026-09-30")["governance"] == "50"
    else:
        with pytest.raises(ValueError, match="REMEDIATION_NOT_DISCLOSED"):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30")


@pytest.mark.parametrize("damage", ["period", "subject", "quote", "adverse", "trustee", "repeated_source"])
def test_positive_state_requires_actual_scope_and_all_originals(inputs, damage):
    raw, root, company = inputs
    raw = deepcopy(raw)
    g = raw["governance"]
    if damage == "period":
        g["statements"].pop()
    elif damage == "subject":
        g["statements"][0]["scope"] = "CUSTODIAN"
    elif damage == "quote":
        g["statements"][0]["quote"] = "未在原件中存在的正常声明"
    elif damage == "adverse":
        g["known_applicable_events"] = [{"actual_event": True}]
    elif damage == "trustee":
        g["trustee_execution"]["quote"] = "只有目录，没有实际托管履职"
    else:
        g["statements"][0]["source_hash"] = g["statements"][1]["source_hash"]
    with pytest.raises(ValueError):
        calculate(raw, root=root, manager_name=company, as_of="2026-09-30")


@pytest.mark.parametrize("damage", [None, "birth_date", "window_start", "source_bytes"])
def test_management_window_starts_at_proven_product_inception(inputs, damage):
    """成立前无管理责任；成立日须原件明示，且不改变其他维度历史要求。"""
    raw, root, company = inputs
    quote = "基金合同生效日为2023年11月17日。"
    content = (company + quote).encode("gb18030")
    key = hashlib.sha256(content).hexdigest()
    path = root / "sources" / (key + ".txt")
    path.write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + path.name, "sha256": key, "kind": "text", "encoding": "gb18030"}
    raw["fund_inception"] = {"date": "2023-11-17", "source_hash": key, "quote": quote}
    raw["assessment_window_start"] = "2023-09-28"
    raw["continuity"] = {
        "period_start": "2023-11-17", "period_end": "2026-09-30",
        "intervals": [{"name": "实际基金经理", "start": "2023-11-17", "end": None, "source_refs": [key]}],
        "source_refs": [key], "timeline_closed": True, "current_role_verified": True,
        "arrangements_review_closed": True, "confirmed_functional_defects": [],
    }
    if damage == "birth_date":
        raw["fund_inception"]["date"] = "2023-11-18"
    elif damage == "window_start":
        raw["continuity"]["period_start"] = "2024-01-01"
    elif damage == "source_bytes":
        path.write_bytes(b"changed")
    if damage:
        with pytest.raises(ValueError):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30")
    else:
        result = calculate(raw, root=root, manager_name=company, as_of="2026-09-30")
        assert result["continuity"] == "75" and result["score"] == "65.0"


@pytest.mark.parametrize("damage", [None, "unrelated_event", "source", "body", "interval"])
def test_partial_manager_event_requires_own_recomputed_dimension(monkeypatch, damage):
    from app.services import fund_rating_reviewed_snapshot as snapshot
    from app.services.fund_market_review import _partial_manager_reviewed

    sha = "a" * 64
    item = {"entry": {"title": "某基金基金经理变更公告"}, "body": "真实任免原文"}
    raw = {"sources": {sha: {}}, "claims": [{"source_hash": sha, "quote": item["body"]}]}
    bundle = {"management_calculation": {"inputs": raw}}
    monkeypatch.setattr(snapshot, "validate_sources", lambda *_: None)
    monkeypatch.setattr(snapshot, "calculate", lambda *_: {
        "dimensions": {"management": {"candidate": None if damage == "interval" else "55"}}
    })
    if damage == "unrelated_event":
        item["entry"]["title"] = "基金暂停赎回公告"
    elif damage == "source":
        raw["sources"] = {}
    elif damage == "body":
        item["body"] = "另一版本正文"
    assert _partial_manager_reviewed(bundle, item, sha) is (damage is None)


@pytest.mark.parametrize("damage", [None, "other_product", "other_share", "other_trustee"])
def test_common_manager_does_not_authorize_borrowing_another_fund_role(inputs, damage):
    raw, root, company = inputs
    role = raw["claims"][0]["quote"]
    content = json.dumps({"body": company + "真实法律产品基金123451" + role}, ensure_ascii=False).encode()
    key = hashlib.sha256(content).hexdigest()
    path = root / "sources" / (key + ".json")
    path.write_bytes(content)
    raw["sources"][key] = {"file": "sources/" + path.name, "sha256": key, "kind": "json"}
    raw["experience"]["source_refs"] = [key]
    raw["claims"][0]["source_hash"] = key
    raw["governance"]["trustee_execution"]["source_hash"] = key
    kwargs = {
        "fund_code": "123452" if damage == "other_share" else "123451",
        "legal_name": "其他法律产品基金" if damage == "other_product" else "真实法律产品基金",
        "primary_report_hash": "b" * 64 if damage == "other_trustee" else key,
    }
    if damage:
        with pytest.raises(ValueError, match="IDENTITY"):
            calculate(raw, root=root, manager_name=company, as_of="2026-09-30", **kwargs)
    else:
        assert calculate(raw, root=root, manager_name=company, as_of="2026-09-30", **kwargs)["lower"] == "50.0"
