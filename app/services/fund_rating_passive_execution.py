"""独立编译直接复制及历史同目标ETF联接的执行项，缺同行不阻塞绝对误差。

只复用R5既有公式和三年窗口，不新增评价规则；其他七维由各自完整原件
单独闭合。编译成功仍不能据此产生完整总分或等级。
"""

import json
import re
from copy import deepcopy
from decimal import localcontext
from functools import lru_cache
from pathlib import Path

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals, _contract, _history, contract_history_refs
from app.services.fund_rating_direct_index_math import performance
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_passive_holdings import _identity
from app.services.fund_rating_r5 import PROTOCOL, number, tracking_score
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy

VERSION = "R5_PASSIVE_EXECUTION_ORIGINALS_20261029_1"


def calculate(inputs: dict, *, root: Path) -> dict:
    """每次复读哈希核准原字节，再复用不可变输入的解析；不联网、不写库。"""
    Originals(inputs, root)
    return deepcopy(_calculate_cached(canonical(inputs), str(root.resolve())))


@lru_cache(maxsize=24)
def _calculate_cached(serialized: str, root_name: str) -> dict:
    inputs, root = json.loads(serialized), Path(root_name)
    if (
        inputs.get("version") != VERSION
        or inputs.get("basis") != PASSIVE_BASIS
        or policy(PASSIVE_BASIS)["terms"]["efficiency"] != [["tracking_error", 100]]
    ):
        raise ValueError("R5_EXECUTION_FROZEN_FORMULA_OR_SCOPE")
    source = Originals(inputs, root)
    refs, code = inputs["target_refs"], inputs["fund_code"]
    if inputs["primary_ref"] != refs["half"] or inputs["holding_date"] > inputs["as_of_date"]:
        raise ValueError("R5_EXECUTION_CURRENT_REPORT_SCOPE")
    pages = _identity(source, refs["half"], code, inputs["legal_name"])
    if str(report_period("\n".join(pages))[1]) != inputs["holding_date"]:
        raise ValueError("R5_EXECUTION_REPORT_PERIOD")
    plain = normalized_text("\n".join(pages))
    feeder = "联接" in inputs["legal_name"]
    if "增强" in inputs["legal_name"] or (not feeder and not re.search(
        r"(?:本基金(?:为|属)被动|本基金采用完全复制|本基金主要采用完全复制)", plain
    )):
        raise ValueError("R5_EXECUTION_DIRECT_PASSIVE_MANDATE_REQUIRED")
    if feeder:
        own = re.search(r"2\.2基金产品说明(.+?)2\.2\.1目标基金产品说明", plain)
        if inputs.get("route") != "FEEDER" or not inputs.get("target_etf") or not own:
            raise ValueError("R5_EXECUTION_FEEDER_OWN_PASSIVE_MANDATE_REQUIRED")
        # 定报可以用联接基金自身的类别和目标说明被动跟踪职责。只检查本基金小节，
        # 历史目标ETF身份及同期合同基准仍由下面的原件检查逐年核准。
        mandate = own[1]
        if "本基金为完全被动式指数基金" not in mandate and not all(
            term in mandate for term in (
                "本基金为ETF联接基金",
                "本基金为指数型基金",
                "本基金通过主要投资于目标ETF,紧密跟踪业绩比较基准,追求跟踪偏离度和跟踪误差最小化",
            )
        ):
            raise ValueError("R5_EXECUTION_FEEDER_OWN_PASSIVE_MANDATE_REQUIRED")
    history = _history(source, inputs)
    if history["dates"][-1] != inputs["as_of_date"]:
        raise ValueError("R5_EXECUTION_HISTORY_CUTOFF")
    annuals = contract_history_refs(
        source, inputs["target_annuals"], {}, history["dates"][0], inputs["as_of_date"], code, {}
    )
    contracts = {}
    for period, digest in {**annuals, "current": refs["half"]}.items():
        if source.sources[digest].get("archive_kind") == "DERIVED":
            raise ValueError("R5_EXECUTION_CONTRACT_ORIGINAL_REQUIRED")
        _identity(source, digest, code, inputs["legal_name"])
        if feeder:
            # 当前持仓正确不代表整个三年路线一致，逐年同一法律目标也须核原文。
            original = normalized_text("\n".join(source.pages(digest)))
            relation = re.search(r"2\.1\.1目标基金基本情况(.{1,2500}?)(?:(?:2\.2)?基金产品说明|"
                                 r"2\.1\.2目标基金产品说明)", original)
            etf = inputs["target_etf"]
            if (not relation or etf["fund_code"] not in relation[1]
                    or normalized_text(etf["legal_name"]) not in relation[1]):
                raise ValueError("R5_EXECUTION_FEEDER_HISTORICAL_TARGET_REQUIRED")
        contracts[period] = _contract(source, digest, code, inputs["index_name"])
    summary = normalized_text("\n".join(source.pages(refs["summary"])))
    main_code = re.search(r"(?<!下属)基金代码(\d{6})", summary)
    if code not in summary or normalized_text(inputs["legal_name"]) not in summary or not main_code:
        raise ValueError("R5_EXECUTION_SHARE_TO_LEGAL_PRODUCT_REQUIRED")
    clause, clause_ref = None, None
    for digest in (refs["summary"], refs["prospectus"]):
        if source.sources[digest].get("archive_kind") == "DERIVED":
            raise ValueError("R5_EXECUTION_CONTRACT_ORIGINAL_REQUIRED")
        text = normalized_text("\n".join(source.pages(digest)))
        # 招募书可以只列主代码；C概要须同时明确主代码、C代码与相同法律全名，
        # 招募书须实列该主代码和C类条款，不能因代码未出现而借另一产品合同。
        if normalized_text(inputs["legal_name"]) not in text or (
            code not in text and not (main_code[1] in text and "C类基金份额" in text)
        ):
            raise ValueError("R5_EXECUTION_TRACKING_DOCUMENT_IDENTITY")
        match = re.search(r"年(?:化)?跟踪误差(?:不超过|控制在|控制)([\d.]+)%", text)
        if match:
            clause, clause_ref = match, digest
            break
    if not clause or number(clause[1]) <= 0:
        raise ValueError("R5_EXECUTION_TRACKING_CLAUSE_REQUIRED")
    values = performance(history["dates"], history["month_ends"], history["adjusted_nav"], history["benchmark_index"])
    with localcontext() as context:
        context.prec = 60
        limit = number(clause[1]) / 100
        anchor = PROTOCOL["parameters"]["tracking_economic_anchor_ratio"]
        point = tracking_score(values["tracking_error"], anchor)
        attainment = number(values["tracking_error"]) / limit
    return {
        "score": str(point),
        "tracking_error_ratio": values["tracking_error"],
        "calculation_revision": PROTOCOL["revision"],
        "economic_anchor_ratio": anchor,
        "contract_limit_ratio": str(limit),
        "contract_attainment_ratio": str(attainment),
        "contract_used_for_score": False,
        "contract_source": clause_ref,
        "actual_clause": clause[0],
        "audit": {
            "contracts": contracts,
            "publisher_cross_check": history["publisher_cross_check"],
            "date_count": len(history["dates"]),
            "month_end_count": len(history["month_ends"]),
            "first": history["dates"][0],
            "last": history["dates"][-1],
            "other_dimensions_calculated": False,
            "independent_peers_required_for_this_term": False,
        },
    }
