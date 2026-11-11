"""政府/政策银行抽样持仓五分项隔离候选；旧四分项包原样走旧编译器。

只评价冻结规则覆盖的发行人类别，不代表逐券成份资格或精确复制。
"""

import copy
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from bs4 import BeautifulSoup

from app.services import fund_rating_bond_index_partial as previous
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_market_review import normalized_text

VERSION = "BOND_INDEX_FIVE_PARTIAL_CATEGORY_EVIDENCE_20261111_V1"
OLD_VERSION = previous.VERSION
CONTRACT_HASH = "92a381ca642ec02940175c0bb0b124f8886bb4612f53edd49d42aab5c1b43eb1"
ISSUERS = {
    "230403": (
        "23农发03",
        "中国农业发展银行",
        "POLICY_BANK",
        "9ed3eddba68973773595a3b85e51ad1d11fb7b3f71e8d5c24c8f119d955a0969",
        "2023-04-18",
        "DISTRIBUTOR_SALES_NOTICE",
    ),
    "260010": (
        "26附息国债10",
        "中华人民共和国财政部",
        "GOVERNMENT",
        "831a7be5e6d25623c820c28c49cf406adc8fcd9ed49a80a278baf35701af9418",
        "2026-05-19",
        "STANDARD_IDENTIFIER_RECORD",
    ),
    "220310": (
        "22进出10",
        "中国进出口银行",
        "POLICY_BANK",
        "7b87da55c5221667601b35d4c81e22c56651ef27d4be7ab86d8f30604f6a867b",
        None,
        "ISSUER_RESULT_ALIAS_WITH_DATED_NOTICE",
    ),
    "230407": (
        "23农发07",
        "中国农业发展银行",
        "POLICY_BANK",
        "39f817dc90a1e45ba3731974dfc8fa28813ca3eab56a8524a9065b8f0bacd596",
        "2023-09-05",
        "DISTRIBUTOR_SALES_NOTICE",
    ),
    "2400001": (
        "24特别国债01",
        "中华人民共和国财政部",
        "GOVERNMENT",
        "90712bce4f7b946245a28b8cfefa393a613c038223fae4d57031e177c76cf688",
        "2024-05-22",
        "STANDARD_IDENTIFIER_RECORD",
    ),
}
DATED_EXIM_HASH = "4137f4c2105b346277294d3d552456213cd91fb454609cb6defe9669dd60b4cb"
LABELS = {
    1: "国家债券",
    2: "央行票据",
    3: "金融债券",
    4: "企业债券",
    5: "企业短期融资券",
    6: "中期票据",
    7: "可转债（可交换债）",
    8: "同业存单",
    9: "其他",
    10: "合计",
}


def require(condition, code):
    if not condition:
        raise ValueError("BOND_INDEX_CATEGORY_" + code)


def _amount(value):
    return Decimal(0) if value in {"-", "－", "—"} else Decimal(value.replace(",", ""))


def categories(text):
    matches = list(re.finditer(r"(?m)^\s*7\.5\s*期末按债券品种分类的债券投资组合(.*?)^\s*7\.6\s", text, re.S))
    require(bool(matches), "TABLE_MISSING")
    body = matches[-1][1]
    rows = {}
    for rank, label, amount in re.findall(r"(?m)^\s*(\d{1,2})\s+([^\d\n]+?)\s+([\d,]+\.\d{2}|[-－—])(?=\s)", body):
        rank = int(rank)
        label = normalized_text(label)
        require(rank not in rows and rank in LABELS, "CATEGORY_DUPLICATE_OR_UNKNOWN")
        require(label == normalized_text(LABELS[rank]), "CATEGORY_UNKNOWN")
        rows[rank] = _amount(amount)
    require(set(rows) == set(LABELS), "CATEGORY_MISSING_OR_UNKNOWN")
    policy = re.findall(r"其中[：:]\s*政策性金融债\s+([\d,]+\.\d{2}|[-－—])", body)
    require(len(policy) == 1, "POLICY_SUBSET_MISSING")
    policy = _amount(policy[0])
    require(all(value >= 0 for value in rows.values()) and 0 <= policy <= rows[3], "CATEGORY_AMOUNT")
    require(sum(rows[i] for i in range(1, 10)) == rows[10] and rows[10] > 0, "CATEGORY_TOTAL_NOT_RECONCILED")
    eligible = rows[1] + policy
    return {
        "government": rows[1],
        "policy_bank": policy,
        "total": rows[10],
        "eligible": eligible,
        "noneligible": rows[10] - eligible,
        "quote": normalized_text(body),
    }


def major_codes(text):
    headings = list(re.finditer(r"7\.6\s*期末按公允价值占基金资产净值比例大小排序的前五名债券投资明细", text))
    require(bool(headings), "MAJOR_SECTION")
    body = text[headings[-1].end() :]
    end = re.search(r"(?m)^\s*7\.7\s*期末", body)
    require(end is not None, "MAJOR_SECTION_END")
    body = body[: end.start()]
    starts = list(re.finditer(r"(?m)^\s*(\d)\s+(\d{6,9})(?=\s)", body))
    require([int(m[1]) for m in starts] == list(range(1, 6)), "MAJOR_FIVE_RANKS")
    result = []
    for i, m in enumerate(starts):
        segment = body[m.end() : starts[i + 1].start() if i + 1 < len(starts) else len(body)]
        # 金额/数量列和比例不参与名称；容忍原报告名称跨行，金额不从此路径取值。
        name = re.sub(r"\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+", "", segment)
        name = normalized_text(name)
        hit = re.match(r"\d{2}(?:(?:附息|超长特别|特别)?国债|国开|农发|进出)\d{1,2}", name)
        require(hit is not None, "MAJOR_ISSUER_CLASS_UNPROVEN")
        result.append({"rank": int(m[1]), "code": m[2], "name": hit[0]})
    require(len({r["code"] for r in result}) == 5, "MAJOR_DUPLICATE_CODE")
    return result


def replay_holdings(text, expected_rows, *, proof_texts=None):
    values = categories(text)
    actual = major_codes(text)
    require(actual == expected_rows, "CODE_NAME_MISMATCH")
    for row in actual:
        expected_kind = "GOVERNMENT" if "国债" in row["name"] else "POLICY_BANK"
        if proof_texts is not None:
            proof = proof_texts[row["code"]]
            raw = normalized_text(proof["text"])
            require(proof["kind"] == expected_kind, "CREDIT_OR_LOCAL_NOT_ELIGIBLE")
            require(row["code"] in raw and row["name"] in raw and proof["issuer"] in raw, "ISSUER_CODE_NAME_PROOF")
            issuer = (
                "中华人民共和国财政部"
                if "国债" in row["name"]
                else "中国农业发展银行"
                if "农发" in row["name"]
                else "中国进出口银行"
                if "进出" in row["name"]
                else "国家开发银行"
            )
            require(proof["issuer"] == issuer, "ISSUER_NAME_CLASS_MISMATCH")
    return 100 * values["eligible"] / values["total"], {
        **{k: str(v) if isinstance(v, Decimal) else v for k, v in values.items()},
        "major_bonds": actual,
        "meaning": "发行人类别金额匹配；并非逐券指数成份或精确权重复制。",
    }


def _proof_text(source, sha):
    ref = source.sources[sha]
    if ref.get("kind") == "pdf":
        return source.plain(sha)  # 项目混合读取器支持同版本PDF提取缓存。
    raw = source.load(sha)
    require(ref.get("kind") in {"html", "htm"}, "IDENTITY_SOURCE_FORMAT")
    return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)


def calculate(inputs, *, root: Path):
    # 旧版本即使附加holding_review也不能改变冻结四分项；不改原模块。
    if inputs.get("version") == OLD_VERSION:
        return previous.calculate(inputs, root=root)
    require(inputs.get("version") == VERSION, "VERSION")
    old = copy.deepcopy(inputs)
    old["version"] = OLD_VERSION
    result = previous.calculate(old, root=root)
    source = Originals(inputs, Path(root), lazy=True)
    try:
        review = inputs["holding_review"]
        require(review["version"] == "GOV_POLICY_CATEGORY_AND_MAJOR_FIVE_V1", "HOLDING_VERSION")
        require(review["primary_ref"] == inputs["primary_ref"], "HOLDING_PRIMARY")
        source.claim(review["contract_claim"])
        clause = normalized_text(review["contract_claim"]["quote"])
        require(
            review["contract_claim"]["source_hash"] == inputs["refs"]["prospectus"] == CONTRACT_HASH
            and inputs["sources"][CONTRACT_HASH].get("published_on") == "2026-05-29"
            and all(term in clause for term in ["分层抽样", "国债", "政策性金融债"]),
            "SAMPLING_CONTRACT",
        )
        full_contract = source.plain(CONTRACT_HASH)
        require(
            all(
                normalized_text(t) in full_contract for t in ["001021", "亚债中国债券指数基金", "华夏基金管理有限公司"]
            ),
            "CONTRACT_PRODUCT",
        )
        require(
            {r["code"]: r["name"] for r in review["major_bonds"]} == {k: v[0] for k, v in ISSUERS.items()},
            "VERIFIED_MAJOR_MANIFEST",
        )
        require(set(review["issuer_proofs"]) == {r["code"] for r in review["major_bonds"]}, "ISSUER_PROOF_SET")
        proofs = {}
        for code, proof in review["issuer_proofs"].items():
            name, issuer, kind, sha, published, role = ISSUERS[code]
            require(
                (proof["issuer"], proof["kind"], proof["source_hash"], proof.get("published_on"), proof.get("role"))
                == (issuer, kind, sha, published, role),
                "VERIFIED_ISSUER_MANIFEST",
            )
            ref = inputs["sources"][sha]
            require(ref.get("published_on") == published and ref.get("source_date_role") == role, "IDENTITY_DATE_ROLE")
            if published is not None:
                require(
                    date.fromisoformat(published) <= date.fromisoformat(inputs["as_of_date"]), "IDENTITY_AFTER_CUTOFF"
                )
            else:
                notice = proof["dated_notice"]
                require(
                    notice
                    == {
                        "source_hash": DATED_EXIM_HASH,
                        "published_on": "2025-06-23",
                        "role": "OFFICIAL_NOTICE_PUBLICATION",
                    },
                    "EXIM_DATED_MANIFEST",
                )
                nr = inputs["sources"][DATED_EXIM_HASH]
                require(
                    nr.get("published_on") == "2025-06-23"
                    and nr.get("source_date_role") == "OFFICIAL_NOTICE_PUBLICATION",
                    "EXIM_PUBLICATION_DATE_ROLE",
                )
                nt = normalized_text(_proof_text(source, DATED_EXIM_HASH))
                require(
                    all(t in nt for t in ["2025-06-2316:29", "220310", "中国进出口银行资金营运部"]),
                    "EXIM_PUBLICATION_ORIGINAL",
                )
            text = _proof_text(source, proof["source_hash"])
            if published is not None:
                d = date.fromisoformat(published)
                require(
                    re.search(rf"{d.year}(?:年|-)0?{d.month}(?:月|-)0?{d.day}(?:日)?", normalized_text(text))
                    is not None,
                    "PUBLICATION_DATE_ORIGINAL",
                )
            require(normalized_text(proof["quote"]) in normalized_text(text), "ISSUER_QUOTE")
            proofs[code] = {**proof, "text": text}
        text = "\n".join(source.pages(inputs["primary_ref"]))
        score, facts = replay_holdings(text, review["major_bonds"], proof_texts=proofs)
        facts.update(
            primary_source_hash=inputs["primary_ref"],
            issuer_source_hashes={k: v["source_hash"] for k, v in review["issuer_proofs"].items()},
            contract_claim=review["contract_claim"],
        )
        result["dimensions"]["holdings"] = str(score)
        result["facts"]["holdings"] = facts
        result["facts"]["unscored_dimensions"].pop("holdings", None)
        result["facts"]["source_sha256"] = sorted(set(result["facts"]["source_sha256"]) | set(source.raw))
    except (ValueError, KeyError) as exc:
        result["dimensions"]["holdings"] = None
        result["facts"]["evidence_errors"]["holdings"] = str(exc)
    result["facts"]["compiler_version"] = VERSION
    result["facts"]["complete_type_validated"] = False
    return result
