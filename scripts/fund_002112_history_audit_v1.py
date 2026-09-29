"""从保存的原始分页独立核总数和逐日覆盖，并补核原件身份；全程不读标签。"""

import json
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from app.services.fund_information_history_v1 import OLD, OUT, normalize, read, save, sha


def independent_catalog_audit():
    """独立实现：不用采集器的分页/覆盖函数；展开每个窗口逐日核完整性。"""
    day_sets = {}
    page_count = 0
    for path in sorted((OUT / "company-catalogs").glob("*.json")):
        item = read(path)
        if not item["catalog_complete"]:
            continue
        ids, total, numbers = [], None, []
        for receipt in item["receipts"]:
            assert sha(receipt["path"]) == receipt["sha256"]
            raw = json.loads(Path(receipt["path"]).read_bytes())
            total = raw["totalAnnouncement"] if total is None else total
            assert total == raw["totalAnnouncement"]
            numbers.append(receipt["params"]["pageNum"])
            for row in raw["announcements"] or []:
                ids.append(row["announcementId"])
            page_count += 1
        assert len(ids) == len(set(ids)) == total
        assert numbers == list(range(1, max(1, (total + 29) // 30) + 1))
        assert set(ids) == {r["announcementId"] for r in item["rows"]}
        window = item["window"]
        start, stop = date.fromisoformat(window["start"]), date.fromisoformat(window["end"])
        days = {str(start + timedelta(days=i)) for i in range((stop - start).days + 1)}
        day_sets.setdefault(window["stock"], set()).update(days)
    before = read(OLD / "event-admission/row-coverage.json")
    after = {(r["fund_code"], r["target"]): r for r in read(OUT / "row-coverage.json")}
    gained = []
    for row in before:
        target = date.fromisoformat(row["target"])
        days = {str(target - timedelta(days=i)) for i in range(1, 31)}
        complete = row["company_coverage_passed"] or (
            bool(row["missing_companies"])
            and all(days.issubset(day_sets.get(code, set())) for code in row["missing_companies"])
        )
        assert complete == after[row["fund_code"], row["target"]]["catalog_complete"]
        if complete and not row["company_coverage_passed"]:
            gained.append({"fund_code": row["fund_code"], "target": row["target"]})
    result = {
        "passed": True,
        "rows_checked": len(before),
        "raw_pages_checked": page_count,
        "gained_rows": gained,
        "gained_count": len(gained),
        "new_fits": 0,
        "scope": "Company catalog coverage only; no semantic-ready assertion",
    }
    save(OUT / "independent-coverage-audit.json", result)
    return result


def identity_audit():
    """公司全称可由同批带证券代码的原件核实，不强制股票代码必须印在封面。"""
    aliases = {
        "000333": ("美的集团股份有限公司", "1202084984"),
        "000423": ("东阿阿胶股份有限公司", "1202202869"),
        "000538": ("云南白药集团股份有限公司", "1202202875"),
        "000895": ("河南双汇投资发展股份有限公司", "1202098793"),
        "000963": ("华东医药股份有限公司", "1202202622"),
        "002032": ("浙江苏泊尔股份有限公司", "1202225678"),
        "002415": ("杭州海康威视数字技术股份有限公司", "1202157218"),
    }
    donors = {}
    for code, (name, donor_id) in aliases.items():
        path = OUT / "company-documents" / (donor_id + ".json")
        donor = read(path)
        text = normalize(donor["pages"][0])
        assert code in text and name in text and donor["row"]["secCode"] == code
        donors[code] = {
            "name": name,
            "path": str(path),
            "sha256": sha(path),
            "page": 1,
            "company_offset": text.index(name),
            "stock_offset": text.index(code),
        }
    results = []
    for path in sorted((OUT / "company-documents").glob("*.json")):
        doc = read(path)
        assert doc["body_saved"] and sha(doc["receipt"]["path"]) == doc["receipt"]["sha256"]
        row, anchors = doc["row"], []
        if any(word in row["title_plain"] for word in ("问询函", "专项说明")):
            results.append(
                {
                    "id": path.stem,
                    "status": "SUPPORTING_DOCUMENT_NOT_PRIMARY_RESULT",
                    "title": row["title_plain"],
                    "not_deleted": True,
                }
            )
            continue
        for page, raw in enumerate(doc["pages"], 1):
            text = normalize(raw)
            match = re.search(
                r"(?:股票代码|证券代码|公司代码|A股代码|[Ss](?:tock|ecurities)[Cc]ode)[^。]{0,25}" + row["secCode"],
                text,
            )
            if match:
                anchors.append({"page": page, "text": match[0], "offset": match.start()})
                break
        title = normalize(row["title_plain"])
        cover = normalize("".join(doc["pages"][:2]))
        name = donors.get(row["secCode"])
        title_match = title in cover or title.replace("全文", "") in cover
        if "英文版" in title and anchors:
            year = re.search(r"20\d{2}", title)[0]
            title_match = year in cover and "AnnualReport" in cover and "年度报告" in title
        company_match = bool(anchors) or bool(name and name["name"] in cover)
        passed = company_match and title_match and not doc["revision_issues"]
        results.append(
            {
                "id": path.stem,
                "title": row["title_plain"],
                "status": "SOURCE_IDENTITY_VERIFIED" if passed else "IDENTITY_UNRESOLVED",
                "stock_anchors": anchors,
                "corporate_name_evidence": name,
                "company_match": company_match,
                "report_identity_match": title_match,
                "semantic_verified": False,
            }
        )
    save(OUT / "company-identity-audit-v3.json", results)
    return dict(Counter(r["status"] for r in results))


def run():
    cov = independent_catalog_audit()
    identities = identity_audit()
    facts = read(OUT / "reviewed-facts-v1.json")
    specs = read(OUT / "review-specs-v1.json")["reviews"]
    assert len(facts) == len(specs) == 12
    fields = 0
    for fact in facts:
        assert sha(fact["source"]["path"]) == fact["source"]["sha256"]
        assert fact["available_at"][:10] > fact["published_date"]
        fields += len(fact["fields"])
    public = read(OUT / "public-history-v2/result.json")
    receipts = [read(p) for p in (OUT / "requests").glob("*.json")]
    result = {
        "passed": True,
        "company_coverage_gained": cov["gained_count"],
        "company_source_identity": identities,
        "reviewed_fields": fields,
        "policy_pages_saved": public["articles_saved_and_identity_passed"],
        "requests_used": dict(Counter(r["group"] for r in receipts)),
        "model_readiness": False,
        "new_fits": 0,
        "cumulative_fits": 70,
    }
    save(OUT / "independent-audit-final-v2.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
