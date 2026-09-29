"""只读取登记的两份独立历史副本，核对旧资料的版本冲突；不自动解除隔离。"""

import hashlib
from datetime import datetime

import httpx
from app.services.fund_earnings_batch_v4 import extract_pdf
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

from scripts.fund_002112_closure_v1 import OUT

SOURCES = (
    {
        "key": "eve-2021-annual-sina",
        "url": "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESZ_STOCK/2022/2022-4/2022-04-26/8073228.PDF",
        "listing": "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=8073228",
        "published_date": "2022-04-26",
        "title": "2021年年度报告",
        "quarantined_document": "1218857644",
    },
    {
        "key": "eve-2022-half-xueqiu",
        "url": "https://stockn.xueqiu.com/SZ300014/20220825493777.pdf",
        "listing": None,
        "published_date": "2022-08-26",
        "title": "2022年半年度报告",
        "quarantined_document": "1218857711",
    },
)


def run():
    """请求前保存不可覆盖计划和消费凭据；失败不重试，原隔离文件保持不变。"""
    target = OUT / "independent-copies"
    plan = {
        "sources": SOURCES,
        "maximum_requests": 2,
        "maximum_bytes_per_document": 16 * 1024 * 1024,
        "maximum_pages": 500,
        "source_discovery": "2026-09-29 public search and observed download links",
        "code_sha256": sha(__file__),
        "no_quarantined_source_redownload": True,
        "new_fits": 0,
    }
    # JSON 固化后列表与元组等价，重入以固化版为准。
    if not (target / "plan.json").exists():
        save(target / "plan.json", plan)
    frozen = read(target / "plan.json")
    if frozen["code_sha256"] != sha(__file__):
        raise ValueError("CODE_CHANGED_AFTER_FREEZE")
    results = []
    with httpx.Client(timeout=httpx.Timeout(35, connect=8), follow_redirects=False, trust_env=False) as client:
        for spec in frozen["sources"]:
            receipt_path = target / (spec["key"] + "-receipt.json")
            if receipt_path.exists():
                results.append(read(receipt_path))
                continue
            started = target / (spec["key"] + "-started.json")
            if started.exists():
                raise ValueError("CONSUMED_SOURCE_WITHOUT_RECEIPT_NO_RETRY")
            save(started, {"url": spec["url"], "at": datetime.now().astimezone().isoformat()})
            result = {**spec, "downloaded": False, "training_ready": False}
            try:
                with client.stream("GET", spec["url"]) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > frozen["maximum_bytes_per_document"]:
                            raise ValueError("SOURCE_SIZE_LIMIT")
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                content_sha = hashlib.sha256(raw).hexdigest()
                path = target / (content_sha + ".pdf")
                if not path.exists():
                    with path.open("xb") as stream:
                        stream.write(raw)
                pages, metadata = extract_pdf(raw, frozen["maximum_pages"])
                body = {"pages": pages, "metadata": metadata}
                save(target / (spec["key"] + "-body.json"), body)
                result.update(
                    downloaded=True,
                    path=str(path),
                    sha256=content_sha,
                    pages=len(pages),
                    metadata=metadata,
                    revision_issues=pdf_revision(metadata, spec["published_date"]),
                    cover_title_verified=spec["title"] in normalize(pages[0]),
                    issuer_name_verified="惠州亿纬锂能股份有限公司" in normalize(pages[0]),
                    original_publication_proof="Independent copy metadata and displayed dates require review",
                )
            except (httpx.HTTPError, ValueError) as exc:
                result["reason"] = str(exc)
            save(receipt_path, result)
            results.append(result)
    save(target / "result.json", {"copies": results, "old_quarantine_preserved": True, "new_fits": 0})
    print(
        [
            {k: r.get(k) for k in ("key", "downloaded", "pages", "metadata", "revision_issues", "reason")}
            for r in results
        ]
    )


if __name__ == "__main__":
    run()
