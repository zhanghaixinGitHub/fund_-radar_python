"""穿透链使用动态目标、复用同版原件，单只失败不阻断其他目标。"""

from contextlib import contextmanager
from types import SimpleNamespace

from app.services import fund_prediction_lookthrough as flow
from app.services.direction_1d_protocol import digest


def test_dynamic_targets_reuse_original_and_isolate_failure(tmp_path, monkeypatch):
    from app.integrations import market_fund_reports as source
    from app.services import fund_exposure_runtime as locks
    from app.services import fund_holdings_publication as publication
    from app.services import market_holdings_parse as parser

    snapshots, downloaded = {}, []

    class Client:
        def __init__(self, root):
            pass

        def close(self):
            pass

        def catalog(self, code):
            return [
                {
                    "id": "AN" + code,
                    "end": "2026-09-30",
                    "published": "2026-10-27",
                    "kind": "QUARTER",
                    "title": code + "目标基金2026年第3季度报告",
                }
            ]

        def body(self, code, entry, previous):
            downloaded.append(code)
            return {"sha256": "original-" + code, "file": code + ".json"}

    @contextmanager
    def locked():
        yield True

    def parse(client, root, fund, entry, ref):
        if fund["fund_code"] == "222222":
            raise ValueError("ACTUAL_PARSE_ERROR")
        return {"id": ref["sha256"], "endDate": entry["end"], "publishedDate": entry["published"], "holdings": []}, []

    def publish(code, report, directory):
        snapshots.setdefault(code, {"reports": [{"id": "old-" + code}]})["reports"].append(report)
        return snapshots[code]

    monkeypatch.setattr(flow, "get_settings", lambda: SimpleNamespace(fund_material_directory=str(tmp_path)))
    monkeypatch.setattr(flow, "snapshot", lambda code: snapshots.get(code))
    monkeypatch.setattr(source, "MarketReportClient", Client)
    monkeypatch.setattr(locks, "execution_lock", locked)
    monkeypatch.setattr(parser, "parse_report", parse)
    monkeypatch.setattr(publication, "publish_report", publish)
    material = {
        "fund_code": "001632",
        "exposure_chain": ["001632", "111111", "222222", "333333"],
        "documents": [
            {
                "id": c,
                "kind": "REPORT",
                "codes": [c],
                "title": c + "目标基金2026年中期报告",
                "published_date": "2026-08-31",
                "body": c + "原报告",
                "body_hash": digest(c + "原报告"),
            }
            for c in ["111111", "222222", "333333"]
        ],
    }
    rows = flow.recover_target_reports(material, tmp_path / "owner")
    assert [r["status"] for r in rows] == ["PUBLISHED", "ACTION_REQUIRED", "PUBLISHED"]
    assert snapshots["111111"]["reports"][0]["id"] == "old-111111"

    # 不重下成功目标；失败目标重试解析也复用先前已经取得的正文回执。
    class RetryClient(Client):
        def body(self, code, entry, previous):
            return previous["source"] if previous and previous.get("source") else super().body(code, entry, previous)

    monkeypatch.setattr(source, "MarketReportClient", RetryClient)
    rows = flow.recover_target_reports(material, tmp_path / "owner")
    assert [r["status"] for r in rows] == ["ORIGINAL_REUSED", "ACTION_REQUIRED", "ORIGINAL_REUSED"]
    assert downloaded == ["111111", "222222", "333333"]


def test_chain_cannot_invent_an_unanchored_target(tmp_path, monkeypatch):
    from app.integrations import market_fund_reports as source
    from app.services import fund_exposure_runtime as locks

    @contextmanager
    def locked():
        yield True

    class Client:
        def __init__(self, root):
            pass

        def close(self):
            pass

        def catalog(self, code):
            raise AssertionError("未核身份不应请求目录")

    monkeypatch.setattr(flow, "get_settings", lambda: SimpleNamespace(fund_material_directory=str(tmp_path)))
    monkeypatch.setattr(source, "MarketReportClient", Client)
    monkeypatch.setattr(locks, "execution_lock", locked)
    rows = flow.recover_target_reports(
        {"fund_code": "001632", "exposure_chain": ["001632", "159736"], "documents": []}, tmp_path
    )
    assert rows[0]["failure"] == "ValueError:PREDICTION_LOOKTHROUGH_ORIGINAL_ANCHOR_REQUIRED"
