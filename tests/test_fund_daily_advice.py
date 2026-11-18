from types import SimpleNamespace

import httpx
from app.services import fund_daily_advice as advice


class Connection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, statement, parameters):
        return SimpleNamespace(scalar_one=lambda: {
            "fundCode": parameters["code"], "strategyVersion": advice.VERSION,
            "reportId": "saved", "adviceState": "INSUFFICIENT", "decision": None,
        })


class Engine:
    def connect(self):
        return Connection()

    def dispose(self):
        pass


def test_pending_new_period_does_not_accept_complete_advice_from_old_inputs(monkeypatch):
    monkeypatch.setattr(advice, "owner_scope", lambda *args: ["001632"])
    calls = []
    value = {"fundCode": "001632", "strategyVersion": advice.VERSION, "reportId": "saved",
             "adviceState": "READY", "decision": "HOLD"}
    class Saved(Connection):
        def execute(self, statement, parameters):
            return SimpleNamespace(scalar_one=lambda: value)
    class SavedEngine(Engine):
        def connect(self):
            return Saved()
    def post(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, request=httpx.Request("POST", url), json=value)
    backend = SimpleNamespace(settings=SimpleNamespace(core_service_base_url="http://local"), owner="owner",
                              client=SimpleNamespace(post=post))
    result = advice.generate_advice(backend, ["001632"], {"001632": [{"step": "MONTHLY_RATING"}]},
                                    engine_factory=SavedEngine)
    assert len(calls) == 1
    assert result["counts"] == {"WAITING_INPUTS": 1}
    assert result["reports_read_back"] == 1 and result["status"] == "ACTION_REQUIRED"


def test_pending_inputs_still_save_qualified_risk_reduction_without_closing_gaps(monkeypatch):
    monkeypatch.setattr(advice, "owner_scope", lambda *args: ["001632"])
    value = {"fundCode": "001632", "strategyVersion": advice.VERSION, "reportId": "saved",
             "adviceState": "QUALIFIED", "decision": "REDUCE", "missingOptionalFactors": ["本期评分仍缺项"]}
    class Saved(Connection):
        def execute(self, statement, parameters):
            return SimpleNamespace(scalar_one=lambda: value)
    class SavedEngine(Engine):
        def connect(self):
            return Saved()
    backend = SimpleNamespace(settings=SimpleNamespace(core_service_base_url="http://local"), owner="owner",
        client=SimpleNamespace(post=lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("POST", url), json=value)))
    result = advice.generate_advice(backend, ["001632"], {"001632": [{"step": "MONTHLY_RATING"}]},
                                   engine_factory=SavedEngine)
    assert result["counts"] == {"QUALIFIED": 1}
    assert result["reports_read_back"] == 1 and result["status"] == "ACTION_REQUIRED"
    assert result["items"][0]["pending_dependencies"] == [{"step": "MONTHLY_RATING"}]
    assert result["items"][0]["current_period_complete"] is False


def test_saved_report_is_not_current_when_authoritative_input_check_changed():
    saved = {"fundCode": "001632", "strategyVersion": advice.VERSION, "reportId": "saved",
             "adviceState": "QUALIFIED", "decision": "REDUCE"}
    class ReadConnection(Connection):
        def execute(self, statement, parameters):
            return SimpleNamespace(scalar_one_or_none=lambda: saved)
    class ReadEngine(Engine):
        def connect(self):
            return ReadConnection()
    backend = SimpleNamespace(settings=SimpleNamespace(core_service_base_url="http://local"),
        get=lambda url: {**saved, "applicability": "CHANGED"})
    receipt = {"items": [{"fund_code": "001632", "report_id": "saved", "decision": "REDUCE",
                          "advice_state": "QUALIFIED", "status": "QUALIFIED"}]}
    row = advice.read_saved_advice("owner", receipt, backend=backend, engine_factory=ReadEngine)[0]
    assert row["read_back"] is True and row["inputs_current"] is False
    assert row["applicability"] == "CHANGED"


def test_one_fund_failure_does_not_prevent_other_funds_database_readback(monkeypatch):
    monkeypatch.setattr(advice, "owner_scope", lambda *args: ["001632", "006730"])

    def post(url, **kwargs):
        code = url.split("/")[-2]
        response = httpx.Response(503 if code == "001632" else 200, request=httpx.Request("POST", url), json={
            "fundCode": code, "strategyVersion": advice.VERSION,
            "reportId": "saved", "adviceState": "INSUFFICIENT", "decision": None,
        })
        return response

    backend = SimpleNamespace(settings=SimpleNamespace(core_service_base_url="http://local"), owner="owner",
                              client=SimpleNamespace(post=post))
    result = advice.generate_advice(backend, ["001632", "006730"], {}, engine_factory=Engine)
    assert result["counts"] == {"FAILED": 1, "INSUFFICIENT": 1}
    assert result["items"][1]["read_back"] is True


def test_qualified_direction_is_not_misreported_as_no_advice(monkeypatch):
    monkeypatch.setattr(advice, "owner_scope", lambda *args: ["013275"])
    value = {"fundCode": "013275", "strategyVersion": advice.VERSION, "reportId": "saved",
             "adviceState": "QUALIFIED", "decision": "REDUCE"}
    class Saved(Connection):
        def execute(self, statement, parameters):
            return SimpleNamespace(scalar_one=lambda: value)
    class SavedEngine(Engine):
        def connect(self):
            return Saved()
    backend = SimpleNamespace(settings=SimpleNamespace(core_service_base_url="http://local"), owner="owner",
        client=SimpleNamespace(post=lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("POST", url), json=value)))
    result = advice.generate_advice(backend, ["013275"], {}, engine_factory=SavedEngine)
    assert result["counts"] == {"QUALIFIED": 1} and result["actionable_count"] == 1
