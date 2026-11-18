import hashlib
import json

import httpx
import pytest
from app.integrations import cmf_public as cmf
from app.services.fund_exposure_common import read

SECRET = "b" * 32
OUTER_ID = "a" * 32
PAGE = '<script>window.__NUXT__={config:{public:{siteno:"web",merchantId:"0"}}};</script>'
CLIENT = (
    'const Ns=()=>"' + OUTER_ID + '";this.merchantKey=s.merchantKey||"thinkive";$s(n,"' + SECRET + '");Dt.sm4.encrypt'
)
PARAMS = {"productCode": "016008", "startDate": "2023-10-31", "endDate": "2026-10-30", "queryMonth": ""}


def test_replays_public_source_protocol_with_fresh_inner_nonce():
    first = cmf._public_query(PAGE, CLIENT, PARAMS)
    second = cmf._public_query(PAGE, CLIENT, PARAMS)
    assert first["params"]["data"] != second["params"]["data"]
    payload = first["params"]
    message = f"secret={SECRET}data={payload['data']}request_id={OUTER_ID}encrypted=true"
    expected = hashlib.new("sm3", message.encode()).hexdigest()
    assert first["headers"]["tk-trans-signature"] == expected
    assert payload["request_id"] == OUTER_ID


def test_changed_client_or_missing_public_ssr_fails_closed():
    with pytest.raises(ValueError, match="NOT_REPLAYABLE"):
        cmf._public_query(PAGE, "no current public protocol", PARAMS)
    with pytest.raises(ValueError, match="NOT_REPLAYABLE"):
        cmf._public_query("ordinary login page", CLIENT, PARAMS)


def test_preserves_raw_response_and_normal_request_identity_without_signatures(tmp_path):
    raw = b'{"code":0,"data":[{"date":"2023-10-31","navYield":"0.00"}]}'
    sent = []

    def respond(request):
        url = str(request.url).split("?", 1)[0]
        if url == cmf.INCOME_URL:
            sent.append(request)
            assert request.url.params["encrypted"] == "true"
            return httpx.Response(200, content=raw)
        text = (
            CLIENT
            if url == cmf.CLIENT_URL
            else ('n.queryMonth="";format("YYYY-MM-DD")' if url == cmf.CHART_URL else PAGE)
        )
        return httpx.Response(200, text=text)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        response = cmf.income_list(client, PARAMS, audit_directory=tmp_path)
    assert response.content == raw
    receipt = read(tmp_path / "anonymous-request-receipt.json")
    assert receipt["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["params"] == PARAMS and receipt["response_reencoded"] is False
    assert len(receipt["public_protocol_sources"]) == 3
    serialized = json.dumps(receipt)
    assert sent[0].headers["tk-trans-signature"] not in serialized
    assert sent[0].url.params["data"] not in serialized


def test_preset_window_never_silently_replaces_explicit_dates(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("unexpected request"))) as client:
        with pytest.raises(ValueError, match="EXPLICIT_DATE_WINDOW"):
            cmf.income_list(client, {**PARAMS, "queryMonth": "36"}, audit_directory=tmp_path)
