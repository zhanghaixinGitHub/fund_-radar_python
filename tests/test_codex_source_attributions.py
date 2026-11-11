from copy import deepcopy

from app.services.fund_codex_analysis import catalog


def test_same_provider_attribution_is_deduplicated_without_changing_frozen_input():
    source = {"title": "本证券实际基准日行情", "url": "https://api.tushare.pro", "publishedDate": "2026-11-10"}
    second = {**source, "publishedDate": "2026-11-09"}
    fact = {
        "category": "市场行情",
        "text": "两只证券的独立原始响应",
        "source": source,
        "sources": [source, source.copy(), second],
        "source_refs": ["stock-a", "stock-b", "previous"],
    }
    data = {"documents": [], "companies": [], "facts": {"current_holdings": fact}}
    original = deepcopy(data)
    facts, _, _ = catalog(data, {"documents": []})
    result = next(iter(facts.values()))
    assert result["sources"] == [source, second]
    assert result["source_refs"] == ["stock-a", "stock-b", "previous"]
    assert data == original
