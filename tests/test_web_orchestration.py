from stella_ml.web_orchestration import acquire_skills_from_results, web_search_orchestration, WebSearchResult


def test_acquire_skills_from_results() -> None:
    results = [
        WebSearchResult(title="Selenium tutorial", url="u1", snippet="selenium html parsing"),
        WebSearchResult(title="PyMC guide", url="u2", snippet="bayesian pymc"),
    ]
    catalog = acquire_skills_from_results(results)
    names = [s.name for s in catalog.list_skills()]
    assert "browser_automation" in names
    assert "bayesian_modeling" in names


def test_web_search_orchestration_with_mock(monkeypatch) -> None:
    from stella_ml import web_orchestration

    monkeypatch.setattr(
        web_orchestration,
        "web_search",
        lambda query, max_results=5: [
            WebSearchResult(title="BeautifulSoup", url="u", snippet="html beautifulsoup")
        ],
    )
    out = web_search_orchestration("html extraction")
    assert out["results"]
    assert out["skills"]
