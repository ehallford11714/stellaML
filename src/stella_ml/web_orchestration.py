"""Web search orchestration and lightweight skill acquisition for agent harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable



@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    tool_hint: str


@dataclass(slots=True)
class SkillCatalog:
    skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def list_skills(self) -> list[Skill]:
        return list(self.skills.values())


def web_search(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """Perform a lightweight web search (DuckDuckGo instant answer API)."""
    endpoint = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
    import requests

    response = requests.get(endpoint, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    results: list[WebSearchResult] = []
    related = data.get("RelatedTopics", [])
    for item in related:
        if "Text" in item and "FirstURL" in item:
            results.append(WebSearchResult(title=item["Text"][:80], url=item["FirstURL"], snippet=item["Text"]))
        for nested in item.get("Topics", []):
            if "Text" in nested and "FirstURL" in nested:
                results.append(WebSearchResult(title=nested["Text"][:80], url=nested["FirstURL"], snippet=nested["Text"]))
        if len(results) >= max_results:
            break

    return results[:max_results]


def acquire_skills_from_results(results: list[WebSearchResult]) -> SkillCatalog:
    """Infer useful skills from search results for agentic tool planning."""
    catalog = SkillCatalog()

    for result in results:
        text = f"{result.title} {result.snippet}".lower()
        if "selenium" in text:
            catalog.register(Skill("browser_automation", "Automate JS-rendered web extraction", "selenium"))
        if "beautifulsoup" in text or "html" in text:
            catalog.register(Skill("html_parsing", "Extract content from raw HTML", "beautifulsoup"))
        if "xml" in text:
            catalog.register(Skill("xml_parsing", "Parse XML into text payloads", "ElementTree"))
        if "hugging face" in text or "llm" in text:
            catalog.register(Skill("model_selection", "Select model family from current web evidence", "huggingface_api"))
        if "pymc" in text or "bayesian" in text:
            catalog.register(Skill("bayesian_modeling", "Probabilistic model inference", "pymc"))

    return catalog


def web_search_orchestration(query: str, max_results: int = 5) -> dict[str, object]:
    results = web_search(query, max_results=max_results)
    skills = acquire_skills_from_results(results)
    return {
        "query": query,
        "results": results,
        "skills": skills.list_skills(),
    }
