# stellaML

`stellaML` is a starter framework for model APIs + agentic orchestration + structured/unstructured data intelligence.

## Install

```bash
pip install -e .
pip install -e '.[analytics]'
pip install -e '.[ecosystem]'
pip install -e '.[unstructured]'
```

## New: local SOTA model selection + API/local switch prompt

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()

# 1) Latest web search of HF SOTA (downloads-based) + hardware fit recommendation.
recommendation = harness.propose_api_vs_local_switch(task="text-generation")
print(recommendation["recommendation"])
print(recommendation["prompt_user"])  # Ask user: continue API or switch local

# 2) If HF API key is missing, request/save key to config.
message = harness.ensure_hf_key(agent_name="analyst", api_key=None)
print(message)
# -> prompts for API key or local mode fallback
```

## Web search orchestration + skill acquisition

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
search_plan = harness.run_web_search_skill_acquisition(
    query="best methods for extracting article text and entity extraction",
    max_results=5,
)
print(search_plan["results"])
print(search_plan["skills"])  # auto-acquired skills for tool planning
```

## Unstructured extraction core (requests, BeautifulSoup, Selenium, HTML/XML/PPTX)

```python
from stella_ml import OpenClawStyleHarness

harness = OpenClawStyleHarness()
out = harness.run_unstructured_data_flow(
    source="https://example.com",
    source_type="url",
    fetch_mode="requests",  # or 'selenium' for JS-rendered pages
    ngram_n=2,
)
print(out["text_eda"])
print(out["ngrams"][:10])
print(out["entities"][:10])
```

## Notes

- Architecture summary + expansion roadmap: `docs/framework_roadmap.md`.
- HF local recommendation is heuristic and uses current web pull from HF model API.
- `ensure_hf_key` can write API keys to config for existing agent profiles.
