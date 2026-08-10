# ClinGrounder Demo

This is an optional local Streamlit demo. It uses the bundled `vi-clinical-small` resource pack,
so the example does not download models or send text to a hosted service.

## Run From PyPI

```bash
python -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements.txt
.venv-demo/bin/streamlit run app.py
```

## Run From A Checkout

```bash
uv sync --extra dev
uv pip install streamlit
uv run streamlit run examples/demo/app.py
```

The demo shows raw-offset-safe entity highlights, assertion status, assigned terminology codes,
top-k candidate provenance and qualification reasons, relation evidence, and per-stage latency.
It is an inspection surface, not a clinical decision-support application.
