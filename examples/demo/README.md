# ClinGrounder demo

This is an optional local Streamlit viewer for the bundled `vi-clinical-small` resource pack. It
is an example application, not part of the core runtime and not a clinical decision-support tool.

## Run

From the repository root:

```bash
python -m venv .venv-demo
.venv-demo/bin/pip install -e "[vi]"
.venv-demo/bin/pip install -r examples/demo/requirements.txt
.venv-demo/bin/streamlit run examples/demo/app.py
```

The app runs offline and shows raw spans, entity types, assertion status, terminology codes,
candidate codes, and relations. Replace the bundled example only with text that is appropriate for
your local environment; the demo does not provide PHI controls or regulatory compliance.
