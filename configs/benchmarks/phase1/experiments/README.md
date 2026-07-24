# Phase 1 Experiment Configs

These profiles preserve historical Phase 1 policy experiments. They are not accepted by the
stable `medical-kg benchmark phase1 submission` command, whose export policies are deliberately
limited to `empty` and `pipeline`.

Use the APIs in `medical_kg_nlp.benchmarks.phase1` when reproducing a selective experiment. Do not
treat the reviewed maps or calibration values here as general clinical defaults: they were derived
from opened Phase 1 data and are not an independent policy holdout.
