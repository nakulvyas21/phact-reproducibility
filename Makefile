# Reproducibility targets for the PHACT paper.
# Everything here runs with the Python standard library alone (pytest for tests).
# No API keys, no network, no language model.

PY ?= python3

.PHONY: help demo tables test calibrate verify clean

help:
	@echo "PHACT reproducibility - available targets:"
	@echo "  make demo       30-second tour of the certification surface (no API key)"
	@echo "  make tables     Regenerate the paper tables from the archived runs"
	@echo "  make calibrate  Verify physics engines match published literature"
	@echo "  make test       Run the full deterministic test suite"
	@echo "  make verify     calibrate + test + tables (the full check)"
	@echo "  make clean      Remove caches"

# A short, runnable tour of the tool specs the model sees + the engine verdict
demo:
	$(PY) demo.py

# Regenerate the paper's result tables from the archived verdicts in results/
tables:
	$(PY) regenerate_tables.py

# Engine-vs-literature calibration only
calibrate:
	$(PY) -m pytest calibration/ -v

# Full deterministic test suite (calibration + adversarial impossibility)
test:
	$(PY) -m pytest calibration/ tests/ -q

# The complete reproducibility check a reviewer should run
verify: calibrate test tables
	@echo ""
	@echo "All checks passed. Engines match the literature, the impossible"
	@echo "goals are verifiably impossible, and the paper tables regenerate."

clean:
	rm -rf .pytest_cache **/__pycache__ */**/__pycache__
