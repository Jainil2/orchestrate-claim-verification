# Shortcut targets for the test -> improve -> finalize -> submit loop.
# Override the interpreter: make run PY=/path/to/venv/bin/python
PY ?= python

.PHONY: install test eval eval-quick smoke run run-fresh trace package clean help

help:
	@echo "install     install Python deps"
	@echo "test        offline engine self-test (no API key)"
	@echo "eval        evaluate on labeled sample_claims.csv + ablations -> evaluation_report.md"
	@echo "eval-quick  fast eval on 5 sample rows"
	@echo "smoke       run on 3 test claims (sanity + cost check)"
	@echo "run         full test set -> output.csv (44 rows)"
	@echo "run-fresh   full run ignoring the result cache (recompute)"
	@echo "trace       full run + dump per-claim reasoning to code/logs/trace/"
	@echo "package     build code.zip for submission"
	@echo "clean       remove cache, traces, and output.csv"

install:
	$(PY) -m pip install -r code/requirements.txt

# --- validation set (sample_claims.csv has gold labels) ---
test:
	$(PY) code/test_engine.py

eval:
	$(PY) code/evaluation/main.py --ablations

eval-quick:
	$(PY) code/evaluation/main.py --limit 5

# --- test set (claims.csv -> output.csv) ---
smoke:
	$(PY) code/main.py --limit 3

run:
	$(PY) code/main.py

run-fresh:
	$(PY) code/main.py --no-cache

trace:
	$(PY) code/main.py --trace

# --- submission ---
package:
	rm -f code.zip
	zip -r code.zip code pyproject.toml -x '*/__pycache__/*' '*/.cache/*' '*/logs/*' '*.pyc' '*/.DS_Store' '*/.env'
	@echo "Built code.zip — submit alongside output.csv and the chat_transcript (\$$HOME/hackerrank_orchestrate/log.txt)."

clean:
	rm -rf code/.cache code/logs output.csv
