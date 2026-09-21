.NOTPARALLEL:
PYTHON ?= .venv/bin/python
DATA_DIR ?=
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS := 4
export VECLIB_MAXIMUM_THREADS := 4

.PHONY: prepare audit benchmark ablations final memory figures verify test all
prepare:
	@test -n "$(DATA_DIR)" || (echo 'Set DATA_DIR to the original Kaggle download directory'; exit 1)
	$(PYTHON) scripts/prepare_data.py --data-dir "$(DATA_DIR)"
audit:
	$(PYTHON) scripts/audit_data.py
benchmark:
	$(PYTHON) scripts/run_benchmark.py --device mps
ablations:
	$(PYTHON) scripts/run_ablations.py
final:
	$(PYTHON) scripts/refit_predict.py --device mps
memory:
	$(PYTHON) scripts/profile_memory.py
figures:
	$(PYTHON) scripts/make_figures.py
verify:
	$(PYTHON) scripts/verify_results.py
test:
	$(PYTHON) -m pytest -q
all: prepare audit benchmark ablations final memory verify figures test
