# Credit risk underwriting pipeline.
#
# `make all` runs the whole thing from nothing. The download is the slow part
# (1.56 GB, about 20 minutes on a home connection) and it resumes, so a failed
# run does not start over.
#
# For a quick look without the full download:
#   make sample   # 200k rows, everything downstream runs in a couple of minutes

PYTHON ?= python

.PHONY: help all sample data warehouse leakage train monitor charts report \
        test lint api app clean clean-data mlflow

help:
	@echo "make all        full pipeline: download, warehouse, experiment, train, monitor, charts"
	@echo "make sample     same but on a 200k-row slice, for a fast first look"
	@echo ""
	@echo "make data       download the Lending Club loan book (1.56 GB, resumable)"
	@echo "make warehouse  load into DuckDB and build the modelling population"
	@echo "make leakage    the 2x2 leakage experiment"
	@echo "make train      baseline, scorecard and LightGBM, with calibration"
	@echo "make monitor    PSI, characteristic analysis and the promotion gate"
	@echo "make charts     regenerate the figures in docs/assets"
	@echo "make report     build the PDF summary"
	@echo ""
	@echo "make test       pytest"
	@echo "make lint       ruff"
	@echo "make api        serve the scoring API on :8000"
	@echo "make app        the interactive Streamlit demo"
	@echo "make mlflow     open the MLflow run history"
	@echo "make clean      remove derived data, keep the download"

all: data warehouse leakage train monitor charts report

# A 200k-row slice is enough to exercise every code path and to see the leakage
# result clearly. The numbers in the docs come from the full file.
sample:
	$(PYTHON) -m src.get_data --rows 200000
	$(PYTHON) -m src.build_warehouse
	$(PYTHON) -m src.leakage_experiment
	$(PYTHON) -m src.train
	$(PYTHON) -m src.monitor
	$(PYTHON) -m src.charts

data:
	$(PYTHON) -m src.get_data

warehouse:
	$(PYTHON) -m src.build_warehouse

leakage:
	$(PYTHON) -m src.leakage_experiment

train:
	$(PYTHON) -m src.train

monitor:
	$(PYTHON) -m src.monitor

charts:
	$(PYTHON) -m src.charts

report:
	$(PYTHON) -m src.report

test:
	$(PYTHON) -m pytest tests -v

lint:
	$(PYTHON) -m ruff check src tests app

api:
	$(PYTHON) -m uvicorn src.serve:app --reload --port 8000

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

mlflow:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# Leaves data/raw alone. Re-downloading 1.56 GB to recover from a typo in a
# chart is not a reasonable thing to ask of anyone.
clean:
	$(PYTHON) -c "import shutil, os; \
	[shutil.rmtree(p, ignore_errors=True) for p in \
	 ('data/warehouse', 'data/models', 'data/reports')]; \
	print('removed warehouse, models and reports')"

clean-data: clean
	$(PYTHON) -c "import shutil; shutil.rmtree('data/raw', ignore_errors=True); \
	print('removed the download too')"
