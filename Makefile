# MSHP Coverage Gap - Makefile
#
# Usage:
#   make all        - Run full pipeline
#   make collect    - Run data collection (scripts 01-04)
#   make process    - Run data processing (scripts 05-07)
#   make analyze    - Run analysis (scripts 08-09)
#   make visualize  - Generate visualizations (script 10)
#   make test       - Run validation tests
#   make clean      - Remove generated files

.PHONY: all collect process analyze visualize test clean help

PYTHON := python
SCRIPTS := scripts

help:
	@echo "MSHP Coverage Gap - Commands"
	@echo ""
	@echo "  make all        Run full pipeline"
	@echo "  make collect    Data collection (01-04)"
	@echo "  make process    Data processing (05-07)"
	@echo "  make analyze    Analysis (08-09)"
	@echo "  make visualize  Visualization (10)"
	@echo "  make test       Run tests"
	@echo "  make clean      Remove generated outputs"

# Individual steps
01-doe-schools:
	$(PYTHON) $(SCRIPTS)/01_collect_doe_schools.py

02-mshp-list:
	$(PYTHON) $(SCRIPTS)/02_collect_mshp_list.py

03-attendance:
	$(PYTHON) $(SCRIPTS)/03_collect_attendance.py

04-asthma:
	$(PYTHON) $(SCRIPTS)/04_collect_asthma_data.py

05-process: 01-doe-schools 02-mshp-list
	$(PYTHON) $(SCRIPTS)/05_process_schools.py

06-match: 05-process
	$(PYTHON) $(SCRIPTS)/06_match_mshp.py

07-spatial: 06-match 03-attendance 04-asthma
	$(PYTHON) $(SCRIPTS)/07_spatial_join.py

08-analyze: 07-spatial
	$(PYTHON) $(SCRIPTS)/08_analyze.py

09-priority: 08-analyze
	$(PYTHON) $(SCRIPTS)/09_priority_ranking.py

10-visualize: 09-priority
	$(PYTHON) $(SCRIPTS)/10_visualize.py

# Aggregate targets
collect: 01-doe-schools 02-mshp-list 03-attendance 04-asthma
	@echo "Data collection complete"

process: 07-spatial
	@echo "Data processing complete"

analyze: 09-priority
	@echo "Analysis complete"

visualize: 10-visualize
	@echo "Visualization complete"

all: visualize
	@echo "Full pipeline complete"

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/*
	rm -rf outputs/figures/*
	rm -rf outputs/interactive/*
	rm -rf outputs/tables/*
	rm -rf logs/*.jsonl
	@echo "Cleaned generated outputs"

