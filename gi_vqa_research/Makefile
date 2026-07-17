PYTHON ?= python3
CONFIG ?= configs/study1/smoke.yaml
RUN_DIR ?= runs/local-smoke

.PHONY: help install install-gpu test check-model-config audit manifest merge-shards docker-build docker-shell

help:
	@echo "install       Install the local package"
	@echo "install-gpu   Install the provisional GPU dependency reference stack"
	@echo "test          Run the standard-library unit-test suite"
	@echo "check-model-config  Validate the complete shared-backend configuration"
	@echo "audit         Audit JSONL splits (set SPLITS='train.jsonl dev.jsonl test.jsonl')"
	@echo "manifest      Capture a run manifest in RUN_DIR"
	@echo "merge-shards  Merge result shards (set SHARDS, OUTPUT and ID_FIELD)"
	@echo "docker-build  Build the pinned GPU research image"
	@echo "docker-shell  Open the repository in the GPU image"

install:
	$(PYTHON) -m pip install -e .

install-gpu:
	$(PYTHON) -m pip install -e ".[gpu]"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

check-model-config:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli config-check \
		--config $(CONFIG) --model-execution

audit:
	PYTHONPATH=src $(PYTHON) scripts/audit_splits.py $(SPLITS)

manifest:
	PYTHONPATH=src $(PYTHON) scripts/capture_run_manifest.py --config $(CONFIG) --run-dir $(RUN_DIR)

merge-shards:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli merge-shards \
		$(foreach shard,$(SHARDS),--shard $(shard)) --output $(OUTPUT) --id-field $(ID_FIELD)

docker-build:
	docker build -f docker/Dockerfile.gpu -t gi-vqa-research:gpu .

docker-shell:
	docker run --rm -it --gpus all -v "$$(pwd):/workspace/gi_vqa_research" \
		-w /workspace/gi_vqa_research gi-vqa-research:gpu bash
