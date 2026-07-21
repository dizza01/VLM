PYTHON ?= python3
CONFIG ?= configs/study1/smoke.yaml
RUN_DIR ?= runs/local-smoke
GATE_WORK_DIR ?= runs/training-gate-work
GATE_ARTIFACT_DIR ?= runs/training-gate-evidence
SMOKE_RUN_DIR ?= runs/development-smoke-20

.PHONY: help install install-data install-gpu test check-model-config prepare-splits materialize-splits split-check prepare-image-cache materialize-image-cache image-cache-check training-gate development-smoke audit manifest merge-shards docker-build docker-shell

help:
	@echo "install       Install the local package"
	@echo "install-data  Install the local package plus metadata dependencies"
	@echo "install-gpu   Install the provisional GPU dependency reference stack"
	@echo "test          Run the standard-library unit-test suite"
	@echo "check-model-config  Validate the complete shared-backend configuration"
	@echo "prepare-splits  Build source-grouped Study 1 splits and smoke-20 selection"
	@echo "materialize-splits  Reconstruct ignored split files from their tracked lock"
	@echo "split-check   Verify split files against the tracked manifest"
	@echo "prepare-image-cache  Lock and cache 20 development plus 20 training images"
	@echo "materialize-image-cache  Restore images from the tracked cache manifest"
	@echo "image-cache-check  Verify every cached image offline"
	@echo "training-gate  Run the two-step CUDA gate (set EXPECTED_COMMIT)"
	@echo "development-smoke  Run/resume the locked 20-item development gate"
	@echo "audit         Audit JSONL splits (set SPLITS='train.jsonl dev.jsonl test.jsonl')"
	@echo "manifest      Capture a run manifest in RUN_DIR"
	@echo "merge-shards  Merge result shards (set SHARDS, OUTPUT and ID_FIELD)"
	@echo "docker-build  Build the pinned GPU research image"
	@echo "docker-shell  Open the repository in the GPU image"

install:
	$(PYTHON) -m pip install -e .

install-data:
	$(PYTHON) -m pip install -e ".[data]"

install-gpu:
	$(PYTHON) -m pip install -e ".[gpu]"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

check-model-config:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli config-check \
		--config $(CONFIG) --model-execution

prepare-splits:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli prepare-splits \
		--config $(CONFIG) --project-root .

materialize-splits:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli materialize-splits \
		--manifest protocols/study1/grouped_split_manifest.json \
		--project-root .

split-check:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli split-check \
		--manifest protocols/study1/grouped_split_manifest.json \
		--project-root .

prepare-image-cache:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli prepare-image-cache \
		--config $(CONFIG) --project-root . --training-source-images 20

materialize-image-cache:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli materialize-image-cache \
		--manifest protocols/study1/smoke_training_image_cache_manifest.json \
		--project-root .

image-cache-check:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.cli image-cache-check \
		--manifest protocols/study1/smoke_training_image_cache_manifest.json \
		--project-root .

training-gate:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.training_gate \
		--config gi_vqa_research/$(CONFIG) \
		--repository-root .. \
		--expected-commit $(EXPECTED_COMMIT) \
		--split-manifest gi_vqa_research/protocols/study1/grouped_split_manifest.json \
		--image-cache-manifest gi_vqa_research/protocols/study1/smoke_training_image_cache_manifest.json \
		--work-dir $(GATE_WORK_DIR) \
		--artifact-dir $(GATE_ARTIFACT_DIR) \
		--required-gpu-substring T4

development-smoke:
	PYTHONPATH=src $(PYTHON) -m gi_vqa.smoke_runner \
		--config $(CONFIG) \
		--project-root . \
		--run-dir $(SMOKE_RUN_DIR) \
		--run-id development-smoke-20 \
		--expected-commit $(EXPECTED_COMMIT) \
		--require-clean-git

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
