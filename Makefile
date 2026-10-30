# 開発でよく使うもの（README の「開発の流れ」）。main の置き場（~/orca/rocm_opencv_server）で動かす。
PYTHON ?= .venv/bin/python
TEST_JOBS ?= 6
export MUJOCO_GL ?= egl

.PHONY: train-explore eval-explore phase-train phase-robot phase scale-check prune run-report help test-fast test lint fmt deploy-detector deploy-map status replay

help:
	@echo "make test-fast        重いテスト（slow）を飛ばす（20 秒ほど）"
	@echo "make test             すべて。$(TEST_JOBS) 個の pytest に分けて同時に（50 秒ほど）"
	@echo "make lint             ruff check と format の確かめ"
	@echo "make deploy-detector  app/ を重ねた snoo を起動し直し、待ち受けと p50/p95 を出す"
	@echo "make deploy-map       地図のサービスを起動し直し、/twin を確かめる"
	@echo "make status           両方のサーバーの様子"
	@echo "make replay SESSION=<id>  保存したセッションの最後の 1 歩を GPU なしで決め直す"
	@echo "make train-explore    作った家で次の 1 歩の選び方を学ぶ（続きから。SIGTERM で止まる）"
	@echo "make eval-explore     学んだ選び方を、学んでいない家で今の決め方と比べる"

test-fast:
	$(PYTHON) -m pytest -q -m "not slow" -p no:cacheprovider $(ARGS)

test:
	PYTHON=$(PYTHON) scripts/test_all.sh $(TEST_JOBS) $(ARGS)

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

fmt:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

deploy-detector:
	$(PYTHON) -m scripts.snoo_dev deploy-detector

deploy-map:
	$(PYTHON) -m scripts.snoo_dev deploy-map

status:
	$(PYTHON) -m scripts.snoo_dev status

# 学ぶ（data/snoo_policy/ の最後のチェックポイントから続ける）。scripts/snoo_phase.sh robot が
# pgrep -f scripts.snoo_explore_train で見つけて止めるので、名前を変えない
train-explore:
	$(PYTHON) -m scripts.snoo_explore_train $(ARGS)

eval-explore:
	$(PYTHON) -m scripts.snoo_explore_eval --out out/explore/eval.json --plot out/explore/curve.png $(ARGS)

phase-train:
	scripts/snoo_phase.sh train

phase-robot:
	scripts/snoo_phase.sh robot

phase:
	scripts/snoo_phase.sh status

scale-check:
	@test -n "$(IMAGE)" -a -n "$(TILT)" || { echo "make scale-check IMAGE=<A4 を撮った画像> TILT=<チルト>"; exit 2; }
	$(PYTHON) -m scripts.snoo_scale_check $(IMAGE) --tilt $(TILT) $(ARGS)

prune:
	$(PYTHON) -m scripts.snoo_prune $(ARGS)

run-report:
	@test -n "$(SESSION)" || { echo "make run-report SESSION=<セッション> [CLEAR=clear.jsonl] [BASE=<セッション>]"; exit 2; }
	$(PYTHON) -m scripts.snoo_run_report $(SESSION) $(if $(CLEAR),--clear $(CLEAR)) $(if $(BASE),--base $(BASE))

replay:
	@test -n "$(SESSION)" || { echo "make replay SESSION=<セッションの名前か置き場>"; exit 2; }
	$(PYTHON) -m scripts.snoo_replay $(SESSION) $(if $(STATION),--station $(STATION)) $(ARGS)
