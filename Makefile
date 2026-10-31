# 開発でよく使うもの（README の「開発の流れ」）。main の置き場（~/orca/rocm_opencv_server）で動かす。
PYTHON ?= .venv/bin/python
TEST_JOBS ?= 6
export MUJOCO_GL ?= egl

.PHONY: bench-plan train-explore eval-explore phase-train phase-offline phase-robot phase scale-check prune run-report help test-fast test lint fmt deploy-detector deploy-map status replay home-build sim-live

help:
	@echo "make test-fast        重いテスト（slow）を飛ばす（20 秒ほど）"
	@echo "make test             すべて。$(TEST_JOBS) 個の pytest に分けて同時に（50 秒ほど）"
	@echo "make lint             ruff check と format の確かめ"
	@echo "make deploy-detector  app/ を重ねた snoo を起動し直し、待ち受けと p50/p95 を出す"
	@echo "make deploy-map       地図のサービスを起動し直し、/twin を確かめる"
	@echo "make status           両方のサーバーの様子"
	@echo "make replay SESSION=<id>  保存したセッションの最後の 1 歩を GPU なしで決め直す"
	@echo "make bench-plan       作った家（白い壁の部屋も）で次の 1 歩の決め方を測る（CPU 3 本）"
	@echo "make train-explore    計画器の重みを場面から出す方策を作った家で学ぶ（続きから。CPU 3 本。SIGTERM で止まる）"
	@echo "make eval-explore     学んだ重みを、bench-plan と同じ家・同じ数で固定の重みと比べる"
	@echo "make home-build OUT=<置き場> [ARGS=\"<セッション>... --anchor <セッション>\"]  家の地図を作る"
	@echo "make sim-live [ARGS=\"--room home:3\"]  猫を手で動かすシミュレーター（http://127.0.0.1:8731/sim）"
	@echo "make phase-train      検出・地図・SLAM・見張りを止め、CPU を学習（train-explore）に使う"
	@echo "make phase-offline    検出・地図・SLAM・見張りを止め、GPU を 1 つの比べる仕事に使う"
	@echo "make phase-robot      見張りを起こす（実機のターン）。make phase で今どちらか"

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

# 次の 1 歩の決め方（app/snoo_maneuver.py）を、作った家で測る。GPU は使わない
bench-plan:
	$(PYTHON) -m scripts.snoo_plan_bench --workers 3 --out out/bench/plan.json $(ARGS)

# 計画器の重みを場面から出す方策（app/snoo_maneuver_policy.py）を学ぶ（data/snoo_policy/<algo>/ の
# 続きから。CPU だけ）。scripts/snoo_phase.sh robot が pgrep -f scripts.snoo_explore_train で見つけて
# 止めるので、名前を変えない
train-explore:
	$(PYTHON) -m scripts.snoo_explore_train $(ARGS)

eval-explore:
	$(PYTHON) -m scripts.snoo_explore_eval --out out/explore/eval.json $(ARGS)

# 切り替え（scripts/snoo_phase.sh）。GPU を握るプロセスを 1 つにし、重い仕事を 1 つずつにする
phase-train:
	scripts/snoo_phase.sh train

phase-offline:
	scripts/snoo_phase.sh offline

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

# 残っているセッションから家の地図（app/snoo_home.py）を作り直す。GPU は使わない。
# 走っている探索のセッションは入れない（ARGS="--exclude <セッション>" か、名前を並べる）
home-build:
	@test -n "$(OUT)" || { echo "make home-build OUT=<置き場> [ARGS=\"<セッション>... --anchor <セッション>\"]"; exit 2; }
	$(PYTHON) -m scripts.snoo_home_build --out $(OUT) $(ARGS)

# MuJoCo のシミュレーターを実機の速さで回し、猫をゲームパッドで動かして車の反応を見る
# （scripts/snoo_sim_live.py。ポート 8731、SNOO_SIM_PORT で変える）。絵は EGL で描き、GPU の計算は使わない
sim-live:
	$(PYTHON) -m scripts.snoo_sim_live $(ARGS)
