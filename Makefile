# TQP-Vortex — Docker entry points.
# A target that runs a compose service is one `docker compose run`; data,
# validate and summary run baselines/ in its own image (make -C baselines;
# build that once with `make -C baselines image`), and bench and smoke-all run
# both.  Nothing builds on the host (cmake runs inside the image).
#
#   make build                   the engine image
#   make doctor                  GPU / memlock / NUMA / data / codec-map readiness
#   make data SF=5               generate (if missing) + validate $DATA_DIR/sf5
#   make validate SF=5           validate $DATA_DIR/sf5
#   make profiling               the codec sweep -> profiling/results/codec_map.json
#   make smoke                   SF1 end-to-end correctness
#   make test                    full pytest suites
#   make q SF=20 Q="1 6 14"      run queries
#   make bench SF="100 300"      the engine comparison, the four baselines and
#                                TQP-Vortex -> baselines/results/all_results.csv
#                                (baselines/AGENTS.md); ENGINES=, QUERIES=
#   make smoke-all               SF1 on all five engines, nothing merged
#   make spill SF=1000           the spill pass of the four GPU engines ->
#                                baselines/results/spill_profile.csv
#                                (baselines/AGENTS.md, "Spill pass"); ENGINES=,
#                                QUERIES=
#   make spill-smoke             SF1 under forced spill, nothing merged; fails
#                                unless every record and identity holds
#   make summary                 pivots of baselines/results/all_results.csv
#   make shell                   interactive shell, host source mounted
#   make plans Q="20"            capture Spark plan text + lower it to pickles
#   make experiments SCRIPT=... | spark
#   make clean | clean-all
#
#   local.env     box settings shared with baselines/ (copy local.env.example):
#                 DATA_DIR, GPU, ...  The environment and the command line win
#                 over it.
#   TQP_DATA_DIR  the data root the engine reads (default $DATA_DIR; set one
#                 of the two)
#   TQP_GPU       physical GPU the container sees (default $GPU, else 0)
#   SF            scale factor: 1 by default for make q; data and validate
#                 require it, bench takes one or more
#   ARGS          extra flags for the entry point, e.g. "--cmp_pool_gb 64".
#                 Needed on a shared card: the pool default is capacity-2GB.
#                 ARGS is expanded by make and then re-split by the shell, so
#                 quoted values survive ("--log 'out/my runs'"); avoid $.
#   DOCKER        e.g. "sudo docker" without docker-group access

# local.env beside this Makefile fills in what the environment and the command
# line leave unset, as in baselines/docker/run.sh, which reads the same file.
LOCAL_ENV := $(dir $(lastword $(MAKEFILE_LIST)))local.env
$(foreach k,$(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' $(LOCAL_ENV) 2>/dev/null),\
  $(eval $(k) ?= $(shell sed -n 's/^$(k)=//p' $(LOCAL_ENV) | head -n 1)))

DOCKER       ?= docker
COMPOSE      ?= $(DOCKER) compose
# Empty without DATA_DIR: data-dir stops on it.
TQP_DATA_DIR ?= $(DATA_DIR)
TQP_GPU      ?= $(or $(GPU),0)
Q            ?=
ARGS         ?=
SCRIPT       ?=
ENGINES      ?= rapids polars_gpu duckdb_cpu sirius tqp_vortex
export TQP_DATA_DIR TQP_GPU

# Batch targets run without a TTY (-T); shell and spark are interactive.
RUN   = $(COMPOSE) run --rm -T
RUNI  = $(COMPOSE) run --rm
QFLAG = $(if $(strip $(Q)),--q $(Q),)

.PHONY: help dirs data-dir build doctor data validate bench smoke-all spill \
        spill-smoke summary smoke test q shell plans profiling experiments \
        spark clean clean-all

help:
	@awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' Makefile

# profiling/results is created here so Docker never creates it root-owned;
# every engine service mounts it (read-only) for the codec map.
dirs:
	@mkdir -p out profiling/results

# The data root, for the services that mount it; created here for the same
# reason (make data writes into it as the host user).
data-dir: dirs
	$(if $(TQP_DATA_DIR),,$(error set DATA_DIR in local.env (copy local.env.example) or TQP_DATA_DIR))
	@mkdir -p "$(TQP_DATA_DIR)"

build:
	$(COMPOSE) --profile run build engine

doctor: data-dir
	$(RUN) -e TQP_SKIP_PREFLIGHT=1 engine python3 docker/preflight.py

# baselines/ makes the dataset, under its own rules: one SF, required.
data validate:
	$(if $(word 2,$(SF)),$(error make $@ takes one SF, e.g. `make $@ SF=1`))
	$(MAKE) -C baselines $@

# The engine comparison (baselines/AGENTS.md), one SF at a time: the baseline
# engines through baselines/run.sh, which frees its ramdisk copy at the end of
# the SF (TQP-Vortex's pinned pool needs that memory), then tqp_vortex through
# bench/run_tqp.sh on the same parquet in $DATA_DIR, made first where it is
# missing.  Both halves take $GPU (a TQP_GPU that differs
# is refused); with tqp_vortex, the bench stops before any engine runs if
# another process holds it.  run.sh starts baselines/results/run.log afresh, so
# each SF's is kept beside it as run_sf<SF>.log.  With RUN_CSV_DIR, a dir under
# $SCRATCH (where the baselines' containers and the host see the same files),
# nothing is merged and the query x engine table ends the run.  The loop names
# no $(MAKE), so `make -n bench` prints it without running it.
BASELINE_ENGINES = $(filter-out tqp_vortex,$(ENGINES))
WITH_TQP = $(filter tqp_vortex,$(ENGINES))
smoke-all: override SF = 1
smoke-all: override ENGINES = rapids polars_gpu duckdb_cpu sirius tqp_vortex
smoke-all: override export RUN_CSV_DIR = $(SCRATCH)/smoke
bench smoke-all: dirs
	$(if $(SF),,$(error make bench needs SF, e.g. `make bench SF="1 10"`))
	$(if $(findstring -,$(SF)),$(error make bench takes bare SFs here, no ranges, e.g. SF="30 50 100"))
	$(if $(RUN_CSV_DIR),$(if $(filter $(SCRATCH)/%,$(RUN_CSV_DIR)),,$(error RUN_CSV_DIR must be a dir under $$SCRATCH ($(SCRATCH)), where the baselines' containers and the host see the same files)))
	$(if $(WITH_TQP),$(if $(filter 1,$(KEEP_RAMDISK)),$(error KEEP_RAMDISK=1 is refused with tqp_vortex: its pinned pool needs the ramdisk's memory)))
	$(if $(WITH_TQP),$(if $(filter $(DATA_DIR),$(TQP_DATA_DIR)),,$(error tqp_vortex runs on $$DATA_DIR, the parquet the other engines read: set DATA_DIR, not TQP_DATA_DIR)))
	$(if $(WITH_TQP),$(if $(and $(GPU),$(filter $(GPU),$(TQP_GPU))),,$(error tqp_vortex runs on $$GPU, the other engines' GPU: set GPU, not TQP_GPU (GPU=$(GPU), TQP_GPU=$(TQP_GPU)))))
	@if [ -n "$(WITH_TQP)" ]; then \
	  busy="$$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i $(TQP_GPU) 2>&1)" && [ -z "$$busy" ] \
	    || { printf 'GPU %s is in use or missing; stopping before any engine runs:\n%s\n' "$(TQP_GPU)" "$$busy" >&2; exit 1; }; \
	fi
	$(if $(WITH_TQP),$(MAKE) build $(foreach s,$(sort 1 $(SF)),&& $(MAKE) data-sf$(s)))
	@rc=0; for sf in $(SF); do \
	  if [ -n "$(BASELINE_ENGINES)" ]; then ENGINES="$(BASELINE_ENGINES)" baselines/docker/run.sh ./run.sh $$sf || rc=1; \
	    mv baselines/results/run.log baselines/results/run_sf$$sf.log; fi; \
	  if [ -n "$(WITH_TQP)" ]; then bench/run_tqp.sh $$sf "$(QUERIES)" || rc=1; fi; \
	done; \
	if [ -n "$$RUN_CSV_DIR" ]; then ENGINES="$(ENGINES)" baselines/docker/run.sh bash -c \
	  '"$$PY" results/smoke_table.py "$$RUN_CSV_DIR" "$$ENGINES" "$(SF)" "$(QUERIES)"' || rc=1; fi; \
	exit $$rc

# The spill pass (baselines/AGENTS.md, "Spill pass"), shaped like bench: per
# SF, the GPU baselines through baselines/run.sh with SPILL=1 (duckdb_cpu has
# no spill instrumentation), then tqp_vortex through bench/run_tqp.sh with
# SPILL=1, each merging its records into baselines/results/spill_profile.csv
# unless SPILL_MERGE=0 (evidence under $SPILL_ROOT, default $SCRATCH/spill).
# spill-smoke is SF1 with each runner's forced-spill settings (SPILL_SMOKE=1;
# evidence in $SCRATCH/spill_smoke, never merged): it fails unless every
# engine's records keep the contract, every query is OK, every identity holds
# and every engine spilled.
SPILL_ENGINES = $(filter-out duckdb_cpu,$(ENGINES))
SPILL_BASELINES = $(filter-out tqp_vortex,$(SPILL_ENGINES))
spill-smoke: override SF = 1
spill-smoke: override export SPILL_SMOKE = 1
spill spill-smoke: dirs
	$(if $(SF),,$(error make spill needs SF, e.g. `make spill SF=1000`))
	$(if $(findstring -,$(SF)),$(error make spill takes bare SFs here, no ranges, e.g. SF="100 1000"))
	$(if $(WITH_TQP),$(if $(filter 1,$(KEEP_RAMDISK)),$(error KEEP_RAMDISK=1 is refused with tqp_vortex: its pinned pool needs the ramdisk's memory)))
	$(if $(WITH_TQP),$(if $(filter $(DATA_DIR),$(TQP_DATA_DIR)),,$(error tqp_vortex runs on $$DATA_DIR, the parquet the other engines read: set DATA_DIR, not TQP_DATA_DIR)))
	$(if $(WITH_TQP),$(if $(and $(GPU),$(filter $(GPU),$(TQP_GPU))),,$(error tqp_vortex runs on $$GPU, the other engines' GPU: set GPU, not TQP_GPU (GPU=$(GPU), TQP_GPU=$(TQP_GPU)))))
	@if [ -n "$(WITH_TQP)" ]; then \
	  busy="$$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i $(TQP_GPU) 2>&1)" && [ -z "$$busy" ] \
	    || { printf 'GPU %s is in use or missing; stopping before any engine runs:\n%s\n' "$(TQP_GPU)" "$$busy" >&2; exit 1; }; \
	fi
	$(if $(WITH_TQP),$(MAKE) build $(foreach s,$(sort 1 $(SF)),&& $(MAKE) data-sf$(s)))
	@rc=0; for sf in $(SF); do \
	  if [ -n "$(SPILL_BASELINES)" ]; then SPILL=1 ENGINES="$(SPILL_BASELINES)" baselines/docker/run.sh ./run.sh $$sf || rc=1; \
	    mv baselines/results/spill_run.log baselines/results/spill_run_sf$$sf.log; fi; \
	  if [ -n "$(WITH_TQP)" ]; then SPILL=1 bench/run_tqp.sh $$sf "$(QUERIES)" || rc=1; fi; \
	done; \
	exit $$rc

# The SF's dataset for bench (and its warmup's, SF1), made where it is missing
# (make data).
data-sf%:
	@p="$(DATA_DIR)/sf$*/parquet"; \
	if [ ! -d "$$p" ]; then echo "SF$*: no $$p: make data"; $(MAKE) data SF=$*; fi

summary:
	$(MAKE) -C baselines summary

smoke: data-dir
	$(RUN) engine python3 main.py --smoke $(ARGS)

test: data-dir
	$(RUN) test python3 test.py $(ARGS)

# SF defaults to 1 here only: the targets that delegate to baselines/ require it.
q: data-dir
	$(RUN) engine python3 main.py --SF $(or $(SF),1) $(QFLAG) $(ARGS)

shell: data-dir
	$(RUNI) dev

plans: dirs
	$(RUN) plans python3 -m parsing.regenerate $(QFLAG) $(ARGS)

profiling: data-dir
	$(RUN) profiling python3 -m profiling.sweep $(ARGS)

experiments: data-dir
	@test -n "$(SCRIPT)" || { echo "set SCRIPT=<path to experiment script>"; exit 2; }
	$(RUN) experiments python3 $(SCRIPT) $(ARGS)

spark: dirs
	$(RUNI) spark

clean:
	$(COMPOSE) down --remove-orphans
	-$(DOCKER) volume rm tqp-vortex_tqp-build

clean-all: clean
	-$(DOCKER) image rm tqp-vortex:latest
