# The one entry point. Every target runs in the pinned container through
# docker/run.sh; the scripts it calls keep their logic, Make only forwards
# variables. Box-specific settings (DATA_DIR, SCRATCH, SHM, GPU) come from
# local.env at the TQP-Vortex root, ../local.env (copy ../local.env.example).
#
#   make image                                 build the image for the current pins
#   make doctor                                versions, GPU, io_uring, free space
#   make data SF=100 / make validate SF=100    generate (if missing) + validate $DATA_DIR/sf100
#   make bench SF=100 [ENGINES="..."]          TPC-H q1-22 -> results/all_results.csv
#   make smoke                                 SF1, all engines, nothing merged
#   make ablation [SF=5 ENCODINGS= COMPRESSIONS= ROUNDS= ENGINES= TOL= OUT=]
#                                              format profiling pass -> format_profile.csv,
#                                              format_map.json, results table on stdout
#   make summary                               pivots of results/all_results.csv
#   make spill SF=1000 [ENGINES="..."]          spill pass (rapids, polars_gpu, sirius) ->
#                                              results/spill_profile.csv, spill_settings.json
#   make spill-smoke                           SF1 under forced spill; fails unless every
#                                              record and identity holds; nothing merged
#   make spill-test                            the spill record builders' unit tests (CPU)
#   make shell                                 a shell in the container

RUN := docker/run.sh
ENCODINGS ?= plain dict delta
COMPRESSIONS ?= snappy zstd lz4raw
ROUNDS ?= 3
TOL ?= 0.05
OUT ?= results
# The format profiling pass runs at SF5 unless SF is given; the other SF
# targets require it.
ablation: SF ?= 5

need = $(if $($(1)),,$(error $(1) is required, e.g. `make $@ $(1)=$(2)`))

.PHONY: image shell doctor data validate bench smoke ablation summary \
        spill spill-smoke spill-test

image:
	$(RUN) build

shell:
	$(RUN) bash

doctor:
	$(RUN) docker/doctor.sh

data:
	$(call need,SF,100)
	$(RUN) datagen/dataset.sh $(SF)

validate:
	$(call need,SF,100)
	$(RUN) bash -c '"$$PY" results/validate_dataset.py "$$DATA_DIR/sf$(SF)/parquet" $(SF)'

bench:
	$(call need,SF,100)
	$(RUN) ./run.sh $(SF)

smoke:
	$(RUN) bash -c 'RUN_CSV_DIR="$$SCRATCH/smoke" ./run.sh 1'

ablation:
	$(RUN) bash -c '"$$PY" ablation/format_profile.py --sf $(SF) --encodings "$(ENCODINGS)" \
	  --compressions "$(COMPRESSIONS)" --rounds $(ROUNDS) --tol $(TOL) --out "$(OUT)" \
	  $${ENGINES:+--engines "$$ENGINES"}'

summary:
	$(RUN) bash -c '"$$PY" results/summary.py'

spill:
	$(call need,SF,1000)
	$(RUN) bash -c 'SPILL=1 ./run.sh $(SF)'

spill-smoke:
	$(RUN) bash -c 'SPILL=1 SPILL_SMOKE=1 ./run.sh 1'

spill-test:
	$(RUN) bash -c '"$$PY" -m unittest discover -s tests -t . -p "test_spill*.py"'
