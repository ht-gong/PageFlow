"""SF1 end-to-end correctness smoke: all 22 TPC-H queries in the shipped
default mode, in-process, against ``tests/verification/SF1_reference.out``.

Also reachable via ``python main.py --smoke``.  The other modes (fixed codecs,
the compression and sizing searches) are covered by ``tests/e2e``, which
drives ``main.py`` as a subprocess per cell.
"""
import pathlib

import main
from IO.cache.config import set_global_buffer_strategy, BufferStrategy
from IO.transfer.data_exchange import set_exchange_to_naive
from utility.logger import set_message_logger
from tests.sigfig_compare import compare_results, SIGFIGS_THRESHOLD

# This smoke suite lives two levels under the repo root
# (tests/smoke/test_main.py); anchor all paths there rather than on CWD.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_main_SF1(sf1_parquet):
    set_message_logger(None, False)
    set_global_buffer_strategy(BufferStrategy.CPU)
    set_exchange_to_naive()

    # Run the SHIPPED DEFAULT mode (adaptive per-type compression) — main.exec
    # bypasses argparse, so without this the smoke would silently keep gating
    # the fixed-RAW debug override instead of the design default.  Restore the
    # module global afterwards so co-collected tests see the pristine default.
    from adaptive.instage_optimization import set_adaptive_compression

    set_adaptive_compression(True)
    try:
        main.exec(1, list(range(1, 23)), None, None, True, None)
    finally:
        set_adaptive_compression(False)

    # main.exec writes results.out into the CWD (which must be the repo root
    # for its config.json read); compare against the checked-in reference.
    res = (_REPO_ROOT / "results.out").read_text(encoding="utf-8")
    ref = (_REPO_ROOT / "tests" / "verification" / "SF1_reference.out").read_text(encoding="utf-8")
    compare_results(res, ref, SIGFIGS_THRESHOLD)
