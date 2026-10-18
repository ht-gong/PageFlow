"""The columns each TPC-H query reads, per table: what sirius_resident pins and
rapids_resident caches.

Vendored from Sirius's test/tpch_performance/tpch_pin_columns.py at the pinned
commit (versions.env, d79d4f97), which the image need not ship. A scan is served
from a pin only when every column it reads (projected or filtered) was pinned;
otherwise Sirius falls back to the parquet reader without a log line, so
run_tpch_sirius.py checks its log for each pinned table's "assigned pinned
entry" line. That check is per table, not per scan, so these lists must cover
every scan: checked against DuckDB 1.5.5's optimized plan of each query of
results/queries/stream_qualification.sql at SF1 (2026-10-12), where each of the
84 parquet scans reads only columns listed here.
"""

QUERY_COLUMNS = {
    1: {
        'lineitem': ['l_returnflag', 'l_linestatus', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_shipdate'],
    },
    2: {
        'part': ['p_partkey', 'p_mfgr', 'p_size', 'p_type'],
        'supplier': ['s_suppkey', 's_acctbal', 's_name', 's_address', 's_phone', 's_comment', 's_nationkey'],
        'partsupp': ['ps_partkey', 'ps_suppkey', 'ps_supplycost'],
        'nation': ['n_nationkey', 'n_name', 'n_regionkey'],
        'region': ['r_regionkey', 'r_name'],
    },
    3: {
        'customer': ['c_mktsegment', 'c_custkey'],
        'orders': ['o_custkey', 'o_orderkey', 'o_orderdate', 'o_shippriority'],
        'lineitem': ['l_orderkey', 'l_extendedprice', 'l_discount', 'l_shipdate'],
    },
    4: {
        'orders': ['o_orderkey', 'o_orderdate', 'o_orderpriority'],
        'lineitem': ['l_orderkey', 'l_commitdate', 'l_receiptdate'],
    },
    5: {
        'orders': ['o_custkey', 'o_orderkey', 'o_orderdate'],
        'lineitem': ['l_orderkey', 'l_suppkey', 'l_extendedprice', 'l_discount'],
        'supplier': ['s_suppkey', 's_nationkey'],
        'nation': ['n_nationkey', 'n_name', 'n_regionkey'],
        'region': ['r_regionkey', 'r_name'],
        'customer': ['c_custkey', 'c_nationkey'],
    },
    6: {
        'lineitem': ['l_extendedprice', 'l_discount', 'l_shipdate', 'l_quantity'],
    },
    7: {
        'supplier': ['s_suppkey', 's_nationkey'],
        'lineitem': ['l_suppkey', 'l_orderkey', 'l_shipdate', 'l_extendedprice', 'l_discount'],
        'orders': ['o_orderkey', 'o_custkey'],
        'customer': ['c_custkey', 'c_nationkey'],
        'nation': ['n_nationkey', 'n_name'],
    },
    8: {
        'lineitem': ['l_partkey', 'l_suppkey', 'l_orderkey', 'l_extendedprice', 'l_discount'],
        'part': ['p_partkey', 'p_type'],
        'supplier': ['s_suppkey', 's_nationkey'],
        'orders': ['o_orderkey', 'o_custkey', 'o_orderdate'],
        'customer': ['c_custkey', 'c_nationkey'],
        'nation': ['n_nationkey', 'n_regionkey', 'n_name'],
        'region': ['r_regionkey', 'r_name'],
    },
    9: {
        'part': ['p_partkey', 'p_name'],
        'supplier': ['s_suppkey', 's_nationkey'],
        'lineitem': ['l_suppkey', 'l_partkey', 'l_orderkey', 'l_extendedprice', 'l_discount', 'l_quantity'],
        'partsupp': ['ps_suppkey', 'ps_partkey', 'ps_supplycost'],
        'orders': ['o_orderkey', 'o_orderdate'],
        'nation': ['n_nationkey', 'n_name'],
    },
    10: {
        'customer': ['c_custkey', 'c_name', 'c_acctbal', 'c_address', 'c_phone', 'c_comment', 'c_nationkey'],
        'orders': ['o_custkey', 'o_orderkey', 'o_orderdate'],
        'lineitem': ['l_orderkey', 'l_extendedprice', 'l_discount', 'l_returnflag'],
        'nation': ['n_nationkey', 'n_name'],
    },
    11: {
        'partsupp': ['ps_partkey', 'ps_suppkey', 'ps_supplycost', 'ps_availqty'],
        'supplier': ['s_suppkey', 's_nationkey'],
        'nation': ['n_nationkey', 'n_name'],
    },
    12: {
        'orders': ['o_orderkey', 'o_orderpriority'],
        'lineitem': ['l_orderkey', 'l_shipmode', 'l_commitdate', 'l_receiptdate', 'l_shipdate'],
    },
    13: {
        'customer': ['c_custkey'],
        'orders': ['o_custkey', 'o_orderkey', 'o_comment'],
    },
    14: {
        'lineitem': ['l_partkey', 'l_extendedprice', 'l_discount', 'l_shipdate'],
        'part': ['p_partkey', 'p_type'],
    },
    15: {
        'lineitem': ['l_suppkey', 'l_extendedprice', 'l_discount', 'l_shipdate'],
        'supplier': ['s_suppkey', 's_name', 's_address', 's_phone'],
    },
    16: {
        'partsupp': ['ps_partkey', 'ps_suppkey'],
        'part': ['p_partkey', 'p_brand', 'p_type', 'p_size'],
        'supplier': ['s_suppkey', 's_comment'],
    },
    17: {
        'lineitem': ['l_partkey', 'l_extendedprice', 'l_quantity'],
        'part': ['p_partkey', 'p_brand', 'p_container'],
    },
    18: {
        'customer': ['c_name', 'c_custkey'],
        'orders': ['o_orderkey', 'o_custkey', 'o_orderdate', 'o_totalprice'],
        'lineitem': ['l_orderkey', 'l_quantity'],
    },
    19: {
        'lineitem': ['l_partkey', 'l_extendedprice', 'l_discount', 'l_quantity', 'l_shipmode', 'l_shipinstruct'],
        'part': ['p_partkey', 'p_brand', 'p_container', 'p_size'],
    },
    20: {
        'supplier': ['s_suppkey', 's_name', 's_address', 's_nationkey'],
        'partsupp': ['ps_suppkey', 'ps_partkey', 'ps_availqty'],
        'part': ['p_partkey', 'p_name'],
        'lineitem': ['l_partkey', 'l_suppkey', 'l_quantity', 'l_shipdate'],
        'nation': ['n_nationkey', 'n_name'],
    },
    21: {
        'supplier': ['s_suppkey', 's_nationkey', 's_name'],
        'lineitem': ['l_suppkey', 'l_orderkey', 'l_receiptdate', 'l_commitdate'],
        'orders': ['o_orderkey', 'o_orderstatus'],
        'nation': ['n_nationkey', 'n_name'],
    },
    22: {
        'customer': ['c_custkey', 'c_phone', 'c_acctbal'],
        'orders': ['o_custkey'],
    },
}
