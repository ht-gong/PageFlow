"""Native-Polars implementations of TPC-H queries 1-22.

These are the same logical TPC-H queries as the Spark/RAPIDS stream
(`stream_qualification.sql`), using the standard TPC-H *validation* (default)
substitution parameters. Polars' SQL frontend cannot run the raw Spark SQL
stream (correlated/scalar subqueries in Q2/Q15/Q17/Q20/Q21/Q22), so we use the
canonical native-Polars expressions (cf. pola-rs/polars-tpch). Monetary Decimal
columns are read as Float64 for robust large-SF aggregation.

Each q<N>(lf) takes the dict of LazyFrames and returns a LazyFrame; q11 also
takes the scale factor, which sets its FRACTION. queries(sf) maps N to q<N>
with that scale factor bound.
"""
import os
import sys
from datetime import date, timedelta
from functools import partial
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queries"))
from stream import fraction


def q1(lf):
    cutoff = date(1998, 12, 1) - timedelta(days=90)
    line = lf["lineitem"]
    disc_price = pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
    charge = disc_price * (1 + pl.col("l_tax"))
    return (
        line.filter(pl.col("l_shipdate") <= cutoff)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.sum("l_quantity").alias("sum_qty"),
            pl.sum("l_extendedprice").alias("sum_base_price"),
            disc_price.sum().alias("sum_disc_price"),
            charge.sum().alias("sum_charge"),
            pl.mean("l_quantity").alias("avg_qty"),
            pl.mean("l_extendedprice").alias("avg_price"),
            pl.mean("l_discount").alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


def q2(lf):
    base = (
        lf["part"].filter((pl.col("p_size") == 15) & pl.col("p_type").str.ends_with("BRASS"))
        .join(lf["partsupp"], left_on="p_partkey", right_on="ps_partkey")
        .join(lf["supplier"], left_on="ps_suppkey", right_on="s_suppkey")
        .join(lf["nation"], left_on="s_nationkey", right_on="n_nationkey")
        .join(lf["region"], left_on="n_regionkey", right_on="r_regionkey")
        .filter(pl.col("r_name") == "EUROPE")
    )
    min_cost = base.group_by("p_partkey").agg(pl.min("ps_supplycost").alias("min_sc"))
    return (
        base.join(min_cost, on="p_partkey")
        .filter(pl.col("ps_supplycost") == pl.col("min_sc"))
        .select("s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr",
                "s_address", "s_phone", "s_comment")
        .sort(["s_acctbal", "n_name", "s_name", "p_partkey"],
              descending=[True, False, False, False])
        .head(100)
    )


def q3(lf):
    d = date(1995, 3, 15)
    return (
        lf["customer"].filter(pl.col("c_mktsegment") == "BUILDING")
        .join(lf["orders"], left_on="c_custkey", right_on="o_custkey")
        .join(lf["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .filter((pl.col("o_orderdate") < d) & (pl.col("l_shipdate") > d))
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("revenue"))
        .select(pl.col("o_orderkey").alias("l_orderkey"), "revenue",
                "o_orderdate", "o_shippriority")
        .sort(["revenue", "o_orderdate"], descending=[True, False])
        .head(10)
    )


def q4(lf):
    d1 = date(1993, 7, 1)
    d2 = date(1993, 10, 1)
    late = lf["lineitem"].filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
    return (
        lf["orders"].filter((pl.col("o_orderdate") >= d1) & (pl.col("o_orderdate") < d2))
        .join(late, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .group_by("o_orderpriority")
        .agg(pl.len().alias("order_count"))
        .sort("o_orderpriority")
    )


def q5(lf):
    d1 = date(1994, 1, 1)
    d2 = date(1995, 1, 1)
    return (
        lf["region"].filter(pl.col("r_name") == "ASIA")
        .join(lf["nation"], left_on="r_regionkey", right_on="n_regionkey")
        .join(lf["customer"], left_on="n_nationkey", right_on="c_nationkey")
        .join(lf["orders"], left_on="c_custkey", right_on="o_custkey")
        .join(lf["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .join(lf["supplier"], left_on=["l_suppkey", "n_nationkey"],
              right_on=["s_suppkey", "s_nationkey"])
        .filter((pl.col("o_orderdate") >= d1) & (pl.col("o_orderdate") < d2))
        .group_by("n_name")
        .agg((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("revenue"))
        .sort("revenue", descending=True)
    )


def q6(lf):
    d1 = date(1994, 1, 1)
    d2 = date(1995, 1, 1)
    return (
        lf["lineitem"].filter(
            (pl.col("l_shipdate") >= d1) & (pl.col("l_shipdate") < d2)
            & (pl.col("l_discount") >= 0.05) & (pl.col("l_discount") <= 0.07)
            & (pl.col("l_quantity") < 24)
        )
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
    )


def q7(lf):
    n = lf["nation"].select("n_nationkey", "n_name")
    sn = n.rename({"n_nationkey": "s_natk", "n_name": "supp_nation"})
    cn = n.rename({"n_nationkey": "c_natk", "n_name": "cust_nation"})
    return (
        lf["lineitem"].filter(
            (pl.col("l_shipdate") >= date(1995, 1, 1)) & (pl.col("l_shipdate") <= date(1996, 12, 31)))
        .join(lf["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(sn, left_on="s_nationkey", right_on="s_natk")
        .join(lf["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .join(lf["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(cn, left_on="c_nationkey", right_on="c_natk")
        .filter(
            ((pl.col("supp_nation") == "FRANCE") & (pl.col("cust_nation") == "GERMANY"))
            | ((pl.col("supp_nation") == "GERMANY") & (pl.col("cust_nation") == "FRANCE")))
        .with_columns(
            pl.col("l_shipdate").dt.year().alias("l_year"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("volume"))
        .group_by("supp_nation", "cust_nation", "l_year")
        .agg(pl.sum("volume").alias("revenue"))
        .sort(["supp_nation", "cust_nation", "l_year"])
    )


def q8(lf):
    n1 = lf["nation"].select("n_nationkey", "n_regionkey")
    n2 = lf["nation"].select(pl.col("n_nationkey").alias("s_natk"),
                             pl.col("n_name").alias("supp_nation"))
    return (
        lf["part"].filter(pl.col("p_type") == "ECONOMY ANODIZED STEEL")
        .join(lf["lineitem"], left_on="p_partkey", right_on="l_partkey")
        .join(lf["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(lf["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .join(lf["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(n1, left_on="c_nationkey", right_on="n_nationkey")
        .join(lf["region"], left_on="n_regionkey", right_on="r_regionkey")
        .filter(pl.col("r_name") == "AMERICA")
        .join(n2, left_on="s_nationkey", right_on="s_natk")
        .filter((pl.col("o_orderdate") >= date(1995, 1, 1)) & (pl.col("o_orderdate") <= date(1996, 12, 31)))
        .with_columns(
            pl.col("o_orderdate").dt.year().alias("o_year"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("volume"))
        .group_by("o_year")
        .agg(
            pl.when(pl.col("supp_nation") == "BRAZIL").then(pl.col("volume"))
              .otherwise(0).sum().alias("num"),
            pl.sum("volume").alias("den"))
        .select("o_year", (pl.col("num") / pl.col("den")).alias("mkt_share"))
        .sort("o_year")
    )


def q9(lf):
    return (
        lf["lineitem"]
        .join(lf["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(lf["partsupp"], left_on=["l_suppkey", "l_partkey"],
              right_on=["ps_suppkey", "ps_partkey"])
        .join(lf["part"], left_on="l_partkey", right_on="p_partkey")
        .join(lf["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .join(lf["nation"], left_on="s_nationkey", right_on="n_nationkey")
        .filter(pl.col("p_name").str.contains("green", literal=True))
        .with_columns(
            pl.col("o_orderdate").dt.year().alias("o_year"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
             - pl.col("ps_supplycost") * pl.col("l_quantity")).alias("amount"))
        .group_by("n_name", "o_year")
        .agg(pl.sum("amount").alias("sum_profit"))
        .sort(["n_name", "o_year"], descending=[False, True])
    )


def q10(lf):
    d1 = date(1993, 10, 1)
    d2 = date(1994, 1, 1)
    return (
        lf["customer"]
        .join(lf["orders"], left_on="c_custkey", right_on="o_custkey")
        .join(lf["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .join(lf["nation"], left_on="c_nationkey", right_on="n_nationkey")
        .filter((pl.col("o_orderdate") >= d1) & (pl.col("o_orderdate") < d2)
                & (pl.col("l_returnflag") == "R"))
        .group_by("c_custkey", "c_name", "c_acctbal", "c_phone", "n_name",
                  "c_address", "c_comment")
        .agg((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("revenue"))
        .select("c_custkey", "c_name", "revenue", "c_acctbal", "n_name",
                "c_address", "c_phone", "c_comment")
        .sort("revenue", descending=True)
        .head(20)
    )


def q11(lf, sf):
    ps = (
        lf["partsupp"]
        .join(lf["supplier"], left_on="ps_suppkey", right_on="s_suppkey")
        .join(lf["nation"], left_on="s_nationkey", right_on="n_nationkey")
        .filter(pl.col("n_name") == "GERMANY")
        .with_columns((pl.col("ps_supplycost") * pl.col("ps_availqty")).alias("value"))
    )
    # FRACTION is 0.0001/SF as qgen prints it, the stream's value; SF500 -> 0.0000002.
    threshold = ps.select((pl.col("value").sum() * float(fraction(sf))).alias("t"))
    return (
        ps.group_by("ps_partkey")
        .agg(pl.sum("value").alias("value"))
        .join(threshold, how="cross")
        .filter(pl.col("value") > pl.col("t"))
        .select("ps_partkey", "value")
        .sort("value", descending=True)
    )


def q12(lf):
    high = ["1-URGENT", "2-HIGH"]
    return (
        lf["lineitem"]
        .join(lf["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .filter(
            pl.col("l_shipmode").is_in(["MAIL", "SHIP"])
            & (pl.col("l_commitdate") < pl.col("l_receiptdate"))
            & (pl.col("l_shipdate") < pl.col("l_commitdate"))
            & (pl.col("l_receiptdate") >= date(1994, 1, 1))
            & (pl.col("l_receiptdate") < date(1995, 1, 1)))
        .group_by("l_shipmode")
        .agg(
            pl.when(pl.col("o_orderpriority").is_in(high)).then(1).otherwise(0).sum().alias("high_line_count"),
            pl.when(~pl.col("o_orderpriority").is_in(high)).then(1).otherwise(0).sum().alias("low_line_count"))
        .sort("l_shipmode")
    )


def q13(lf):
    no_special = lf["orders"].filter(
        ~pl.col("o_comment").str.contains("special.*requests"))
    return (
        lf["customer"]
        .join(no_special, left_on="c_custkey", right_on="o_custkey", how="left")
        .group_by("c_custkey")
        .agg(pl.col("o_orderkey").count().alias("c_count"))
        .group_by("c_count")
        .agg(pl.len().alias("custdist"))
        .sort(["custdist", "c_count"], descending=[True, True])
    )


def q14(lf):
    d1 = date(1995, 9, 1)
    d2 = date(1995, 10, 1)
    disc_price = pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
    return (
        lf["lineitem"]
        .join(lf["part"], left_on="l_partkey", right_on="p_partkey")
        .filter((pl.col("l_shipdate") >= d1) & (pl.col("l_shipdate") < d2))
        .select(
            (100.0 * pl.when(pl.col("p_type").str.starts_with("PROMO"))
             .then(disc_price).otherwise(0).sum() / disc_price.sum()).alias("promo_revenue"))
    )


def q15(lf):
    d1 = date(1996, 1, 1)
    d2 = date(1996, 4, 1)
    revenue = (
        lf["lineitem"].filter((pl.col("l_shipdate") >= d1) & (pl.col("l_shipdate") < d2))
        .group_by("l_suppkey")
        .agg((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("total_revenue"))
    )
    max_rev = revenue.select(pl.max("total_revenue").alias("mx"))
    return (
        lf["supplier"]
        .join(revenue, left_on="s_suppkey", right_on="l_suppkey")
        .join(max_rev, how="cross")
        .filter(pl.col("total_revenue") == pl.col("mx"))
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
    )


def q16(lf):
    bad_supp = lf["supplier"].filter(
        pl.col("s_comment").str.contains("Customer.*Complaints")).select("s_suppkey")
    return (
        lf["part"].filter(
            (pl.col("p_brand") != "Brand#45")
            & ~pl.col("p_type").str.starts_with("MEDIUM POLISHED")
            & pl.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9]))
        .join(lf["partsupp"], left_on="p_partkey", right_on="ps_partkey")
        .join(bad_supp, left_on="ps_suppkey", right_on="s_suppkey", how="anti")
        .group_by("p_brand", "p_type", "p_size")
        .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
        .sort(["supplier_cnt", "p_brand", "p_type", "p_size"],
              descending=[True, False, False, False])
    )


def q17(lf):
    parts = lf["part"].filter((pl.col("p_brand") == "Brand#23") & (pl.col("p_container") == "MED BOX"))
    joined = lf["lineitem"].join(parts, left_on="l_partkey", right_on="p_partkey")
    avg_q = joined.group_by("l_partkey").agg((0.2 * pl.mean("l_quantity")).alias("avg_q"))
    return (
        joined.join(avg_q, on="l_partkey")
        .filter(pl.col("l_quantity") < pl.col("avg_q"))
        .select((pl.col("l_extendedprice").sum() / 7.0).alias("avg_yearly"))
    )


def q18(lf):
    big = (
        lf["lineitem"].group_by("l_orderkey")
        .agg(pl.sum("l_quantity").alias("sum_q"))
        .filter(pl.col("sum_q") > 300)
    )
    return (
        lf["orders"].join(big, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .join(lf["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(lf["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        .agg(pl.sum("l_quantity").alias("sum_qty"))
        .select(pl.col("c_name"), pl.col("o_custkey").alias("c_custkey"),
                pl.col("o_orderkey"), pl.col("o_orderdate"),
                pl.col("o_totalprice"), pl.col("sum_qty"))
        .sort(["o_totalprice", "o_orderdate"], descending=[True, False])
        .head(100)
    )


def q19(lf):
    disc_price = pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
    return (
        lf["part"].join(lf["lineitem"], left_on="p_partkey", right_on="l_partkey")
        .filter(
            pl.col("l_shipmode").is_in(["AIR", "AIR REG"])
            & (pl.col("l_shipinstruct") == "DELIVER IN PERSON"))
        .filter(
            ((pl.col("p_brand") == "Brand#12")
             & pl.col("p_container").is_in(["SM CASE", "SM BOX", "SM PACK", "SM PKG"])
             & (pl.col("l_quantity") >= 1) & (pl.col("l_quantity") <= 11)
             & (pl.col("p_size") >= 1) & (pl.col("p_size") <= 5))
            | ((pl.col("p_brand") == "Brand#23")
               & pl.col("p_container").is_in(["MED BAG", "MED BOX", "MED PKG", "MED PACK"])
               & (pl.col("l_quantity") >= 10) & (pl.col("l_quantity") <= 20)
               & (pl.col("p_size") >= 1) & (pl.col("p_size") <= 10))
            | ((pl.col("p_brand") == "Brand#34")
               & pl.col("p_container").is_in(["LG CASE", "LG BOX", "LG PACK", "LG PKG"])
               & (pl.col("l_quantity") >= 20) & (pl.col("l_quantity") <= 30)
               & (pl.col("p_size") >= 1) & (pl.col("p_size") <= 15)))
        .select(disc_price.sum().alias("revenue"))
    )


def q20(lf):
    parts = lf["part"].filter(pl.col("p_name").str.starts_with("forest")).select("p_partkey")
    lqty = (
        lf["lineitem"].filter(
            (pl.col("l_shipdate") >= date(1994, 1, 1)) & (pl.col("l_shipdate") < date(1995, 1, 1)))
        .group_by("l_partkey", "l_suppkey")
        .agg((0.5 * pl.sum("l_quantity")).alias("qty_threshold"))
    )
    ok_supp = (
        lf["partsupp"]
        .join(parts, left_on="ps_partkey", right_on="p_partkey", how="semi")
        .join(lqty, left_on=["ps_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"])
        .filter(pl.col("ps_availqty") > pl.col("qty_threshold"))
        .select("ps_suppkey").unique()
    )
    return (
        lf["supplier"].join(ok_supp, left_on="s_suppkey", right_on="ps_suppkey", how="semi")
        .join(lf["nation"], left_on="s_nationkey", right_on="n_nationkey")
        .filter(pl.col("n_name") == "CANADA")
        .select("s_name", "s_address")
        .sort("s_name")
    )


def q21(lf):
    line = lf["lineitem"]
    n_all = line.group_by("l_orderkey").agg(pl.n_unique("l_suppkey").alias("n_all"))
    late = line.filter(pl.col("l_receiptdate") > pl.col("l_commitdate"))
    n_late = late.group_by("l_orderkey").agg(pl.n_unique("l_suppkey").alias("n_late"))
    return (
        late.join(lf["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(lf["nation"], left_on="s_nationkey", right_on="n_nationkey")
        .filter(pl.col("n_name") == "SAUDI ARABIA")
        .join(lf["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .filter(pl.col("o_orderstatus") == "F")
        .join(n_all, on="l_orderkey").filter(pl.col("n_all") > 1)
        .join(n_late, on="l_orderkey").filter(pl.col("n_late") == 1)
        .group_by("s_name")
        .agg(pl.len().alias("numwait"))
        .sort(["numwait", "s_name"], descending=[True, False])
        .head(100)
    )


def q22(lf):
    codes = ["13", "31", "23", "29", "30", "18", "17"]
    cust = lf["customer"].with_columns(pl.col("c_phone").str.slice(0, 2).alias("cntrycode")) \
        .filter(pl.col("cntrycode").is_in(codes))
    avg_bal = cust.filter(pl.col("c_acctbal") > 0.0).select(pl.mean("c_acctbal").alias("avg_bal"))
    has_orders = lf["orders"].select("o_custkey").unique()
    return (
        cust.filter(pl.col("c_acctbal") > 0.0)
        .join(has_orders, left_on="c_custkey", right_on="o_custkey", how="anti")
        .join(avg_bal, how="cross")
        .filter(pl.col("c_acctbal") > pl.col("avg_bal"))
        .group_by("cntrycode")
        .agg(pl.len().alias("numcust"), pl.sum("c_acctbal").alias("totacctbal"))
        .sort("cntrycode")
    )


def queries(sf):
    """{N: q<N>} at scale factor sf; only q11 reads it."""
    qs = {i: globals()[f"q{i}"] for i in range(1, 23)}
    qs[11] = partial(q11, sf=sf)
    return qs
