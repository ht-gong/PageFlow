//! ndsgen: TPC-H straight to parquet in the NDS-H layout, with no dbgen text and no Spark.
//!
//! The rows come from tpchgen-rs (`tpchgen`, the dbgen 2.17.3 port, chunk i of
//! PARALLEL = `dbgen -C PARALLEL -S i`). Everything the NDS-H dbgen -> Spark
//! transcode (the retired datagen/gen_tpch.sh) added on top of dbgen is reproduced here:
//!
//! * Batches: BATCH chunks per batch; every batch writes FILES (200) part-files
//!   per table, so files per table = FILES x batches, as `repartition(200)` per
//!   batch does. nation/region come once, from the batch holding chunk 1.
//! * Row order: Spark's CSV read of a batch's chunk files is replayed. Each chunk
//!   is cut into input splits of maxSplitBytes = min(128 MB, max(4 MB, bytes /
//!   cores)) of its dbgen text (a line belongs to the split its first byte
//!   follows, as Hadoop's LineRecordReader does); the splits are sorted by size,
//!   largest first, and packed into map tasks; each task's rows are scrambled
//!   (Spark sorts them by row hash before a round-robin repartition) and dealt
//!   round-robin from a random start. A part-file is its share of every task, in
//!   task order. The text lengths are tpchgen's own .tbl lines, so the split
//!   points are the ones Spark saw.
//! * Encoding: parquet-mr defaults (1 MB pages and dictionaries, 20,000-row
//!   pages, 128 MB row groups, zstd level 3, every column OPTIONAL, the trailing
//!   all-null `ignore` column), including parquet-mr's first-page dictionary
//!   check: a dictionary column whose first page does not get smaller than
//!   PLAIN is written PLAIN for the whole chunk.
//!
//! Formats: uniform (`--compression`, `--dictionary`, `--writer-version`, the
//! retired transcode's knobs), or per column from the profiling pass's
//! format_map.json (`--format-map`, `--engine`).

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{new_null_array, Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::{cast, interleave};
use arrow::datatypes::{
    DataType, Date32Type, Decimal128Type, Field, Int32Type, Int64Type, Schema, SchemaRef,
};
use clap::Parser;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;
use rayon::prelude::*;
use tpchgen::generators::{
    CustomerGenerator, LineItemGenerator, NationGenerator, OrderGenerator, PartGenerator,
    PartSuppGenerator, RegionGenerator, SupplierGenerator,
};
use tpchgen_arrow::{
    CustomerArrow, LineItemArrow, NationArrow, OrderArrow, PartArrow, PartSuppArrow,
    RecordBatchIterator, RegionArrow, SupplierArrow,
};

const TABLES: [&str; 8] = [
    "lineitem", "orders", "partsupp", "part", "customer", "supplier", "nation", "region",
];
const MB: u64 = 1 << 20;
// parquet-mr 1.13 defaults
const PAGE_BYTES: usize = 1 << 20;
const PAGE_ROWS: usize = 20_000;
const DICT_BYTES: usize = 1 << 20;
const ROW_GROUP_BYTES: usize = 128 << 20;
const ZSTD_LEVEL: i32 = 3;

#[derive(Parser, Debug)]
#[command(version, about = "TPC-H -> NDS-H-layout parquet (tpchgen-rs rows, Spark's batch shuffle)")]
struct Args {
    /// Scale factor
    #[arg(long)]
    sf: u32,
    /// Output root: writes <out>/parquet/<table>/ (built in <out>/_parquet.partial)
    #[arg(long)]
    out: PathBuf,
    /// dbgen chunks (-C); default max(20, 2*SF), as datagen/dataset.sh
    #[arg(long)]
    parallel: Option<i32>,
    /// Chunks per batch (one Spark transcode per batch in NDS-H)
    #[arg(long, default_value_t = 25)]
    batch: i32,
    /// Part-files per table per batch (the transcode's repartition)
    #[arg(long, default_value_t = 200)]
    files: usize,
    /// Tables (comma-separated); default all
    #[arg(long, value_delimiter = ',')]
    tables: Option<Vec<String>>,
    /// Seed of the row scramble
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Spark's defaultParallelism in the replayed read (local[*] cores); default this process's
    #[arg(long)]
    spark_cores: Option<u64>,
    /// Worker threads; default this process's available parallelism
    #[arg(long)]
    threads: Option<usize>,
    /// Batches in flight at once (memory: ~15 GB each for a lineitem batch of SF12.5)
    #[arg(long, default_value_t = 6)]
    concurrent_batches: usize,
    /// Also format every row as .tbl text and check the split lengths against it (slower)
    #[arg(long)]
    verify_text_lens: bool,
    /// Uniform codec: snappy | zstd | lz4raw | none
    #[arg(long, default_value = "snappy")]
    compression: String,
    /// Uniform dictionary encoding (with parquet-mr's PLAIN fallback)
    #[arg(long, default_value = "true")]
    dictionary: String,
    /// Uniform page version: v1 | v2 (v2 without dictionary = the delta encodings)
    #[arg(long, default_value = "v1")]
    writer_version: String,
    /// Per-column formats from format_map.json (overrides the uniform knobs)
    #[arg(long)]
    format_map: Option<PathBuf>,
    /// Engine whose columns to take from --format-map
    #[arg(long)]
    engine: Option<String>,
    /// Key in --format-map (default: the only one, e.g. sf5)
    #[arg(long)]
    map_key: Option<String>,
    /// Page version of --format-map files
    #[arg(long, default_value = "v1")]
    map_page_version: String,
}

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ------------------------------------------------------------------ rows
/// Rows per generated piece; every piece of a chunk but its last is exactly this long.
const PIECE: usize = 1 << 16;

/// One dbgen chunk as NDS-schema pieces of PIECE rows (kept apart: no concatenation copy).
fn gen_arrow(table: &str, sf: f64, part: i32, parts: i32) -> Vec<RecordBatch> {
    const BS: usize = PIECE;
    let it: Box<dyn RecordBatchIterator> = match table {
        "lineitem" => Box::new(LineItemArrow::new(LineItemGenerator::new(sf, part, parts)).with_batch_size(BS)),
        "orders" => Box::new(OrderArrow::new(OrderGenerator::new(sf, part, parts)).with_batch_size(BS)),
        "partsupp" => Box::new(PartSuppArrow::new(PartSuppGenerator::new(sf, part, parts)).with_batch_size(BS)),
        "part" => Box::new(PartArrow::new(PartGenerator::new(sf, part, parts)).with_batch_size(BS)),
        "customer" => Box::new(CustomerArrow::new(CustomerGenerator::new(sf, part, parts)).with_batch_size(BS)),
        "supplier" => Box::new(SupplierArrow::new(SupplierGenerator::new(sf, part, parts)).with_batch_size(BS)),
        "nation" => Box::new(NationArrow::new(NationGenerator::new(sf, part, parts)).with_batch_size(BS)),
        "region" => Box::new(RegionArrow::new(RegionGenerator::new(sf, part, parts)).with_batch_size(BS)),
        t => panic!("unknown table {t}"),
    };
    it.map(|b| to_nds(&b)).collect()
}

/// Bytes of each row's dbgen .tbl line (tpchgen's Display is the .tbl format), newline included.
fn text_lens(table: &str, sf: f64, part: i32, parts: i32) -> Vec<u32> {
    let mut buf = String::with_capacity(512);
    let mut out = Vec::new();
    macro_rules! lens {
        ($g:expr) => {{
            let g = $g;
            for row in g.iter() {
                buf.clear();
                write!(buf, "{row}").unwrap();
                out.push(buf.len() as u32 + 1);
            }
        }};
    }
    match table {
        "lineitem" => lens!(LineItemGenerator::new(sf, part, parts)),
        "orders" => lens!(OrderGenerator::new(sf, part, parts)),
        "partsupp" => lens!(PartSuppGenerator::new(sf, part, parts)),
        "part" => lens!(PartGenerator::new(sf, part, parts)),
        "customer" => lens!(CustomerGenerator::new(sf, part, parts)),
        "supplier" => lens!(SupplierGenerator::new(sf, part, parts)),
        "nation" => lens!(NationGenerator::new(sf, part, parts)),
        "region" => lens!(RegionGenerator::new(sf, part, parts)),
        t => panic!("unknown table {t}"),
    }
    out
}

fn digits(v: u64) -> u32 {
    if v == 0 { 1 } else { v.ilog10() + 1 }
}

/// The same lengths as text_lens, from the Arrow columns (no second generation pass):
/// integers in decimal, money as [-]int.dd (l_quantity is a plain integer in .tbl),
/// dates as YYYY-MM-DD, a '|' after every field, then the newline.
fn text_lens_arrow(b: &RecordBatch) -> Vec<u32> {
    let schema = b.schema();
    let fields = schema.fields().iter().zip(b.columns()).filter(|(f, _)| f.name() != "ignore");
    let mut lens = vec![1u32; b.num_rows()];
    for (f, c) in fields {
        let add: Box<dyn Fn(usize) -> u32> = match c.data_type() {
            DataType::Int64 => {
                let a = c.as_primitive::<Int64Type>();
                Box::new(move |i| digits(a.value(i) as u64))
            }
            DataType::Int32 => {
                let a = c.as_primitive::<Int32Type>();
                Box::new(move |i| digits(a.value(i) as u64))
            }
            DataType::Date32 => Box::new(|_| 10),
            DataType::Decimal128(..) if f.name() == "l_quantity" => {
                let a = c.as_primitive::<Decimal128Type>();
                Box::new(move |i| digits((a.value(i) / 100) as u64))
            }
            DataType::Decimal128(..) => {
                let a = c.as_primitive::<Decimal128Type>();
                Box::new(move |i| {
                    let v = a.value(i);
                    (v < 0) as u32 + digits((v.unsigned_abs() / 100) as u64) + 3
                })
            }
            DataType::Utf8 => {
                let o = c.as_string::<i32>().value_offsets();
                Box::new(move |i| (o[i + 1] - o[i]) as u32)
            }
            dt => panic!("no .tbl length for {dt}"),
        };
        for (i, l) in lens.iter_mut().enumerate() {
            *l += add(i) + 1;
        }
    }
    lens
}

/// tpchgen's Arrow types -> the NDS-H Spark schema: DECIMAL(11,2), plain strings,
/// every column nullable, plus the trailing all-null `ignore` column.
fn to_nds(b: &RecordBatch) -> RecordBatch {
    let mut fields = Vec::new();
    let mut cols: Vec<ArrayRef> = Vec::new();
    for (f, c) in b.schema().fields().iter().zip(b.columns()) {
        let (dt, arr): (DataType, ArrayRef) = match f.data_type() {
            DataType::Decimal128(_, s) => {
                let a = c.as_primitive::<Decimal128Type>().clone().with_precision_and_scale(11, *s).unwrap();
                (DataType::Decimal128(11, *s), Arc::new(a))
            }
            DataType::Utf8View | DataType::LargeUtf8 => (DataType::Utf8, cast(c, &DataType::Utf8).unwrap()),
            dt => (dt.clone(), c.clone()),
        };
        fields.push(Field::new(f.name(), dt, true));
        cols.push(arr);
    }
    fields.push(Field::new("ignore", DataType::Utf8, true));
    cols.push(new_null_array(&DataType::Utf8, b.num_rows()));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
}

// ------------------------------------------------------------------ Spark's read + repartition
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
    fn of(parts: &[u64]) -> Rng {
        let mut r = Rng(0x5EED);
        for &p in parts {
            r.0 ^= p;
            r.0 = r.next();
        }
        r
    }
}

struct SparkRead {
    cores: u64,
    max_partition_bytes: u64,
    open_cost: u64,
}

/// The map tasks of one Spark CSV read of `lens` (row text lengths per input file):
/// each task is a list of (file, first row, end row), in read order.
fn map_tasks(lens: &[Vec<u32>], sp: &SparkRead) -> Vec<Vec<(usize, usize, usize)>> {
    let sizes: Vec<u64> = lens.iter().map(|l| l.iter().map(|&x| x as u64).sum()).collect();
    let total: u64 = sizes.iter().map(|s| s + sp.open_cost).sum();
    let max_split = sp.max_partition_bytes.min(sp.open_cost.max(total / sp.cores));
    let mut splits = Vec::new(); // (bytes, file, first row, end row)
    for (f, l) in lens.iter().enumerate() {
        let n = sizes[f].div_ceil(max_split) as usize;
        let mut first = vec![l.len(); n + 1];
        first[0] = 0;
        let (mut off, mut k) = (0u64, 0usize);
        for (r, &len) in l.iter().enumerate() {
            // LineRecordReader: split [s, s+M) skips the line holding byte s-1... and reads
            // every line that starts at or before its end, so a line starting at offset o
            // belongs to split ceil(o / M) - 1 (split 0 for o = 0).
            let s = if off == 0 { 0 } else { ((off - 1) / max_split) as usize };
            while k < s {
                k += 1;
                first[k] = r;
            }
            off += len as u64;
        }
        for k in 0..n {
            let bytes = max_split.min(sizes[f] - k as u64 * max_split);
            splits.push((bytes, f, first[k], first[k + 1]));
        }
    }
    splits.sort_by(|a, b| b.0.cmp(&a.0)); // stable: largest first, then file order
    let (mut tasks, mut cur, mut cur_size) = (Vec::new(), Vec::new(), 0u64);
    for (bytes, f, a, b) in splits {
        if cur_size + bytes > max_split && !cur.is_empty() {
            tasks.push(std::mem::take(&mut cur));
            cur_size = 0;
        }
        cur.push((f, a, b));
        cur_size += bytes + sp.open_cost;
    }
    if !cur.is_empty() {
        tasks.push(cur);
    }
    tasks
}

/// Round-robin repartition of the tasks' rows into `n_files`: (file, row) per output file.
fn deal(tasks: &[Vec<(usize, usize, usize)>], n_files: usize, seed: &[u64]) -> Vec<Vec<(u32, u32)>> {
    let scrambled: Vec<(usize, Vec<(u32, u32)>)> = tasks
        .par_iter()
        .enumerate()
        .map(|(t, task)| {
            let mut rows: Vec<(u32, u32)> = task
                .iter()
                .flat_map(|&(f, a, b)| (a..b).map(move |r| (f as u32, r as u32)))
                .collect();
            let mut rng = Rng::of(&[seed, &[t as u64]].concat());
            for i in (1..rows.len()).rev() {
                rows.swap(i, rng.below(i as u64 + 1) as usize);
            }
            (rng.below(n_files as u64) as usize, rows)
        })
        .collect();
    // Row j of a task goes to file (start + 1 + j) % n_files, so file f takes every
    // n_files-th row from offset (f - start - 1) mod n_files of each task, in task order.
    let total: usize = scrambled.iter().map(|(_, r)| r.len()).sum();
    (0..n_files)
        .into_par_iter()
        .map(|f| {
            let mut out = Vec::with_capacity(total / n_files + scrambled.len());
            for (start, rows) in &scrambled {
                let first = (f + n_files - (start + 1) % n_files) % n_files;
                out.extend(rows.iter().skip(first).step_by(n_files));
            }
            out
        })
        .collect()
}

// ------------------------------------------------------------------ formats
#[derive(Clone, Copy, Debug, PartialEq)]
enum Enc {
    Plain,
    Dict,
    Delta,
}

#[derive(Clone, Copy, Debug)]
struct ColFmt {
    enc: Enc,
    codec: Compression,
}

fn codec(name: &str) -> Compression {
    match name {
        "snappy" => Compression::SNAPPY,
        "zstd" => Compression::ZSTD(ZstdLevel::try_new(ZSTD_LEVEL).unwrap()),
        "lz4raw" | "lz4_raw" => Compression::LZ4_RAW,
        "none" | "uncompressed" => Compression::UNCOMPRESSED,
        c => panic!("unknown compression {c}"),
    }
}

enum Formats {
    Uniform { col: ColFmt, v2: bool, suffix: String },
    Map { map: serde_json::Value, v2: bool },
}

impl Formats {
    fn col(&self, table: &str, column: &str) -> ColFmt {
        match self {
            Formats::Uniform { col, .. } => *col,
            Formats::Map { map, .. } => match map.get(table).and_then(|t| t.get(column)) {
                Some(serde_json::Value::Object(f)) => ColFmt {
                    enc: match f["encoding"].as_str().unwrap() {
                        "plain" => Enc::Plain,
                        "dict" => Enc::Dict,
                        "delta" => Enc::Delta,
                        e => panic!("unknown encoding {e}"),
                    },
                    codec: codec(f["compression"].as_str().unwrap()),
                },
                // `ignore` (never profiled) and columns with no OK format: parquet-mr's default
                _ => ColFmt { enc: Enc::Dict, codec: Compression::SNAPPY },
            },
        }
    }
    fn v2(&self) -> bool {
        match self {
            Formats::Uniform { v2, .. } | Formats::Map { v2, .. } => *v2,
        }
    }
    fn suffix(&self) -> &str {
        match self {
            Formats::Uniform { suffix, .. } => suffix,
            Formats::Map { .. } => "mixed",
        }
    }
}

/// parquet-mr's dictionary decision for a column chunk: FallbackValuesWriter cuts the
/// first page at 20,000 values or 1 MB of PLAIN bytes, and keeps the dictionary only if
/// the dictionary plus the bit-packed ids are smaller than PLAIN (and the dictionary
/// stays under 1 MB). parquet-rs has no such check; it is applied here per file.
fn mr_keeps_dictionary(a: &ArrayRef) -> bool {
    fn scan<K: std::hash::Hash + Eq>(vals: impl Iterator<Item = (K, usize)>) -> Option<(usize, usize, usize, usize)> {
        let (mut seen, mut raw, mut dict, mut n) = (HashSet::new(), 0usize, 0usize, 0usize);
        for (k, plain) in vals {
            raw += plain;
            n += 1;
            if seen.insert(k) {
                dict += plain;
                if dict > DICT_BYTES {
                    return None;
                }
            }
            if n >= PAGE_ROWS || raw >= PAGE_BYTES {
                break;
            }
        }
        Some((n, raw, dict, seen.len()))
    }
    let r = match a.data_type() {
        DataType::Int64 => scan(a.as_primitive::<Int64Type>().iter().flatten().map(|v| (v, 8))),
        DataType::Int32 => scan(a.as_primitive::<Int32Type>().iter().flatten().map(|v| (v, 4))),
        DataType::Date32 => scan(a.as_primitive::<Date32Type>().iter().flatten().map(|v| (v, 4))),
        DataType::Decimal128(p, _) if *p <= 18 => scan(a.as_primitive::<Decimal128Type>().iter().flatten().map(|v| (v, 8))),
        DataType::Utf8 => scan(a.as_string::<i32>().iter().flatten().map(|v| (v, 4 + v.len()))),
        _ => return false,
    };
    let Some((n, raw, dict, distinct)) = r else { return false };
    if n == 0 {
        return false;
    }
    let bw = (usize::BITS - (distinct - 1).leading_zeros()) as usize;
    let ids = 1 + if bw == 0 { 3 } else { n.div_ceil(8) * bw + n.div_ceil(504) };
    ids + dict < raw
}

fn delta_encoding(dt: &DataType) -> Encoding {
    match dt {
        DataType::Utf8 | DataType::Binary => Encoding::DELTA_BYTE_ARRAY,
        _ => Encoding::DELTA_BINARY_PACKED,
    }
}

fn writer_props(table: &str, batch: &RecordBatch, fmts: &Formats) -> WriterProperties {
    let mut b = WriterProperties::builder()
        .set_writer_version(if fmts.v2() { WriterVersion::PARQUET_2_0 } else { WriterVersion::PARQUET_1_0 })
        .set_max_row_group_row_count(None)
        .set_max_row_group_bytes(Some(ROW_GROUP_BYTES))
        .set_data_page_size_limit(PAGE_BYTES)
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_dictionary_page_size_limit(DICT_BYTES);
    for (i, f) in batch.schema().fields().iter().enumerate() {
        let path = ColumnPath::from(f.name().as_str());
        let c = fmts.col(table, f.name());
        b = b.set_column_compression(path.clone(), c.codec);
        b = match c.enc {
            Enc::Plain => b.set_column_dictionary_enabled(path.clone(), false).set_column_encoding(path, Encoding::PLAIN),
            Enc::Delta => b
                .set_column_dictionary_enabled(path.clone(), false)
                .set_column_encoding(path, delta_encoding(f.data_type())),
            Enc::Dict => {
                let keep = mr_keeps_dictionary(batch.column(i));
                // v1 dictionary columns fall back to PLAIN, as parquet-mr's v1 writer does
                b.set_column_dictionary_enabled(path.clone(), keep).set_column_encoding(path, Encoding::PLAIN)
            }
        };
    }
    b.build()
}

fn write_file(path: &Path, table: &str, pieces: &[RecordBatch], base: &[usize], rows: &[(u32, u32)], fmts: &Formats) -> u64 {
    let idx: Vec<(usize, usize)> = rows
        .iter()
        .map(|&(c, r)| (base[c as usize] + r as usize / PIECE, r as usize % PIECE))
        .collect();
    let schema: SchemaRef = pieces[0].schema();
    let cols: Vec<ArrayRef> = (0..schema.fields().len())
        .map(|c| {
            let arrs: Vec<&dyn Array> = pieces.iter().map(|b| b.column(c).as_ref()).collect();
            interleave(&arrs, &idx).expect("interleave")
        })
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let props = writer_props(table, &batch, fmts);
    let file = BufWriter::with_capacity(4 << 20, File::create(path).expect("create"));
    let opts = ArrowWriterOptions::new().with_properties(props).with_skip_arrow_metadata(true);
    let mut w = ArrowWriter::try_new_with_options(file, schema, opts).expect("writer");
    w.write(&batch).expect("write");
    w.close().expect("close");
    fs::metadata(path).unwrap().len()
}

// ------------------------------------------------------------------ driver
struct Ctx {
    sf: f64,
    files: usize,
    seed: u64,
    spark: SparkRead,
    fmts: Formats,
    dir: PathBuf,
    verify_lens: bool,
    bytes: AtomicU64,
    rows: AtomicU64,
}

fn run_batch(cx: &Ctx, table: &str, b: usize, chunks: &[(i32, i32)]) {
    let t0 = Instant::now();
    let gen: Vec<(Vec<RecordBatch>, Vec<u32>)> = chunks
        .par_iter()
        .map(|&(p, n)| {
            let d = gen_arrow(table, cx.sf, p, n);
            let l: Vec<u32> = d.iter().flat_map(text_lens_arrow).collect();
            if cx.verify_lens {
                assert!(l == text_lens(table, cx.sf, p, n), "{table} chunk {p}: .tbl lengths differ from tpchgen's text");
            }
            (d, l)
        })
        .collect();
    let (chunk_pieces, lens): (Vec<Vec<RecordBatch>>, Vec<Vec<u32>>) = gen.into_iter().unzip();
    let mut base = Vec::new(); // first piece of each chunk in `pieces`
    let mut pieces: Vec<RecordBatch> = Vec::new();
    for cp in chunk_pieces {
        assert!(cp.iter().rev().skip(1).all(|p| p.num_rows() == PIECE));
        base.push(pieces.len());
        pieces.extend(cp);
    }
    let t_gen = t0.elapsed().as_secs_f64();
    let tasks = map_tasks(&lens, &cx.spark);
    let tid = TABLES.iter().position(|t| *t == table).unwrap() as u64;
    let files = deal(&tasks, cx.files, &[cx.seed, tid, b as u64]);
    let t_deal = t0.elapsed().as_secs_f64() - t_gen;
    let dir = cx.dir.join(table);
    let bytes: u64 = files
        .par_iter()
        .enumerate()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(f, rows)| {
            let name = format!("part-{f:05}-b{b:05}-c000.{}.parquet", cx.fmts.suffix());
            write_file(&dir.join(name), table, &pieces, &base, rows, &cx.fmts)
        })
        .sum();
    let rows: usize = pieces.iter().map(|d| d.num_rows()).sum();
    cx.bytes.fetch_add(bytes, Ordering::Relaxed);
    cx.rows.fetch_add(rows as u64, Ordering::Relaxed);
    eprintln!(
        "  {table} batch {b}: {rows} rows, {} map tasks, {:.0} MB in {:.1}s (gen {:.1}s, deal {:.1}s, write {:.1}s)",
        tasks.len(),
        bytes as f64 / MB as f64,
        t0.elapsed().as_secs_f64(),
        t_gen,
        t_deal,
        t0.elapsed().as_secs_f64() - t_gen - t_deal
    );
}

fn main() {
    let a = Args::parse();
    let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = a.threads.unwrap_or(avail);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    let parallel = a.parallel.unwrap_or((2 * a.sf as i32).max(20));
    let fmts = match &a.format_map {
        Some(p) => {
            let doc: serde_json::Value = serde_json::from_reader(File::open(p).expect("format map")).unwrap();
            let key = a.map_key.clone().unwrap_or_else(|| {
                let keys: Vec<_> = doc.as_object().unwrap().keys().cloned().collect();
                assert_eq!(keys.len(), 1, "--map-key: the map has {keys:?}");
                keys[0].clone()
            });
            let engine = a.engine.as_deref().expect("--engine is required with --format-map");
            let map = doc[&key]["engines"][engine].clone();
            assert!(map.is_object(), "no engine {engine} under {key} in the map");
            Formats::Map { map, v2: a.map_page_version == "v2" }
        }
        None => {
            let dict = matches!(a.dictionary.as_str(), "true" | "1" | "yes");
            let v2 = a.writer_version == "v2";
            assert!(!(dict && v2), "--dictionary true with --writer-version v2 is not a profiled format");
            let enc = if dict { Enc::Dict } else if v2 { Enc::Delta } else { Enc::Plain };
            let suffix = if a.compression == "lz4raw" { "lz4raw".into() } else { a.compression.clone() };
            Formats::Uniform { col: ColFmt { enc, codec: codec(&a.compression) }, v2, suffix }
        }
    };
    let partial = a.out.join("_parquet.partial");
    let fin = a.out.join("parquet");
    assert!(!fin.exists(), "{} exists", fin.display());
    let _ = fs::remove_dir_all(&partial);
    let tables: Vec<String> = a.tables.clone().unwrap_or_else(|| TABLES.iter().map(|s| s.to_string()).collect());
    let cx = Ctx {
        sf: a.sf as f64,
        files: a.files,
        seed: a.seed,
        spark: SparkRead { cores: a.spark_cores.unwrap_or(avail as u64), max_partition_bytes: 128 * MB, open_cost: 4 * MB },
        fmts,
        dir: partial.clone(),
        verify_lens: a.verify_text_lens,
        bytes: AtomicU64::new(0),
        rows: AtomicU64::new(0),
    };
    eprintln!(
        "ndsgen SF{} PARALLEL={parallel} BATCH={} FILES={} seed={} spark_cores={} threads={threads} -> {}",
        a.sf, a.batch, a.files, a.seed, cx.spark.cores, fin.display()
    );
    // One work item per (table, batch), biggest tables first. A few driver threads take
    // items in order and run each one's stages on the shared pool, so one batch's
    // generation overlaps another's writing and in-flight memory stays bounded.
    let mut work: Vec<(&str, usize, Vec<(i32, i32)>)> = Vec::new();
    for table in &tables {
        fs::create_dir_all(partial.join(table)).unwrap();
        if table == "nation" || table == "region" {
            work.push((table, 0, vec![(1, 1)]));
        } else {
            let chunks: Vec<i32> = (1..=parallel).collect();
            for (b, c) in chunks.chunks(a.batch as usize).enumerate() {
                work.push((table, b, c.iter().map(|&i| (i, parallel)).collect()));
            }
        }
    }
    let t0 = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..a.concurrent_batches.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((table, b, chunks)) = work.get(i) else { break };
                run_batch(&cx, table, *b, chunks);
            });
        }
    });
    fs::rename(&partial, &fin).unwrap();
    let meta = serde_json::json!({
        "generator": "ndsgen", "tpchgen": "3.0.0", "scale_factor": a.sf, "parallel": parallel,
        "batch": a.batch, "files": a.files, "seed": a.seed, "spark_cores": cx.spark.cores,
        "format": match &a.format_map { Some(p) => format!("map {} {}", p.display(), a.engine.clone().unwrap()),
                                        None => format!("{} dict={} {}", a.compression, a.dictionary, a.writer_version) },
    });
    fs::write(a.out.join("NDSGEN.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    eprintln!(
        "done: {} rows, {:.2} GB, {:.1}s",
        cx.rows.load(Ordering::Relaxed),
        cx.bytes.load(Ordering::Relaxed) as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
}
