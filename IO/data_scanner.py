"""The scan: each column a query reads, loaded from the datagen parquet into
the pinned host pool, as RAW plus at most one compressed representation,
before the query's timed region.

Each representation is a record table of regions
(``IO/pages/record_table.py``, ``variable.TableVariable``): one region per
scan batch of at most ``COMPRESSION_BATCH_SIZE_B`` raw bytes, in the pinned
pool, given back when the table is freed.  Records are cut on rows
(``region_producer.record_rows``): each a power-of-two number of rows
within the record cap, cut from row 0, and a batch is a whole number of
``region_producer.row_quantum`` rows (2^20 under the record row cap, at least one
quantum), so batch boundaries are record boundaries of every column of the
table.  Every representation is written by a region producer
(``IO/codec/region_producer.py``), the path every pipe's outputs take: each
batch is one tick, its records of whole rows compressed on the GPU (or
copied, for RAW) and packed into the batch's region in its home, carved
from the producer's pinned chunks.  A slice of a column (a tick's rows)
holds no reference and frees nothing.

The scan reads each table once per query, the columns the query needs from
it together, decoded on the GPU in chunks (``IO/parquet_reader.py``), and
cuts each column's rows into the batches across chunk and part-file
boundaries (``_ParquetColumn``), so no batch is staged in pageable memory.
The scan marks no column sorted: a column's compressed rep is its type's
unsorted codec class, the class the dataflow graph gives its scan instance.
"""
import functools
import torch
from variable import TableVariable
from utility.logger import message_logger
import constants
from IO.pages.combined_column_rep import CombinedColumnRep
from collections import defaultdict
from IO import parquet_reader
from TQPlib.tqpcompress import mgr_release_scratch
from utility import rmm_arena


TYPE_MAPPING = {
    "identifier": "int",
    "integer": "int",
    "decimal": "float",
    "date": "date",
    "variable text": "string",
    "fixed text": "string"
}

COMPRESSION_BATCH_SIZE_B = 256 * 1024 ** 2


def engine_dtype(read_dtype, tensor_type):
    """The engine's dtype for a column of ``tensor_type`` the reader gives
    as ``read_dtype``."""
    if tensor_type == 'date':
        return constants.date_dtype
    elif read_dtype.is_floating_point:
        return constants.float_dtype
    elif tensor_type == 'string':
        return constants.string_dtype
    else:
        return constants.int_dtype


def cast(read, tensor_type):
    """A column in the reader's encoding, cast to the engine's dtype for
    ``tensor_type``."""
    return read.to(engine_dtype(read.dtype, tensor_type))


def column_type(schema_table, table, col_name):
    """The engine's type name (``TYPE_MAPPING``) of ``table``'s column
    ``col_name`` (upper case) in the schema."""
    return TYPE_MAPPING[schema_table[table][col_name].split(',')[0]]


def scan_occurrences(to_scan):
    """``{table: {COLUMN: occurrences}}`` of the columns ``to_scan`` reads, in
    the order it first names them."""
    occurrences = defaultdict(lambda: defaultdict(int))
    for scanned_tuple in to_scan:
        occurrences[scanned_tuple.table][scanned_tuple.name.upper()] += 1
    return occurrences


def ensure_2d(tensor):
    """A column as a 2-D tensor (rows x width)."""
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    return tensor


@functools.cache
def parquet_table(root, SF, table):
    """``table``'s ``ParquetTable`` under ``root`` at scale factor ``SF``,
    its footers read once per process."""
    return parquet_reader.ParquetTable(root, SF, table.lower())


def scan_to_ram(config_json, schema_table, SF, cpu_mem_pool, to_scan, policy=None):
    """Load ``to_scan`` into the pinned pool as record tables (module
    docstring).

    Per table, one chunked read of the columns the query needs from it, on
    a reader stream; per column, a ``_ParquetColumn`` whose producers
    compress its batches on a producer stream and ship them home on the
    home writer's stream.  A chunk's batches are issued before the next
    chunk is read, so the next read and decode overlap their compress and
    copies; after a chunk's batches the host waits for the copies issued up
    to the end of the chunk before, so at most one chunk's copies are in
    flight while the next one is read.  A table's producers are drained
    before the next table is read: until its pack, a producer's last tick
    holds its batch, a view of the table's last chunk, and the tick's
    compress slabs."""
    root = config_json.get('tables path')

    # The active compression mode decides which reps to materialize: instead
    # of RAW + all three codecs unconditionally, build RAW + at most ONE
    # compressed format — fixed_X -> {RAW, X}; RAW -> {RAW}; adaptive ->
    # {RAW, class-format(type, unsorted)}.  Same CompressionPolicy the
    # applier selects from, so the materialized compressed rep always matches
    # the codec the runtime requests.
    if policy is None:
        from adaptive.instage_optimization import active_compression_policy
        policy = active_compression_policy()

    from IO.codec.home_writer import writer_stream
    from utility import cuda_streams
    reader_stream = cuda_streams.compute_stream()
    producer_stream = cuda_streams.compute_stream()
    occurrences = scan_occurrences(to_scan)
    columns = []
    shipped = []
    scan_group = defaultdict(lambda: defaultdict(list))
    try:
        # cuDF's small buffers (one per part file, for a whole read pass)
        # share arena blocks.
        with torch.cuda.stream(reader_stream), rmm_arena.small_buffers():
            for table, counts in occurrences.items():
                pt = parquet_table(root, SF, table)
                feeds = {}
                for col_name, n in counts.items():
                    tensor_type = column_type(schema_table, table, col_name)
                    dtype, width = pt.encoding(col_name)
                    feeds[col_name] = _ParquetColumn(
                        tensor_type, policy.formats_for(tensor_type, False),
                        n, pt.rows, width * dtype.itemsize, cpu_mem_pool,
                        producer_stream)
                    columns.append(feeds[col_name])
                    message_logger().info("scanned %s", col_name)
                for _, chunk in pt.batches(list(feeds)):
                    for name, tensor in chunk.items():
                        feed = feeds[name]
                        feed.feed(ensure_2d(cast(tensor, feed.tensor_type)))
                    event = torch.cuda.Event()
                    event.record(writer_stream())
                    shipped.append(event)
                    if len(shipped) > 1:
                        shipped.pop(0).synchronize()
                for feed in feeds.values():
                    feed.flush()
                for col_name, feed in feeds.items():
                    scan_group[table][col_name] += feed.finish()
    except BaseException:
        for feed in columns:
            feed.abort()
        raise
    # The producers' nvCOMP managers keep their scratch in the arena until it
    # is released, and they are keyed by this scan's producer stream, which
    # nothing after the scan uses: release it, so the query starts with the
    # arena the scan found.
    mgr_release_scratch()
    return dict(scan_group)


class _ParquetColumn:
    """One column of a table the scan reads, and its occurrences in the
    query.  Its decoded rows arrive chunk by chunk (``feed``) and are cut
    into the scan's batches, ``_batch_rows`` each from row 0 and the last
    one shorter, across chunk boundaries: a batch inside a chunk is a view
    of it, one that spans chunks is concatenated, and one that does not
    start ``COMPRESSION_ALIGNMENT``-aligned is copied, so every record the
    producer cuts from it starts aligned.  Each batch is one tick of every
    occurrence's region producer, whose outputs are the occurrence's
    representations (instance ``i``: ``formats[i]``), its RAW one included.
    A producer packs a tick's compressed records at the next tick, so the
    host waits for a compress only one batch later.  Each occurrence has a
    producer of its own: one producer for all of them returned a wrong or
    empty Q15 (which scans lineitem twice) in whole SF100 passes, for a
    reason not known.

    The producers run on ``stream``, current around each call, behind the
    caller's stream, which cut the batch from the reader's chunks.  A
    batch's memory goes back to the arena ordered on the stream it came
    from only (``IO/parquet_reader.py``), so the column holds each batch
    until every producer has packed it, its compresses then ended; the home
    writer holds a RAW output's batch until its copies end."""

    def __init__(self, tensor_type, formats, occurrences, rows, row_bytes,
                 pool, stream):
        from IO.codec.region_producer import RegionProducer
        self.tensor_type = tensor_type
        self.formats = formats
        self.stream = stream
        self.batch_rows = _batch_rows(row_bytes)
        self.ticks = -(-rows // self.batch_rows)
        self.done = 0
        self.parts = []          # the next batch's rows so far (views)
        self.part_rows = 0
        self.held = None         # the last batch, until every producer packs it
        self.drained = 0         # producers drained so far
        self.producers = []
        for _ in range(occurrences):
            producer = RegionProducer(pool, "scan", spills=False,
                                      pool_tag="scan")
            # What is in flight is bounded per chunk (``scan_to_ram``), not
            # per tick: the writer never waits for a tick's copies.
            producer.writer.cap = self.ticks + 1
            self.producers.append(producer)

    def feed(self, tensor):
        """The column's next decoded rows (2-D, cast), on the current
        stream."""
        rows = tensor.shape[0]
        at = 0
        while at < rows:
            take = min(self.batch_rows - self.part_rows, rows - at)
            self.parts.append(tensor[at:at + take])
            self.part_rows += take
            at += take
            if self.part_rows == self.batch_rows:
                self._tick()

    def flush(self):
        """The column's last rows have been fed: its last batch."""
        if self.part_rows:
            self._tick()

    def _tick(self):
        parts, self.parts, self.part_rows = self.parts, [], 0
        batch = parts[0] if len(parts) == 1 else torch.cat(parts)
        if batch.data_ptr() % constants.COMPRESSION_ALIGNMENT:
            batch = batch.clone()
        items = [(i, batch, [batch.shape[0]], algo)
                 for i, algo in enumerate(self.formats)]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for producer in self.producers:
                # The ticks still to produce, the one a pack packs included.
                producer.ticks_left = self.ticks - self.done + 1
                if producer.has_pending():
                    producer.pack()
                producer.compress(items)
        # Every producer has packed the last batch: it can go.
        self.held = batch
        self.done += 1

    def finish(self):
        """Pack the last ticks, wait for the copies home and return each
        occurrence's ``CombinedColumnRep``."""
        out = []
        for producer in self.producers:
            with torch.cuda.stream(self.stream):
                producer.drain()
            self.drained += 1
            tables = producer.tables()
            reps = []
            for base in range(len(self.formats)):
                table = tables[base]
                table.logical_type = self.tensor_type
                table.base = True
                reps.append(TableVariable(table, self.tensor_type))
            out.append(CombinedColumnRep(reps))
        self.held = None
        return out

    def abort(self):
        for producer in self.producers[self.drained:]:
            producer.abort()
        self.held = None


def _batch_rows(row_bytes):
    """Rows per scan batch: the most whole ``row_quantum`` row runs that
    fit ``COMPRESSION_BATCH_SIZE_B``, at least one."""
    from IO.codec.region_producer import row_quantum
    q = row_quantum(row_bytes)
    return max(1, COMPRESSION_BATCH_SIZE_B // row_bytes // q) * q
