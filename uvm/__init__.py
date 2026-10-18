"""The UVM mode's executor (``main.py --uvm``).

Every column and intermediate lives in CUDA managed memory (the arena over
one ``cudaMallocManaged`` range, ``memory/``), and the driver migrates its
pages to the GPU when a kernel touches them and evicts them when the GPU
budget is full.  The executor runs the same plans with the streaming
engine's leaf functions, as loops over morsels of plain tensor slices:

- ``column.py``: ``UColumn``, a managed tensor and its type name, and
  ``Payloads``, a query occurrence's columns keyed by dataflow instance;
- ``morsel.py``: the rows one kernel call takes;
- ``scan.py``: ``scan_to_managed``, the scan into managed tensors left on
  the GPU's host NUMA node (outside the timed region);
- ``executor.py``: one function per operator, dispatched by ``run``;
- ``aggregate.py``: the keyed aggregation, partitioned by hash;
- ``join.py``: the join, partitioned by hash;
- ``buckets.py``: bucketed morsel outputs regrouped into the calls that
  read them (pass-2 rounds, join core calls);
- ``gather.py``: the gather nodes (a join's rearrange, a lineage compose,
  a deferred gather).

It imports none of the streaming machinery (pipelines, record tables, the
pinned and slot pools, the cache, transfers, the searches, codecs):
``tests/uvm/test_imports.py`` checks the imports of every module here.
"""
