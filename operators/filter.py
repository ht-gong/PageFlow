import torch

from IO.pipeline.vortex_pipeline import VortexPipeline
from utility.logger import perf_logger, message_logger
from IO.pages.combined_column_rep import CombinedColumnRep
from operators.operator_utils import evaluate_filter
import constants
import sizing_resolver

def filter_chunk(cols, tree, subquery_result):
  """The filter's mask over one chunk: ``filter tree`` evaluated over
  ``cols`` (by id, the chunk's rows), a 1-D bool tensor."""
  rows = next(iter(cols.values())).size(0)
  mask = evaluate_filter('cuda', cols, tree, rows, subquery_result)
  assert mask.ndim != 0, f"Filter mask is 0-dim; filter tree: {tree}"
  return mask


def tqp_filter(tensor_group, args, subquery_result, cpu_mem_pool, cache_region, name, applier=None, page_cache=None):
  perf_logger().start(name)

  assert 'mask map' not in args, "Filter not accepting masks"

  tree = args['filter tree']
  _chunk_sums = []
  _chunk_counts = []

  if len(args['materialize_in']) == 0:
    message_logger().debug("Filter not dispatched")
    perf_logger().stop(name)
    return -1
  # The filter's graph node: its inputs are the predicate columns' current
  # instances, its one output the mask instance.
  node = args['nodes']['pipe']

  def filter_wrapper(cols):
    mask = filter_chunk(cols, tree, subquery_result)
    _chunk_sums.append(mask.sum())      # stays on GPU, no sync
    _chunk_counts.append(mask.numel())
    return (mask,)


  filter_pipe_name = node.name
  pipe = VortexPipeline(
    node=node,
    inputs=node.input_pairs(),
    output_columns_type=[torch.bool],
    operator=filter_wrapper,
    chunk_size=sizing_resolver.resolve("filter", filter_pipe_name).chunk_b,
    cpu_mem_pool=cpu_mem_pool,
    sizing_key=("filter", filter_pipe_name),
    applier=applier,
    page_cache=page_cache,
  )
  pipe.do_exchange()

  pipe.get_result()

  # Store mask in mask id — pipeline is done, safe to sync
  mask_true_count = int(torch.stack(_chunk_sums).sum().item()) if _chunk_sums else 0
  mask_total_count = sum(_chunk_counts)
  selectivity = mask_true_count / mask_total_count if mask_total_count > 0 else 1.0
  message_logger().debug("filter selectivity: %.4f (true=%d, total=%d)", selectivity, mask_true_count, mask_total_count)
  # The mask instance carries its selectivity (the zero-copy mask decision).
  mask_inst = node.outputs[0]
  mask_inst.ccr = CombinedColumnRep([pipe.output_variable(
      0, 'bool', selectivity=selectivity)])
  tensor_group[args['out mask id']] = mask_inst.ccr

  for input_col in args['input ids']:
  # Free projected out tensors
    if input_col in tensor_group and input_col not in args['output ids']:
      tensor_group[input_col].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=cache_region)
      del tensor_group[input_col]


  perf_logger().stop(name)
  return -1
