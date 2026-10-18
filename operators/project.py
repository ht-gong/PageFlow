import torch
from IO.pipeline.vortex_pipeline import VortexPipeline
from operators.operator_utils import evaluate_expr, get_evaluable_project_expr, infer_project_output_types
from variable import torch_to_type
from utility.logger import message_logger
from IO.pages.combined_column_rep import CombinedColumnRep
import constants
import sizing_resolver

def project_exprs(used_exprs):
  """The project's computed expressions, in output order: its ``used
  expr`` trees but the pass-through columns and renames."""
  return [expr for expr in map(get_evaluable_project_expr, used_exprs)
          if expr is not None]


def project_chunk(cols, exprs, out_types):
  """The project's computed columns over one chunk: each of ``exprs``
  (``project_exprs``) evaluated over ``cols`` (by id, the chunk's rows),
  checked to be of its ``out_types`` dtype."""
  outs = []
  for expr, dtype in zip(exprs, out_types):
    result = evaluate_expr('cuda', cols, expr, None, None, None)
    assert result.dtype == dtype, f"Unexpected project expr dtype: {result.dtype} != {dtype}"
    outs.append(result)
  return outs


def tqp_project(tensor_group, args, cpu_mem_pool, cache_region, name="project", applier=None, page_cache=None):
  ### Handle renames
  for orig, new in args['rename map'].items():
    tensor_group.alias(orig, new)

  if len(args['used expr']) > 0:
    # The project's graph node: inputs are the expression columns' current
    # instances, outputs the computed columns.
    node = args['nodes']['pipe']
    input_type_map = {col_id: tensor_group[col_id].get_type_torch() for col_id in args['materialize_in']}
    output_columns_type = infer_project_output_types(args['used expr'], input_type_map)

    exprs = project_exprs(args['used expr'])

    def project_wrapper(columns):
      # Expression evaluation returns the inferred dtype directly.
      return [t.contiguous()
              for t in project_chunk(columns, exprs, output_columns_type)]

    assert 'mask map' not in args

    project_pipe_name = node.name
    assert [inst.local for inst in node.outputs] == list(args['materialize_now'])
    pipe = VortexPipeline(
      node=node,
      inputs=node.input_pairs(),
      output_columns_type=[constants.float_dtype] * len(args['materialize_now']),
      operator=project_wrapper,
      chunk_size=sizing_resolver.resolve("project", project_pipe_name).chunk_b,
      cpu_mem_pool=cpu_mem_pool,
      sizing_key=("project", project_pipe_name),
      applier=applier,
      page_cache=page_cache,
    )
    pipe.do_exchange()
    pipe.get_result()

    # Re-wrap each output instance's payload under the inferred type name
    # (the pipeline declares float slots); the cross-operator map points at
    # the instances.
    for i, col_id in enumerate(args['materialize_now']):
      inst = node.outputs[i]
      inst.ccr = CombinedColumnRep([pipe.output_variable(
          i, torch_to_type(output_columns_type[i]))])
      tensor_group[col_id] = inst.ccr

  rename_map = args.get('rename map', {}) or {}
  for input_col in args['input ids']:
    if input_col not in tensor_group or input_col in args['output ids']:
      continue
    # Aliased inputs share their CCR object with the renamed output id
    # (see the ``tensor_group.alias(orig, new)`` block above).  Freeing
    # via the old key here would release the backing memory while the
    # new key — and every downstream consumer — still references it,
    # so the cpu_mem_pool's free list immediately reuses those blocks
    # and the next allocation corrupts the column's bytes.  Drop the
    # stale key without releasing storage.
    if input_col in rename_map and rename_map[input_col] in tensor_group:
      del tensor_group[input_col]
    else:
      tensor_group[input_col].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=cache_region)
      del tensor_group[input_col]

  return -1
