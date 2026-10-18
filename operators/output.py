import time
import torch
import pandas as pd
from conversion import num_to_str, float_to_date
from parsing import get_id, get_name
from utility.logger import message_logger
from IO.pages.combined_column_rep import CombinedColumnRep
from IO.pipeline.common import col_to_gpu
from IO.telemetry import gates, movement


def result_to_host(gpu_tensors: dict[int, torch.Tensor],
                   output_ids) -> dict[int, torch.Tensor]:
  """The output columns' copy to the host (the result's D2H), synchronized:
  the part of the output step the query's timer keeps."""
  out_tensors = {output_id: gpu_tensors[output_id].to('cpu', non_blocking=True)
                 for output_id in output_ids}
  torch.cuda.synchronize()
  return out_tensors


def host_frame(out_tensors: dict[int, torch.Tensor], out_types: dict[int, str],
               args) -> pd.DataFrame:
  """The result frame from the host copies of the output columns
  (``result_to_host``) and their type names (``out_types``): each column
  converted by its type, on the CPU.  The timer leaves this part out."""
  output_ids = args['output ids']
  result_names = args['output names']
  result = pd.DataFrame()

  tensor, type = None, None
  for output_id, name in zip(output_ids, result_names):
    tensor, type = out_tensors[output_id], out_types[output_id]
    if type == 'int':
      result[name] = tensor.squeeze() if tensor.dim() != 1 else tensor
    elif type == 'float':
      result[name] = tensor.squeeze() if tensor.dim() != 1 else tensor
    elif type == 'string':
      if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
      result[name] = [num_to_str(row) for row in tensor]
    elif type == 'date':
      result[name] = [float_to_date(row.item()) for row in tensor]
    else:
      raise ValueError(f"Unsupported tensor type: {type}")

  del tensor, type

  return result


def output_frame(gpu_tensors: dict[int, torch.Tensor], out_types: dict[int, str],
                 args) -> pd.DataFrame:
  """The result frame from the output columns' GPU tensors
  (``gpu_tensors``, by output id) and their type names (``out_types``):
  every column copied to the host, then converted by its type."""
  return host_frame(result_to_host(gpu_tensors, args['output ids']),
                    out_types, args)


def tqp_output_to_host(tensors: dict[int, CombinedColumnRep], args):
  """The output columns on the GPU (``col_to_gpu``), then copied to the host:
  (host tensors, type names) by output id, for ``host_frame``."""
  gpu_tensors = {}
  out_types = {}
  for output_id in args['output ids']:
    gpu_tensors[output_id] = col_to_gpu(tensors[output_id])
    out_types[output_id] = tensors[output_id].reps[0].tensor_type
  out_tensors = result_to_host(gpu_tensors, args['output ids'])
  if gates.EXPERIMENT:
    # The result's copy to the host, one page per column.
    nbytes = sum(t.numel() * t.element_size() for t in gpu_tensors.values())
    movement.record("result_d2h", len(gpu_tensors), nbytes, nbytes)
  return out_tensors, out_types


def tqp_output(tensors: dict[int, CombinedColumnRep], args) -> pd.DataFrame:
  return host_frame(*tqp_output_to_host(tensors, args), args)
