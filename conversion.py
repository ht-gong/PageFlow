import pandas as pd
import numpy as np

import torch
from torch import Tensor
from itertools import accumulate
import constants

YEAR_START = 1990
YEAR_END = 2000

epoch = pd.Timestamp(f'{YEAR_START}-01-01')
year_range = list(range(YEAR_START, YEAR_END))
year_edges = list(accumulate([365 if year % 4 else 366 for year in year_range]))

# num_of_ns_per_year = 365.2 * 24 * 60 * 60 * (10**9)

def is_date(s: str):
  return len(s) == 10 and s.count('-') == 2 and s.replace('-', '').isdigit()

def is_float(s: str):
  return s.replace('.', '').replace('-', '').replace('E', '').isdigit()

def date_to_float(date_str: str):
  date = pd.Timestamp(date_str)
  diff = (date - epoch).days
  return diff

def float_to_date(diff: float):
  return (epoch + pd.Timedelta(days=diff)).strftime('%Y-%m-%d')

def str_to_np(s, n):
  return np.pad([ord(c) for c in s], (0, n - len(s)), constant_values=0)

def num_to_str(tensor: Tensor):
  s = [chr(int(ele)) if int(ele) != 0 else '~' for ele in tensor]
  return "".join(s).rstrip('~')

def index_with_null(data, idx):
  """``data``'s rows at ``idx`` (1-D), with a null row (every element
  ``constants.null``, a 2-D row's included) wherever ``idx`` is
  ``constants.null``; all null rows when ``data`` has no row."""
  if data.shape[0] == 0:
    return torch.full((idx.numel(),) + tuple(data.shape[1:]), constants.null,
                      dtype=data.dtype, device=data.device)
  mask = idx == constants.null
  safe_idx = idx.masked_fill(mask, 0)          # 0 just needs to be a valid index
  data = data[safe_idx]
  data.masked_fill_(mask.view((-1,) + (1,) * (data.dim() - 1)), constants.null)
  return data
