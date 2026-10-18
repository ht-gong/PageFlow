import torch
import time
import os
import constants
from variable import Variable

def tqp_scan(scanned_group, tensor_group, args, config_json, schema_table, SF, mem_pool, name="scan"):
  table_name, col_names, col_ids = args['table name'], args['output names'], args['output ids'] 
  for col_name, col_id in zip(col_names, col_ids):
    tensor_group[col_id] = scanned_group[table_name][col_name.upper()].pop()
  