import contextlib
import io
import json
import os
import re
import shutil

from .expressions import (
    extract_number_after_hash,
    get_aggregation_key_and_values,
    get_id,
    get_name,
    get_project_output_ids,
)
from .materialization import apply_simple_materialization, seed_materialization
from .models import QueryPlanRep, ScannedTuple

class ParsingLayer:
    MASK_COL_START = 10_000
    # Placeholder for the scratch directory in captured plan text, so a
    # Location line reads the same wherever the capture ran.
    SCRATCH_PLACEHOLDER = "./tmp"

    def __init__ (self, scratch_dir="./tmp"):
        # pyspark is imported here, not at module level, so lowering a
        # captured plan (ParsingLayer.lower) needs neither pyspark nor a JVM.
        from pyspark.sql import SparkSession
        from pyspark.sql.types import StructType, DateType, StructField, IntegerType, StringType, DoubleType

        with open('./queries/schema.json', 'r') as f:
            schema_json = json.load(f)

        self.scratch_dir = scratch_dir
        self.spark = SparkSession.builder   \
        .config("spark.local.dir", scratch_dir) \
        .getOrCreate()
        
        # .config("spark.driver.memory", "1g") \
        # .config("spark.executor.memory", "1g") \
        # .config("spark.executor.memoryOverhead", "1g") \
        
        self.schema = {}
        type_mapping = {
            "identifier": IntegerType(),
            "integer": IntegerType(),
            "decimal": DoubleType(),
            "date": DateType(),
            "variable text": StringType(),
            "fixed text": StringType()
        }
        for table_name, table_schema in schema_json.items():
            fields = []
            for column_name, column_type in table_schema.items():
                if column_name != 'PRIMARY_KEY':
                    fields.append(StructField(column_name.lower(), type_mapping[column_type.split(',')[0]], False))

            self.schema[table_name.lower()] = StructType(fields)
            
        self.df = {}

        # Make sure the dataset is parquet (column store), not RDD (row store)
        # Scan parquest will discard unrelated columns

        for table, table_schema in self.schema.items():
            df_empty = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), table_schema)
            output_path = os.path.join(scratch_dir, f"tmpfile-{table}")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            df_empty.write.parquet(output_path)
            self.df[table] = self.spark.read.parquet(output_path)
            self.df[table].createOrReplaceTempView(table)

    def explain(self, query):
        """Spark's formatted physical plan for ``query``.

        The scratch directory in each scan's Location line is replaced by
        ``SCRATCH_PLACEHOLDER``.  The expression ids (``#NNN``) come from a
        counter in the JVM, so they depend on the pyspark version and on
        everything this JVM ran before; ``parsing/regenerate.py`` fixes both.
        """
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            self.spark.sql(query).explain(mode='formatted')
        plan_string = captured_output.getvalue()
        for path in {os.path.realpath(self.scratch_dir), os.path.abspath(self.scratch_dir)}:
            plan_string = plan_string.replace(f"file:{path}", f"file:{self.SCRATCH_PLACEHOLDER}")
        return plan_string

    def generate(self, query, qid):
        """Capture ``query``'s physical plan with Spark and lower it."""
        plan_string = self.explain(query)
        print(plan_string)
        return ParsingLayer.lower(plan_string, qid)

    @staticmethod
    def lower(plan_string, qid):
        """
        Parse a captured Spark formatted physical plan into QueryPlanReps, one per
        plan (a subquery plan first).  Pure Python: no Spark session is needed.
        Args:
            plan_string (str): output of ``explain(mode='formatted')``.
            qid (int): the TPC-H query id (selects the per-query special cases).
        Returns:
            list[QueryPlanRep]
        """
        # The expression grammar (pyparsing, the raised recursion limit and the
        # packrat cache) loads only in a process that lowers a plan; importing
        # ``parsing`` stays free of it.
        from .expression_parser import parse

        # Mask column ids are allocated per query, across all of its plans.
        mask_col_id = ParsingLayer.MASK_COL_START

        def get_next_mask_id(used_set):
            nonlocal mask_col_id
            while mask_col_id in used_set:
                mask_col_id += 1
            ret = mask_col_id
            mask_col_id += 1
            return ret

        def split_plan(plan_lines):
            tree_start_pos = next(i for i, line in enumerate(plan_lines) if line.startswith("+-"))
            tree_lines = plan_lines[tree_start_pos:]
            tree_end_pos = next(i for i, line in enumerate(tree_lines) if not line.strip())
            item_lines = tree_lines[tree_end_pos:]
            tree_lines = tree_lines[:tree_end_pos]
            return tree_lines, item_lines
        
        # Split the physical plan into "tree" and "item" parts
        subquery_mark = '===== Subqueries ====='
        sub_tree_lines, sub_item_lines = None, None
        if subquery_mark in plan_string:
            plan_string, sub_plan_string = plan_string.split(subquery_mark) # Throw it away
            sub_plan_lines = sub_plan_string.split("\n")[3:]
            sub_tree_lines, sub_item_lines = split_plan(sub_plan_lines)
            
        plan_lines = plan_string.strip().split("\n")
        tree_lines, item_lines = split_plan(plan_lines)


        class TreeNode:
            def __init__(self, id, depth):
                self.id = id
                self.depth = depth
                self.children = []

            def add_child(self, child_node):
                self.children.append(child_node)

        def parse_tree(lines, children, args):
            root = None
            stack = []
            for line in lines:
                depth = len(line) - len(line.lstrip(" -:+"))
                id = int(line[depth:-1].split('(')[-1])
                node = TreeNode(id, depth)
                if 'BuildRight' in line:
                    args[id]['build'] = 'right'
                elif 'BuildLeft' in line:
                    args[id]['build'] = 'left'
                
                if root == None:
                    root = node
                    stack = [node]
                else:
                    while stack and stack[-1].depth >= depth:
                        stack.pop()
                    stack[-1].add_child(node)
                    children.setdefault(stack[-1].id, []).append(node.id)
                    stack.append(node)

        def parse_item(item_lines):
            ops = []
            args = {}
            scanned = []
            op = None
            for line in item_lines:
                if not line:
                    continue
                if line.startswith('('): # (ID) Operator Name
                    op = int(line[1:].split(')')[0])
                    ops.append(op)
                    name = line.split(')')[1].strip()
                    args[op] = {'name': name}
                else: # [key] : value
                    # Results [1]: [((100.0 * sum(CASE WHEN StartsWith(p_type#31, PROMO) THEN (l_extendedprice#243 * (1.0 - l_discount#244)) ELSE 0.0 END)#306) / sum((l_extendedprice#243 * (1.0 - l_discount#244)))#307) AS promo_revenue#305]
                    if line.startswith('Subquery:2'):
                        break
                    key = line.split('[')[0].split(':')[0].strip()

                    sp = 'Results [1]:'
                    
                    # This is a special case for TPC-H Q15, where a comma is present inside a substring function argument list.
                    # The default comma-based splitting logic fails for this line.
                    # Example line: ...: [substring(c_phone#133, 1, 2) AS cntrycode#305, c_acctbal#134]
                    match = re.search(r'substring\(c_phone#(\d+), 1, 2\) AS cntrycode#(\d+), (c_acctbal#\d+)\]$', line)
                    if match:
                        args[op][key] = [f'substring(c_phone#{match.group(1)}, 1, 2) AS cntrycode#{match.group(2)}', match.group(3)]
                        continue

                    match = re.search(
                        r'\(0\.5 \* cast\(sum\(l_quantity#(\d+)\)#(\d+)L as decimal\(20,0\)\)\) AS '
                        r'\(0\.5 \* sum\(l_quantity\)\)#(\d+), l_partkey#(\d+), l_suppkey#(\d+)\]$',
                        line
                    )
                    if match:
                        args[op][key] = [
                            f'(0.5 * (sum(l_quantity#{match.group(1)})#{match.group(2)}L)) AS (0.5 * sum(l_quantity))#{match.group(3)})',
                            f'l_partkey#{match.group(4)}',
                            f'l_suppkey#{match.group(5)}',
                        ]
                        continue

                    match = re.search(
                        r'cast\(ps_availqty#(\d+) as decimal\(\d+,\d+\)\) > \(0\.5 \* sum\(l_quantity\)\)#(\d+)\)$', 
                        line
                    )

                    if match:
                        args[op][key] = [
                            f'(ps_availqty#{match.group(1)}) > (0.5 * sum(l_quantity))#{match.group(2)})'
                        ]
                        continue

                    if line.startswith(sp):
                        args[op][key] = [line[len(sp):].strip('[ ]')]
                    elif key == 'Condition' or key == 'Join condition':
                        args[op][key] = [line.split(':')[1].strip()]
                    else:
                        args[op][key] = [item.strip() for item in line.split(':', 1)[1].strip('[ ]').split(',')]
            
            for u in list(args.keys()):
                args_u = args[u]
                name = args_u['name']

                if name == 'Scan parquet':
                    args_u['name'] = 'Scan'
                    # Location: InMemoryFileIndex [file:../tmp/tmpfile-lineitem]
                    args_u['table name'] = args_u['Location'][-1].split('-')[-1].upper()
                    # Output [7]: [l_quantity#242, l_extendedprice#243, l_discount#244, l_tax#245, l_returnflag#246, l_linestatus#247, l_shipdate#248]
                    args_u['output names'] = [item.split('#')[0] for item in args_u['Output']]
                    args_u['output ids'] = [get_id(item.split('#')[1]) for item in args_u['Output']]
                    scanned.extend([ScannedTuple(name, identifier, args_u['table name']) for name, identifier in zip(args_u['output names'], args_u['output ids'])])
                    args_u.pop('Output')

                elif name == 'Filter':
                    args_u['input ids'] = [get_id(col.split('#')[-1]) for col in args_u.pop('Input')]
                    args_u['filter tree'] = parse(args_u.pop('Condition')[0])
                    args_u['output ids'] = args_u['input ids'].copy()

                elif name == 'Project':
                    if args_u['Output'] == ['']:
                        args_u['output'] = []
                    else:
                        args_u['output'] = [parse(col) for col in args_u.pop('Output')]
                    args_u['input ids'] =  [get_id(col) for col in args_u.pop('Input')]
                    args_u['output ids'] = get_project_output_ids(args_u['output'])
                
                elif name.endswith('Aggregate'):
                    args_u['name'] = 'Aggregate'
                    args_u['is partial'] = args_u.pop('Functions')[0].startswith("partial")
                    if args_u['Keys'] != ['']:
                        args_u['keys'] = args_u.pop('Keys')
                    else:
                        args_u['keys'] = None

                    args_u['input ids'] =  [get_id(col) for col in args_u.pop('Input')]
                    args_u['results'] = [parse(col) for col in args_u.pop('Results')]
                    args_u['in key ids'], args_u['out key ids'], args_u['out agg ids'], args_u['agg ops'] = get_aggregation_key_and_values(args_u['results'])
                    args_u['output ids'] = args_u['out key ids'] + args_u['out agg ids']
                    print('Results', args_u['results'])
                 
                elif name.endswith('Exchange'):
                    args_u['name'] = 'Exchange'
                
                elif name == 'Sort':
                    args_u['input ids'] = [get_id(col) for col in args_u.pop('Input')]
                                # args_u['Arguments'],
                    # # [l_returnflag#56 ASC NULLS FIRST, l_linestatus#57 ASC NULLS FIRST], true, 0
                    keys = args_u.pop('Arguments')[:-2]
                    args_u['key ids']  = [get_id(key.split(' ')[0]) for key in keys]
                    args_u['key orders'] = [key.split(' ')[1] == 'DESC' for key in keys] # ASC or DESC
                    args_u['output ids'] = args_u['input ids'].copy()

                elif name == 'TakeOrderedAndProject':
                    args_u['name'] = 'take ordered'
                    args_u['input ids'] = [get_id(col) for col in args_u.pop('Input')]
                    # Arguments: 10, "[revenue#305 DESC NULLS LAST, o_orderdate#111 ASC NULLS FIRST]", "[l_orderkey#48, revenue#305, o_orderdate#111, o_shippriority#114]"
                    ag_str = args_u.pop('Arguments')
                    indices = [i for i, s in enumerate(ag_str) if '[' in s]

                    # Reconstruct the key and output strings based on bracket positions
                    args_u['limit'] = int(ag_str[0])
                    keys_str = ag_str[indices[0]: indices[1]]
                    outputs_str = ag_str[indices[1]:]
                    args_u['key ids'] = [get_id(item.strip().split(' ')[0]) for item in keys_str]
                    args_u['key orders']  = [item.strip().split('#')[-1].split(' ')[1] == 'DESC' for item in keys_str]
                    args_u['output ids'] = [get_id(item.strip()) for item in outputs_str]

                elif name.endswith('Join'):
                    args_u['name'] = 'Join'
                    args_u['left keys'] = [extract_number_after_hash(ele) for ele in args_u.pop('Left keys')]
                    args_u['right keys'] = [extract_number_after_hash(ele) for ele in args_u.pop('Right keys')]
                    args_u['join type'] = args_u.pop('Join type')[0]

                    if 'Join condition' in args_u.keys():
                        if args_u['Join condition'][0] == 'None':
                            args_u.pop('Join condition')
                        else:
                            args_u['join condition'] = parse(args_u.pop('Join condition')[0])
                
                elif name == 'AdaptiveSparkPlan':     # Treated as output operator
                    args_u['output ids'] = [get_id(col) for col in args_u['Output']]
                    args_u['output names'] = [get_name(col) for col in args_u['Output']]
                    args_u.pop('Output')
                
            return ops, args, scanned
        
        def get_child_tree(plan: QueryPlanRep):
            successor_tree = {} # Maps node in plan to successor
            for i in plan.ops_idx:
                for parent in plan.query_parent_map.get(i, []):
                    assert parent not in successor_tree, "One node should not have multiple parents in plan"
                    successor_tree[parent] = i
            return successor_tree

        def remove_exchange(plan: QueryPlanRep):
            marked_del = []
            for i in plan.ops_idx:
                if plan.ops_dict[i]['name'] == 'Exchange':
                    parent = plan.query_parent_map[i]
                    assert len(parent) == 1, "Exchange should only have 1 parent"
                    
                    child = plan.query_child_map[i]
                    plan.query_child_map[parent[0]] = child
                    ind = plan.query_parent_map[child].index(i)
                    plan.query_parent_map[child][ind] = parent[0]
                    marked_del.append(i)
            
            for d in marked_del:
                plan.ops_idx.remove(d)
                plan.query_child_map.pop(d)
                plan.query_parent_map.pop(d)
                plan.ops_dict.pop(d)
                        
        def is_partial(plan: QueryPlanRep, i):
            # Spark marks a partial aggregate by its "partial_" functions.  A
            # group-only aggregate, such as the dedup on (p_brand, p_type,
            # p_size, ps_suppkey) below Q16's count(DISTINCT ps_suppkey), has
            # no function to mark, so its partial phase is the one with an
            # Exchange and an aggregate on the same keys, also without
            # functions, right above it.
            args = plan.ops_dict[i]
            if args['is partial']:
                return True
            final = plan.ops_dict.get(i + 2, {})
            return (not args['agg ops']
                    and plan.ops_dict.get(i + 1, {}).get('name') == 'Exchange'
                    and plan.query_parent_map.get(i + 1) == [i]
                    and final.get('name') == 'Aggregate'
                    and plan.query_parent_map.get(i + 2) == [i + 1]
                    and not final['agg ops'] and final['keys'] == args['keys'])

        def clean_agg(plan: QueryPlanRep):
            ## Collapse each partial/final aggregate pair onto the final aggregate
            marked_del = []
            for i in plan.ops_idx:
                if plan.ops_dict[i]['name'] == 'Aggregate' and is_partial(plan, i):
                    assert plan.ops_dict[i+1]['name'] == 'Exchange' and plan.ops_dict[i+2]['name'] == 'Aggregate'
                    plan.ops_dict[i+2]['input ids'] = plan.ops_dict[i]['input ids'] # Only need to swap input ids
                    plan.query_parent_map[i+2] = plan.query_parent_map[i]
                    marked_del.extend([i+1, i])
                    

            for d in marked_del:
                plan.ops_idx.remove(d)
                plan.query_parent_map.pop(d)
                plan.ops_dict.pop(d)

        parsed = []
        for trees, lines in zip([sub_tree_lines, tree_lines], [sub_item_lines, item_lines]):
            if lines is not None:
                plan = QueryPlanRep()
                plan.ops_idx, plan.ops_dict, plan.disk_read_cols = parse_item(lines)
                parse_tree(trees, plan.query_parent_map, plan.ops_dict)
                clean_agg(plan)
                plan.query_child_map = get_child_tree(plan)
                remove_exchange(plan)
                seed_materialization(plan)
                apply_simple_materialization(plan, get_next_mask_id)
                parsed.append(plan)
        if qid == 11 and len(parsed) > 1:
            # TPC-H defines Q11's fraction as 0.0001 / SF; tpch-q11.sql hard-codes
            # the SF1 value, so the subquery's scalar is rescaled at runtime
            # (parsing.pipeline.subquery_value).
            parsed[0].subquery_scale = "1/SF"
        return parsed

        
