"""PyParsing grammar for Spark physical-plan expressions."""

import re
import sys
import time

from pyparsing import (
    Combine,
    DelimitedList,
    Forward,
    Group,
    Keyword,
    Literal,
    OpAssoc,
    Optional,
    ParserElement,
    Suppress,
    Word,
    alphanums,
    infix_notation,
    nums,
    one_of,
)

current_limit = sys.getrecursionlimit()
sys.setrecursionlimit(4096)

# print(f"Recursion limit: {current_limit} --> 4096")

ParserElement.enable_packrat(cache_size_limit=None, force=True) # Accelerate

year, month, day = Word(nums, exact=4), Word(nums, exact=2), Word(nums, exact=2)
date_expr = Combine(year + '-' + month + '-' + day)

identifier = Word(alphanums + "_%#")
# level = Combine(nums + '-' + Word(alphas))
integer = Combine(Word(nums) + Optional(Literal('.') + Word(nums)) + Optional('L') + Optional(one_of("e E") + Optional(one_of("+ -")) + Word(nums)))
lparen, rparen = Suppress("("), Suppress(")")

comparison_op = one_of("= > < >= <= <>")
logical_op = one_of("AND OR")
not_op = Keyword("NOT")
arithmetic_op = one_of("+ - * /")
in_op = Keyword("IN")
like_op = Keyword("LIKE")
# as_op = one_of("AS as", as_keyword=True)
as_op = Keyword("AS")
hash_op = Literal("#")
# case_op = one_of("CASE WHEN THEN ELSE END")
aggregate = one_of("sum avg count year max min isnotnull", as_keyword=True)

multiple_word = DelimitedList(~(logical_op | not_op | in_op | like_op | as_op) + identifier, delim=" ", combine=True)

parsed_expr = Forward()
case_when_expr = Group(
    Keyword("CASE")
    + Keyword("WHEN") + ((lparen + parsed_expr + rparen) | parsed_expr)
    + Keyword("THEN") + ((lparen + parsed_expr + rparen) | parsed_expr)
    + Keyword("ELSE") + ((lparen + parsed_expr + rparen) | parsed_expr)
    + Keyword("END")
)
func_call = Group(identifier + Group(lparen + DelimitedList(case_when_expr | date_expr | multiple_word | integer | identifier) + rparen))
tuples    = Group(             lparen + DelimitedList(multiple_word | date_expr | integer | identifier) + rparen) 

operand = case_when_expr | date_expr | integer | func_call | multiple_word | tuples | identifier
parsed_expr <<= infix_notation(
    operand,
    [
        (aggregate, 1, OpAssoc.RIGHT),
        (hash_op, 2, OpAssoc.LEFT),
        (arithmetic_op, 2, OpAssoc.LEFT),
        (in_op | like_op, 2, OpAssoc.LEFT),
        (not_op, 1, OpAssoc.RIGHT),
        (comparison_op, 2, OpAssoc.LEFT),
        (logical_op, 2, OpAssoc.LEFT),
        (as_op, 2, OpAssoc.LEFT)
    ]
)

tot = 0

def parse(s: str):
  global tot
  print ("s=", s)
  
  #### TPC-H SPECIAL CASES ####
  # Case 1
  match = re.match(r'^\(isnotnull\(value#(\d+)\) AND \(value#\d+ > Subquery subquery#(\d+), \[id=#(\d+)\]\)\)$', s)
  if match:
    n1 = match.group(1)
    return ['isnotnull', [f'value#{n1}']], 'AND', [[f'value#{n1}'], '>', ['Subquery']]

  # Case 2
  match = re.match(r'^\(isnotnull\(total_revenue#(\d+)\) AND \(total_revenue#\d+ = Subquery subquery#(\d+), \[id=#(\d+)\]\)\)$', s)
  if match:
    n1 = match.group(1)
    return [['isnotnull', [f'total_revenue#{n1}']], 'AND', [[f'total_revenue#{n1}'], '=', ['Subquery']]]

  # Case 3
  match = re.match(r'^\(sum\(CASE WHEN \(nation#(\d+) = BRAZIL\) THEN volume#(\d+) ELSE 0\.0 END\)#(\d+) / sum\(volume#\d+\)#(\d+)\) AS mkt_share#(\d+)$', s)
  if match:
    n1, n2, n3, n4, n5 = match.groups()
    return [[[['sum', ['CASE', 'WHEN', [f'nation#{n1}', '=', 'BRAZIL'], 'THEN', f'volume#{n2}', 'ELSE', '0.0', 'END']], '#', n3], '/', [['sum', f'volume#{n2}'], '#', n4]], 'AS', f'mkt_share#{n5}']

  # Case 4
  match = re.match(r'^\(\(isnotnull\(c_acctbal#(\d+)\) AND substring\(c_phone#(\d+), 1, 2\) IN \(13,31,23,29,30,18,17\)\) AND \(c_acctbal#\d+ > Subquery subquery#(\d+), \[id=#(\d+)\]\)\)$', s)
  if match:
    n1, n2, _, _ = match.groups()
    return [[[['isnotnull', [f'c_acctbal#{n1}']], 'AND', [['substring', [f'c_phone#{n2}', '1', '2']], 'IN', ['13', '31', '23', '29', '30', '18', '17']]]], 'AND', [f'c_acctbal#{n1}', '>', 'Subquery']]
  #### TPC-H SPECIAL CASES ####

  assert 'Subquery' not in s
  parse_start = time.time()
  try:
    result = parsed_expr.parse_string(s).as_list()
  except Exception as e:
    print ("error")
    result = parsed_expr.parse_string(s[1:-1]).as_list()

  parse_end = time.time()
  tot += parse_end - parse_start
  print(f"Parsing {s} Time = {parse_end - parse_start:.6f} seconds\n")
  print(f"Parsing Time Total = {tot:.6f} seconds\n")
  return result
