SELECT 
  o_orderpriority, 
  COUNT(*) AS order_count
FROM 
  orders
WHERE 
  o_orderdate >= '1993-07-01'
  AND o_orderdate < DATE_ADD('1993-07-01', 31 + 31 + 30)
  AND EXISTS (
    SELECT 
      *
    FROM 
      lineitem
    WHERE 
      l_orderkey = o_orderkey 
      AND l_commitdate < l_receiptdate
  )
GROUP BY 
  o_orderpriority
ORDER BY 
  o_orderpriority;