SELECT reading_id, node_id, value
  FROM reading
 WHERE node_id NOT IN ('nd-1522', 'nd-1526', 'nd-1529', 'nd-1530', 'nd-1537', 'nd-1545')
 ORDER BY reading_id
