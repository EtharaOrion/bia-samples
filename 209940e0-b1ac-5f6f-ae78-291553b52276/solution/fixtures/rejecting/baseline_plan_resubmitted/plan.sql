WITH RECURSIVE quarantined(node_id) AS (
    SELECT node_id
      FROM lineage
     WHERE declared_check <> (p_alpha * 31 + p_beta * 17 + p_gamma * 7) % 1000003
    UNION
    SELECT l.node_id FROM lineage l JOIN quarantined q ON l.parent_id = q.node_id
    UNION
    SELECT d.target_node FROM derives d JOIN quarantined q ON d.source_node = q.node_id
)
SELECT reading_id, node_id, value
  FROM reading
 WHERE node_id NOT IN (SELECT node_id FROM quarantined)
 ORDER BY reading_id
