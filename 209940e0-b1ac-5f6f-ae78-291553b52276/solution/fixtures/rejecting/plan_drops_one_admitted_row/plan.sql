SELECT reading_id, node_id, value
  FROM reading
 WHERE node_id NOT IN ('nd-0498', 'nd-1522', 'nd-1526', 'nd-1529', 'nd-1530', 'nd-1534', 'nd-1537', 'nd-1538', 'nd-1541', 'nd-1544', 'nd-1545', 'nd-1557', 'nd-1560', 'nd-1562', 'nd-1568', 'nd-1569', 'nd-1570', 'nd-1573', 'nd-1578', 'nd-1582', 'nd-1583', 'nd-1585', 'nd-1586', 'nd-1588', 'nd-1593', 'nd-1594', 'nd-1595', 'nd-1596', 'nd-1597', 'nd-1598', 'nd-1599')
 ORDER BY reading_id
