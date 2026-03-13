# SQL Fitness Evaluation Report
Generated on: 2025-09-26 11:20:57

## Evaluation for NL Query #1: `List the top 3 distribution centers by the number of returned items.`

### Test Case: Correct Query

**Generated SQL:**
```sql
SELECT dc.name, COUNT(oi.id) AS returned_items_count FROM distribution_centers AS dc JOIN inventory_items AS ii ON dc.id = ii.product_distribution_center_id JOIN order_items AS oi ON ii.id = oi.inventory_item_id WHERE oi.status = 'Returned' GROUP BY dc.name ORDER BY returned_items_count DESC LIMIT 3
```

**Evaluation Report:**
- **Final SQL Score:** 1.000
- **Confidence Score:** 1.000
- **Needs Human Review:** No

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 1.000

---

### Test Case: Wrong Status

**Generated SQL:**
```sql
SELECT dc.name, COUNT(oi.id) AS returned_items_count FROM distribution_centers AS dc JOIN inventory_items AS ii ON dc.id = ii.product_distribution_center_id JOIN order_items AS oi ON ii.id = oi.inventory_item_id WHERE oi.status = 'Shipped' GROUP BY dc.name ORDER BY returned_items_count DESC LIMIT 3
```

**Evaluation Report:**
- **Final SQL Score:** 0.600
- **Confidence Score:** 0.600
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 0.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 0.000
- `operations_accuracy (AO)`: 1.000

---

### Test Case: Incorrect Aggregation

**Generated SQL:**
```sql
SELECT dc.name, SUM(oi.id) AS returned_items_count FROM distribution_centers AS dc JOIN inventory_items AS ii ON dc.id = ii.product_distribution_center_id JOIN order_items AS oi ON ii.id = oi.inventory_item_id WHERE oi.status = 'Returned' GROUP BY dc.name ORDER BY returned_items_count DESC LIMIT 3
```

**Evaluation Report:**
- **Final SQL Score:** 0.970
- **Confidence Score:** 0.940
- **Needs Human Review:** No

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.700

---

### Test Case: Missing Limit

**Generated SQL:**
```sql
SELECT dc.name, COUNT(oi.id) AS returned_items_count FROM distribution_centers AS dc JOIN inventory_items AS ii ON dc.id = ii.product_distribution_center_id JOIN order_items AS oi ON ii.id = oi.inventory_item_id WHERE oi.status = 'Returned' GROUP BY dc.name ORDER BY returned_items_count DESC
```

**Evaluation Report:**
- **Final SQL Score:** 0.990
- **Confidence Score:** 0.980
- **Needs Human Review:** No

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.900

---

## Evaluation for NL Query #2: `What are the top 5 product categories by total sales for female users from California?`

### Test Case: Correct Query

**Generated SQL:**
```sql
SELECT p.category, SUM(oi.sale_price) AS total_sales FROM users AS u JOIN order_items AS oi ON u.id = oi.user_id JOIN products AS p ON oi.product_id = p.id WHERE u.gender = 'F' AND u.state = 'California' GROUP BY p.category ORDER BY total_sales DESC LIMIT 5
```

**Evaluation Report:**
- **Final SQL Score:** 0.900
- **Confidence Score:** 0.800
- **Needs Human Review:** No

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.000

---

### Test Case: Incorrect Join Condition

**Generated SQL:**
```sql
SELECT p.category, SUM(oi.sale_price) AS total_sales FROM users AS u JOIN orders AS o ON u.id = o.user_id JOIN order_items AS oi ON o.order_id = oi.order_id JOIN products AS p ON oi.inventory_item_id = p.id WHERE u.gender = 'F' AND u.state = 'California' GROUP BY p.category ORDER BY total_sales DESC LIMIT 5
```

**Evaluation Report:**
- **Final SQL Score:** 0.789
- **Confidence Score:** 0.698
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 0.818
- `join_accuracy (AJ)`: 0.670
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.000

---

### Test Case: Missing Filter Condition

**Generated SQL:**
```sql
SELECT p.category, SUM(oi.sale_price) AS total_sales FROM users AS u JOIN orders AS o ON u.id = o.user_id JOIN order_items AS oi ON o.order_id = oi.order_id JOIN products AS p ON oi.product_id = p.id WHERE u.gender = 'F' GROUP BY p.category ORDER BY total_sales DESC LIMIT 5
```

**Evaluation Report:**
- **Final SQL Score:** 0.750
- **Confidence Score:** 0.800
- **Needs Human Review:** No

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 0.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 1.000

---

### Test Case: Wrong Aggregation

**Generated SQL:**
```sql
SELECT p.category, COUNT(oi.sale_price) AS total_sales FROM users AS u JOIN orders AS o ON u.id = o.user_id JOIN order_items AS oi ON o.order_id = oi.order_id JOIN products AS p ON oi.product_id = p.id WHERE u.gender = 'F' AND u.state = 'California' GROUP BY p.category ORDER BY total_sales DESC LIMIT 5
```

**Evaluation Report:**
- **Final SQL Score:** 0.950
- **Confidence Score:** 0.900
- **Needs Human Review:** No

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.500

---

## Evaluation for NL Query #3: `How many users from China have placed more than 2 orders?`

### Test Case: Correct Query

**Generated SQL:**
```sql
SELECT COUNT(DISTINCT user_id) FROM orders WHERE user_id IN (SELECT id FROM users WHERE country = 'China') GROUP BY user_id HAVING COUNT(order_id) > 2
```

**Evaluation Report:**
- **Final SQL Score:** 0.700
- **Confidence Score:** 0.600
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 0.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.000

---

### Test Case: Incorrect Count

**Generated SQL:**
```sql
SELECT COUNT(T1.id) FROM users AS T1 JOIN (SELECT user_id, COUNT(order_id) AS order_count FROM orders GROUP BY user_id) AS T2 ON T1.id = T2.user_id WHERE T1.country = 'China' AND T2.order_count >= 2
```

**Evaluation Report:**
- **Final SQL Score:** 0.750
- **Confidence Score:** 0.600
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 0.000
- `operations_accuracy (AO)`: 0.000

---

### Test Case: Wrong Join Logic

**Generated SQL:**
```sql
SELECT COUNT(u.id) FROM users AS u WHERE u.country = 'China' AND (SELECT COUNT(o.order_id) FROM orders AS o WHERE o.user_id = u.id) > 2
```

**Evaluation Report:**
- **Final SQL Score:** 0.486
- **Confidence Score:** 0.457
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 0.286
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 0.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.000

---

### Test Case: Missing Group By

**Generated SQL:**
```sql
SELECT COUNT(T1.id) FROM users AS T1 JOIN orders AS T2 ON T1.id = T2.user_id WHERE T1.country = 'China' HAVING COUNT(T2.order_id) > 2
```

**Evaluation Report:**
- **Final SQL Score:** 0.900
- **Confidence Score:** 0.800
- **Needs Human Review:** No

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 1.000
- `operations_accuracy (AO)`: 0.000

---

## Evaluation for NL Query #4: `What is the total profit from products in the 'Jeans' category sold in '2023'?`

### Test Case: Correct Query

**Generated SQL:**
```sql
SELECT SUM(oi.sale_price - p.cost) AS total_profit FROM order_items AS oi JOIN products AS p ON oi.product_id = p.id WHERE p.category = 'Jeans' AND STRFTIME('%Y', oi.created_at) = '2023'
```

**Evaluation Report:**
- **Final SQL Score:** 0.320
- **Confidence Score:** 0.360
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 0.000
- `column_accuracy (AC)`: 0.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 0.800
- `operations_accuracy (AO)`: 0.000

---

### Test Case: Wrong Profit Calculation

**Generated SQL:**
```sql
SELECT SUM(oi.sale_price) FROM order_items AS oi JOIN products AS p ON oi.product_id = p.id WHERE p.category = 'Jeans' AND STRFTIME('%Y', oi.created_at) = '2023'
```

**Evaluation Report:**
- **Final SQL Score:** 0.642
- **Confidence Score:** 0.590
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 1.000
- `column_accuracy (AC)`: 0.000
- `join_accuracy (AJ)`: 1.000
- `filter_accuracy (AF)`: 0.950
- `operations_accuracy (AO)`: 0.000

---

### Test Case: Incorrect Date Filter

**Generated SQL:**
```sql
SELECT SUM(oi.sale_price - p.cost) AS total_profit FROM order_items AS oi JOIN products AS p ON oi.product_id = p.id WHERE p.category = 'Jeans' AND oi.created_at = '2023'
```

**Evaluation Report:**
- **Final SQL Score:** 0.395
- **Confidence Score:** 0.460
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 0.000
- `column_accuracy (AC)`: 1.000
- `join_accuracy (AJ)`: 0.000
- `filter_accuracy (AF)`: 0.300
- `operations_accuracy (AO)`: 1.000

---

### Test Case: Missing Join

**Generated SQL:**
```sql
SELECT SUM(sale_price) FROM order_items WHERE product_id IN (SELECT id FROM products WHERE category = 'Jeans') AND STRFTIME('%Y', created_at) = '2023'
```

**Evaluation Report:**
- **Final SQL Score:** 0.000
- **Confidence Score:** 0.000
- **Needs Human Review:** Yes

**Score Breakdown:**
- `table_accuracy (AT)`: 0.000
- `column_accuracy (AC)`: 0.000
- `join_accuracy (AJ)`: 0.000
- `filter_accuracy (AF)`: 0.000
- `operations_accuracy (AO)`: 0.000

---

