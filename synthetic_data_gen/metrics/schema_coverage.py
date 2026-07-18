import json
import sqlglot
from sqlglot import exp
import pandas as pd

# --- 1. Load Ground Truth Schema from Spider ---
def load_spider_schemas(tables_json_path):
    """Parses Spider's tables.json to get total tables and columns per db_id."""
    with open(tables_json_path, 'r') as f:
        spider_tables = json.load(f)
        
    schemas = {}
    for db in spider_tables:
        db_id = db['db_id']
        # Extract original table names
        tables = [t.lower() for t in db['table_names_original']]
        
        # Extract original column names (skipping the '*' column at index 0)
        columns = []
        for col in db['column_names_original']:
            if col[1] != "*": 
                columns.append(col[1].lower())
                
        schemas[db_id] = {
            'total_tables': set(tables),
            'total_columns': set(columns)
        }
    return schemas

# --- 2. Extract Used Entities from Generated SQL ---
def extract_tables_and_columns_from_sql(sql_query):
    """Uses sqlglot to parse the SQL and extract table and column names."""
    used_tables = set()
    used_columns = set()
    
    try:
        # Parse the SQL query
        parsed = sqlglot.parse_one(sql_query, read="sqlite") 
        
        # Find all tables
        for table in parsed.find_all(exp.Table):
            if table.name:
                used_tables.add(table.name.lower())
                
        # Find all columns
        for column in parsed.find_all(exp.Column):
            if column.name:
                used_columns.add(column.name.lower())
                
    except Exception as e:
        # If the LLM generates invalid SQL, sqlglot will fail to parse it.
        # We catch the error and return empty sets for this specific query.
        pass 
        
    return used_tables, used_columns

# --- 3. Calculate Schema Coverage ---
def calculate_schema_coverage(df, tables_json_path):
    """
    df: A pandas DataFrame containing at least ['db_id', 'sql']
    """
    spider_schemas = load_spider_schemas(tables_json_path)
    
    coverage_results = []
    df = pd.DataFrame(df)
    
    # Group the generated queries by database
    grouped = df.groupby('db_id')
    
    for db_id, group in grouped:
        if db_id not in spider_schemas:
            continue
            
        ground_truth = spider_schemas[db_id]
        all_used_tables = set()
        all_used_columns = set()
        
        # Aggregate all used tables and columns across all 5 queries for this db
        for sql in group['sql']:
            tables, columns = extract_tables_and_columns_from_sql(sql)
            all_used_tables.update(tables)
            all_used_columns.update(columns)
            
        # Optional: Intersect with ground truth to ignore hallucinated tables/columns
        valid_used_tables = all_used_tables.intersection(ground_truth['total_tables'])
        valid_used_columns = all_used_columns.intersection(ground_truth['total_columns'])
        
        # Calculate percentages with weights (90% tables, 10% columns)
        total_tables = len(ground_truth['total_tables'])
        total_columns = len(ground_truth['total_columns'])
        tables_used = len(valid_used_tables)
        columns_used = len(valid_used_columns)
        
        table_coverage = tables_used / total_tables if total_tables > 0 else 0
        column_coverage = columns_used / total_columns if total_columns > 0 else 0
        
        coverage_score = (0.9 * table_coverage) + (0.1 * column_coverage)
        
        coverage_results.append({
            'db_id': db_id,
            'tables_used': tables_used,
            'total_tables': total_tables,
            'columns_used': columns_used,
            'total_columns': total_columns,
            'schema_coverage': coverage_score
        })
        
    # Create a report DataFrame
    report_df = pd.DataFrame(coverage_results)
    
    # Calculate the final average across all databases
    average_sc = report_df['schema_coverage'].mean()
    
    return report_df, average_sc

# --- Example Usage ---
# df = pd.DataFrame({'db_id': ['concert_singer', 'concert_singer'], 'sql': ['SELECT ...', 'SELECT ...']})
# report, avg_score = calculate_schema_coverage(df, 'spider/tables.json')
# print(report)
# print(f"Average Schema Coverage: {avg_score:.2%}")