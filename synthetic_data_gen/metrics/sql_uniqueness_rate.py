import pandas as pd
import sqlglot
from sqlglot import exp

def mask_literals(node):
    """
    A transformer function that replaces values with a placeholder.
    Catches strings, numbers, booleans, and NULLs.
    """
    if isinstance(node, (exp.Literal, exp.Boolean, exp.Null)):
        return exp.Placeholder() # Replaces the value with '?'
    return node

def normalize_and_mask_sql(sql_query):
    """
    Parses the SQL, masks all literal values, and standardizes formatting.
    """
    try:
        # 1. Parse the SQL into an Abstract Syntax Tree (AST)
        parsed = sqlglot.parse_one(sql_query, read="sqlite")
        
        # 2. Walk the tree and replace literals
        masked_ast = parsed.transform(mask_literals)
        
        # 3. Return the clean, masked SQL string
        return masked_ast.sql()
    except Exception:
        # Fallback for LLM hallucinations that generate unparsable SQL
        return str(sql_query).strip().lower()

def calculate_sur_masked(df):
    """
    Calculates the strict SQL Uniqueness Rate using masked skeletons.
    """
    df = pd.DataFrame(df)
    # Apply the new masking function
    df['masked_sql'] = df['sql'].apply(normalize_and_mask_sql)
    
    sur_results = []
    grouped = df.groupby('db_id')
    
    for db_id, group in grouped:
        total_queries = len(group)
        # Count unique structural skeletons
        unique_queries = group['masked_sql'].nunique() 
        
        sur_score = unique_queries / total_queries if total_queries > 0 else 0
        
        sur_results.append({
            'db_id': db_id,
            'total_queries': total_queries,
            'unique_queries': unique_queries,
            'sur_masked': sur_score
        })
        
    report_df = pd.DataFrame(sur_results)
    average_sur = report_df['sur_masked'].mean()
    
    return report_df, average_sur