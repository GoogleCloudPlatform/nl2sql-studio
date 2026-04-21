import json
from collections import Counter

def analyze_stage1_pipeline(json_filepath, tables_json_path):
    """
    Calculates comprehensive metrics for the Stage 1 pipeline results.
    
    Metrics:
    - Total Databases, Tables, Columns (Stage 1 Scope)
    - Total SQL Generated vs Executed Successfully
    - Complexity Distribution (Generated vs Executed)
    - Successful Execution Rate
    """
    try:
        # 1. Load Data
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(tables_json_path, 'r', encoding='utf-8') as f:
            schemas = json.load(f)
            
        if not isinstance(data, list):
            print("❌ Error: JSON results must be a list of query objects.")
            return

        # 2. Schema Metrics (Databases, Tables, Columns)
        active_db_ids = set(q.get("db_id") for q in data if q.get("db_id"))
        total_dbs_stage1 = len(active_db_ids)
        
        # Filter master schemas to only those databases present in the results
        active_schemas = [s for s in schemas if s.get('db_id') in active_db_ids]
        total_tables_stage1 = sum(len(s.get('table_names_original', [])) for s in active_schemas)
        # Note: Excluding '*' from column counts for accuracy
        total_columns_stage1 = sum(len([c for c in s.get('column_names_original', []) if c[1] != '*']) for s in active_schemas)
        
        # 3. SQL Execution Metrics
        total_sql_generated = len([q for q in data if "sql" in q])
        # Support backward compatibility: If 'success' key is missing, assume query was successful
        successful_queries = [q for q in data if q.get("success", True)]
        total_sql_success = len(successful_queries)
        
        success_rate = (total_sql_success / total_sql_generated * 100) if total_sql_generated > 0 else 0
        
        # 4. Complexity Distributions
        gen_complexities = Counter(q.get("complexity", "Unknown") for q in data)
        exec_complexities = Counter(q.get("complexity", "Unknown") for q in successful_queries)
        
        # 5. Print Formal Report
        print("\n" + "="*70)
        print("📊 STAGE 1 PIPELINE: COMPREHENSIVE METRICS REPORT")
        print("="*70)
        
        print(f"{'DATABASE METRICS':<45}")
        print(f"  - Total Databases (Stage 1):{' ':<15} {total_dbs_stage1}")
        print(f"  - Total Tables (Stage 1):{' ':<18} {total_tables_stage1}")
        print(f"  - Total Columns (Stage 1):{' ':<17} {total_columns_stage1}")
        
        print("\n" + f"{'SQL GENERATION & EXECUTION':<45}")
        print(f"  - Total SQL Generated (Stage 1):{' ':<11} {total_sql_generated}")
        print(f"  - Total SQL Executed successfully (Stage 1):{' ':<2} {total_sql_success}")
        print(f"  - Successful Execution Rate (Stage 1):{' ':<6} {success_rate:.1f}%")
        
        print("\n" + f"{'GENERATED COMPLEXITY DISTRIBUTION':<45}")
        for complexity, count in gen_complexities.most_common():
            perc = (count / total_sql_generated * 100) if total_sql_generated > 0 else 0
            print(f"  - {complexity:<15} : {count:<5} ({perc:.1f}%)")

        print("\n" + f"{'EXECUTED COMPLEXITY DISTRIBUTION (SUCCESS ONLY)':<45}")
        for complexity, count in exec_complexities.most_common():
            perc = (count / total_sql_success * 100) if total_sql_success > 0 else 0
            print(f"  - {complexity:<15} : {count:<5} ({perc:.1f}%)")
            
        print("="*70 + "\n")

    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"❌ Unexpected Error during metrics analysis: {e}")
