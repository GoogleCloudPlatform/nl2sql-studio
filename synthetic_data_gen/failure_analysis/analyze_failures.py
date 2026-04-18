import sys
import os
from collections import Counter, defaultdict

def analyze_failures(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Metrics storage
    overall_counts = Counter()
    level_counts = defaultdict(Counter)
    query_types = Counter()
    failure_types = Counter()
    schema_failures = Counter()
    keyword_failures = Counter()

    current_item = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # Process completed item if any
                    if current_item:
                        process_item(current_item, overall_counts, level_counts, query_types, failure_types, schema_failures, keyword_failures)
                        current_item = {}
                    continue

                if line.startswith("question:"):
                    # If we had a previous item without empty line separator, process it
                    if current_item and 'result' in current_item:
                        process_item(current_item, overall_counts, level_counts, query_types, failure_types, schema_failures, keyword_failures)
                        current_item = {}
                    current_item['question'] = line[len("question:"):].strip()
                elif line.startswith("sql:"):
                    current_item['sql'] = line[len("sql:"):].strip()
                elif line.startswith("level:"):
                    current_item['level'] = line[len("level:"):].strip().lower()
                elif line.startswith("result:"):
                    current_item['result'] = line[len("result:"):].strip().lower()
                elif line.startswith("schema:"):
                    current_item['schema'] = line[len("schema:"):].strip().lower()

            # Process the last item if it didn't end with an empty line
            if current_item:
                process_item(current_item, overall_counts, level_counts, query_types, failure_types, schema_failures, keyword_failures)

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Reporting
    print("=" * 50)
    print(f"Failure Analysis Report for: {os.path.basename(file_path)}")
    print("=" * 50)
    
    total = sum(overall_counts.values())
    print(f"Total Queries: {total}")
    print(f"Passed (True): {overall_counts['true']} ({overall_counts['true']/total*100:.2f}%)" if total else "No queries")
    print(f"Failed (False): {overall_counts['false']} ({overall_counts['false']/total*100:.2f}%)" if total else "")
    print(f"Syntax Failure: {overall_counts['syntax failure']} ({overall_counts['syntax failure']/total*100:.2f}%)" if total else "")

    print("\n" + "=" * 50)
    print("Results Breakdown by Level")
    print("=" * 50)
    
    for lvl in ['easy', 'medium', 'complex']:
        counts = level_counts[lvl]
        lvl_total = sum(counts.values())
        if lvl_total == 0:
            print(f"\nLevel: {lvl.capitalize()} (No queries)")
            continue
            
        print(f"\nLevel: {lvl.capitalize()} (Total: {lvl_total})")
        print(f"  - Passed: {counts['true']} ({counts['true']/lvl_total*100:.2f}%)")
        print(f"  - Failed: {counts['false']} ({counts['false']/lvl_total*100:.2f}%)")
        print(f"  - Syntax Failure: {counts['syntax failure']} ({counts['syntax failure']/lvl_total*100:.2f}%)")

    print("\n" + "=" * 50)
    print("Successful Query Types Breakdown")
    print("=" * 50)
    for qt, count in query_types.most_common():
        print(f"- {qt}: {count}")

    print("\n" + "=" * 50)
    print("Failure Breakdown (Query Type + Level + Error Type)")
    print("=" * 50)
    for ft, count in failure_types.most_common():
        print(f"- {ft}: {count}")

    print("\n" + "=" * 50)
    print("Top 10 Schemas with Most Failures")
    print("=" * 50)
    for schema, count in schema_failures.most_common(10):
        print(f"- {schema}: {count}")

    print("\n" + "=" * 50)
    print("Failing SQL Keywords Frequency")
    print("=" * 50)
    for kw, count in keyword_failures.most_common():
        print(f"- {kw}: {count}")

def process_item(item, overall_counts, level_counts, query_types, failure_types, schema_failures, keyword_failures):
    level = item.get('level', 'unknown')
    result = item.get('result', 'unknown')
    schema = item.get('schema', 'unknown')
    sql = item.get('sql', '').upper()
    
    # Identify Query Type
    q_type = "Simple"
    if "JOIN" in sql:
        q_type = "Join"
    if "GROUP BY" in sql or "HAVING" in sql:
        q_type = "Aggregation"
    if "INTERSECT" in sql or "EXCEPT" in sql or "UNION" in sql:
        q_type = "Set Operation"
    if "SELECT" in sql and sql.count("SELECT") > 1:
         q_type = "Subquery"

    overall_counts[result] += 1
    level_counts[level][result] += 1
    
    if result == 'true':
        query_types[q_type] += 1
    else:
        failure_types[f"{q_type} - {level.capitalize()} - {result.capitalize()}"] += 1
        schema_failures[schema] += 1
        
        # Keyword analysis
        keywords = ['HAVING', 'EXISTS', 'INTERSECT', 'UNION', 'EXCEPT', 'LIMIT', 'ORDER BY', 'AVG', 'SUM', 'COUNT']
        for kw in keywords:
            if kw in sql:
                keyword_failures[kw] += 1

if __name__ == "__main__":
    default_file = "/Users/sanchitlatawa/Desktop/nl2sql-studio/synthetic_data_gen/failure_analysis/sample_data.txt"
    target_file = sys.argv[1] if len(sys.argv) > 1 else default_file
    
    print(f"Analyzing: {target_file}\n")
    analyze_failures(target_file)
