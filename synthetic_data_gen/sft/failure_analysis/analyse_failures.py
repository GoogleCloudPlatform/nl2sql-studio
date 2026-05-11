import json
import os
import sys
import re
from collections import Counter

def analyze_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    total = len(data)
    
    # Difficulty-wise stats
    # difficulty -> {stat_name -> count}
    diff_stats = {}
    
    # Global counters for query types (Whole Analytics)
    query_types = Counter()
    failure_types = Counter()
    schema_failures = Counter()
    keyword_failures = Counter()

    for item in data:
        difficulty = item.get("difficulty", "Unknown")
        db_id = item.get("db_id", "Unknown")
        if difficulty not in diff_stats:
            diff_stats[difficulty] = {
                "total": 0,
                "passed": 0,
                "incorrect_results": 0,
                "syntax_failed": 0,
                "other_failed": 0
            }
        
        diff_stats[difficulty]["total"] += 1
        
        status = item.get("status", "")
        success = bool(re.match(r"^Correct", status))
        error_msg = item.get("error", "")
        sql = item.get("ai_generated_sql", "").upper()

        # Identify Query Type (Heuristic)
        q_type = "Simple"
        if "JOIN" in sql:
            q_type = "Join"
        if "GROUP BY" in sql or "HAVING" in sql:
            q_type = "Aggregation"
        if "INTERSECT" in sql or "EXCEPT" in sql or "UNION" in sql:
            q_type = "Set Operation"
        if "SELECT" in sql and sql.count("SELECT") > 1:
             q_type = "Subquery"

        if success:
            diff_stats[difficulty]["passed"] += 1
            query_types[q_type] += 1
        else:
            if item.get("status") == "Incorrect":
                diff_stats[difficulty]["incorrect_results"] += 1
                failure_types[f"{q_type} - Incorrect Results"] += 1
            elif "SYNTAX ERROR" in error_msg.upper() or "NO SUCH" in error_msg.upper():
                diff_stats[difficulty]["syntax_failed"] += 1
                failure_types[f"{q_type} - Syntax/Schema"] += 1
            else:
                diff_stats[difficulty]["other_failed"] += 1
                failure_types[f"{q_type} - Other"] += 1

            # Track schema failures
            schema_failures[db_id] += 1

            # Track keyword failures
            keywords = ['HAVING', 'EXISTS', 'INTERSECT', 'UNION', 'EXCEPT', 'LIMIT', 'ORDER BY', 'AVG', 'SUM', 'COUNT']
            for kw in keywords:
                if kw in sql:
                    keyword_failures[kw] += 1

    # Calculate whole analytics
    whole_stats = {
        "total": total,
        "passed": sum(s["passed"] for s in diff_stats.values()),
        "incorrect_results": sum(s["incorrect_results"] for s in diff_stats.values()),
        "syntax_failed": sum(s["syntax_failed"] for s in diff_stats.values()),
        "other_failed": sum(s["other_failed"] for s in diff_stats.values())
    }

    def print_stats(title, stats):
        print("=" * 50)
        print(f"{title}")
        print("=" * 50)
        print(f"Total Queries Attempted: {stats['total']}")
        if stats['total'] > 0:
            print(f"✅ Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.2f}%)")
            print(f"❌ Incorrect Results: {stats['incorrect_results']} ({stats['incorrect_results']/stats['total']*100:.2f}%)")
            print(f"🚨 Syntax/Schema Failed: {stats['syntax_failed']} ({stats['syntax_failed']/stats['total']*100:.2f}%)")
            print(f"❓ Other Failures: {stats['other_failed']} ({stats['other_failed']/stats['total']*100:.2f}%)")
        else:
            print("No queries for this level.")

    print("=" * 50)
    print(f"Analysis Report for: {os.path.basename(file_path)}")
    print("=" * 50)
    
    print_stats("Whole Analytics", whole_stats)
    
    # Print breakdowns for Whole Analytics
    print("\n" + "=" * 50)
    print("Successful Query Types Breakdown:")
    print("=" * 50)
    for qt, count in query_types.most_common():
        print(f"- {qt}: {count}")

    print("\n" + "=" * 50)
    print("Failure Breakdown (Query Type + Error Type):")
    print("=" * 50)
    for ft, count in failure_types.most_common():
        print(f"- {ft}: {count}")
    
    print("\n" + "=" * 50)
    print("Top 10 Schemas with Most Failures:")
    print("=" * 50)
    for schema, count in schema_failures.most_common(10):
        print(f"- {schema}: {count}")

    print("\n" + "=" * 50)
    print("Failing SQL Keywords Frequency:")
    print("=" * 50)
    for kw, count in keyword_failures.most_common():
        print(f"- {kw}: {count}")
    
    for difficulty in sorted(diff_stats.keys()):
        print("\n")
        print_stats(f"Difficulty Level: {difficulty}", diff_stats[difficulty])

if __name__ == "__main__":
    # You can pass a file path as an argument, or it will use a default one
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.abspath(os.path.join(current_dir, "../../results/sft/spider_test_set_ai_gemma4-26b-base.json"))
    target_file = sys.argv[1] if len(sys.argv) > 1 else default_file
    
    print(f"Analyzing: {target_file}")
    analyze_file(target_file)