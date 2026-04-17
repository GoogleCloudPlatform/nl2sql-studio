"""
SQL Completeness Checker
------------------------
This script scans a JSON file containing AI-generated SQL queries to identify
potential truncation or malformed queries. It performs basic heuristic checks
such as unclosed quotes, unmatched parentheses, and dangling keywords.
"""

import json

def check_sql_completeness(file_path: str):
    """
    Scans the specified JSON file for potentially incomplete or malformed SQL queries.
    
    Args:
        file_path (str): Path to the JSON file containing the queries.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    suspicious_records = []
    count = 0
    
    # Keywords that should NEVER be the very last word of a valid SQL query.
    # If a query ends with one of these, it was likely truncated.
    dangling_keywords = (
        "SELECT", "FROM", "WHERE", "AND", "OR", "ON", "JOIN", 
        "AS", "IN", "LIKE", "IS", "NOT", "GROUP", "ORDER", "BY", "HAVING", "="
    )

    # Iterate through each record in the loaded JSON data
    for item in data:
        q_id = item.get("question_id", "Unknown")
        # Extract the AI-generated SQL query, defaulting to an empty string if missing
        sql = item.get("ai_generated_sql", "").strip()
        
        # Check 1: Catch completely empty generations
        if not sql:
            suspicious_records.append((q_id, "Empty query", sql))
            continue
            
        # Check 2: Catch unclosed single quotes (a classic sign of LLM truncation)
        if sql.count("'") % 2 != 0:
            suspicious_records.append((q_id, "Unclosed single quote", sql))
            continue
            
        # Check 3: Catch unmatched parentheses (another strong indicator of truncation)
        if sql.count("(") != sql.count(")"):
            suspicious_records.append((q_id, "Unmatched parentheses", sql))
            continue
            
        # Check 4: Catch dangling keywords (e.g., ends abruptly with "WHERE" or "AND")
        # We strip out the trailing semicolon first if it exists to check the actual last word
        clean_sql = sql.rstrip(";").strip().upper()
        if clean_sql.endswith(dangling_keywords):
            suspicious_records.append((q_id, "Dangling keyword at end", sql))
            continue

    # --- REPORTING ---
    print(f"--- Scan Complete ---")
    print(f"Total queries scanned: {len(data)}")
    print(f"Potentially incomplete queries found: {len(suspicious_records)}\n")
    
    if suspicious_records:
        
        for q_id, reason, sql in suspicious_records:
            count += 1
            print(f"[Question ID {q_id}] - {reason}")
            print(f"SQL: {sql}\n")
    else:
        print("Looks good! No obvious truncations found.")
    print(count)

if __name__ == "__main__":
    # Execution entry point
    # Point this to the JSON file you want to scan
    json_file = '../results/sft/filtered_dev_ai_qwen.json' 
    
    print(f"Scanning file: {json_file}")
    check_sql_completeness(json_file)