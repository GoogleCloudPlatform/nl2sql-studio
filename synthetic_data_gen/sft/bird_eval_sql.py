"""
BIRD SQL Execution Evaluator
----------------------------
This script evaluates AI-generated SQL queries against ground truth queries
by executing both on the target SQLite database and comparing the results.
It handles order-insensitive comparisons when ORDER BY is missing.
"""

import json
import sqlite3
import os
from typing import List, Any, Tuple, Dict, Optional
from tqdm import tqdm

def execute_sql(db_path: str, query: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Executes a given SQL query on a specified SQLite database.
    
    Args:
        db_path (str): Path to the .sqlite file.
        query (str): The SQL query to execute.
        
    Returns:
        tuple: (Result list if successful, Error message if failed)
    """
    if not os.path.exists(db_path):
        return None, f"Database not found at {db_path}"

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result, None
    except sqlite3.Error as e:
        return None, str(e)

def are_results_equal(
    ground_truth_query: str,
    ground_truth_result: List[Any],
    ai_result: List[Any]
) -> bool:
    """
    Compares two SQL query results for equality, handling order sensitivity.
    
    Args:
        ground_truth_query (str): The original ground truth SQL string.
        ground_truth_result (list): Results from the ground truth query.
        ai_result (list): Results from the AI-generated query.
        
    Returns:
        bool: True if results are equivalent, False otherwise.
    """
    if ai_result is None:
        return False

    if len(ground_truth_result) != len(ai_result):
        return False

    # If there's no ORDER BY in the ground truth, the order of results is not guaranteed.
    # We sort the results to compare them content-wise.
    if "order by" not in ground_truth_query.lower():
        try:
            ground_truth_result = sorted(list(set(ground_truth_result)))
            ai_result = sorted(list(set(ai_result)))
        except TypeError:
            # Fallback for unorderable types (e.g., lists of dicts) using simple set comparison
            return set(ground_truth_result) == set(ai_result)

    return ground_truth_result == ai_result

def evaluate_bird_queries(file_path: str, db_root_path: str):
    """
    Evaluates AI-generated SQL queries against ground truth for the BIRD dataset.
    Processes each item in the JSON file, executes queries, and prints summary stats.
    
    Args:
        file_path (str): Path to the JSON file containing AI generated SQL.
        db_root_path (str): Directory where SQLite databases are stored.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        with open(file_path, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)

        correct_count = 0
        total_count = 0
        execution_failed_count = 0
        incorrect_result_count = 0

        for item in tqdm(data, desc="Evaluating queries"):
            db_id = item.get("db_id")
            ground_truth_sql = item.get("SQL") # BIRD dataset uses capitalized 'SQL'
            ai_sql = item.get("ai_generated_sql")
            question = item.get("question")

            if not all([db_id, ground_truth_sql, ai_sql]):
                continue

            total_count += 1
            db_path = os.path.join(db_root_path, f"{db_id}.sqlite")

            # Execute ground truth query to get the expected result
            gt_result, gt_error = execute_sql(db_path, ground_truth_sql)
            if gt_error:
                # We log it but still count it; dataset issues might cause this.
                pass

            # Execute AI query to get the actual result
            ai_result, ai_error = execute_sql(db_path, ai_sql)
            
            if ai_error:
                execution_failed_count += 1
                continue

            # Compare results
            if are_results_equal(ground_truth_sql, gt_result or [], ai_result):
                correct_count += 1
            else:
                incorrect_result_count += 1

        # Print final metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"\n--- Evaluation Results ---")
        print(f"Total Queries Evaluated: {total_count}")
        print(f"Correct Queries: {correct_count}")
        print(f"Execution Failed: {execution_failed_count}")
        print(f"Incorrect Results: {incorrect_result_count}")
        print(f"Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    # Execution entry point
    # This assumes you have run gen_bird_ai_sql.py first to generate the file
    generated_file = '/Users/roopayk/Documents/nl-sql/nl2sql-studio/synthetic_data_gen/results/sft/test_set_ai_qwen.json'
    db_root = '/Users/roopayk/Documents/nl-sql/nl2sql-studio/synthetic_data_gen/database'
    
    evaluate_bird_queries(generated_file, db_root)
