import json
import sqlite3
import os
from typing import List, Any, Tuple, Dict, Optional
from tqdm import tqdm

def execute_sql(db_path: str, query: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Executes a given SQL query on a specified SQLite database.
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
    Compares two SQL query results for equality.
    """
    if ai_result is None:
        return False

    if len(ground_truth_result) != len(ai_result):
        return False

    # If there's no ORDER BY in the ground truth, the order of results is not guaranteed.
    if "order by" not in ground_truth_query.lower():
        try:
            ground_truth_result = sorted(list(set(ground_truth_result)))
            ai_result = sorted(list(set(ai_result)))
        except TypeError:
            # Fallback for unorderable types
            return set(ground_truth_result) == set(ai_result)

    return ground_truth_result == ai_result

def evaluate_bird_queries(file_path: str, db_root_path: str):
    """
    Evaluates AI-generated SQL queries against ground truth for BIRD dataset.
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
            ground_truth_sql = item.get("SQL") # BIRD uses 'SQL'
            ai_sql = item.get("ai_generated_sql")
            question = item.get("question")

            if not all([db_id, ground_truth_sql, ai_sql]):
                continue

            total_count += 1
            db_path = os.path.join(db_root_path, f"{db_id}.sqlite")

            # Execute ground truth query
            gt_result, gt_error = execute_sql(db_path, ground_truth_sql)
            if gt_error:
                # print(f"Error executing ground truth SQL for DB {db_id}: {gt_error}")
                # We still count it as total, but maybe it's a dataset issue
                pass

            # Execute AI query
            ai_result, ai_error = execute_sql(db_path, ai_sql)
            
            if ai_error:
                execution_failed_count += 1
                continue

            if are_results_equal(ground_truth_sql, gt_result or [], ai_result):
                correct_count += 1
            else:
                incorrect_result_count += 1

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
    # This assumes you have run gen_bird_ai_sql.py first
    generated_file = '/Users/roopayk/Downloads/filtered_dev_ai_qwen.json'
    db_root = '/Users/roopayk/Documents/nl-sql/nl2sql-studio/synthetic_data_gen/database'
    
    evaluate_bird_queries(generated_file, db_root)
