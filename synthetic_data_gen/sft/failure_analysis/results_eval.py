"""
BIRD/Spider SQL Execution Evaluator & Equivalent Analyzer
---------------------------------------------------------
This script evaluates AI-generated SQL queries against ground truth queries
by executing both on the target SQLite database, comparing the results,
and performing advanced equivalence analysis (reordering, extra columns, tie-breakers).
"""

import json
import sqlite3
import os
import re
import itertools
from typing import List, Any, Tuple, Dict, Optional
from tqdm import tqdm
from collections import Counter

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
    Compares two SQL query results for equality, handling order sensitivity.
    """
    if ai_result is None:
        return False

    if len(ground_truth_result) != len(ai_result):
        return False

    # If there's no ORDER BY in the ground truth, compare order-insensitively using Counter
    if "order by" not in ground_truth_query.lower():
        return Counter(ground_truth_result) == Counter(ai_result)

    # If there is an ORDER BY, compare order-sensitively
    return ground_truth_result == ai_result

def check_reordering(res_gt: List[Any], res_ai: List[Any]) -> Tuple[bool, Optional[Tuple[int, ...]]]:
    """
    Checks if AI result is equal to GT result under some column reordering (permutation).
    """
    if not res_gt or not res_ai:
        return False, None
    if len(res_gt) != len(res_ai):
        return False, None
    
    len_gt = len(res_gt[0])
    len_ai = len(res_ai[0])
    if len_gt != len_ai:
        return False, None
    
    for perm in itertools.permutations(range(len_gt)):
        permuted_ai = []
        for row in res_ai:
            permuted_row = tuple(row[i] for i in perm)
            permuted_ai.append(permuted_row)
        
        if Counter(res_gt) == Counter(permuted_ai):
            return True, perm
            
    return False, None

def check_extra_columns(res_gt: List[Any], res_ai: List[Any]) -> Tuple[bool, Optional[bool], Optional[Tuple[int, ...]]]:
    """
    Checks if results are equivalent but one has extra columns.
    Returns (is_extra_cols, swapped, perm) where swapped=True means GT has extra columns.
    """
    if not res_gt or not res_ai:
        return False, None, None
        
    len_gt = len(res_gt[0])
    len_ai = len(res_ai[0])
    if len_gt == len_ai:
        return False, None, None
        
    if len_gt < len_ai:
        smaller, larger = res_gt, res_ai
        swapped = False
    else:
        smaller, larger = res_ai, res_gt
        swapped = True
        
    len_small = len(smaller[0])
    len_large = len(larger[0])
    
    if len(smaller) != len(larger):
        return False, None, None
        
    for cols in itertools.combinations(range(len_large), len_small):
        for perm in itertools.permutations(cols):
            projected_large = []
            for row in larger:
                projected_row = tuple(row[i] for i in perm)
                projected_large.append(projected_row)
                
            if Counter(smaller) == Counter(projected_large):
                return True, swapped, perm
                
    return False, None, None

def check_tie_breaker(db_path: str, sql_gt: str, sql_ai: str) -> Tuple[bool, Optional[str]]:
    """
    Checks if queries are equivalent after stripping the LIMIT clause.
    """
    def strip_limit(sql):
        cleaned = re.sub(r'\bLIMIT\s+\d+\b', '', sql, flags=re.IGNORECASE).strip()
        if cleaned.endswith(';'):
            cleaned = cleaned[:-1].strip()
        return cleaned

    clean_gt = strip_limit(sql_gt)
    clean_ai = strip_limit(sql_ai)
    
    if clean_gt == sql_gt and clean_ai == sql_ai:
        return False, None
        
    res_gt_unlimited, err_gt = execute_sql(db_path, clean_gt)
    res_ai_unlimited, err_ai = execute_sql(db_path, clean_ai)
    
    if err_gt or err_ai or not res_gt_unlimited or not res_ai_unlimited:
        return False, None
        
    if Counter(res_gt_unlimited) == Counter(res_ai_unlimited):
        return True, "exact_match_unlimited"
        
    is_reordered, _ = check_reordering(res_gt_unlimited, res_ai_unlimited)
    if is_reordered:
        return True, "reordered_unlimited"
        
    is_extra, _, _ = check_extra_columns(res_gt_unlimited, res_ai_unlimited)
    if is_extra:
        return True, "extra_columns_unlimited"
        
    return False, None

def evaluate_queries(file_path: str, db_root_path: str):
    """
    Evaluates AI-generated SQL queries against ground truth for the Spider dataset,
    performing exact checks and advanced equivalence analysis.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        with open(file_path, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)

        exact_correct_count = 0
        reordered_count = 0
        extra_columns_count = 0
        tie_breaker_count = 0
        execution_failed_count = 0
        pure_incorrect_count = 0
        total_count = 0

        for item in tqdm(data, desc="Evaluating queries"):
            db_id = item.get("db_id")
            ground_truth_sql = item.get("SQL")
            ai_sql = item.get("ai_generated_sql")

            if not all([db_id, ground_truth_sql, ai_sql]):
                continue

            total_count += 1
            db_path = os.path.join(db_root_path, f"{db_id}.sqlite")

            # Execute ground truth query to get the expected result
            gt_result, gt_error = execute_sql(db_path, ground_truth_sql)

            # Execute AI query to get the actual result
            ai_result, ai_error = execute_sql(db_path, ai_sql)
            
            if ai_error:
                execution_failed_count += 1
                item["status"] = "Execution Failure"
                item["error"] = ai_error
                continue

            # 1. Compare results (Exact match)
            if are_results_equal(ground_truth_sql, gt_result or [], ai_result):
                exact_correct_count += 1
                item["status"] = "Correct"
                item["error"] = None
                continue

            # 2. Category 1: Column Reordering
            is_reordered, perm = check_reordering(gt_result or [], ai_result)
            if is_reordered:
                reordered_count += 1
                item["status"] = "Correct (Column Reordered)"
                item["error"] = f"Equivalent under column permutation {perm}"
                continue

            # 3. Category 2: Extra Columns
            is_extra, swapped, perm = check_extra_columns(gt_result or [], ai_result)
            if is_extra:
                extra_columns_count += 1
                who_has_extra = "Ground Truth" if swapped else "AI-generated SQL"
                item["status"] = "Correct (Extra Columns)"
                item["error"] = f"Equivalent. {who_has_extra} has extra columns. Permutation {perm}"
                continue

            # 4. Category 3: Different Row during Tie Breaker
            is_tie, tie_type = check_tie_breaker(db_path, ground_truth_sql, ai_sql)
            if is_tie:
                tie_breaker_count += 1
                item["status"] = "Correct (Tie Breaker)"
                item["error"] = f"Equivalent unlimited queries (type: {tie_type})"
                continue

            # If none of the equivalence checks succeeded, it is pure incorrect
            pure_incorrect_count += 1
            item["status"] = "Incorrect"
            if len(ai_result) == 0:
                item["error"] = "Returned 0 rows"
            else:
                item["error"] = "Incorrect results"

        # Save the updated data back to the input file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated evaluation results saved to '{file_path}'")

        # Calculations
        accuracy_exact = exact_correct_count / total_count if total_count > 0 else 0.0
        total_correct = exact_correct_count + reordered_count + extra_columns_count + tie_breaker_count
        accuracy_joint = total_correct / total_count if total_count > 0 else 0.0

        # Print final metrics
        print(f"\n--- Combined Evaluation Results ---")
        print(f"Total Queries Evaluated: {total_count}")
        print(f"Exact Correct Queries: {exact_correct_count}")
        print(f"Equivalent (Column Reordered): {reordered_count}")
        print(f"Equivalent (Extra Columns): {extra_columns_count}")
        print(f"Equivalent (Tie Breaker): {tie_breaker_count}")
        print(f"Execution Failed: {execution_failed_count}")
        print(f"Pure Incorrect Results: {pure_incorrect_count}")
        print(f"-----------------------------------")
        print(f"Total Correct (Exact + Equivalents): {total_correct}")
        print(f"Accuracy (Exact): {accuracy_exact:.4f}")
        print(f"Accuracy (Joint/Equivalents): {accuracy_joint:.4f}")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    # Default entry point
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generated_file = os.path.abspath(os.path.join(current_dir, "../../results/sft/spider_test_set_ai_gemma3-4b-sft-cot-9k-0405-ep.json"))
    db_root = os.path.abspath(os.path.join(current_dir, "../../database"))
    
    evaluate_queries(generated_file, db_root)
