import json
import sqlite3
import argparse
import os
from typing import List, Any, Tuple, Dict, Optional

def execute_sql(db_path: str, query: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Executes a given SQL query on a specified SQLite database.

    Args:
        db_path: The file path to the SQLite database.
        query: The SQL query string to execute.

    Returns:
        A tuple containing the query result as a list of tuples and an error string.
        If execution is successful, the error string is None.
        If execution fails, the result is None.
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

    If the ground truth query does not contain an ORDER BY clause, the results
    are sorted before comparison to handle non-deterministic ordering.

    Args:
        ground_truth_query: The original ground truth SQL query string.
        ground_truth_result: The result from executing the ground truth query.
        ai_result: The result from executing the AI-generated query.

    Returns:
        True if the results are considered equal, False otherwise.
    """
    # If the AI-generated query produced an error, the results are not equal.
    if ai_result is None:
        return False

    # If the number of rows is different, they are not equal.
    if len(ground_truth_result) != len(ai_result):
        return False

    # If there's no ORDER BY in the ground truth, the order of results is not guaranteed.
    # We sort both results to ensure a fair comparison.
    if "order by" not in ground_truth_query.lower():
        try:
            # Sort using a stable sorting algorithm.
            # The set conversion handles potential duplicate rows before sorting.
            ground_truth_result = sorted(list(set(ground_truth_result)))
            ai_result = sorted(list(set(ai_result)))
        except TypeError:
            # This can happen if results contain unorderable types like None.
            # In such cases, we fall back to comparing the sets of tuples directly.
            return set(ground_truth_result) == set(ai_result)


    return ground_truth_result == ai_result

def evaluate_queries(data: List[Dict[str, Any]], db_root_path: str, verbose: bool = False) -> float:
    """
    Evaluates AI-generated SQL queries against ground truth queries.

    Args:
        data: A list of dictionaries, each containing query information.
        db_root_path: The root directory containing the database folders.
        verbose: If True, prints detailed results for each query.

    Returns:
        The execution accuracy as a float between 0.0 and 1.0.
    """
    correct_count = 0
    total_count = 0
    execution_failed_count = 0
    incorrect_result_count = 0

    for i, item in enumerate(data):
        db_id = item.get("db_id")
        ground_truth_sql = item.get("query")
        ai_sql = item.get("ai_generated_sql")
        question = item.get("question")

        if not all([db_id, ground_truth_sql, ai_sql]):
            # print(f"Skipping item {i+1} due to missing 'db_id', 'query', or 'ai_generated_sql'.")
            continue

        total_count += 1
        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

        # Execute ground truth query
        gt_result, gt_error = execute_sql(db_path, ground_truth_sql)
        if gt_error:
            print(f"Error in ground truth query for item {i+1} ({question}): {gt_error}")
            # We can't evaluate if the ground truth is broken.
            continue

        # Execute AI-generated query
        ai_result, ai_error = execute_sql(db_path, ai_sql)

        if ai_error:
            execution_failed_count += 1
            print(f"---------------------- Error in AI-generated query for item {i+1} ({question}): {ai_error}")


        is_correct = are_results_equal(ground_truth_sql, gt_result, ai_result)

        if is_correct:
            correct_count += 1
        elif not ai_error:
            incorrect_result_count += 1
            print(f"---------------------- Incorrect result for item {i+1} ({db_id}) \nQUESTION = {question} \nGT_RESULT = {gt_result} \nAI_RESULT = {ai_result}.")

        if verbose:
            print("-" * 40)
            print(f"Query #{i+1} for DB: {db_id}")
            print(f"Question: {item['question']}")
            print(f"Ground Truth SQL: {ground_truth_sql}")
            print(f"AI Generated SQL: {ai_sql}")
            if ai_error:
                print(f"AI Execution Error: {ai_error}")
            print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            print("-" * 40 + "\n")


    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n" + "=" * 40)
    print("Evaluation Summary")
    print(f"Total Queries Evaluated: {total_count}")
    print(f"  - Correct Results: {correct_count}")
    print(f"  - Incorrect Results: {incorrect_result_count}")
    print(f"  - Execution Failed: {execution_failed_count}")
    print(f"Execution Accuracy: {accuracy:.2f}%")
    print("=" * 40)

    return accuracy

def main(json_file, db_root, verbose=False):
    """Main function to run the evaluation script."""

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file}'.")
        return

    # --- Placeholder for AI-generated SQL ---
    # In a real scenario, this key would already be in your JSON file.
    # For demonstration, we'll add it to the first few entries.
    if data and 'ai_generated_sql' not in data[0]:
        print("Warning: 'ai_generated_sql' key not found. Using ground truth for demonstration.")
        # Correct query
        data[0]['ai_generated_sql'] = "SELECT count(*) FROM singer"
        data[1]['ai_generated_sql'] = "SELECT count(*) FROM singer"
        # Incorrect query (wrong column)
        data[2]['ai_generated_sql'] = "SELECT name ,  country FROM singer ORDER BY age DESC"
        # Query that will cause a syntax error
        data[4]['ai_generated_sql'] = "SELECT avg(age) min(age) max(age) FROM singer WHERE country  =  'France'"


    evaluate_queries(data, db_root, verbose)

if __name__ == "__main__":
    db_root = 'database'
    verbose = False
    eval_out_path = 'eval_out/'

    # json_file = eval_out_path + 'dev_filtered_ai_gemini-2.5-flash.json'
    json_file = eval_out_path + 'dev_filtered_ai_gemini-2.5-pro.json'

    main(json_file, db_root, verbose)
