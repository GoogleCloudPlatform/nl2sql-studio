"""
Post-Evaluation Analysis Pipeline
---------------------------------
This script orchestrates the post-evaluation pipeline by sequentially invoking:
1. results_eval.py (evaluate_queries): Executes and compares generated SQL with ground truth, 
   detecting semantic equivalents (column permutations, extra columns, tie-breakers).
2. analyse_failures.py (analyze_file): Computes and displays structured statistics on success rates,
   failure modes, query complexity, db schemas, and keywords.
"""

import sys
import os

# Ensure local imports (results_eval, analyse_failures) can be resolved
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from results_eval import evaluate_queries
from analyse_failures import analyze_file

def run_evaluation_pipeline(input_file_path: str, db_root_path: str = None):
    """
    Sequentially runs query execution evaluation followed by failure analysis.

    Args:
        input_file_path (str): Path to the JSON file containing AI generated SQL queries.
        db_root_path (str, optional): Path to the directory where the SQLite databases are stored.
                                      Defaults to "../../database" relative to this script.
    """
    if db_root_path is None:
        # Default fallback to "../../database" relative to script location
        db_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database"))

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        sys.exit(1)

    # Step 1: Execute SQL queries and determine semantic correctness categories
    print(f"Step 1: Evaluating SQL queries in: {input_file_path}")
    evaluate_queries(input_file_path, db_root_path)

    # Step 2: Extract statistical breakdowns (success, failure modes, complexity, etc.)
    print(f"\nStep 2: Analyzing evaluation results for: {input_file_path}")
    analyze_file(input_file_path)

if __name__ == "__main__":
    # Setup path resolution for local dev testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.abspath(os.path.join(current_dir, "../../results/sft/spider_test_set_ai_gemma3-4b-sft-cot-8k.json"))
    db_root = os.path.abspath(os.path.join(current_dir, "../../database"))

    # Execute the post-evaluation pipeline
    run_evaluation_pipeline(input_file, db_root)