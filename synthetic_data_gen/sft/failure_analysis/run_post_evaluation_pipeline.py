import sys
import os

# Ensure imports can be resolved relative to this file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from results_eval import evaluate_queries
from analyse_failures import analyze_file

def run_evaluation_pipeline(input_file_path: str, db_root_path: str = None):
    """
    Sequentially runs results_eval.py and analyse_results.py on the input file path.
    
    Args:
        input_file_path (str): Path to the JSON file containing AI generated SQL queries.
        db_root_path (str): Path to the directory where the SQLite databases are stored.
    """
    if db_root_path is None:
        db_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database"))

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        sys.exit(1)

    print(f"Step 1: Evaluating SQL queries in: {input_file_path}")
    evaluate_queries(input_file_path, db_root_path)

    print(f"\nStep 2: Analyzing evaluation results for: {input_file_path}")
    analyze_file(input_file_path)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.abspath(os.path.join(current_dir, "../../results/sft/spider_test_set_ai_gemma3-4b-sft-cot-8k.json"))
    db_root = os.path.abspath(os.path.join(current_dir, "../../database"))

    run_evaluation_pipeline(input_file, db_root)
# nl2sql-studio/synthetic_data_gen/results/sft/spider_test_set_ai_gemma3-4b-sft-cot-8k.json