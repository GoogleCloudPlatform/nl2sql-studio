import os
import sys

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bird_eval_sql import evaluate_bird_queries

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input file containing AI generated SQL
    # Using relative paths resolved to absolute for robustness
    generated_file = os.path.abspath(os.path.join(script_dir, '../../results/sft/filtered_dev_ai_qwen.json'))
    
    # Database root path
    db_root = os.path.abspath(os.path.join(script_dir, '../../database'))
    
    print(f"Running bird_eval on: {generated_file}")
    evaluate_bird_queries(generated_file, db_root)
