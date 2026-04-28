import os
import sys

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_completeness import check_sql_completeness

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.abspath(os.path.join(current_dir, "../results/sft/filtered_dev_ai_qwen.json"))
    
    print(f"Running check_completeness on: {json_file}")
    check_sql_completeness(json_file)
