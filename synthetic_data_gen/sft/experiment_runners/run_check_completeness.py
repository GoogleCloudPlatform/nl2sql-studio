import os
import sys

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_completeness import check_sql_completeness

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Point to the JSON file relative to this script
    # The original pointed to '../results/sft/filtered_dev_ai_qwen.json' relative to sft/
    # From sft/experiment_runners/, it is '../../results/sft/filtered_dev_ai_qwen.json'
    json_file = os.path.abspath(os.path.join(script_dir, '../../results/sft/filtered_dev_ai_qwen.json'))
    
    print(f"Running check_completeness on: {json_file}")
    check_sql_completeness(json_file)
