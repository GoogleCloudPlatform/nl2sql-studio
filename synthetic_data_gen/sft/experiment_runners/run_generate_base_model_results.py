import os
import sys

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_base_model_results import add_ai_sql_to_json

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input file containing BIRD questions
    # The original pointed to 'filtered_dev.json' relative to sft/
    # From sft/experiment_runners/, it is '../filtered_dev.json'
    json_file = os.path.abspath(os.path.join(script_dir, '../filtered_dev.json'))
    
    print(f"Running gen_bird_ai_sql on: {json_file}")
    add_ai_sql_to_json(json_file)
