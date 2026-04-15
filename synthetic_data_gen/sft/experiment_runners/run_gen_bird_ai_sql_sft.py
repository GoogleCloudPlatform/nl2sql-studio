import os
import sys

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_bird_ai_sql_sft import add_ai_sql_to_json

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input file containing BIRD questions
    # The original pointed to '../results/sft/filtered_dev.json' relative to sft/
    # From sft/experiment_runners/, it is '../../results/sft/filtered_dev.json'
    json_file = os.path.abspath(os.path.join(script_dir, '../../results/sft/filtered_dev.json'))
    
    print(f"Running gen_bird_ai_sql_sft on: {json_file}")
    add_ai_sql_to_json(json_file)
