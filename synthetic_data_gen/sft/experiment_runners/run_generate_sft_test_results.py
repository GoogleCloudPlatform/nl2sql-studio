import os
import sys
import argparse

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

from generate_sft_test_results import add_ai_sql_to_json

if __name__ == "__main__":
    # Default Vertex AI Configuration
    PROJECT_ID = config.PROJECT
    LOCATION = config.LOCATION
    
    parser = argparse.ArgumentParser(description="Generate SQL queries using SFT fine-tuned models.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file containing questions.")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to evaluate.")
    parser.add_argument("--endpoint-id", type=str, required=True, help="Vertex AI Endpoint ID.")
    parser.add_argument("--project", type=str, default=PROJECT_ID, help="Vertex AI Project ID.")
    parser.add_argument("--location", type=str, default=LOCATION, help="Vertex AI Location.")
    args = parser.parse_args()

    print(f"Running gen_bird_ai_sql_sft on: {args.input} with model: {args.model_name}")
    add_ai_sql_to_json(args.input, args.model_name, endpoint_id=args.endpoint_id, project=args.project, location=args.location)
