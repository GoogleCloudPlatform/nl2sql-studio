"""
BIRD AI SQL Generator
---------------------
This script calls a hosted model on Vertex AI to generate SQL queries for
the BIRD dataset. It uses a specific container configuration that expects
parameters outside the instance list. 
"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from google.cloud import aiplatform
from get_schema_details import get_schema_details
import re

# Configuration for Vertex AI Endpoint
ENDPOINT_ID = "mg-endpoint-a6b654ee-0d0f-4fef-acd8-be194241f9b0"
PROJECT_ID = "862253555914"
LOCATION = "asia-southeast1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

def get_ai_sql_gemma(endpoint_obj: aiplatform.Endpoint, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using Gemma's native prompt formatting.
    """
    
    # 1. Smuggle the evidence directly into the question so the model doesn't ignore it
    if evidence:
        combined_question = f"{question} (Context to use: {evidence})"
    else:
        combined_question = question

    # 2. Construct the prompt using Gemma's native control tokens.
    # Gemma ONLY recognizes 'user' and 'model' roles. System instructions must be bundled into the user turn.
    prompt = f"""<start_of_turn>user
You are a powerful text-to-SQL model. Your role is to answer user questions by generating SQL queries against a given database schema. First, provide a step-by-step chain of thought that explains your reasoning, and then provide the final SQL query in a markdown code block.

Schema: 
{schema_details}

Question: {combined_question}<end_of_turn>
<start_of_turn>model
"""

    final_sql = ""

    try:
        # Pass the prompt and configuration into the instances array payload
        instances = [{
            "prompt": prompt,
            "max_tokens": 8192, 
            "temperature": 0.0,
            "frequency_penalty": 0.0,  # Set to 0.0 to prevent the model from going into a rambling loop
            "stop": ["<end_of_turn>"]  # Updated to Gemma's native stop token
        }]

        # Call the endpoint object
        response = endpoint_obj.predict(instances=instances)

        if response.predictions:
            prediction = response.predictions[0]
            print(prediction)
            raw_content = prediction.get("text", "") if isinstance(prediction, dict) else str(prediction)

            # Strip the container's echo payload aggressively (Output split first)
            if "Output:" in raw_content:
                raw_content = raw_content.split("Output:")[-1]
            elif "<start_of_turn>model" in raw_content:
                raw_content = raw_content.split("<start_of_turn>model")[-1]

            raw_output = raw_content.replace("<end_of_turn>", "").strip()
            
            # Extract SQL from markdown block
            sql_match = re.search(r"```[sS][qQ][lL]\s*(.*?)\s*```", raw_output, re.DOTALL)
            
            if sql_match:
                final_sql = sql_match.group(1).strip()
            else:
                # Fallback Regex ensuring we capture the full query if it forgot markdown
                fallback_match = re.search(r'(?i)\b(SELECT\b[^"]*)', raw_output, re.DOTALL)
                if fallback_match:
                    final_sql = fallback_match.group(1).strip()
                else:
                    print(f"Extraction failed.")
                    final_sql = "EXTRACTION_FAILED"

            # BIRD safety check: Strip trailing semicolons to prevent SQLite crashes
            if final_sql and final_sql.endswith(";"):
                final_sql = final_sql[:-1].strip()

    except Exception as e:
        print(f"Error processing SQL: {e}")
        
    return final_sql

def add_ai_sql_to_json(file_path: str):
    """
    Reads a JSON file of questions, generates SQL for each, and saves a new JSON.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return
        
    # Initialize Vertex AI once for the session
    aiplatform.init(project=PROJECT_ID, location=LOCATION, api_endpoint=API_ENDPOINT)
    vertex_endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    out_filename = file_path.replace('.json', f'_ai_{MODEL_NAME}.json')
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        processed_data = []

        # Process each item with a progress bar
        for item in tqdm(data, desc="Generating SQL queries"):
            db_id = item.get('db_id')
            
            ai_sql = get_ai_sql_gemma(
                vertex_endpoint, 
                json.dumps(get_schema_details(db_id), indent=2), # Indent the schema to prevent format collapse
                item.get('question'), 
                item.get('evidence', '')
            )

            item["ai_generated_sql"] = ai_sql
            processed_data.append(item)

        # Save the updated data with the new AI-generated SQL field
        with open(out_filename, 'w') as f:
            json.dump(processed_data, f, indent=4)

        print(f"\nSuccessfully created '{out_filename}'")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Input file containing BIRD questions
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.abspath(os.path.join(current_dir, "../results/sft/spider_test_set.json"))
    MODEL_NAME = 'gemma4-26b-base'
    add_ai_sql_to_json(json_file)