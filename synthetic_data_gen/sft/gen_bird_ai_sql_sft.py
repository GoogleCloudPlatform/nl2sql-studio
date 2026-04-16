import json
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm
from google.cloud import aiplatform
from get_schema_details import get_schema_details

# Configuration
ENDPOINT_ID = "4143612923366866944"
PROJECT_ID = "862253555914"
LOCATION = "asia-southeast1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

def get_ai_sql_qwen(endpoint_obj: aiplatform.Endpoint, schema_details_dict: dict, question: str, evidence: str = "") -> str:
    """
    Generates a SQL query ensuring 1:1 byte-parity with the SFT training data.
    """
    # 1. Match Training Indentation exactly
    schema_json_string = json.dumps(schema_details_dict, indent=2)
    
    # 2. Inject evidence directly into the question so the model doesn't ignore it
    if evidence:
        combined_question = f"{question} (Context to use: {evidence})"
    else:
        combined_question = question

    # 3. Match the exact training System Prompt
    system_prompt = "You are a powerful text-to-SQL model. Your role is to answer user questions by generating SQL queries against a given database schema. First, provide a step-by-step chain of thought that explains your reasoning, and then provide the final SQL query in a markdown code block."
    
    # 4. Match the exact training User Content format (including the \njson\n literal)
    user_content = f"DATABASE SCHEMA:\njson\n{schema_json_string}\n\n\nQuestion: {combined_question}"
    
    prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_content}\n<|im_end|>\n<|im_start|>assistant\n"

    final_sql = ""

    try:
        instances = [{
            "prompt": prompt,
            "max_tokens": 8192, 
            "temperature": 0.0,
            "frequency_penalty": 0.0,
            "stop": ["<|im_end|>"] 
        }]

        response = endpoint_obj.predict(instances=instances)

        if response.predictions:
            prediction = response.predictions[0]
            raw_content = prediction.get("text", "") if isinstance(prediction, dict) else str(prediction)

            # Strip the container's echo payload
            if "<|im_start|>assistant" in raw_content:
                raw_content = raw_content.split("<|im_start|>assistant")[-1]
            elif "Output:" in raw_content:
                raw_content = raw_content.split("Output:")[-1]

            raw_output = raw_content.replace("<|im_end|>", "").strip()
            
            # Extract SQL from markdown
            sql_match = re.search(r"```[sS][qQ][lL]\s*(.*?)\s*```", raw_output, re.DOTALL)
            
            if sql_match:
                final_sql = sql_match.group(1).strip()
            else:
                fallback_match = re.search(r"(?i)\b(SELECT\b.*?;?)", raw_output, re.DOTALL)
                if fallback_match:
                    final_sql = fallback_match.group(1).strip()
                else:
                    print(f"Extraction failed.")
                    final_sql = "EXTRACTION_FAILED"

            # BIRD safety check: Strip trailing semicolons as they can occasionally crash SQLite runners
            if final_sql and final_sql.endswith(";"):
                final_sql = final_sql[:-1].strip()

    except Exception as e:
        print(f"Error processing SQL: {e}")
        
    return final_sql

def add_ai_sql_to_json(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return
        
    aiplatform.init(project=PROJECT_ID, location=LOCATION, api_endpoint=API_ENDPOINT)
    vertex_endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    out_filename = file_path.replace('.json', '_ai_qwen.json')
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        base_db_path = os.path.join("../database")
        processed_data = []

        for item in tqdm(data):
            db_id = item.get('db_id')
            
            # THE FIX: Pass the RAW DICTIONARY to the function, do NOT json.dump it here!
            # The function needs the dict to apply the `indent=2` formatting.
            schema_dict = get_schema_details(db_id, base_db_path)
            
            ai_sql = get_ai_sql_qwen(
                vertex_endpoint, 
                schema_dict,  # Passed as dict
                item.get('question'), 
                item.get('evidence', '')
            )

            item["ai_generated_sql"] = ai_sql
            processed_data.append(item)

        with open(out_filename, 'w') as f:
            json.dump(processed_data, f, indent=4)

        print(f"\nSuccessfully created '{out_filename}'")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    json_file = '../results/sft/test_set.json' 
    add_ai_sql_to_json(json_file)