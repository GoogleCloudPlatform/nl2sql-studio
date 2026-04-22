"""
BIRD AI SQL Generator
---------------------
This script calls a hosted model on Vertex AI to generate SQL queries for
the BIRD dataset. It uses a specific container configuration that expects
parameters outside the instance list. It uses prompt pre-filling to force
a JSON response containing the SQL query.
"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from pydantic import BaseModel, Field
from google.cloud import aiplatform
from get_schema_details import get_schema_details
import re

# Configuration for Vertex AI Endpoint
ENDPOINT_ID = "mg-endpoint-ed425d0b-3cde-4d67-bb71-a0efa63f17df"
PROJECT_ID = "862253555914"
LOCATION = "asia-southeast1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

class SQLResponse(BaseModel):
    """Schema for the AI's response to ensure a single SQL query is returned."""
    sql_query: str = Field(description="The executable SQLite query.")

def get_ai_sql_gemma(endpoint_obj: aiplatform.Endpoint, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using Pydantic-style JSON forcing on a model.
    This version uses a container that expects 'text' in instances and a separate 'parameters' dict.
    
    Args:
        endpoint_obj (aiplatform.Endpoint): The initialized Vertex AI endpoint.
        schema_details (str): JSON string of the database schema.
        question (str): The natural language question.
        evidence (str): Additional context or hints for the query.
        
    Returns:
        str: The extracted and validated SQL query.
    """
    # Create the schema hint for the model
    json_schema = SQLResponse.model_json_schema()
    
    # Construct the prompt. We pre-fill the assistant response with the start of the JSON
    # to lock the model into generating valid JSON matching our schema.
    prompt = f"""<|im_start|>system
    You are a powerful text-to-SQL model. Your role is to answer user questions by generating SQL queries against a given database schema. First, provide a step-by-step chain of thought that explains your reasoning, and then provide the final SQL query in a markdown code block.
    <|im_end|>\n
    <|im_start|>user
    Schema: {schema_details}
    Question: {question}
    <|im_end|>\n
    <|im_start|>assistant
    {{"sql_query": \""""

    final_sql = ""

    try:
        # This specific container expects the prompt in a "text" field
        instances = [{
            "prompt": prompt,
            "max_tokens": 8192, 
            "temperature": 0.0,
            "stop": ["<|im_end|>"] 
        
        }]

        # Call the endpoint object with both instances and parameters
        response = endpoint_obj.predict(instances=instances)

        if response.predictions:
            prediction = response.predictions[0]
            print(f"\n--- Prediction Response ---")
            print(prediction)
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
                fallback_match = re.search(r'(?i)\b(SELECT\b[^"]*)', raw_output, re.DOTALL)
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
    """
    Reads a JSON file of questions, generates SQL for each, and saves a new JSON.
    
    Args:
        file_path (str): Path to the input JSON file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return
        
    # Initialize Vertex AI once for the session
    aiplatform.init(project=PROJECT_ID, location=LOCATION, api_endpoint=API_ENDPOINT)
    # Create the Endpoint Object once to reuse connections
    vertex_endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    out_filename = file_path.replace('.json', f'_ai_{MODEL_NAME}.json')
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Resolve database path relative to this script
        # base_db_path = os.path.join(os.path.dirname(__file__), "database")
        processed_data = []

        # Process each item with a progress bar
        for item in tqdm(data, desc="Generating SQL queries"):
            db_id = item.get('db_id')
            
            ai_sql = get_ai_sql_gemma(
                vertex_endpoint, 
                json.dumps(get_schema_details(db_id)), 
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
    json_file = '../results/sft/spider_test_set.json' 
    MODEL_NAME = 'gemma-26b-base'
    add_ai_sql_to_json(json_file)