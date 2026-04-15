"""
BIRD AI SQL Generator (SFT Preparation)
---------------------------------------
This script calls a hosted Qwen model on Vertex AI to generate SQL queries for
the BIRD dataset. It uses strict JSON schema enforcement (via Pydantic) and
prompt pre-filling to ensure the model returns a clean, parseable JSON object
containing ONLY the SQL query.
"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from pydantic import BaseModel, Field
from google.cloud import aiplatform
from get_schema_details import get_schema_details

# Configuration for Vertex AI Endpoint
ENDPOINT_ID = "4143612923366866944"
PROJECT_ID = "862253555914"
LOCATION = "asia-southeast1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

class SQLResponse(BaseModel):
    """Schema for the AI's response to ensure a single SQL query is returned."""
    sql_query: str = Field(description="The executable SQLite query.")

def get_ai_sql_qwen(endpoint_obj: aiplatform.Endpoint, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using Pydantic-style JSON forcing on a Qwen model.
    
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
     Given the database schema details below and a natural language query, generate the corresponding SQL query.
    Make sure the SQL query is compatible with SQLite.
    Double check all the table names are matching with schema and all the column names are matching for the corresponding table in the schema.
    Think step by step and ensure the SQL query is syntactically correct and executable and You are a SQL expert. You must respond ONLY with a JSON object matching this schema:
    {json.dumps(json_schema)}
    Do not include explanations, thinking tags, or multiple queries.
    <|im_end|>
    <|im_start|>user
    Schema: {schema_details}
    Evidence: {evidence}
    Question: {question}
    Generate the JSON:
    <|im_end|>
    <|im_start|>assistant
    {{"sql_query": \""""

    final_sql = ""

    try:
        # Use 'prompt' key as required by the Qwen container on Vertex
        instances = [{
            "prompt": prompt,
            "max_tokens": 2048, 
            "temperature": 0.0,
            "stop": ["\"}", "<|im_end|>"]
        }]

        # Call the endpoint object. We drop the 'parameters' argument completely
        # as this specific container expects everything inside 'instances'.
        response = endpoint_obj.predict(instances=instances)

        if response.predictions:
            prediction = response.predictions[0]
            print(f"DEBUG - Raw prediction: {prediction}")
            
            # Handle dictionary or string response from Vertex depending on container version
            raw_content = prediction.get("text", "") if isinstance(prediction, dict) else str(prediction)
            print(f"DEBUG - Raw content: {raw_content}")
 
            # Extract the part after "Output:" if the container echoes the prompt header
            if "Output:" in raw_content:
                raw_content = raw_content.split("Output:")[1]

            # Preserve the full multi-line SQL block
            clean_content = raw_content.strip()
            
            # Remove trailing quote and brace if present (due to greedy generation)
            if clean_content.endswith('"}'):
                clean_content = clean_content[:-2]
            elif clean_content.endswith('"'):
                clean_content = clean_content[:-1]
            clean_content = clean_content.strip()
            
            print(f"DEBUG - Clean content: {clean_content}")
            
            # Escape internal newlines so JSON parsing doesn't crash on multi-line SQL
            clean_content_escaped = clean_content.replace('\n', ' ')
            
            # Reconstruct full JSON string for validation
            full_json_str = f'{{"sql_query": "{clean_content_escaped}"}}'

            # Validate with Pydantic to ensure schema adherence
            parsed = SQLResponse.model_validate_json(full_json_str)
            final_sql = parsed.sql_query

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

    out_filename = file_path.replace('.json', '_ai_qwen.json')
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        base_db_path = os.path.join("../database")
        processed_data = []

        # Process each item with a progress bar
        for item in tqdm(data, desc="Generating SQL queries"):
            db_id = item.get('db_id')
            
            # Fetch schema details dynamically for each database
            schema_json = json.dumps(get_schema_details(db_id, base_db_path))
            
            ai_sql = get_ai_sql_qwen(
                vertex_endpoint, 
                schema_json, 
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
    # Input file containing BIRD dev set questions
    json_file = '../results/sft/filtered_dev.json' 
    add_ai_sql_to_json(json_file)