"""
BIRD AI SQL Generator
---------------------
This script calls a hosted Qwen model on Vertex AI to generate SQL queries for
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

# Configuration for Vertex AI Endpoint
ENDPOINT_ID = "561280894770348032"
PROJECT_ID = "862253555914"
LOCATION = "asia-southeast1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

class SQLResponse(BaseModel):
    """Schema for the AI's response to ensure a single SQL query is returned."""
    sql_query: str = Field(description="The executable SQLite query.")

def get_ai_sql_qwen(endpoint_obj: aiplatform.Endpoint, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using Pydantic-style JSON forcing on a Qwen model.
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
        # This specific container expects the prompt in a "text" field
        instances = [{"text": prompt}]
        
        # And parameters in a separate dict, wrapped in sampling_params
        parameters = {
            "sampling_params": {
                "max_new_tokens": 150,
                "temperature": 0.0,  # Zero randomness for deterministic code generation
                "stop": ["\"}", "\n", "<|im_end|>"] # Stop as soon as query or JSON ends
            }
        }

        # Call the endpoint object with both instances and parameters
        response = endpoint_obj.predict(instances=instances, parameters=parameters)

        if response.predictions:
            prediction = response.predictions[0]
            # Handle dictionary or string response from Vertex
            raw_content = prediction.get("text", "") if isinstance(prediction, dict) else str(prediction)

            # Reconstruct the JSON since we pre-filled the start.
            # We take the first line to be safe and split on the expected stop sequence.
            clean_content = raw_content.split("\n")[0].split("\"}")[0].strip()
            full_json_str = f'{{"sql_query": "{clean_content}"}}'

            # Validate with Pydantic
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

        # Resolve database path relative to this script
        base_db_path = os.path.join(os.path.dirname(__file__), "database")
        processed_data = []

        # Process each item with a progress bar
        for item in tqdm(data, desc="Generating SQL queries"):
            db_id = item.get('db_id')
            
            ai_sql = get_ai_sql_qwen(
                vertex_endpoint, 
                json.dumps(get_schema_details(db_id, base_db_path)), 
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
    json_file = 'filtered_dev.json' 
    add_ai_sql_to_json(json_file)