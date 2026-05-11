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
from google import genai
from google.genai import types
from get_schema_details import get_schema_details
import re
import time
import random
import config

class SQLResponse(BaseModel):
    """Schema for the AI's response to ensure a single SQL query is returned with reasoning."""
    reasoning: str = Field(description="Step-by-step chain of thought explaining the reasoning.")
    sql_query: str = Field(description="The executable SQLite query.")

def get_ai_sql_gemini(client: genai.Client, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using the Gemini model with structured JSON output.
    
    Args:
        client (genai.Client): The initialized GenAI client.
        schema_details (str): JSON string of the database schema.
        question (str): The natural language question.
        evidence (str): Additional context or hints for the query.
        
    Returns:
        str: The extracted and validated SQL query.
    """
    if evidence:
        combined_question = f"{question} (Context to use: {evidence})"
    else:
        combined_question = question

    system_instruction = (
        "You are a powerful text-to-SQL model. Your role is to answer user questions by generating "
        "valid SQL queries against a given database schema. Provide your step-by-step reasoning, "
        "and then provide the final executable SQLite query."
    )

    prompt = f"Schema:\n{schema_details}\n\nQuestion: {combined_question}"
    final_sql = ""

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=SQLResponse,
                    temperature=0.0,
                )
            )

            if response.text:
                raw_output = response.text.strip()
                try:
                    parsed_response = json.loads(raw_output)
                    final_sql = parsed_response.get("sql_query", "").strip()
                except Exception as parse_err:
                    print(f"Failed to parse JSON structured output from Gemini response: {parse_err}")
                    # Fallback regex if somehow it returns non-strict json but text exists
                    sql_match = re.search(r"```[sS][qQ][lL]\s*(.*?)\s*```", raw_output, re.DOTALL)
                    if sql_match:
                        final_sql = sql_match.group(1).strip()
                    else:
                        fallback_match = re.search(r'(?i)\b(SELECT\b[^"]*)', raw_output, re.DOTALL)
                        if fallback_match:
                            final_sql = fallback_match.group(1).strip()
                        else:
                            final_sql = "EXTRACTION_FAILED"

                # BIRD safety check: Strip trailing semicolons
                if final_sql and final_sql.endswith(";"):
                    final_sql = final_sql[:-1].strip()

            break

        except Exception as e:
            print(f"Error processing SQL on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                break
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                sleep_time = (2 ** attempt) * 5 + random.uniform(1, 5)
            else:
                sleep_time = 2 ** attempt
            print(f"Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        
    return final_sql

def add_ai_sql_to_json(file_path: str, db_path: str = None):
    """
    Reads a JSON file of questions, generates SQL for each, and saves a new JSON.
    
    Args:
        file_path (str): Path to the input JSON file.
        db_path (str): Path to the SQLite databases directory.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    if not db_path:
        # Infer database directory path relative to this script file (../database)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.abspath(os.path.join(script_dir, "../database"))
        
    # Initialize the Gemini Client with Vertex AI enabled from config
    try:
        llm_client = genai.Client(
            vertexai=config.VERTEXAI,
            project=config.PROJECT,
            location=config.LOCATION
        )
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}")
        return

    out_filename = file_path.replace('.json', f'_ai_{MODEL_NAME}.json')
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        processed_data = []

        # Process each item with a progress bar
        for item in tqdm(data, desc="Generating SQL queries"):
            db_id = item.get('db_id')
            
            ai_sql = get_ai_sql_gemini(
                llm_client, 
                json.dumps(get_schema_details(db_id, base_db_path=db_path)), 
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
    # Input file containing BIRD/Spider questions
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.abspath(os.path.join(current_dir, "../results/sft/spider_test_set.json"))
    MODEL_NAME = 'gemini-2.5-flash'
    add_ai_sql_to_json(json_file)