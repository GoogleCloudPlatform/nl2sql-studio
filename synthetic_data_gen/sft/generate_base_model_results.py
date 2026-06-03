"""
Unified Foundation/Base Model SQL Generator
--------------------------------------------
This script calls Vertex AI hosted foundation/base models (Gemma 3, Gemma 4)
and native Gemini models via the google-genai SDK to generate SQL queries.
It handles connection pooling, progress checkpointing, and Ctrl+C recovery gracefully.
"""

import json
import os
import re
import time
import random
import sys
from typing import List, Dict, Any
from tqdm import tqdm
from pydantic import BaseModel, Field
import argparse

# Vertex AI & Cloud imports
from google.cloud import aiplatform

# GenAI imports
from google import genai
from google.genai import types

from utils.get_schema_details import get_schema_details



class SQLResponse(BaseModel):
    """Schema for Structured LLM Responses (Gemini) to ensure clean output formatting."""
    reasoning: str = Field(description="Step-by-step chain of thought explaining the reasoning.")
    sql_query: str = Field(description="The executable SQLite query.")


# ==========================================
# MODEL GENERATOR FUNCTIONS
# ==========================================

def get_ai_sql_gemini(client: genai.Client, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using Gemini 2.5 Flash with structured JSON response forcing.

    Args:
        client (genai.Client): GenAI SDK client instance configured for Vertex AI.
        schema_details (str): Serialized schema of the target database tables.
        question (str): Natural language user question.
        evidence (str, optional): Additional database context/hints. Defaults to "".

    Returns:
        str: Generated and extracted SQL query, or error placeholder.
    """
    # Append schema/evidence context if present
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

    # Call Gemini API with exponential backoff for rate limits/exceptions
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
                    # Enforce strict response schema validation
                    parsed_response = json.loads(raw_output)
                    final_sql = parsed_response.get("sql_query", "").strip()
                except Exception as parse_err:
                    print(f"Failed to parse JSON structured output from Gemini response: {parse_err}")
                    # Fallback 1: Extract SQL blocks from markdown format
                    sql_match = re.search(r"```[sS][qQ][lL]\s*(.*?)\s*```", raw_output, re.DOTALL)
                    if sql_match:
                        final_sql = sql_match.group(1).strip()
                    else:
                        # Fallback 2: Pull out anything starting with SELECT
                        fallback_match = re.search(r'(?i)\b(SELECT\b[^"]*)', raw_output, re.DOTALL)
                        if fallback_match:
                            final_sql = fallback_match.group(1).strip()
                        else:
                            final_sql = "EXTRACTION_FAILED"

                # BIRD dataset safety check: Trailing semicolons can cause execution bugs under some wrappers
                if final_sql and final_sql.endswith(";"):
                    final_sql = final_sql[:-1].strip()

            break

        except Exception as e:
            print(f"Error processing SQL on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                break
            err_str = str(e)
            # Differentiate between quota exhaustions and standard connection/syntax failures
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Exponential backoff with jitter for Rate Limits
                sleep_time = (2 ** attempt) * 5 + random.uniform(1, 5)
            else:
                sleep_time = 2 ** attempt
            print(f"Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        
    return final_sql


def get_ai_sql_gemma3(endpoint_obj: aiplatform.Endpoint, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using Gemma 3's container prompt format and prefix pre-filling.

    Uses Vertex AI Endpoint predictions. Pre-fills the assistant response with `{"sql_query": "`
    to force JSON-structured output formatting from Gemma 3.

    Args:
        endpoint_obj (aiplatform.Endpoint): Initialized Vertex AI model endpoint object.
        schema_details (str): Serialized schema of the target database tables.
        question (str): Natural language user question.
        evidence (str, optional): Additional database context/hints. Defaults to "".

    Returns:
        str: Generated SQL query string.
    """
    # Prefix pre-filling forces JSON structured outputs on non-schema models
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

    max_retries = 10
    for attempt in range(max_retries):
        try:
            instances = [{
                "prompt": prompt,
                "max_tokens": 8192, 
                "temperature": 0.0,
                "frequency_penalty": 1.0,
                "stop": ["<|im_end|>"] 
            }]

            response = endpoint_obj.predict(instances=instances)

            if response.predictions:
                prediction = response.predictions[0]
                raw_content = prediction.get("text", "") if isinstance(prediction, dict) else str(prediction)

                # Strip container echo headers
                if "<|im_start|>assistant" in raw_content:
                    raw_content = raw_content.split("<|im_start|>assistant")[-1]
                elif "Output:" in raw_content:
                    raw_content = raw_content.split("Output:")[-1]

                raw_output = raw_content.replace("<|im_end|>", "").strip()
                
                # Extract SQL
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


def get_ai_sql_gemma4(endpoint_obj: aiplatform.Endpoint, schema_details: str, question: str, evidence: str = "") -> str:
    """
    Generates a single SQL query using Gemma 4's native control tokens prompt format.

    Args:
        endpoint_obj (aiplatform.Endpoint): Initialized Vertex AI model endpoint object.
        schema_details (str): Serialized schema of the target database tables.
        question (str): Natural language user question.
        evidence (str, optional): Additional database context/hints. Defaults to "".

    Returns:
        str: Generated SQL query string.
    """
    if evidence:
        combined_question = f"{question} (Context to use: {evidence})"
    else:
        combined_question = question

    # Uses standard Gemma 4 chat delimiters <start_of_turn> and <end_of_turn>
    prompt = f"""<start_of_turn>user
You are a powerful text-to-SQL model. Your role is to answer user questions by generating SQL queries against a given database schema. First, provide a step-by-step chain of thought that explains your reasoning, and then provide the final SQL query in a markdown code block.

Schema: 
{schema_details}

Question: {combined_question}<end_of_turn>
<start_of_turn>model
"""

    final_sql = ""

    max_retries = 10
    for attempt in range(max_retries):
        try:
            instances = [{
                "prompt": prompt,
                "max_tokens": 8192, 
                "temperature": 0.0,
                "frequency_penalty": 0.0,
                "stop": ["<end_of_turn>"]
            }]

            response = endpoint_obj.predict(instances=instances)

            if response.predictions:
                prediction = response.predictions[0]
                raw_content = prediction.get("text", "") if isinstance(prediction, dict) else str(prediction)

                # Strip container echo headers
                if "Output:" in raw_content:
                    raw_content = raw_content.split("Output:")[-1]
                elif "<start_of_turn>model" in raw_content:
                    raw_content = raw_content.split("<start_of_turn>model")[-1]

                raw_output = raw_content.replace("<end_of_turn>", "").strip()
                
                # Extract SQL
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


# ==========================================
# UNIFIED CORE PIPELINE
# ==========================================

def add_ai_sql_to_json(file_path: str, model_name: str, endpoint_id: str, project: str, location: str):
    """
    Unified execution pipeline that loads database schemas, queries models based on name routing,
    safeguards checkpoint progress, and supports graceful termination.

    Loads the input JSON containing NL2SQL evaluation cases, queries the respective model routing
    with exponential backoff, periodically saves progress to a checkpoint file, and writes the
    final output list to disk.

    Args:
        file_path (str): Path to input JSON containing questions and metadata.
        model_name (str): Selector string for the model (e.g., 'gemini', 'gemma3', 'gemma4').
        endpoint_id (str): Vertex AI Custom Endpoint ID for Gemma endpoints.
        project (str): GCP Project ID.
        location (str): GCP region (e.g., 'us-central1').
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    # 1. Determine Model Type from Name Routing
    model_lower = model_name.lower()
    if "gemini" in model_lower:
        model_type = "gemini"
    elif "gemma4" in model_lower:
        model_type = "gemma4"
    else:
        model_type = "gemma3" # Default base model

    print(f"Initializing model environment for: '{model_name}' (type: {model_type})")

    # 2. Initialize Connection Clients based on model type
    if model_type == "gemini":
        try:
            # Use native Google GenAI SDK (Vertex AI backend configuration)
            llm_client = genai.Client(
                vertexai=True,
                project=project,
                location=location
            )
            endpoint_obj = llm_client
        except Exception as e:
            print(f"Error initializing Gemini Client: {e}")
            return
    else:
        try:
            # Initialize Vertex AI Platform connection for custom container endpoints
            api_endpoint = f"{location}-aiplatform.googleapis.com"
            aiplatform.init(project=project, location=location, api_endpoint=api_endpoint)
            endpoint_obj = aiplatform.Endpoint(endpoint_id)
        except Exception as e:
            print(f"Error initializing Vertex Endpoint: {e}")
            return

    # 3. Prepare Outputs & File Paths
    out_filename = file_path.replace('.json', f'_ai_{model_name}.json')
    checkpoint_filename = file_path.replace('.json', f'_ai_{model_name}_checkpoint.json')
    
    # Resolve database directory path containing SQLite files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_db_path = os.path.abspath(os.path.join(current_dir, "../database"))

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Load checkpoints if resuming from a crashed or interrupted run
        completed_queries = {}
        if os.path.exists(checkpoint_filename):
            try:
                with open(checkpoint_filename, 'r') as f:
                    checkpoint_data = json.load(f)
                completed_queries = {
                    item['question']: item['ai_generated_sql'] 
                    for item in checkpoint_data 
                    if 'ai_generated_sql' in item
                }
                print(f"Loaded checkpoint: resuming from {len(completed_queries)}/{len(data)} queries.")
            except Exception as cp_err:
                print(f"Could not load checkpoint: {cp_err}. Starting fresh.")

        processed_data = []

        # 4. Generation Loop with tqdm Progress Bar
        for idx, item in enumerate(tqdm(data, desc="Generating SQL queries")):
            question = item.get('question')
            
            # Skip calls for questions that were completed and stored in checkpoint files
            if question in completed_queries:
                item["ai_generated_sql"] = completed_queries[question]
                processed_data.append(item)
                continue

            db_id = item.get('db_id')
            # Extract database structure to build prompt schema block
            schema_details = get_schema_details(db_id, base_db_path)
            schema_json_str = json.dumps(schema_details, indent=2)

            # Route calls to the corresponding prompt generators based on model type
            if model_type == "gemini":
                ai_sql = get_ai_sql_gemini(endpoint_obj, schema_json_str, question, item.get('evidence', ''))
            elif model_type == "gemma4":
                ai_sql = get_ai_sql_gemma4(endpoint_obj, schema_json_str, question, item.get('evidence', ''))
            else:
                ai_sql = get_ai_sql_gemma3(endpoint_obj, schema_json_str, question, item.get('evidence', ''))

            item["ai_generated_sql"] = ai_sql
            processed_data.append(item)

            # Save checkpoint every 5 items to minimize loss on execution termination
            if (idx + 1) % 5 == 0 or (idx + 1) == len(data):
                with open(checkpoint_filename, 'w') as f:
                    json.dump(processed_data, f, indent=4)

        # Save final complete dataset results
        with open(out_filename, 'w') as f:
            json.dump(processed_data, f, indent=4)

        # Clean up temporary checkpoint file on success
        if os.path.exists(checkpoint_filename):
            os.remove(checkpoint_filename)

        print(f"\nSuccessfully created '{out_filename}'")

    except KeyboardInterrupt:
        # Handle Ctrl+C interruptions gracefully by persisting progress before exit
        print(f"\nExecution interrupted by user (Ctrl+C). Saving progress to checkpoint...")
        try:
            with open(checkpoint_filename, 'w') as f:
                json.dump(processed_data, f, indent=4)
            print(f"Progress successfully saved to: {checkpoint_filename}")
        except Exception as save_err:
            print(f"Failed to save checkpoint on interrupt: {save_err}")
        sys.exit(130)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser(description="Generate SQL queries using Vertex AI base models.")
#     parser.add_argument("--input", type=str, required=True, help="Path to input JSON file containing questions.")
#     parser.add_argument("--model-name", type=str, required=True, help="Model name to evaluate.")
#     parser.add_argument("--endpoint-id", type=str, required=True, help="Vertex AI Endpoint ID (only for custom Gemma models).")
#     parser.add_argument("--project", type=str, default=PROJECT_ID, help="Vertex AI Project ID.")
#     parser.add_argument("--location", type=str, default=LOCATION, help="Vertex AI Location.")
#     args = parser.parse_args()

#     add_ai_sql_to_json(args.input, args.model_name, endpoint_id=args.endpoint_id, project=args.project, location=args.location)