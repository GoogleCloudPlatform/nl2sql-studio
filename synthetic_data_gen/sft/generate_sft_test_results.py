import json
import re
from typing import List, Dict, Any
from tqdm import tqdm
from google.cloud import aiplatform
from utils.get_schema_details import get_schema_details
import time
import random
import os

def get_ai_sql_qwen_or_gemma(endpoint_obj: Any, schema_details_dict: dict, question: str, evidence: str = "", model: str = "gemma", endpoint_id: str = None, project: str = None, location: str = None) -> str:
    """
    Generates a SQL query ensuring 1:1 byte-parity with the SFT training data format.

    This ensures that test inputs match the exact prompt structure (indentation, system instructions,
    special tags, and delimiters) that the model saw during fine-tuning.

    Args:
        endpoint_obj: Initialized client object (genai.Client or aiplatform.Endpoint).
        schema_details_dict (dict): Database schema structure as a dict.
        question (str): User question.
        evidence (str, optional): Additional contextual clues. Defaults to "".
        model (str): Name/type category ('gemini' or 'gemma'). Defaults to "gemma".
        endpoint_id (str, optional): Deployment endpoint id string.
        project (str, optional): GCP Project ID.
        location (str, optional): GCP Region location.

    Returns:
        str: Cleaned, executable SQL query string.
    """
    # 1. Match Training Indentation exactly (2-space indent for JSON schemas)
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
    
    # 5. Format base prompt with model-specific chat template syntax (e.g. ChatML tags)
    if model.lower() == "gemini":
        prompt = f"{system_prompt}\n\n{user_content}"
    else:
        prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_content}\n<|im_end|>\n<|im_start|>assistant\n"

    final_sql = ""

    # Call target endpoint client with backoff retries
    max_retries = 10
    for attempt in range(max_retries):
        try:
            if model.lower() == "gemini":
                from google.genai import types
                if "/" in endpoint_id:
                    model_resource = endpoint_id
                else:
                    # Construct full Vertex model endpoint path
                    model_resource = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
                
                response = endpoint_obj.models.generate_content(
                    model=model_resource,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        # Disable thinking config since the SFT base model handles reasoning explicitly
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,
                        )
                    )
                )
                raw_content = response.text if response.text else ""
                raw_output = raw_content.strip()
            else:
                # Custom/Gemma fine-tuned models deployed to Vertex Endpoint
                instances = [{
                    "prompt": prompt,
                    "max_tokens": 8192, 
                    "temperature": 0.0,
                    "stop": ["<|im_end|>"] 
                }]

                response = endpoint_obj.predict(instances=instances)

                if response.predictions:
                    prediction = response.predictions[0]
                    print(prediction)
                    
                    # Handle different possible formats in the prediction payload depending on container setup
                    if isinstance(prediction, dict):
                        raw_content = prediction.get("text", "") or prediction.get("content", "") or str(prediction)
                    else:
                        raw_content = str(prediction)

                    # Strip container prompt echo prefixes
                    if "<|im_start|>assistant" in raw_content:
                        raw_content = raw_content.split("<|im_start|>assistant")[-1]
                    elif "Output:" in raw_content:
                        raw_content = raw_content.split("Output:")[-1]

                    raw_output = raw_content.replace("<|im_end|>", "").strip()
                else:
                    raw_output = ""
                
            # Extract SQL statements nested inside Markdown tags
            sql_match = re.search(r"```[sS][qQ][lL]\s*(.*?)\s*```", raw_output, re.DOTALL)
            
            if sql_match:
                final_sql = sql_match.group(1).strip()
            else:
                # Fallback to extraction of SELECT query if block tags are missing
                fallback_match = re.search(r"(?i)\b(SELECT\b.*?;?)", raw_output, re.DOTALL)
                if fallback_match:
                    final_sql = fallback_match.group(1).strip()
                else:
                    print(f"Extraction failed.")
                    final_sql = "EXTRACTION_FAILED"

            # BIRD safety check: Strip trailing semicolons as they can occasionally crash SQLite runners
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

def add_ai_sql_to_json(file_path: str, model_name: str, endpoint_id: str, project: str = None, location: str = None):
    """
    Unified execution pipeline that loads database schemas, queries fine-tuned models,
    safeguards checkpoint progress, and supports graceful termination.

    This ensures test generation prompt formats are aligned with the SFT models.

    Args:
        file_path (str): Path to input JSON containing questions and metadata.
        model_name (str): Model identifier string (e.g. includes 'gemini' or is gemma/qwen).
        endpoint_id (str): Vertex AI Custom Endpoint ID or model resource path.
        project (str, optional): GCP Project ID.
        location (str, optional): GCP Region location.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return
        
    model_type = "gemini" if "gemini" in model_name.lower() else "gemma"
    
    # Initialize connection clients based on the routed model architecture
    if model_type == "gemini":
        from google import genai
        endpoint_obj = genai.Client(
            vertexai=True,
            project=project,
            location=location
        )
    else:
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        aiplatform.init(project=project, location=location, api_endpoint=api_endpoint)
        endpoint_obj = aiplatform.Endpoint(endpoint_id)

    out_filename = file_path.replace('.json', f'_ai_{model_name}.json')
    checkpoint_filename = file_path.replace('.json', f'_ai_{model_name}_checkpoint.json')
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_db_path = os.path.abspath(os.path.join(current_dir, "../database"))
        
        # Load checkpoints to support resuming previous execution state
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

        # Process each item in the dataset sequentially
        for idx, item in enumerate(tqdm(data, desc="Generating SQL queries")):
            question = item.get('question')
            
            # Skip if already processed in this checkpointed run
            if question in completed_queries:
                item["ai_generated_sql"] = completed_queries[question]
                processed_data.append(item)
                continue

            db_id = item.get('db_id')
            schema_dict = get_schema_details(db_id, base_db_path)
            
            model_type = "gemini" if "gemini" in model_name.lower() else "gemma"
            
            ai_sql = get_ai_sql_qwen_or_gemma(
                endpoint_obj, 
                schema_dict, 
                question, 
                item.get('evidence', ''),
                model=model_type,
                endpoint_id=endpoint_id,
                project=project,
                location=location
            )

            item["ai_generated_sql"] = ai_sql
            processed_data.append(item)

            # Save progress every 5 records to guard against execution errors or timeouts
            if (idx + 1) % 5 == 0 or (idx + 1) == len(data):
                with open(checkpoint_filename, 'w') as f:
                    json.dump(processed_data, f, indent=4)

        # Save final complete SFT output results file
        with open(out_filename, 'w') as f:
            json.dump(processed_data, f, indent=4)

        # Cleanup temporary checkpoint file on success
        if os.path.exists(checkpoint_filename):
            os.remove(checkpoint_filename)

        print(f"\nSuccessfully created '{out_filename}'")
        
    except KeyboardInterrupt:
        # Save progress gracefully on Ctrl+C interrupt
        print(f"\nExecution interrupted by user (Ctrl+C). Saving progress so far to checkpoint...")
        try:
            with open(checkpoint_filename, 'w') as f:
                json.dump(processed_data, f, indent=4)
            print(f"Progress successfully saved to: {checkpoint_filename}")
        except Exception as save_err:
            print(f"Failed to save checkpoint on interrupt: {save_err}")
        import sys
        sys.exit(130)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     json_file = os.path.abspath(os.path.join(current_dir, "../results/sft/spider_test_set.json"))
#     model_name = 'gemini-2.5-flash-sft-new-cot-53k'
#     add_ai_sql_to_json(json_file, model_name)