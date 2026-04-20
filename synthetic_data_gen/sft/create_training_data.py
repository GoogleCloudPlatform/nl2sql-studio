"""
Synthetic Data Generation - Training Data Creator
--------------------------------------------------
This script processes Stage 2 output (synthetic NL2SQL data) and prepares it
for fine-tuning various LLMs (Llama, Gemini, Qwen). It can optionally generate
Chain of Thought (CoT) reasoning steps for each query using a large model (Gemini 2.5 Pro)
to create a high-quality SFT dataset. Finally, it splits the dataset into
training and validation sets.
"""

import asyncio
import json
from functools import partial
from tqdm import tqdm
import textwrap
import os
import sys

# Add the path to the spider_eval directory to import nl2sql utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fine_tuning/spider_eval')))
from nl2sql import generate
from get_schema_details import get_schema_details

# Prompt template used to ask Gemini to generate the reasoning steps (Chain of Thought)
COT_GENERATION_PROMPT = """Given the following database schema, a natural language question and its corresponding ground truth SQL query, provide a detailed, step-by-step chain of thought that leads to the correct SQL query. Focus on the logical steps to translate the natural language into SQL, including identifying tables, columns, joins, filters, aggregations, and ordering. Do NOT provide the SQL query itself, only the reasoning process.

DATABASE SCHEMA:
```json
{schema_json_string}
```\n
Question: {question}\n
ground truth sql: {ground_truth_sql}\n
Chain of Thought:
"""

def create_llama_tuning_record(system_prompt: str, schema: dict, question: str, cot_reasoning: str, ground_truth_sql: str) -> dict:
    """
    Formats the inputs and ground truth SQL into the Llama chat JSONL structure.
    
    Args:
        system_prompt (str): The system instructions for the model.
        schema (dict): The database schema as a dictionary.
        question (str): The natural language question.
        cot_reasoning (str): The generated reasoning steps.
        ground_truth_sql (str): The correct SQL query.
        
    Returns:
        dict: A dictionary structured for Llama fine-tuning.
    """
    schema_json_string = json.dumps(schema, indent=2)
    user_content = f"{system_prompt}\n\nDATABASE SCHEMA:\njson\n{schema_json_string}\n\n\nQuestion: {question}"

    record = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}]
            },
            {
                "role": "model",
                "parts": [{"text": f"""{cot_reasoning}
```sql
{ground_truth_sql}
```"""}]
            }
        ]
    }
    return record


def create_gemini_tuning_record(system_prompt: str, schema: dict, question: str, cot_reasoning: str, ground_truth_sql: str) -> dict:
    """
    Formats the inputs and ground truth SQL into the Gemini JSONL structure.
    
    Args:
        system_prompt (str): The system instructions for the model.
        schema (dict): The database schema as a dictionary.
        question (str): The natural language question.
        cot_reasoning (str): The generated reasoning steps.
        ground_truth_sql (str): The correct SQL query.
        
    Returns:
        dict: A dictionary structured for Gemini fine-tuning.
    """
    schema_json_string = json.dumps(schema, indent=2)
    user_content = f"{system_prompt}\n\nDATABASE SCHEMA:\njson\n{schema_json_string}\n\n\nQuestion: {question}"

    assistant_content = f"""{cot_reasoning}
```sql
{ground_truth_sql}
```"""

    record = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}]
            },
            {
                "role": "model",
                "parts": [{"text": assistant_content}]
            }
        ]
    }
    return record


def create_qwen_or_gemma_tuning_record(system_prompt: str, schema: dict, question: str, cot_reasoning: str, ground_truth_sql: str) -> dict:
    """
    Formats the inputs and ground truth SQL into the Qwen JSONL structure.
    
    Args:
        system_prompt (str): The system instructions for the model.
        schema (dict): The database schema as a dictionary.
        question (str): The natural language question.
        cot_reasoning (str): The generated reasoning steps.
        ground_truth_sql (str): The correct SQL query.
        
    Returns:
        dict: A dictionary structured for Qwen fine-tuning.
    """
    schema_json_string = json.dumps(schema, indent=2)
    user_content = f"DATABASE SCHEMA:\njson\n{schema_json_string}\n\n\nQuestion: {question}"

    assistant_content = f"""{cot_reasoning}
```sql
{ground_truth_sql}
```"""

    record = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }
    return record


async def main(input_file_path: str, output_file_path: str, model_type: str, generate_cot: bool, batch_size: int):
    """
    Main function to generate fine-tuning data in JSONL format.
    Supports dynamic CoT generation and multi-model formatting.
    """
    # Define system prompts based on whether CoT is enabled
    system_prompt_cot = "You are a powerful text-to-SQL model. Your role is to answer user questions by generating SQL queries against a given database schema. First, provide a step-by-step chain of thought that explains your reasoning, and then provide the final SQL query in a markdown code block."
    system_prompt_no_cot = "You are a powerful text-to-SQL model. Your role is to answer user questions by generating SQL queries against a given database schema. Provide the final SQL query in a markdown code block."
    system_prompt = system_prompt_cot if generate_cot else system_prompt_no_cot

    # Check for existing output to support resuming
    start_index = 0
    try:
        with open(output_file_path, 'r') as f:
            start_index = sum(1 for _ in f)
        if start_index > 0:
            print(f"Found {start_index} existing records in {output_file_path}. Resuming generation.")
    except FileNotFoundError:
        print(f"Output file not found at {output_file_path}. Starting from scratch.")

    # Map model types to their respective record creator functions
    record_creators = {
        "llama": create_llama_tuning_record,
        "gemini": create_gemini_tuning_record,
        "qwen": create_qwen_or_gemma_tuning_record,
        "gemma": create_qwen_or_gemma_tuning_record
    }
    create_record_func = record_creators.get(model_type)
    if not create_record_func:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from {list(record_creators.keys())}")

    try:
        # Open input in read mode and output in append mode to support resuming
        with open(input_file_path, 'r') as infile, open(output_file_path, 'a') as outfile:
            spider_data = json.load(infile)

            if start_index >= len(spider_data):
                print("All items have already been processed.")
            
            loop = asyncio.get_running_loop()

            # Process in batches to control concurrency
            with tqdm(total=len(spider_data), initial=start_index, desc="Processing Spider data") as pbar:
                for i in range(start_index, len(spider_data), batch_size):
                    batch_items = spider_data[i:i+batch_size]
                    tasks = []
                    
                    for item in batch_items:
                        if generate_cot:
                            db_id = item['db_id']
                            schema = get_schema_details(db_id)
                            question = item['nl_question']
                            ground_truth_sql = item['sql']
                            schema_json_string = json.dumps(schema, indent=2)

                            cot_generation_prompt = COT_GENERATION_PROMPT.format(
                                schema_json_string=schema_json_string,
                                question=question,
                                ground_truth_sql=ground_truth_sql
                            )

                            # Run synchronous `generate` in a thread pool to avoid blocking the event loop
                            task = loop.run_in_executor(None, partial(generate, cot_generation_prompt, model="gemini-2.5-pro"))
                            tasks.append(task)
                        else:
                            # If CoT is not required, create a completed future with empty content
                            future = asyncio.Future()
                            future.set_result("")
                            tasks.append(future)

                    # Wait for all tasks in the batch to complete
                    generated_cots = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results and write to file
                    for item, cot_result in zip(batch_items, generated_cots):
                        if isinstance(cot_result, Exception):
                            tqdm.write(f"Error generating CoT for question '{item.get('nl_question', item.get('question'))}': {cot_result}. Skipping this item.")
                            pbar.update(1)
                            continue

                        item['cot_reasoning'] = cot_result
                        
                        db_id = item['db_id']
                        schema = get_schema_details(db_id)
                        
                        question = item.get('nl_question', item.get('question'))
                        query = item.get('sql', item.get('query'))
                        
                        # Create the model-specific record
                        tuning_record = create_record_func(
                            system_prompt, schema, question, item['cot_reasoning'], query
                        )
                        outfile.write(json.dumps(tuning_record) + '\n')
                        pbar.update(1)
        
        print(f"Successfully created {model_type} tuning data with generated CoT at: {output_file_path}")

        # Split the final dataset into train and val (80/20 split)
        print(f"Splitting {output_file_path} into training and validation sets...")
        with open(output_file_path, 'r') as f:
            records = f.readlines()
        
        if not records:
            print("No records found to split.")
            return
            
        import random
        # Use a fixed seed for reproducibility
        random.seed(42)
        random.shuffle(records)
        
        val_count = int(len(records) * 0.2)
        val_records = records[:val_count]
        train_records = records[val_count:]
        
        train_file_path = output_file_path.replace('.jsonl', '_train.jsonl')
        val_file_path = output_file_path.replace('.jsonl', '_val.jsonl')
        
        with open(train_file_path, 'w') as f:
            f.writelines(train_records)
            
        with open(val_file_path, 'w') as f:
            f.writelines(val_records)
            
        print(f"Created training set with {len(train_records)} records at: {train_file_path}")
        print(f"Created validation set with {len(val_records)} records at: {val_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}. Please ensure the file exists and is accessible.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Configuration for the generation run
    INPUT_FILE_PATH = '../results/stage2/s2_flash_synthetic_data_0_51_20260417_103303.json'
    
    MODEL_TYPE = "gemma" # Target model format: "llama", "gemini", or "qwen"
    GENERATE_COT = True # Set to True to generate reasoning steps
    CONCURRENT_BATCH_SIZE = 10 # Number of parallel calls to Gemini

    # Derive output file name based on model type and CoT setting
    OUTPUT_FILE_PATH = INPUT_FILE_PATH[:-5]+f"_{MODEL_TYPE}_{'COT' if GENERATE_COT else 'no_COT'}.jsonl"

    # Run the async main function
    asyncio.run(
        main(
            INPUT_FILE_PATH,
            OUTPUT_FILE_PATH,
            model_type=MODEL_TYPE,
            generate_cot=GENERATE_COT,
            batch_size=CONCURRENT_BATCH_SIZE
        )
    )
