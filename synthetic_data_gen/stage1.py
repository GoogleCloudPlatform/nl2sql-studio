from google import genai
from google.genai import types
import base64
import os
import sqlite3
import json
import time
from pydantic import BaseModel, Field
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from metrics.sql_generation_metrics import analyze_stage1_pipeline

class SmartSampler:
    """
    Utility class to load database schemas and sample real rows for prompt engineering.
    
    Attributes:
        schemas (list): List of database schema dictionaries.
        base_db_path (str): Root directory where SQLite files are stored.
    """
    def __init__(self, tables_json_path, base_db_path):
        """
        Initializes the sampler by loading the master schema JSON.
        
        Args:
            tables_json_path (str): Path to the tables-all.json file.
            base_db_path (str): Path to the directory containing .sqlite files.
        """
        try:
            with open(tables_json_path, 'r') as f:
                self.schemas = json.load(f)
            self.base_db_path = base_db_path
        except Exception as e:
            print(f"❌ Error loading schema file: {e}")
            raise

    def get_formatted_schema_with_samples(self, db_id):
        """
        Extracts table structures and sample rows from a database and formats them for an LLM prompt.

        Args:
            db_id (str): The unique identifier of the database to sample.

        Returns:
            str: A formatted string containing the schema and sample data, or an error message.
        """
        target_db = next((db for db in self.schemas if db['db_id'] == db_id), None)
        if not target_db:
            return "Database not found."

        table_names = target_db['table_names_original']
        column_data = target_db['column_names_original'] 
        
        # Connect to the actual SQLite database for this db_id
        db_path = os.path.join(self.base_db_path, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            return f"Error: Database file not found at {db_path}"

        schema_text = f"Database: {db_id}\n\n"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            for table_idx, table_name in enumerate(table_names):
                # print(f"   - Sampling table: {table_name}")
                schema_text += f"Table: {table_name}\n"
                
                # 1. Get Columns
                columns_for_table = [col[1] for col in column_data if col[0] == table_idx]
                schema_text += f"Columns: {', '.join(columns_for_table)}\n"
                
                # 2. Get Sample Data (LIMIT 3)
                try:
                    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
                    sample_rows = cursor.fetchall()
                    
                    schema_text += "Sample Rows (Limit 3):\n"
                    if sample_rows:
                        for row in sample_rows:
                            schema_text += f"- {row}\n"
                    else:
                        schema_text += "- (Table is empty)\n"
                except sqlite3.Error as e:
                    schema_text += f"- (Could not fetch samples: {e})\n"
                
                schema_text += "\n" 

            conn.close()
            
        except sqlite3.Error as e:
            return f"Error connecting to database {db_id}: {e}"

        return schema_text.strip()


class DatabaseExecutor:
    """
    Handles the execution of SQL queries against local SQLite databases for validation.
    
    Attributes:
        base_db_path (str): Root directory where SQLite files are stored.
    """
    def __init__(self, base_db_path):
        """
        Args:
            base_db_path (str): Path to the 'database' folder containing .sqlite files.
        """
        self.base_db_path = base_db_path

    def execute_query(self, db_id, sql_query):
        """
        Runs a SQL query and returns the results or a descriptive discard message.

        Args:
            db_id (str): The database identifier.
            sql_query (str): The SQL string to execute.

        Returns:
            tuple: (bool success, str message, list results, list column_names)
        """
        db_path = os.path.join(self.base_db_path, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            return False, f"Discarded: Database file not found at {db_path}", None, None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            conn.close()

            if not results:
                return False, "Discarded: Executed successfully but returned 0 rows.", None, None
                
            return True, "Success", results, column_names

        except sqlite3.Error as e:
            return False, f"Discarded: SQL Runtime Error - {e}", None, None


class ResultSummarizer:
    """
    Agent responsible for describing the 'shape' of data returned by a query using an LLM.
    """
    def __init__(self, client, model_name, prompt_template):
        """
        Args:
            client: The initialized Google GenAI client.
            model_name (str): The ID of the Gemini model to use.
            prompt_template (str): The template string for the summarization prompt.
        """
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template

    def summarize_shape(self, sql_query, column_names, sample_results):
        """
        Calls the LLM to generate a natural language summary of the query output.

        Args:
            sql_query (str): The query that was run.
            column_names (list): The headers of the returned data.
            sample_results (list): The first few rows of the actual data.

        Returns:
            str: A one-sentence description of the data shape.
        """
        preview_data = sample_results[:3]
        total_rows = len(sample_results)
        
        prompt = self.prompt_template.format(
            sql_query=sql_query,
            column_names=column_names,
            total_rows=total_rows,
            preview_data=preview_data
        )
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ]
            ),
        ]
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = contents
        )
        return response.text.strip()


class BatchArchitectAgent:
    """
    Agent responsible for generating a batch of diverse SQL queries based on a database schema.
    """
    def __init__(self, client, model_name, prompt_template):
        """
        Args:
            client: The initialized Google GenAI client.
            model_name (str): The ID of the Gemini model to use.
            prompt_template (str): The template string for the generation prompt.
        """
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
    
    def generate_batch_sql(self, schema_with_samples_text):
        """
        Requests the LLM to generate multiple SQL queries in one batch.

        Args:
            schema_with_samples_text (str): The formatted schema and sample rows.

        Returns:
            list: A list of dictionaries, each containing 'complexity' and 'sql'.
        """
        
        prompt = self.prompt_template.format(
            schema_with_samples_text=schema_with_samples_text
        )
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ]
            ),
        ]
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = contents
        )
        
        # Clean the response to ensure it's parseable JSON
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse JSON from LLM: {e}")
            return []


def parse_db_selection(selection_str, schemas):
    """
    Parses a user-provided selection string to determine which databases to process.

    Supported Formats:
    - Exact Match: 'concert_singer' (returns that specific ID)
    - Slice: '0:10' (first 10), '5:' (from index 5 to end)
    - End Index: '10' (assumes 0:10)

    Args:
        selection_str (str): The raw selection string from configuration.
        schemas (list): The list of all available database schemas.

    Returns:
        list: A list of db_id strings to be processed.
    """
    all_ids = [db['db_id'] for db in schemas]
    selection_str = selection_str.strip()
    
    # 1. Priority: Check if it's an exact DB ID match
    if selection_str in all_ids:
        return [selection_str]
    
    # 2. Check if it's a slice (contains :)
    if ":" in selection_str:
        parts = selection_str.split(":")
        try:
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else len(all_ids)
            return all_ids[start:end]
        except ValueError:
            pass # Not a numeric slice, move on
    
    # 3. Check if it's just an end index (assume start 0)
    if selection_str.isdigit():
        end = int(selection_str)
        return all_ids[0:end]
        
    return []


def process_single_database(db_id, tables_path, db_dir, client, model_name, prompts, batch_prompt_name, summarize_prompt_name):
    """
    The end-to-end workflow for a single database: Sample -> Generate -> Validate -> Summarize.

    Args:
        db_id (str): Database to process.
        tables_path (str): Path to master schema file.
        db_dir (str): Root directory of SQLite files.
        client: Google GenAI client.
        model_name (str): LLM model ID.
        prompts (dict): Dictionary of loaded prompt templates.
        batch_prompt_name (str): Key for the generation prompt.
        summarize_prompt_name (str): Key for the summarization prompt.

    Returns:
        list: A list of all attempted query objects (success and fail).
    """
    sampler = SmartSampler(tables_path, db_dir)
    architect = BatchArchitectAgent(client, model_name, prompts[batch_prompt_name])
    executor = DatabaseExecutor(db_dir)
    summarizer = ResultSummarizer(client, model_name, prompts[summarize_prompt_name])
    
    results_batch = []
    
    schema_text = sampler.get_formatted_schema_with_samples(db_id)
    if "Error" in schema_text or "Database not found" in schema_text:
        return results_batch

    # 1. Sample and Generate
    print(f"🔍 [{db_id}] Sampling schema and calling Architect...")
    generated_queries = architect.generate_batch_sql(schema_text)
    
    # 2. Validate and Summarize
    print(f"🧪 [{db_id}] Validating {len(generated_queries)} generated queries...")
    for item in generated_queries:
        sql = item.get("sql")
        complexity = item.get("complexity")
        
        if not sql:
            continue
            
        # Execute against the local SQLite DB
        is_valid, msg, results, columns = executor.execute_query(db_id, sql)
        
        # Initialize the query entry (tracking success/failure)
        query_entry = {
            "db_id": db_id,
            "complexity": complexity,
            "sql": sql,
            "success": is_valid
        }
        
        if is_valid:
            # Generate the natural language summary of the output shape
            summary = summarizer.summarize_shape(sql, columns, results)
            query_entry["result_summary"] = summary
        else:
            query_entry["error_message"] = msg
            
        results_batch.append(query_entry)
            
    return results_batch


def run_multithreaded_pipeline(tables_path, db_dir, db_ids, client, model_name, prompts, batch_prompt_name, summarize_prompt_name, max_workers=5):
    """
    Orchestrates the concurrent processing of multiple databases.

    Args:
        tables_path (str): Path to master schema file.
        db_dir (str): Root directory of SQLite files.
        db_ids (list): List of databases to process.
        client: Google GenAI client.
        model_name (str): LLM model ID.
        prompts (dict): Dictionary of all loaded prompt templates.
        batch_prompt_name (str): Key for the generation prompt.
        summarize_prompt_name (str): Key for the summarization prompt.
        max_workers (int): Maximum number of concurrent database threads.

    Returns:
        list: The complete dataset of verified queries.
    """
    print(f"🚀 Starting Multithreaded Run on {len(db_ids)} DBs with {max_workers} workers...")
    start_time = time.time()
    
    master_dataset = []
    
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        future_to_db = {
            executor.submit(process_single_database, db_id, tables_path, db_dir, client, model_name, prompts, batch_prompt_name, summarize_prompt_name): db_id 
            for db_id in db_ids
        }
        
        # As each thread completes, gather the results
        for future in as_completed(future_to_db):
            db_id = future_to_db[future]
            try:
                # Get the verified list of queries from that specific thread
                verified_queries = future.result() 
                master_dataset.extend(verified_queries)
                success_count = sum(1 for q in verified_queries if q.get("success"))
                print(f"✅ DB '{db_id}' completed. Total: {len(verified_queries)}, Success: {success_count}")
            except Exception as exc:
                print(f"❌ DB '{db_id}' generated an exception: {exc}")

    end_time = time.time()
    print("\n" + "="*50)
    print(f"🎉 Run Complete in {round(end_time - start_time, 2)} seconds!")
    
    return master_dataset




'''
TODO: Suggested Optimizations for "Big Schema" Scenarios:

A. Table Subsetting (A "Two-Step" Agent)
Instead of sending all 1,000 tables, implement a "Table Selector" step. An agent would first look at the 1,000 table names/descriptions and select groups of 5–10 related tables (e.g., "Accounting module," "User Management module") to send to the generator.

B. Sub-Batch Concurrency
Modify the logic to parallelize within the database. Instead of one thread for the DB, you could spawn multiple threads that each target different "clusters" of tables within that same database.

C. Incremental Schema Loading
Instead of a fixed tables-all.json, use a RAG-based approach (Retrieval-Augmented Generation). The system would search for relevant table schemas based on a high-level goal, rather than dumping the entire data dictionary into the prompt.

D. Connection Pooling
For 1,000 tables, you would need to ensure the DatabaseExecutor handles SQLite connections efficiently, perhaps by keeping a single connection open per thread rather than opening/closing for every execute_query call.

'''

# ==========================================
# SAMPLE EXECUTION
# ==========================================
if __name__ == "__main__":  
    # --- CONFIGURATION AREA ---
    # GCP Project and Location details for Vertex AI
    PROJECT_ID = "mystic-bank-352905"
    LOCATION = "us-central1"
    MODEL_ID = "gemini-2.5-pro"
    
    # Paths to the input data and prompt templates
    TABLES_JSON_PATH = "synthetic_data_gen/tables-all.json"
    DATABASE_DIR_PATH = "synthetic_data_gen/database/"
    PROMPTS_DIR = "synthetic_data_gen/prompts/"
    
    # Specify which prompt files to use (using filenames without .txt)
    BATCH_PROMPT_NAME = "batch_sql_generation_few_shot" 
    SUMMARIZE_PROMPT_NAME = "summarize_sql"
    
    # Selection logic for which databases to process
    # Examples: '0:5' (range), 'perpetrator' (single ID), '10' (first 10)
    DB_SELECTION = "1:2"
    
    # Setup for timestamped output to prevent overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_selection = DB_SELECTION.replace(":", "_").replace(",", "_").replace(" ", "")
    OUTPUT_FILENAME = f"synthetic_data_gen/results/stage1/synthetic_data_{clean_selection}_{timestamp}.json"
    
    # --- INITIALIZATION ---
    print(f"🎬 Initializing Synthetic Data Pipeline [Stage 1]...")
    
    # 1. Load all external prompts from the prompts directory
    print(f"📂 Loading prompt templates from {PROMPTS_DIR}...")
    prompts = {}
    if os.path.exists(PROMPTS_DIR):
        for filename in os.listdir(PROMPTS_DIR):
            if filename.endswith(".txt"):
                prompt_name = filename[:-4]
                with open(os.path.join(PROMPTS_DIR, filename), 'r') as f:
                    prompts[prompt_name] = f.read()
    
    # 2. Initialize the shared Gemini client
    print(f"☁️  Initializing Vertex AI Client (Project: {PROJECT_ID})...")
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )

    # 3. Initialize the sampler to get the list of available databases
    temp_sampler = SmartSampler(TABLES_JSON_PATH, DATABASE_DIR_PATH)
    
    # 4. Filter databases based on user selection
    target_dbs = parse_db_selection(DB_SELECTION, temp_sampler.schemas)
    if not target_dbs:
        print(f"❌ Error: No databases matched selection '{DB_SELECTION}'")
        exit(1)
        
    print(f"🎯 Selected {len(target_dbs)} database(s): {', '.join(target_dbs)}")
    
    # --- EXECUTION ---
    # Run the multithreaded generation pipeline
    final_dataset = run_multithreaded_pipeline(
        tables_path=TABLES_JSON_PATH, 
        db_dir=DATABASE_DIR_PATH,
        db_ids=target_dbs,
        client=client,
        model_name=MODEL_ID,
        prompts=prompts,
        batch_prompt_name=BATCH_PROMPT_NAME,
        summarize_prompt_name=SUMMARIZE_PROMPT_NAME,
        max_workers=8 
    )

    # --- FINALIZATION ---
    # Ensure the results directory exists
    print(f"💾 Saving results to {OUTPUT_FILENAME}...")
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    
    # Write the verified dataset to JSON
    with open(OUTPUT_FILENAME, "w") as f:
        json.dump(final_dataset, f, indent=4)
        
    print("\n" + "="*50)
    print(f"🎉 Stage 1 Complete! Saved {len(final_dataset)} verified queries.")
    print(f"📍 Output: {OUTPUT_FILENAME}")
    print("="*50 + "\n")
    analyze_stage1_pipeline(OUTPUT_FILENAME, TABLES_JSON_PATH)