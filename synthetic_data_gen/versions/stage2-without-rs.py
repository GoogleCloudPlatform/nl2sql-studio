import random
import logging
import time
import json
import os
import sqlite3
import concurrent.futures
from collections import defaultdict
from google import genai
from google.genai import types
import sys

# Set up simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PERSONAS = [
    {
        "name": "The Executive",
        "description": "Focuses on high-level KPIs and trends (e.g., 'How is our Q3 revenue looking?')."
    },
    {
        "name": "The Engineering Manager",
        "description": "Focuses on operational status (e.g., 'List tickets blocked by infra issues')."
    },
    {
        "name": "The Data Analyst",
        "description": "Focuses on precise cuts (e.g., 'Select top 5 users by spend, grouped by region')."
    },
    {
        "name": "The Product Manager",
        "description": "Focuses on user engagement and feature adoption (e.g., 'How many active users engaged with the new dashboard last week?')."
    },
    {
        "name": "The Customer Support Lead",
        "description": "Focuses on user issues and resolution times (e.g., 'What is the average resolution time for severity 1 tickets this month?')."
    }
]

class SmartSampler:
    def __init__(self, tables_json_path, base_db_path):
        """Loads the Spider dataset schemas."""
        with open(tables_json_path, 'r') as f:
            self.schemas = json.load(f)
        self.base_db_path = base_db_path

    def get_formatted_schema_with_samples(self, db_id):
        """Formats tables, columns, and 3 actual rows of data into a prompt-ready string."""
        target_db = next((db for db in self.schemas if db['db_id'] == db_id), None)
        if not target_db:
            return "Database not found."

        db_path = os.path.join(self.base_db_path, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return f"Error: Database file not found at {db_path}"

        schema_parts = [f"Database: {db_id}\n"]
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                for table_idx, table_name in enumerate(target_db['table_names_original']):
                    schema_parts.append(f"Table: {table_name}")
                    
                    columns = [col[1] for col in target_db['column_names_original'] if col[0] == table_idx]
                    schema_parts.append(f"Columns: {', '.join(columns)}")
                    
                    schema_parts.append("Sample Rows (Limit 3):")
                    try:
                        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
                        rows = cursor.fetchall()
                        if rows:
                            schema_parts.extend([f"- {row}" for row in rows])
                        else:
                            schema_parts.append("- (Table is empty)")
                    except sqlite3.Error as e:
                        schema_parts.append(f"- (Could not fetch samples: {e})")
                    schema_parts.append("") # Spacing
        except sqlite3.Error as e:
            return f"Error connecting to database {db_id}: {e}"

        return "\n".join(schema_parts).strip()


class ContextTranslatorAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.personas = PERSONAS

    def generate_questions(self, sql_items: list, schemas_dict: dict, persona: dict) -> str:
        """Stage 2: Context-Aware Translation (Batched by Persona)"""
        db_ids = {item['db_id'] for item in sql_items}
        schemas_context = "\n".join(
            f"--- Schema for Database: {db_id} ---\n{schemas_dict.get(db_id, 'Schema not provided.')}\n" 
            for db_id in db_ids
        )

        queries_context = "\n".join(
            f"--- Query {idx} ---\nDatabase ID: {item.get('db_id')}\nSQL: {item.get('sql', '')}\n"
            for idx, item in enumerate(sql_items)
        )

        system_prompt = f"""
        You are an expert natural language generation agent.
        Your persona for this task is: {persona['name']}.
        Persona Description: {persona['description']}
        
        Your goal is Reverse Translation: Given the inputs below, generate the precise, pure Natural Language question that would have produced each exact SQL query and result.
        Condition your generation heavily on the ACTUAL result shape to avoid vague or hallucinated intents. (e.g., if the query limits to 5, the question must say 'Which 5...').
        Importantly, craft each question to match your exact persona and refer to the specific Database Schema corresponding to each query.
        
        [CONTEXT]
        - Schemas Used:
        {schemas_context}
        
        - Queries to Translate:
        {queries_context}
        
        CRITICAL: Please generate ONLY a valid JSON array of strings, where the string at index `i` is the Natural Language question matching your persona's perspective for Query `i`.
        The length of the JSON array MUST be exactly {len(sql_items)}. 
        Do not include extra conversational filler or markdown formatting outside of the JSON array.
        """

        if not self.llm_client:
            logger.info("Prompt constructed successfully (LLM client not provided).")
            return None

        try:
            logger.info("Calling Gemini model to generate natural language questions...")
            response = self.llm_client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating questions: {e}"


def clean_json_response(response_text: str) -> str:
    """Cleans up potential markdown from the LLM JSON response."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def process_batch(batch_items, schemas_dict, persona, translator_agent, batch_num, total_batches, max_retries):
    persona_name = persona['name']
    print(f"\n  📦 [{persona_name}] Processing batch {batch_num}/{total_batches} ({len(batch_items)} items)...")
    
    for attempt in range(1, max_retries + 1):
        print(f"    [{persona_name} - Batch {batch_num}] Attempt {attempt}/{max_retries}...")
        response_text = translator_agent.generate_questions(batch_items, schemas_dict, persona)
        
        if not response_text or response_text.startswith("Error"):
            print(f"    ⚠️ [{persona_name} - Batch {batch_num}] {response_text}")
            time.sleep(1)
            continue

        try:
            nl_questions = json.loads(clean_json_response(response_text))
            
            if isinstance(nl_questions, list) and len(nl_questions) == len(batch_items):
                print(f"    ✅ [{persona_name} - Batch {batch_num}] Translation Success!")
                results = []
                for idx, item in enumerate(batch_items):
                    print(f"    📝 [{persona_name}] Generated Question (DB: {item['db_id']}): {nl_questions[idx]}")
                    results.append({
                        "db_id": item['db_id'],
                        "complexity": item['complexity'],
                        "sql": item['sql'],
                        "schema": schemas_dict[item['db_id']],
                        "persona": persona_name,
                        "nl_question": nl_questions[idx]
                    })
                return results
            
            print(f"    ⚠️ [{persona_name} - Batch {batch_num}] Expected {len(batch_items)} questions, got {len(nl_questions) if isinstance(nl_questions, list) else type(nl_questions)}.")
        except Exception as e:
            print(f"    ⚠️ [{persona_name} - Batch {batch_num}] Error parsing response: {e}\nResponse was:\n{response_text[:100]}...")
        
        time.sleep(1)
        
    print(f"  ❌ [{persona_name}] Failed to translate batch {batch_num} after {max_retries} attempts.")
    return []


def run_stage_2_pipeline(stage_1_dataset, schemas_dict, llm_client=None, max_retries=3, max_workers=5):
    print(f"🚀 Starting Stage 2 Pipeline (Multi-Threaded)\n" + "="*50)
    translator_agent = ContextTranslatorAgent(llm_client)
    final_dataset = []

    # Assign random personas and group by persona_name
    persona_groups = defaultdict(list)
    for item in stage_1_dataset:
        item['assigned_persona'] = random.choice(translator_agent.personas)
        persona_groups[item['assigned_persona']['name']].append(item)

    batch_size = 30
    jobs = []
    
    for persona_name, items in persona_groups.items():
        persona = items[0]['assigned_persona']
        print(f"\n⚙️ Queuing {len(items)} Queries for Persona '{persona_name}'...")
        
        total_batches = (len(items) + batch_size - 1) // batch_size
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            jobs.append((batch_items, schemas_dict, persona, translator_agent, batch_num, total_batches, max_retries))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                if batch_results:
                    final_dataset.extend(batch_results)
            except Exception as e:
                print(f"⚠️ Exception generated during batch processing: {e}")
            
    print("\n" + "="*50)
    print(f"🎉 Stage 2 Complete. Generated {len(final_dataset)} final NL2SQL pairs.")
    return final_dataset


# ==========================================
# EXECUTION CELL
# ==========================================
if __name__ == "__main__":
    TABLES_FILE = "./tables-new.json"
    DATABASE_PATH = "./database/"
    INPUT_FILE_PATH = "./results/merged_synthetic_dataset.json"
    OUTPUT_FILE_PATH = "./results/merged_stage2_results_without_rs.json"
    
    try:
        with open(INPUT_FILE_PATH, 'r') as f:
            stage_1_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE_PATH}")
        sys.exit(1)


    sampler = SmartSampler(TABLES_FILE, DATABASE_PATH)
    
    # Extract unique DB IDs to prevent duplicate schema formatting
    db_ids = {item['db_id'] for item in stage_1_data}
    schemas = {db_id: sampler.get_formatted_schema_with_samples(db_id) for db_id in db_ids}

    try:
        llm_client = genai.Client(
            vertexai=True,
            project="sl-test-project-353312",
            location="us-central1"
        )
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        llm_client = None
    
    final_golden_data = run_stage_2_pipeline(stage_1_data, schemas, llm_client)
    
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(final_golden_data, f, indent=4)
        
    print(f"Results saved to {OUTPUT_FILE_PATH}")
