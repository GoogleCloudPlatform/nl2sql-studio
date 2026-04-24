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
        """
        Initializes the SmartSampler with schema and database paths.

        Args:
            tables_json_path (str): Path to the JSON file containing Spider dataset schemas.
            base_db_path (str): Directory containing the actual SQLite databases.
        """
        with open(tables_json_path, 'r') as f:
            self.schemas = json.load(f)
        self.base_db_path = base_db_path

    def get_formatted_schema_with_samples(self, db_id):
        """
        Fetches schema details and sample data rows for a specific database.

        Args:
            db_id (str): The identifier for the database.

        Returns:
            str: A formatted block of text containing tables, columns, and sample rows,
                 suitable for inclusion in an LLM prompt.
        """
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

def build_system_prompt(sql_items: list, schemas_dict: dict, persona: dict, include_result_summary: bool = True, include_schema: bool = True) -> str:
    """
    Builds the unified system prompt for both Stage 2 (batch generation) and Stage 3 (single-item evaluation recreation).

    Args:
        sql_items (list): Items to translate.
        schemas_dict (dict): Database schemas lookup.
        persona (dict): Target persona.
        include_result_summary (bool): Include result summary in context.
        include_schema (bool): Include schema in context.

    Returns:
        str: The generated system prompt.
    """
    schema_section = ""
    if include_schema:
        db_ids = {item['db_id'] for item in sql_items}
        schemas_text = "\n".join(
            f"--- Schema for Database: {db_id} ---\n{schemas_dict.get(db_id, 'Schema not provided.')}\n" 
            for db_id in db_ids
        )
        schema_section = f"- Schemas Used:\n{schemas_text}\n"

    queries_context = "\n".join(
        f"--- Query {idx} ---\n"
        f"Database ID: {item.get('db_id')}\n"
        f"SQL: {item.get('sql', '')}\n"
        + (f"Result Summary: {item.get('result_summary', '')}\n" if include_result_summary else "")
        for idx, item in enumerate(sql_items)
    )

    dynamic_instructions = []
    if include_schema:
        dynamic_instructions.append("You MUST refer to the specific Database Schema corresponding to each query.")
    if include_result_summary:
        dynamic_instructions.append("You MUST refer to the expected Result Summary corresponding to each query.")
    dynamic_instruction = " ".join(dynamic_instructions)

    system_prompt = f"""
    [ROLE]
    You are an expert natural language generation agent specializing in Reverse Translation (SQL to Natural Language queries).
    Your persona for this task is: {persona['name']}.
    Persona Description: {persona['description']}

    [GOAL]
    Translate the provided SQL queries into the exact Natural Language questions that would have generated them, matching your assigned persona's perspective.

    {schema_section}
    - Queries to Translate:
    {queries_context}

    [GUIDELINES & CONSTRAINTS]
    1. Exact Match: Generate questions that correspond strictly to the SQL logic and output shape (e.g., if the query limits to 5, the question must reflect 'Which 5...').
    2. Persona Alignment: Use the tone, terminology, and focus described in your persona.
    3. No Hallucinations: Do not assume or invent data constraints not present in the SQL.
    4. No Open-Endedness: Avoid vague inquiries if the result set is small or heavily constrained.
    5. {dynamic_instruction}

    [OUTPUT FORMAT]
    You MUST return ONLY a valid JSON array of strings. 
    - The array length MUST be exactly {len(sql_items)}.
    - Do not include extra conversational filler or markdown formatting outside of the JSON array.
    """
    return system_prompt


class ContextTranslatorAgent:
    def __init__(self, llm_client=None):
        """
        Initializes the ContextTranslatorAgent.

        Args:
            llm_client (genai.Client, optional): The Google GenAI client instance.
        """
        self.llm_client = llm_client
        self.personas = PERSONAS

    def generate_questions(self, sql_items: list, schemas_dict: dict, persona: dict, include_result_summary: bool = True, include_schema: bool = True) -> str:
        """
        Generates natural language questions for a batch of SQL queries using the Gemini API.

        Args:
            sql_items (list): List of dictionaries containing SQL queries and metadata.
            schemas_dict (dict): Map of db_id to its formatted schema string.
            persona (dict): The target persona configuration.
            include_result_summary (bool): Whether to include the expected result summary.
            include_schema (bool): Whether to include schema definitions.

        Returns:
            str: JSON string containing the generated questions.
        """
        system_prompt = build_system_prompt(sql_items, schemas_dict, persona, include_result_summary, include_schema)

        if not self.llm_client:
            logger.info("Prompt constructed successfully (LLM client not provided).")
            return None

        max_retries = 10
        for attempt in range(max_retries):
            try:
                logger.info("Calling Gemini model to generate natural language questions...")
                response = self.llm_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)])],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                if attempt == max_retries - 1:
                    return f"Error generating questions: {e}"
                
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    sleep_time = (2 ** attempt) * 5 + random.uniform(1, 5)
                else:
                    sleep_time = 2 ** attempt
                
                logger.info(f"Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)


def clean_json_response(response_text: str) -> str:
    """
    Cleans up potential markdown code block wrappers (e.g., ```json) from the LLM response.

    Args:
        response_text (str): Raw response text.

    Returns:
        str: Stripped JSON string.
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def process_batch(batch_items, schemas_dict, persona, translator_agent, batch_num, total_batches, max_retries, include_result_summary, include_schema):
    """
    Processes a single batch of queries for translation, with retry logic.

    Args:
        batch_items (list): Items in this batch.
        schemas_dict (dict): Database schemas lookup.
        persona (dict): The target persona.
        translator_agent (ContextTranslatorAgent): The translation worker.
        batch_num (int): Current batch index.
        total_batches (int): Total batches for this persona.
        max_retries (int): Retry limit.

    Returns:
        list: Consolidated results for the batch.
    """
    persona_name = persona['name']
    print(f"\n  📦 [{persona_name}] Processing batch {batch_num}/{total_batches} ({len(batch_items)} items)...")
    
    for attempt in range(1, max_retries + 1):
        print(f"    [{persona_name} - Batch {batch_num}] Attempt {attempt}/{max_retries}...")
        response_text = translator_agent.generate_questions(batch_items, schemas_dict, persona, include_result_summary=include_result_summary, include_schema=include_schema)
        
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
                    # print(f"    📝 [{persona_name}] Generated Question (DB: {item['db_id']}): {nl_questions[idx]}")
                    results.append({
                        "db_id": item['db_id'],
                        "complexity": item['complexity'],
                        "sql": item['sql'],
                        "result_summary": item.get('result_summary', ''),
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


def run_stage_2_pipeline(stage_1_dataset, schemas_dict, llm_client=None, max_retries=3, max_workers=5, include_result_summary=True, include_schema=True):
    """
    Orchestrates the Stage 2 pipeline across all personas using multi-threading.

    Args:
        stage_1_dataset (list): Output from Stage 1.
        schemas_dict (dict): Pre-formatted schemas.
        llm_client (genai.Client): GenAI Client.
        max_retries (int): Retry limit for failed batches.
        max_workers (int): Thread pool size.

    Returns:
        list: The enriched dataset.
    """
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
            jobs.append((batch_items, schemas_dict, persona, translator_agent, batch_num, total_batches, max_retries, include_result_summary, include_schema))

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



