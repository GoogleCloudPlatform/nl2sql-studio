import random
import logging
import time
from google import genai
from google.genai import types
import json
import os
import sqlite3

class SmartSampler:
    def __init__(self, tables_json_path, base_db_path):
        """Loads the Spider dataset schemas."""
        with open(tables_json_path, 'r') as f:
            self.schemas = json.load(f)
        self.base_db_path = base_db_path

    def get_formatted_schema_with_samples(self, db_id):
        """
        Formats tables, columns, and 3 actual rows of data into a prompt-ready string.
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
                schema_text += f"Table: {table_name}\n"
                
                # 1. Get Columns
                columns_for_table = [col[1] for col in column_data if col[0] == table_idx]
                schema_text += f"Columns: {', '.join(columns_for_table)}\n"
                
                # 2. Get Sample Data (LIMIT 3)
                try:
                    # Wrapping table name in quotes in case of SQL keywords/spaces
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
                
                schema_text += "\n" # Add spacing between tables

            conn.close()
            
        except sqlite3.Error as e:
            return f"Error connecting to database {db_id}: {e}"

        return schema_text.strip()

# Set up simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PERSONAS = [
    {
        "name": "The Executive",
        "weight": 1.0,
        "description": "Focuses on high-level KPIs and trends (e.g., 'How is our Q3 revenue looking?')."
    },
    {
        "name": "The Engineering Manager",
        "weight": 1.0,
        "description": "Focuses on operational status (e.g., 'List tickets blocked by infra issues')."
    },
    {
        "name": "The Data Analyst",
        "weight": 1.0,
        "description": "Focuses on precise cuts (e.g., 'Select top 5 users by spend, grouped by region')."
    },
    {
        "name": "The Product Manager",
        "weight": 1.0,
        "description": "Focuses on user engagement and feature adoption (e.g., 'How many active users engaged with the new dashboard last week?')."
    },
    {
        "name": "The Customer Support Lead",
        "weight": 1.0,
        "description": "Focuses on user issues and resolution times (e.g., 'What is the average resolution time for severity 1 tickets this month?')."
    }
]

class ContextTranslatorAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.personas = PERSONAS

    def generate_question(self, validated_sql: str, result_summary: str, schema_tables: str):
        """
        Stage 2: Context-Aware Translation
        
        Generates a high-fidelity Natural Language question grounded in data context by performing Reverse Translation.
        The agent receives the Validated SQL, Result Summary, and schemas used, then randomly selects a target persona
        to condition the generated question.
        """
        # 1. Random Persona Selection using weighted choices
        population = [p for p in self.personas]
        weights = [p["weight"] for p in self.personas]
        
        selected_persona = random.choices(population, weights=weights, k=1)[0]
        logger.info(f"Selected Persona: {selected_persona['name']}")
        
        # 2. Context Injection and Reverse Translation Prompt Construction
        system_prompt = f"""
        You are an expert natural language generation agent.
        Your persona for this task is: {selected_persona['name']}.
        Persona Description: {selected_persona['description']}
        
        Your goal is Reverse Translation: Given the inputs below, generate the precise, pure Natural Language question that would have produced this exact SQL query and result.
        Condition your generation heavily on the ACTUAL result shape to avoid vague or hallucinated intents. (e.g., if the query limits to 5, the question must say 'Which 5...').
        
        [CONTEXT]
        - Schema / Tables Used:
        {schema_tables}
        
        - Validated SQL:
        {validated_sql}
        
        - Result Summary (Shape and sample output):
        {result_summary}
        
        Please generate ONLY the Natural Language question matching your persona's perspective. Do not include extra conversational filler.
        """

        # 3. LLM call using Gemini (google-genai SDK or equivalent)
        if self.llm_client:
            try:
                logger.info("Calling Gemini model to generate natural language question...")
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=system_prompt)
                        ]
                    ),
                ]
                
                response = self.llm_client.models.generate_content(
                    model='gemini-2.5-pro',
                    contents=contents,
                )
                return selected_persona, system_prompt, response.text.strip()
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return selected_persona, system_prompt, f"Error generating question: {e}"
        
        logger.info("Prompt constructed successfully (LLM client not provided).")
        return selected_persona, system_prompt, None


def run_stage_2_pipeline(stage_1_dataset, schemas_dict, llm_client=None, max_retries=3):
    """
    Orchestrates Stage 2: Generates Context-Aware Natural Language questions for each verified SQL query.
    
    stage_1_dataset: List of dicts returned by Stage 1 (db_id, complexity, sql, result_summary).
    schemas_dict: A dictionary mapping db_id to schema_text.
    llm_client: Initialized Gemini client (e.g. genai.Client instance).
    """
    print(f"🚀 Starting Stage 2 Pipeline\n" + "="*50)
    
    # 1. Initialize our Translator Agent
    translator_agent = ContextTranslatorAgent(llm_client)
    final_dataset = []
    

    # 2. Master Translation Loop
    for item in stage_1_dataset:
        db_id = item['db_id']
        complexity = item['complexity']
        sql = item['sql']
        summary = item['result_summary']
        
        # Look up schema for context
        schema_text = schemas_dict.get(db_id, "Schema not provided.")
        
        print(f"\n⚙️ Translating '{complexity}' Query for DB '{db_id}'...")
        
        success = False
        attempts = 0
        
        while not success and attempts < max_retries:
            attempts += 1
            print(f"  Attempt {attempts}/{max_retries}...")
            
            persona, system_prompt, nl_question = translator_agent.generate_question(sql, summary, schema_text)
            
            # Simple fault tolerance check: if there is an error string returned
            if not nl_question.startswith("Error"):
                print(f"  ✅ Translation Success! Persona: {persona['name']}")
                print(f"  📝 Generated Question: {nl_question}")
                
                # Assemble the finalized Stage 2 payload
                stage_2_payload = {
                    "db_id": db_id,
                    "complexity": complexity,
                    "sql": sql,
                    "result_summary": summary,
                    "persona": persona['name'],
                    "system_prompt": system_prompt,
                    "nl_question": nl_question
                }
                
                final_dataset.append(stage_2_payload)
                success = True
            else:
                print(f"  ⚠️ {nl_question}")
                time.sleep(1) # Backoff
                
        if not success:
            print(f"❌ Failed to translate query after {max_retries} attempts.")
            
    print("\n" + "="*50)
    print(f"🎉 Stage 2 Complete. Generated {len(final_dataset)} final NL2SQL pairs.")
    return final_dataset


# ==========================================
# EXECUTION CELL (Analogous to Stage 1)
# ==========================================
if __name__ == "__main__":
    import json
    
    # Read data arriving from Stage 1
    with open('synthetic_data_batch.json', 'r') as f:
        stage_1_data = json.load(f)

    # Dynamically extract schemas using the sampler
    TABLES_JSON_PATH = "./tables.json"
    DATABASE_DIR_PATH = "./database/"
    sampler = SmartSampler(TABLES_JSON_PATH, DATABASE_DIR_PATH)
    schemas = {}
    for item in stage_1_data:
        db_id = item['db_id']
        if db_id not in schemas:
            schemas[db_id] = sampler.get_formatted_schema_with_samples(db_id)

    llm_client = genai.Client(
        vertexai=True,
        project="sl-test-project-353312",
        location="us-central1"
    )
    
    # Execute Stage 2
    final_golden_data = run_stage_2_pipeline(
        stage_1_dataset=stage_1_data, 
        schemas_dict=schemas,
        llm_client=llm_client
    )
    
    # Save the output to a JSON file so Stage 3 can pick it up
    with open('stage2_results.json', 'w') as f:
        json.dump(final_golden_data, f, indent=4)
    print("Results saved to stage2_results.json")
    
    # Preview the complete, verified payload
    for payload in final_golden_data:
        print("\n--- Final Verified Payload [Stage 2] ---")
        for key, val in payload.items():
            print(f"{key.upper()}: {val}")
