from google import genai
from google.genai import types
import base64
import os
import sqlite3
import json
import time

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


class ArchitectAgent:
    def __init__(self):
        pass

    def generate_sql(self, schema_with_samples_text, complexity_level):
        """Prompts Gemini to generate a data-aware SQL query based on schema and sample rows."""
        
        prompt = f"""
        You are an expert SQL Architect building a synthetic dataset. 
        Your task is to generate a highly accurate, valid SQL query based on the provided database schema and sample data.

        CRITICAL DIRECTIVE: The target databases are extremely small. To ensure your query returns actual data (more than 0 rows), you MUST adhere strictly to the following rules:

        RULES:
        1. NO MARKDOWN: Only output the raw SQL code. No explanations, no formatting blocks.
        2. ALIASING: Use table aliasing (e.g., T1, T2) for clarity in all queries involving more than one table.
        3. DATA-AWARE FILTERING: If you use a WHERE or HAVING clause, you may ONLY filter using exact data values explicitly shown in the "Sample Rows" for that specific column. Do not invent, guess, or assume values exist.
        4. STRUCTURAL COMPLEXITY: To achieve "Medium" or "Complex" queries, do NOT use highly restrictive, multi-condition WHERE clauses. Instead, build complexity structurally using:
            - Multiple JOINs
            - GROUP BY with aggregations (COUNT, SUM, AVG)
            - Subqueries in the FROM or WHERE clause
            - Set operations (INTERSECT, UNION, EXCEPT)
        5. BROAD FILTERS: If you cannot find a good sample value to filter on, use broad structural filters like `IS NOT NULL` or numeric comparisons like `> 0`.

        FEW-SHOT EXAMPLES:
        - Simple: SELECT Name, Country, Age FROM singer ORDER BY Age DESC
        - Medium: SELECT T2.name, count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id GROUP BY T1.stadium_id
        - Complex: SELECT T1.name FROM student AS T1 JOIN student_course_attendance AS T2 ON T1.student_id = T2.student_id JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.course_name = 'English' INTERSECT SELECT T1.name FROM student AS T1 JOIN student_course_attendance AS T2 ON T1.student_id = T2.student_id JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.course_name = 'Math'

        ---
        TARGET SCHEMA & SAMPLE DATA:
        {schema_with_samples_text}

        REQUESTED COMPLEXITY: {complexity_level}
        
        SQL QUERY:
        """

        client = genai.Client(
            vertexai=True,
            project="mystic-bank-352905",
            location="us-central1")

        model = "gemini-2.5-pro"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ]
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            seed = 0,
            max_output_tokens = 65535,
            safety_settings = [types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )],
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
        )

        response = client.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
        )
        return response.text.strip()


class DatabaseExecutor:
    def __init__(self, base_db_path):
        """
        base_db_path: The path to the 'database' folder you uploaded to Workbench.
        """
        self.base_db_path = base_db_path

    def execute_query(self, db_id, sql_query):
        """Executes SQL against the specific SQLite file. Fails fast on errors or empty results."""
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
    def __init__(self):
        pass

    def summarize_shape(self, sql_query, column_names, sample_results):
        """Prompts Gemini to describe the metadata/shape of the output."""
        preview_data = sample_results[:3]
        total_rows = len(sample_results)
        
        prompt = f"""
        You are a Data Summarization Agent in a synthetic data pipeline.
        Your task is to describe the "shape" and metadata of a SQL query's output.
        
        RULES:
        1. Strictly describe what the SQL query looks like in natural language.
        2. Keep it to one concise sentence to help understand what scenario this data and query describe.
        
        Example Output: "Returns 5 rows containing the name and capacity of stadiums, ordered by capacity descending."
        
        ---
        SQL QUERY: {sql_query}
        RETURNED COLUMNS: {column_names}
        TOTAL ROWS RETURNED: {total_rows}
        SAMPLE DATA (First 3 rows): {preview_data}
        
        RESULT SUMMARY:
        """
        client = genai.Client(
            vertexai=True,
            project="mystic-bank-352905",
            location="us-central1")
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ]
            ),
        ]
        response = client.models.generate_content(
            model = "gemini-2.5-pro",
            contents = contents
        )
        return response.text.strip()


def run_batch_generation(sampler, spider_db_dir, num_dbs=10, max_retries=3):
    """
    Runs Stage 1 across multiple databases to generate a batch dataset.
    """
    architect = ArchitectAgent()
    executor = DatabaseExecutor(spider_db_dir)
    summarizer = ResultSummarizer()
    
    db_ids = [db['db_id'] for db in sampler.schemas][:num_dbs]
    print(f"📦 Starting batch run for {len(db_ids)} databases...\n" + "="*50)
    
    query_distribution = ["Medium", "Complex", "Complex"]
    master_dataset = []

    for db_id in db_ids:
        print(f"\n🚀 Processing Database: '{db_id}'")
        schema_text = sampler.get_formatted_schema_with_samples(db_id)
        
        if "Database not found" in schema_text:
            print(f"❌ Skipping {db_id}: Schema not found.")
            continue
            
        for i, complexity in enumerate(query_distribution):
            print(f"\n  ⚙️ Generating Query {i+1}/{len(query_distribution)} ({complexity})...")
            success = False
            attempts = 0
            
            while not success and attempts < max_retries:
                attempts += 1
                
                try:
                    generated_sql = architect.generate_sql(schema_text, complexity)
                    is_valid, msg, results, columns = executor.execute_query(db_id, generated_sql)
                    
                    if not is_valid:
                        print(f"    ⚠️ Attempt {attempts}: {msg}")
                        time.sleep(1)
                        continue 
                        
                    summary = summarizer.summarize_shape(generated_sql, columns, results)
                    
                    master_dataset.append({
                        "db_id": db_id,
                        "complexity": complexity,
                        "sql": generated_sql,
                        "result_summary": summary
                    })
                    success = True
                    print(f"    ✅ Success! ({len(results)} rows) -> {summary}")
                    
                except Exception as e:
                    print(f"    🚨 Unexpected API/Runtime Error: {e}")
                    time.sleep(2)
                    
            if not success:
                print(f"  ❌ Failed to generate a valid {complexity} query for {db_id} after {max_retries} attempts.")

        print(f"✅ Finished '{db_id}'. Accumulated {len(master_dataset)} total queries so far.")

    return master_dataset

if __name__ == "__main__":
    TABLES_JSON_PATH = "./tables.json"
    DATABASE_DIR_PATH = "./database/"   
    
    sampler = SmartSampler(TABLES_JSON_PATH, DATABASE_DIR_PATH)
    final_batch_data = run_batch_generation(
        sampler=sampler, 
        spider_db_dir=DATABASE_DIR_PATH,
        num_dbs=10
    )
    
    output_filename = "synthetic_data_batch.json"
    with open(output_filename, "w") as f:
        json.dump(final_batch_data, f, indent=4)
        
    print("\n" + "="*50)
    print(f"🎉 Batch Complete! Saved {len(final_batch_data)} verified queries to '{output_filename}'.")
