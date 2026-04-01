import os
import json
from datetime import datetime
from google import genai
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1 import SmartSampler, parse_db_selection, run_multithreaded_pipeline
from metrics.sql_generation_metrics import analyze_stage1_pipeline

# ==========================================
# EXPERIMENT RUNNER STAGE1
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
    DB_SELECTION = "0:5"
    
    # Setup for timestamped output to prevent overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_selection = DB_SELECTION.replace(":", "_").replace(",", "_").replace(" ", "")
    OUTPUT_FILENAME = f"synthetic_data_gen/results/stage1/synthetic_data_{clean_selection}_{timestamp}.json"
    
    # --- INITIALIZATION ---
    print(f"🎬 Initializing Synthetic Data Pipeline [Stage 1] (Experiment Runner)...")
    
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
