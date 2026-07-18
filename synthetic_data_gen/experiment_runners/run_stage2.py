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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import SmartSampler, run_stage_2_pipeline


if __name__ == "__main__":
    PROJECT_NAME = "sl-test-project-353312"
    LOCATION = "us-central1"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    TABLES_FILE = os.path.abspath(os.path.join(script_dir, "../tables-all.json"))
    DATABASE_PATH = os.path.abspath(os.path.join(script_dir, "../database/"))
    INPUT_FILE_PATH = os.path.abspath(os.path.join(script_dir, "../results/stage1/4k.json"))
    OUTPUT_FILE_PATH = os.path.abspath(os.path.join(script_dir, "../results/stage2/s2_flash_4k.json"))
    INCLUDE_RESULT_SUMMARY = True
    INCLUDE_SCHEMA = True
    
    try:
        with open(INPUT_FILE_PATH, 'r') as f:
            stage_1_data = json.load(f)
        
        # Filter for successful items
        original_count = len(stage_1_data)
        stage_1_data = [item for item in stage_1_data if item.get('success') is True]
        print(f"📋 Loaded {original_count} items from Stage 1. Kept {len(stage_1_data)} successful items for translation.")
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
            project=PROJECT_NAME,
            location=LOCATION
        )
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        llm_client = None
    
    final_golden_data = run_stage_2_pipeline(stage_1_data, schemas, llm_client, include_result_summary=INCLUDE_RESULT_SUMMARY, include_schema=INCLUDE_SCHEMA)
    
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(final_golden_data, f, indent=4)
        
    print(f"Results saved to {OUTPUT_FILE_PATH}")