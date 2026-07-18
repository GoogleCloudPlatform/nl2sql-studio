import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1 import SmartSampler, DatabaseExecutor
from metrics.sql_generation_metrics import analyze_stage1_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_NAME = "sl-test-project-353312"
LOCATION = "us-central1"

# Target the NEXT 5 failing schemas as requested
TARGET_DBS = ["tvshow", "wta_1", "flight_2", "pets_1", "cre_Doc_Template_Mgt"]

# Prompt configuration
BATCH_PROMPT_NAME = "batch_sql_generation_cot_v4"
QPM_LIMIT = 60

# ==========================================
# ASYNC RATE LIMITER (Token Bucket Pattern)
# ==========================================
class AsyncRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.capacity = float(requests_per_minute)
        self.refill_rate = self.capacity / 60.0
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    async def wait_for_token(self):
        while True:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            await asyncio.sleep((1.0 - self.tokens) / self.refill_rate)

# ==========================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ==========================================
class QueryItem(BaseModel):
    complexity: str = Field(description="Simple or Medium")
    thought: str = Field(description="Step-by-step reasoning for why the query is valid and will yield results based on sample data")
    sql: str = Field(description="The valid SQLite query")

class BatchSQLResponse(BaseModel):
    queries: list[QueryItem]

# ==========================================
# ASYNC GENERATION FUNCTION
# ==========================================
async def generate_batch_for_db(db_id: str, schema_text: str, prompt_template: str, llm_client, limiter: AsyncRateLimiter):
    print(f"🔍 [{db_id}] Queuing for Architect...")
    
    # Wait for rate limiter token
    await limiter.wait_for_token()
    
    print(f"🚀 [{db_id}] Calling Architect...")
    
    prompt = prompt_template.replace("{schema_with_samples_text}", schema_text)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = await llm_client.aio.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchSQLResponse,
                    temperature=0.7,
                ),
            )
            
            # Parse the JSON response
            result = json.loads(response.text)
            queries = result.get("queries", [])
            print(f"✅ [{db_id}] Successfully generated {len(queries)} queries.")
            return db_id, queries
            
        except Exception as e:
            print(f"⚠️ [{db_id}] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"❌ [{db_id}] Failed after {max_retries} attempts.")
                return db_id, []
            
            # Exponential backoff with jitter
            sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            await asyncio.sleep(sleep_time)

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    TABLES_FILE = os.path.abspath(os.path.join(script_dir, "../tables-all.json"))
    DATABASE_PATH = os.path.abspath(os.path.join(script_dir, "../database/"))
    PROMPT_FILE = os.path.abspath(os.path.join(script_dir, f"../prompts/{BATCH_PROMPT_NAME}.txt"))
    
    # Load prompt template
    try:
        with open(PROMPT_FILE, 'r') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {PROMPT_FILE}")
        sys.exit(1)
        
    # Initialize Sampler and Executor
    sampler = SmartSampler(TABLES_FILE, DATABASE_PATH)
    executor = DatabaseExecutor(DATABASE_PATH)
    
    # Initialize Gemini Client (Async)
    try:
        llm_client = genai.Client(
            vertexai=True,
            project=PROJECT_NAME,
            location=LOCATION
        )
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        sys.exit(1)
        
    limiter = AsyncRateLimiter(requests_per_minute=QPM_LIMIT)
    
    print(f"🎬 Initializing Async Augmentation Pipeline ({BATCH_PROMPT_NAME})...")
    print(f"🎯 Targeting: {','.join(TARGET_DBS)}")
    print(f"⏱️  Rate Limit: {QPM_LIMIT} QPM")
    
    # Create tasks for all target DBs
    tasks = []
    for db_id in TARGET_DBS:
        schema_text = sampler.get_formatted_schema_with_samples(db_id)
        tasks.append(generate_batch_for_db(db_id, schema_text, prompt_template, llm_client, limiter))
        
    # Run all tasks concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"\n⌛ All LLM calls completed in {end_time - start_time:.2f} seconds.")
    
    # Process and validate results
    all_synthetic_data = []
    
    for db_id, queries in results:
        if not queries:
            continue
            
        print(f"\n🧪 [{db_id}] Validating {len(queries)} queries...")
        valid_count = 0
        
        for q in queries:
            sql = q.get("sql")
            complexity = q.get("complexity")
            
            if not sql:
                continue
                
            success, msg, res, columns = executor.execute_query(db_id, sql)
            
            if success:
                valid_count += 1
                result_summary = f"Returns {len(res)} rows with columns: {', '.join(columns)}."
            else:
                result_summary = f"Discarded: {msg}"
            
            all_synthetic_data.append({
                "db_id": db_id,
                "complexity": complexity,
                "sql": sql,
                "success": success,
                "result_summary": result_summary
            })
                
        print(f"💾 Validated {valid_count}/{len(queries)} queries for {db_id}")
        
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILENAME = f"synthetic_data_gen/results/stage1/synthetic_data_aug_{timestamp}.json"
    
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(all_synthetic_data, f, indent=4)
        
    print(f"\n==================================================")
    print(f"🎉 Augmentation Complete! Saved {len(all_synthetic_data)} queries.")
    print(f"📍 Output: {OUTPUT_FILENAME}")
    print(f"==================================================")
    
    # Added metrics report call
    print("📊 Generating Comprehensive Metrics Report...")
    analyze_stage1_pipeline(OUTPUT_FILENAME, TABLES_FILE)

if __name__ == "__main__":
    asyncio.run(main())
