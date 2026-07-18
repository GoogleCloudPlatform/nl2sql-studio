import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from collections import Counter, defaultdict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Add parent directory to path to import from stage1 and metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1 import SmartSampler, DatabaseExecutor
from metrics.sql_generation_metrics import analyze_stage1_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# STRATEGIES DEFINITION
# ==========================================
STRATEGIES = {
    "Schema-Anchoring": "Generate queries that strictly use only column names explicitly listed in the provided schema. Avoid hallucinating or guessing column names not present in the schema.",
    "Strict Set Operations": "Generate queries that require comparing or combining two sets of data using UNION, INTERSECT, or EXCEPT. Ensure the column types match for these operations.",
    "Precision Aggregation & Grouping": "Generate queries that require basic aggregations (COUNT, SUM, AVG), GROUP BY, and HAVING clauses with specific conditions.",
    "Complex Alias Enforcement": "Generate queries that involve at least 3 tables and strictly enforce table aliasing (e.g., T1, T2, etc.) for all column references to avoid ambiguity."
}

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
# ASYNC RATE LIMITER
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
# ANALYZER COMPONENT
# ==========================================
def analyze_failures(inference_file_path):
    if not os.path.exists(inference_file_path):
        print(f"Error: Inference file not found at {inference_file_path}")
        return None

    try:
        with open(inference_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

    failures = [item for item in data if item.get('status') != 'Correct']
    db_counts = Counter([item['db_id'] for item in failures])
    sorted_dbs = db_counts.most_common()
    
    # Group into tiers
    tier1 = [db[0] for db in sorted_dbs[:5]]
    tier2 = [db[0] for db in sorted_dbs[5:10]]
    tier3 = [db[0] for db in sorted_dbs[10:]]
    
    return {
        "total_failures": len(failures),
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "full_counts": sorted_dbs
    }

def print_analysis_report(analysis):
    if not analysis:
        return
        
    print("\n" + "="*50)
    print("📊 SMART AUGMENTOR: FAILURE ANALYSIS REPORT")
    print("="*50)
    print(f"Total Failed Queries analyzed: {analysis['total_failures']}")
    
    print("\n🔴 TIER 1: Critical Priority (Top 5 worst performers)")
    for db_id in analysis['tier1']:
        count = next(c[1] for c in analysis['full_counts'] if c[0] == db_id)
        print(f"   - {db_id}: {count} failures")
        
    print("\n🟡 TIER 2: High Priority (Next 5 worst performers)")
    for db_id in analysis['tier2']:
        count = next(c[1] for c in analysis['full_counts'] if c[0] == db_id)
        print(f"   - {db_id}: {count} failures")
        
    print("\n🟢 TIER 3: Normal Priority (Rest of failing schemas)")
    print(f"   Count: {len(analysis['tier3'])} schemas")
    print("="*50 + "\n")

# ==========================================
# GENERATOR COMPONENT
# ==========================================
async def generate_batch_with_strategy(db_id: str, schema_text: str, strategy_name: str, strategy_desc: str, count: int, llm_client, limiter: AsyncRateLimiter):
    print(f"🔍 [{db_id}] Strategy: {strategy_name} | Queuing for Architect...")
    await limiter.wait_for_token()
    print(f"🚀 [{db_id}] Strategy: {strategy_name} | Calling Architect for {count} queries...")
    
    prompt = f"""
TARGET SCHEMA & SAMPLE DATA:
{schema_text}

You are an expert SQL Architect building a diverse synthetic dataset. 
Your task is to generate exactly {count} highly accurate, valid SQL queries based on the provided schema and sample data.

SPECIFIC STRATEGY FOCUS:
Your primary goal is to address failures related to: **{strategy_name}**
Instruction: {strategy_desc}

DISTRIBUTION REQUIREMENT:
- Queries should be mostly "Simple" and some "Medium" complexity.

RULES:
1. NO MARKDOWN: Only output raw SQL in the JSON.
2. ALIASING: Use table aliasing (e.g., T1, T2) for clarity in all queries involving more than one table.
3. DATA-AWARE FILTERING: Only filter using exact data values explicitly shown in the "Sample Rows".
4. OUTPUT FORMAT: You MUST output ONLY a valid JSON object with a key named "queries" containing the array of query objects.

Format Example:
{{
    "queries": [
        {{"complexity": "Simple", "thought": "...", "sql": "..."}},
        ...
    ]
}}
"""
    
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
            
            result = json.loads(response.text)
            queries = result.get("queries", [])
            print(f"✅ [{db_id}] [{strategy_name}] Generated {len(queries)} queries.")
            return db_id, strategy_name, queries
            
        except Exception as e:
            print(f"⚠️ [{db_id}] [{strategy_name}] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                return db_id, strategy_name, []
            await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    parser = argparse.ArgumentParser(description='Run Error-Driven Data Augmentation')
    parser.add_argument('--inference', type=str, required=True, help='Path to infrence.json')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    parser.add_argument('--qpm', type=int, default=30, help='Queries Per Minute limit')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    TABLES_FILE = os.path.abspath(os.path.join(script_dir, "../tables-all.json"))
    DATABASE_PATH = os.path.abspath(os.path.join(script_dir, "../database/"))
    
    if not args.output_dir:
        args.output_dir = os.path.abspath(os.path.join(script_dir, "../results/stage1/error_driven"))

    # 1. Run Analysis
    print(f"Analyzing failures from: {args.inference}")
    analysis = analyze_failures(args.inference)
    print_analysis_report(analysis)
    
    if not analysis:
        sys.exit(1)
        
    # 2. Initialize Components
    sampler = SmartSampler(TABLES_FILE, DATABASE_PATH)
    executor = DatabaseExecutor(DATABASE_PATH)
    
    try:
        llm_client = genai.Client(
            vertexai=True,
            project="sl-test-project-353312",
            location="us-central1"
        )
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        sys.exit(1)
        
    limiter = AsyncRateLimiter(requests_per_minute=args.qpm)
    
    # 3. Formulate Generation Plan
    tasks = []
    
    # Tier 1: 100 queries per DB (25 per strategy)
    for db_id in analysis['tier1']:
        schema_text = sampler.get_formatted_schema_with_samples(db_id)
        for strat_name, strat_desc in STRATEGIES.items():
            tasks.append(generate_batch_with_strategy(db_id, schema_text, strat_name, strat_desc, 25, llm_client, limiter))
            
    # Tier 2: 50 queries per DB (approx 12 per strategy)
    for db_id in analysis['tier2']:
        schema_text = sampler.get_formatted_schema_with_samples(db_id)
        for strat_name, strat_desc in STRATEGIES.items():
            tasks.append(generate_batch_with_strategy(db_id, schema_text, strat_name, strat_desc, 12, llm_client, limiter))
            
    # Tier 3: 10 queries per DB (split randomly or just pick one)
    for db_id in analysis['tier3']:
        schema_text = sampler.get_formatted_schema_with_samples(db_id)
        strat_name = random.choice(list(STRATEGIES.keys()))
        strat_desc = STRATEGIES[strat_name]
        tasks.append(generate_batch_with_strategy(db_id, schema_text, strat_name, strat_desc, 10, llm_client, limiter))

    print(f"🚀 Starting execution of {len(tasks)} generation tasks...")
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"⌛ All LLM calls completed in {end_time - start_time:.2f} seconds.")
    
    # 4. Validate and Save
    all_synthetic_data = []
    
    db_results = defaultdict(list)
    for db_id, strat_name, queries in results:
        db_results[db_id].extend(queries)
        
    for db_id, queries in db_results.items():
        print(f"\n🧪 [{db_id}] Validating {len(queries)} total queries...")
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
        
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILENAME = os.path.join(args.output_dir, f"error_driven_aug_{timestamp}.json")
    
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(all_synthetic_data, f, indent=4)
        
    print(f"\n==================================================")
    print(f"🎉 Error-Driven Augmentation Complete! Saved {len(all_synthetic_data)} queries.")
    print(f"📍 Output: {OUTPUT_FILENAME}")
    print(f"==================================================")
    
    # Added metrics report call
    print("📊 Generating Comprehensive Metrics Report...")
    analyze_stage1_pipeline(OUTPUT_FILENAME, TABLES_FILE)

if __name__ == "__main__":
    asyncio.run(main())
