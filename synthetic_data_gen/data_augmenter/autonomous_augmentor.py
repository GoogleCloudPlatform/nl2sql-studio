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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1 import SmartSampler, DatabaseExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# PYDANTIC MODELS FOR AUTONOMOUS ANALYSIS
# ==========================================
class Strategy(BaseModel):
    name: str = Field(description="Short name of the strategy (e.g., 'Alias Enforcement')")
    instruction: str = Field(description="Detailed instruction for the query generator on what kind of queries to create to address this failure mode.")

class AnalysisResponse(BaseModel):
    strategies: list[Strategy]

class QueryItem(BaseModel):
    complexity: str = Field(description="Simple or Medium")
    thought: str = Field(description="Step-by-step reasoning for why the query is valid and will yield results based on sample data")
    sql: str = Field(description="The valid SQLite query")

class BatchSQLResponse(BaseModel):
    queries: list[QueryItem]

# ==========================================
# HELPER: SAMPLE FAILURES
# ==========================================
def sample_failures(inference_file_path, samples_per_db=5):
    """
    Reads inference results and samples failed queries for analysis.
    """
    if not os.path.exists(inference_file_path):
        return None

    with open(inference_file_path, 'r') as f:
        data = json.load(f)

    failures = [item for item in data if item.get('status') != 'Correct']
    
    # Group by DB
    db_failures = defaultdict(list)
    for item in failures:
        db_failures[item['db_id']].append(item)
        
    # Sample
    sampled_data = []
    for db_id, items in db_failures.items():
        sampled = random.sample(items, min(len(items), samples_per_db))
        sampled_data.extend(sampled)
        
    return sampled_data, Counter([item['db_id'] for item in failures])

# ==========================================
# AGENT COMPONENT: THE ANALYZER
# ==========================================
async def formulate_strategies(sampled_failures, llm_client):
    """
    Calls Gemini to analyze failures and formulate strategies.
    """
    print("🧠 Calling Analyzer Agent to formulate strategies...")
    
    # Prepare the context for the LLM
    failures_context = ""
    for idx, item in enumerate(sampled_failures):
        failures_context += f"""
--- Failure {idx+1} ---
Database: {item.get('db_id')}
Complexity: {item.get('complexity')}
Generated SQL (Wrong): {item.get('sql')}
Error/Status: {item.get('status')}
"""

    prompt = f"""
You are an expert AI Data Scientist analyzing failures in a Text-to-SQL model.
Below is a sample of queries that the model generated incorrectly (they failed execution or produced wrong results).

FAILURES TO ANALYZE:
{failures_context}

YOUR TASK:
1. Analyze these failures to understand what logical concepts or schema patterns the model is struggling with.
2. Formulate a list of 3 to 5 specific "Data Augmentation Strategies".
3. Each strategy must be a clear, actionable instruction that we can give to a query generator to create targeted training data to fix these weaknesses.

Output a JSON object adhering to the requested schema.
"""

    try:
        response = await llm_client.aio.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResponse,
                temperature=0.2, # Low temperature for analytical focus
            ),
        )
        
        result = json.loads(response.text)
        return result.get("strategies", [])
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return []

# ==========================================
# GENERATOR COMPONENT
# ==========================================
async def generate_batch_with_dynamic_strategy(db_id: str, schema_text: str, strategy: Strategy, count: int, llm_client):
    print(f"🚀 [{db_id}] Strategy: {strategy.name} | Calling Architect for {count} queries...")
    
    prompt = f"""
TARGET SCHEMA & SAMPLE DATA:
{schema_text}

You are an expert SQL Architect building a diverse synthetic dataset. 
Your task is to generate exactly {count} highly accurate, valid SQL queries based on the provided schema and sample data.

SPECIFIC STRATEGY FOCUS:
Your primary goal is to address failures related to: **{strategy.name}**
Instruction: {strategy.instruction}

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
        return db_id, strategy.name, result.get("queries", [])
    except Exception as e:
        print(f"⚠️ [{db_id}] [{strategy.name}] Failed: {e}")
        return db_id, strategy.name, []

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    parser = argparse.ArgumentParser(description='Autonomous Data Augmentation Agent')
    parser.add_argument('--inference', type=str, required=True, help='Path to infrence.json')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    TABLES_FILE = os.path.abspath(os.path.join(script_dir, "../../tables-all.json"))
    DATABASE_PATH = os.path.abspath(os.path.join(script_dir, "../../database/"))
    
    # 1. Sample Failures
    sampled_failures, failure_counts = sample_failures(args.inference, samples_per_db=3)
    if not sampled_failures:
        print("No failures found or file missing.")
        sys.exit(1)
        
    print(f"📋 Sampled {len(sampled_failures)} failed queries for analysis.")
    
    # Initialize Gemini Client
    try:
        llm_client = genai.Client(
            vertexai=True,
            project="sl-test-project-353312",
            location="us-central1"
        )
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        sys.exit(1)
        
    # 2. Formulate Strategies Dynamically
    strategies = await formulate_strategies(sampled_failures, llm_client)
    
    print("\n" + "="*50)
    print("🤖 DYNAMICALLY FORMULATED STRATEGIES")
    print("="*50)
    for s in strategies:
        print(f"\n▶️ Strategy: {s.get('name')}")
        print(f"  Instruction: {s.get('instruction')}")
    print("="*50 + "\n")
    
    if not strategies:
        print("Failed to formulate strategies. Exiting.")
        sys.exit(1)
        
    # Convert to Pydantic models for safety
    strat_models = [Strategy(name=s['name'], instruction=s['instruction']) for s in strategies]
    
    # 3. Determine Tiers based on counts
    sorted_dbs = failure_counts.most_common()
    tier1 = [db[0] for db in sorted_dbs[:5]]
    tier2 = [db[0] for db in sorted_dbs[5:10]]
    
    # 4. Generate (Small scale test run)
    # For this agent, let's just run a small test batch to show it works
    # Generating 2 queries per strategy for Tier 1 DBs
    sampler = SmartSampler(TABLES_FILE, DATABASE_PATH)
    
    tasks = []
    for db_id in tier1[:2]: # Just take top 2 DBs for quick demo
        schema_text = sampler.get_formatted_schema_with_samples(db_id)
        for strat in strat_models:
            tasks.append(generate_batch_with_dynamic_strategy(db_id, schema_text, strat, 2, llm_client))
            
    print(f"🚀 Launching test generation for {len(tasks)} batches...")
    results = await asyncio.gather(*tasks)
    
    # Summarize generated queries
    total_gen = 0
    for db_id, strat_name, queries in results:
        total_gen += len(queries)
        print(f"✅ Generated {len(queries)} queries for {db_id} using strategy '{strat_name}'")
        
    print(f"\n🎉 Autonomous Agent Demo Complete. Generated {total_gen} queries.")

if __name__ == "__main__":
    asyncio.run(main())
