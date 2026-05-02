import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from collections import defaultdict
from google import genai
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2 import SmartSampler, build_system_prompt, clean_json_response, PERSONAS

# Set up simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

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
# ASYNC TRANSLATOR AGENT
# ==========================================
class AsyncContextTranslatorAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.personas = PERSONAS

    async def generate_questions_async(self, sql_items: list, schemas_dict: dict, persona: dict, include_result_summary: bool = True, include_schema: bool = True) -> str:
        system_prompt = build_system_prompt(sql_items, schemas_dict, persona, include_result_summary, include_schema)

        if not self.llm_client:
            logger.info("Prompt constructed successfully (LLM client not provided).")
            return None

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.aio.models.generate_content(
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
                await asyncio.sleep(sleep_time)
        return "Error generating questions"

# ==========================================
# ASYNC BATCH PROCESSOR
# ==========================================
async def process_batch_async(batch_items, schemas_dict, persona, translator_agent, batch_num, total_batches, max_retries, include_result_summary, include_schema, limiter):
    persona_name = persona['name']
    print(f"\n  📦 [{persona_name}] Processing batch {batch_num}/{total_batches} ({len(batch_items)} items)...")
    
    for attempt in range(1, max_retries + 1):
        print(f"    [{persona_name} - Batch {batch_num}] Attempt {attempt}/{max_retries}...")
        
        await limiter.wait_for_token()
        
        response_text = await translator_agent.generate_questions_async(
            batch_items, schemas_dict, persona, 
            include_result_summary=include_result_summary, include_schema=include_schema
        )
        
        if not response_text or response_text.startswith("Error"):
            print(f"    ⚠️ [{persona_name} - Batch {batch_num}] {response_text}")
            await asyncio.sleep(1)
            continue

        try:
            nl_questions = json.loads(clean_json_response(response_text))
            
            if isinstance(nl_questions, list) and len(nl_questions) == len(batch_items):
                print(f"    ✅ [{persona_name} - Batch {batch_num}] Translation Success!")
                results = []
                for idx, item in enumerate(batch_items):
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
        
        await asyncio.sleep(1)
        
    print(f"  ❌ [{persona_name}] Failed to translate batch {batch_num} after {max_retries} attempts.")
    return []

# ==========================================
# ASYNC PIPELINE ORCHESTRATOR
# ==========================================
async def run_stage_2_pipeline_async(stage_1_dataset, schemas_dict, llm_client=None, max_retries=3, include_result_summary=True, include_schema=True, qpm=60):
    print(f"🚀 Starting Stage 2 Pipeline (Async with {qpm} QPM)\n" + "="*50)
    translator_agent = AsyncContextTranslatorAgent(llm_client)
    limiter = AsyncRateLimiter(requests_per_minute=qpm)
    final_dataset = []

    # Assign random personas and group by persona_name
    persona_groups = defaultdict(list)
    for item in stage_1_dataset:
        item['assigned_persona'] = random.choice(translator_agent.personas)
        persona_groups[item['assigned_persona']['name']].append(item)

    # Changed batch size to 10 as requested
    batch_size = 10
    tasks = []
    
    for persona_name, items in persona_groups.items():
        persona = items[0]['assigned_persona']
        print(f"\n⚙️ Queuing {len(items)} Queries for Persona '{persona_name}'...")
        
        total_batches = (len(items) + batch_size - 1) // batch_size
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            tasks.append(
                process_batch_async(
                    batch_items, schemas_dict, persona, translator_agent, 
                    batch_num, total_batches, max_retries, 
                    include_result_summary, include_schema, limiter
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for res in results:
        if isinstance(res, list):
            final_dataset.extend(res)
        elif isinstance(res, Exception):
            print(f"⚠️ Task failed with exception: {res}")
            
    print("\n" + "="*50)
    print(f"🎉 Stage 2 Complete. Generated {len(final_dataset)} final NL2SQL pairs.")
    return final_dataset

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    parser = argparse.ArgumentParser(description='Run Stage 2 Translation (Async)')
    parser.add_argument('--input', type=str, required=True, help='Path to Stage 1 output JSON file')
    parser.add_argument('--output', type=str, help='Path to save output JSON file')
    parser.add_argument('--qpm', type=int, default=60, help='Queries Per Minute limit')
    args = parser.parse_args()

    PROJECT_NAME = "sl-test-project-353312"
    LOCATION = "us-central1"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    TABLES_FILE = os.path.abspath(os.path.join(script_dir, "../tables-all.json"))
    DATABASE_PATH = os.path.abspath(os.path.join(script_dir, "../database/"))
    
    if not args.output:
        base_name = os.path.basename(args.input)
        name_part, _ = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.abspath(os.path.join(script_dir, f"../results/stage2/{name_part}_questions_{timestamp}.json"))

    try:
        with open(args.input, 'r') as f:
            stage_1_data = json.load(f)
        
        original_count = len(stage_1_data)
        stage_1_data = [item for item in stage_1_data if item.get('success') is True]
        print(f"📋 Loaded {original_count} items from Stage 1. Kept {len(stage_1_data)} successful items for translation.")
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input}")
        sys.exit(1)

    sampler = SmartSampler(TABLES_FILE, DATABASE_PATH)
    
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
    
    final_golden_data = await run_stage_2_pipeline_async(
        stage_1_data, schemas, llm_client, 
        include_result_summary=True, include_schema=True, qpm=args.qpm
    )
    
    print(f"💾 Saving results to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(final_golden_data, f, indent=4)
        
    print(f"🎉 Stage 2 Complete! Results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
