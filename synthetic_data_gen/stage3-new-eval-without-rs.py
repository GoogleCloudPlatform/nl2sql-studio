
import os
import vertexai
import json
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

class CategoryEvaluation(BaseModel):
    passed: bool = Field(description="True if the clip passes this category (Score 8+), False otherwise.")
    score: int = Field(description="Score for this category from 0 to 10.")
    details: str = Field(description="Explanation for the score and pass/fail status.")

def get_persona_description(persona_name):
    """Fetches the description of a given persona by name."""
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
    return next((p["description"] for p in PERSONAS if p["name"] == persona_name), "Persona not found.")

class GenAIReport(BaseModel):
    technical_accuracy: CategoryEvaluation
    persona_alignment: CategoryEvaluation
    schema_adherence: CategoryEvaluation
    groundedness: CategoryEvaluation
    conciseness_clarity: CategoryEvaluation
    fluency: CategoryEvaluation
    genai_total_score: int = Field(description="Total score across all 6 GenAI categories (out of 60).")


from typing import List

class BatchEvaluationReport(BaseModel):
    evaluations: List[GenAIReport] = Field(description="A list of evaluation reports, one for exactly each input record provided.")


def evaluate_batch(model: GenerativeModel, batch_records: list) -> BatchEvaluationReport:
    """
    Evaluates a batch of NL responses against their corresponding SQL contexts.
    batch_records: list of dicts, each containing 'user_prompt' and 'nl_question'
    """
    try:
        # Construct the batched prompt payload
        records_text = ""
        for i, rec in enumerate(batch_records):
            records_text += f"\n--- RECORD {i+1} ---\n"
            records_text += f"[CONTEXT/INSTRUCTIONS]:\n{rec.get('system_prompt', '')}\n"
            records_text += f"[GENERATED QUESTION TO EVALUATE]:\n{rec.get('nl_question', '')}\n"
            
        eval_prompt = f"""
        You are an expert evaluator assessing the quality of synthetic Reverse Translation (SQL to Natural Language).
        You will evaluate a batch of {len(batch_records)} records. First, read the context and the generated question for each record.
        Then, evaluate each question according to the following 6 rubric criteria on a 0-10 scale.
        
        [RUBRIC]
        1. Technical Accuracy (SQL Translation):
           - 10/10: The generated question accurately captures all clauses, conditions, groupings, ranges, limits, and joins present in the SQL.
           - Fail criteria (< 8): Misses an explicitly defined condition (e.g. 'WHERE age > 10', 'LIMIT 5'), hallucinates a condition not in SQL, or misinterprets an aggregation (e.g. asking for 'total' when SQL says 'average').
        2. Persona Alignment:
           - 10/10: The tone, terminology, and business focus perfectly match the provided persona.
           - Fail criteria (< 8): Disregards the persona description (e.g. highly technical schema jargon used when persona is 'The Executive', or missing analytical depth when persona is 'Data Analyst').
        3. Schema Adherence:
           - 10/10: Only uses terminology, tables, column names, and domain concepts logically present in the provided schema.
           - Fail criteria (< 8): References fictional tables/columns, or asks about data dimensions completely unsupported by the schema (e.g. asking for 'company revenue' when the schema only contains employee demographics).
        4. Groundedness (Result Shape):
           - 10/10: The question explicitly asks for the specific output shape described in the 'Result Summary' (e.g. 'Which 3...').
           - Fail criteria (< 8): Asks an open-ended question when the result is heavily constrained (e.g., SQL limits to 1, but question is 'List all...'), or asks for detailed rows when the output is a single aggregate number.
        5. Conciseness & Clarity:
           - 10/10: The question is direct, professional, and gets straight to the point without redundant filler.
           - Fail criteria (< 8): The question is excessively wordy, unclear, repetitious, or requires multiple reads to understand the actual intent.
        6. Fluency:
           - 10/10: The generated NL question is grammatically correct, well-organized, and reads naturally to a native speaker.
           - Fail criteria (< 8): The sentence contains noticeable grammatical mistakes, awkward phrasing, or reads like mechanical translation rather than fluent human language.

        [BATCH DATA TO EVALUATE]
        {records_text}

        You MUST process exactly {len(batch_records)} records.
        Return a JSON object that strictly conforms to the following JSON structure. Do NOT output the schema itself. You MUST return exactly {len(batch_records)} evaluations.
        {{
            "evaluations": [
                {{
                    "technical_accuracy": {{ "passed": true, "score": 9, "details": "string explanation" }},
                    "persona_alignment": {{ "passed": true, "score": 9, "details": "string explanation" }},
                    "schema_adherence": {{ "passed": true, "score": 9, "details": "string explanation" }},
                    "groundedness": {{ "passed": true, "score": 9, "details": "string explanation" }},
                    "conciseness_clarity": {{ "passed": true, "score": 9, "details": "string explanation" }},
                    "fluency": {{ "passed": true, "score": 9, "details": "string explanation" }},
                    "genai_total_score": 54
                }}
            ]
        }}
        """

        # Force the model to output clean JSON matching our Pydantic schema
        config = GenerationConfig(
            temperature=0.1, 
            response_mime_type="application/json"
        )
        max_retries = 6
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content([eval_prompt], generation_config=config)
                # Parse the JSON response directly into the Pydantic object
                ai_evaluation = BatchEvaluationReport.model_validate_json(response.text)
                return ai_evaluation
            except Exception as e:
                import time
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Batch evaluation exception (Attempt {attempt+1}/{max_retries}) [429/503 Quota]. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error evaluating batch after {max_retries} attempts: {e}")
                    return None
    except Exception as e:
        print(f"Critical error preparing batch payload: {e}")
        return None


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    # Initialize Vertex AI
    vertexai.init() 
    
    # Instantiate the Generative Model 
    # Use gemini-2.5-pro for high-quality evaluation reasoning
    eval_model = GenerativeModel("gemini-2.5-pro")

    INPUT_FILE_PATH = "./results/results-without-result-summary/merged_stage2_results_without_rs.json"
    OUTPUT_FILE_PATH = "./results/results-without-result-summary/merged_stage3_eval_results_without_rs_pro.json"

    print(f"Loading data from {INPUT_FILE_PATH}")
    try:
        with open(INPUT_FILE_PATH, "r") as f:
            stage2_data = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find any input files.")
        stage2_data = []
    
    # Inject system_prompt dynamically
    for row in stage2_data:
        persona_desc = get_persona_description(row.get('persona', ''))
        row['system_prompt'] = f"""You are an expert natural language generation agent.
        Your persona for this task is: {row.get('persona', '')}.
        Persona Description: {persona_desc}
        
        Your goal is Reverse Translation: Given the inputs below, generate the precise, pure Natural Language question that would have produced each exact SQL query.
        Importantly, craft each question to match your exact persona and refer to the specific Database Schema corresponding to each query.
        
        [CONTEXT]
        - Schema Used: {row.get('schema', '')}

        - Query to Translate:
        --- Query ---
        SQL: {row.get('sql', '')}
        """

    if stage2_data:
        # Remove items that do not have necessary prompts
        valid_data = [r for r in stage2_data if r.get("system_prompt") and r.get("nl_question")]
        print(f"Starting evaluation of {len(valid_data)} valid items via batched LLM calls...")
        
        BATCH_SIZE = 20 # Process 20 records per LLM call
        MAX_WORKERS = 5 # Number of parallel threads
        
        batches = list(chunk_list(valid_data, BATCH_SIZE))
        
        def process_batch(batch_idx, batch):
            print(f"Starting Batch {batch_idx + 1}/{len(batches)} (Records: {len(batch)})...")
            try:
                batch_result = evaluate_batch(eval_model, batch)
                
                if batch_result and len(batch_result.evaluations) == len(batch):
                    for i, record in enumerate(batch):
                        eval_obj = batch_result.evaluations[i]
                        record["evaluation"] = eval_obj.model_dump()
                else:
                    print(f"Warning: Batch returned {len(batch_result.evaluations) if batch_result else 0} evaluations, expected {len(batch)}. Skipping these results.")
                    for record in batch:
                        record["evaluation"] = None
                        
            except Exception as e:
                print(f"Batch {batch_idx + 1} failed: {e}")
                for record in batch:
                    record["evaluation"] = None
            print(f"Completed Batch {batch_idx + 1}/{len(batches)}")

        print(f"Processing {len(batches)} batches using {MAX_WORKERS} threads...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_batch, i, b) for i, b in enumerate(batches)]
            for future in as_completed(futures):
                pass # wait for all to complete
                
        # Calculate scores after all threads finish
        complexity_scores = {"Simple": [], "Medium": [], "Complex": []}
        categories = ["technical_accuracy", "persona_alignment", "schema_adherence", "groundedness", "conciseness_clarity", "fluency"]
        metric_scores = {cat: [] for cat in categories}
        
        for record in valid_data:
            eval_data = record.get("evaluation")
            if eval_data and "genai_total_score" in eval_data:
                total_score = eval_data["genai_total_score"]
                comp = record.get("complexity")
                if comp in complexity_scores:
                    complexity_scores[comp].append(total_score)
                    
                for cat in categories:
                    if cat in eval_data and "score" in eval_data[cat]:
                        metric_scores[cat].append(eval_data[cat]["score"])

        # Print avg scores
        print("\n--- Average Scores by Complexity ---")
        for comp, scores in complexity_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"{comp}: {avg:.2f} / 60 ({len(scores)} records)")
            else:
                print(f"{comp}: N/A")
                
        print("\n--- Average Scores by Category (Out of 10) ---")
        for cat, scores in metric_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"{cat}: {avg:.2f}")
            else:
                print(f"{cat}: N/A")

        # Save to output file
        print(f"\nSaving results to {OUTPUT_FILE_PATH}...")
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        with open(OUTPUT_FILE_PATH, "w") as f:
            json.dump(stage2_data, f, indent=4)
            
        print("Done!")
