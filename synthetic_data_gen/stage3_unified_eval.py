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
    """
    Fetches the description of a given persona by name.

    Args:
        persona_name (str): The name of the persona (e.g., 'The Executive').

    Returns:
        str: The description if found, otherwise 'Persona not found.'.
    """
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
    self_debate: str
    technical_accuracy: CategoryEvaluation
    persona_alignment: CategoryEvaluation
    schema_adherence: CategoryEvaluation
    groundedness: CategoryEvaluation
    conciseness_clarity: CategoryEvaluation
    information_density_clarity: CategoryEvaluation
    fluency: CategoryEvaluation
    genai_total_score: int = Field(description="Total score across all 7 GenAI categories (out of 35).")


from typing import List

class BatchEvaluationReport(BaseModel):
    evaluations: List[GenAIReport] = Field(description="A list of evaluation reports, one for exactly each input record provided.")


def evaluate_batch(model: GenerativeModel, batch_records: list) -> BatchEvaluationReport:
    """
    Evaluates a batch of generated questions using the Gemini API.

    Args:
        model (GenerativeModel): The Vertex AI GenerativeModel instance.
        batch_records (list): A list of dictionaries containing 'system_prompt' and 'nl_question'.

    Returns:
        BatchEvaluationReport: A Pydantic model representing the evaluation results for the batch.
                               Returns None if evaluation fails after all retries.
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

        Before assigning any scores, you MUST perform a "Self-Debate" where you briefly argue at least one reason why the generated question might be technically inaccurate, robotic, vague, or missing a constraint.

        Then, evaluate each question according to the 7 criteria below on a strict 1-5 scale based on the detailed rubrics provided.

        [INDIVIDUAL SCORING RUBRICS]

        1. Technical Accuracy (SQL Translation):
        - 5: Flawlessly captures all SQL logic, clauses, aggregations, groupings, ranges, limits, and joins.
        - 4: Captures primary logic but misses a minor, non-critical condition that doesn't drastically change the outcome.
        - 3: Captures general intent but misses an explicitly defined specific constraint (e.g., a WHERE filter).
        - 2: Fundamentally misinterprets SQL logic, misrepresents an aggregation (e.g., 'total' instead of 'average'), or hallucinates a condition not in the SQL.
        - 1: Completely wrong, disconnected from the SQL, or missing the majority of the logical operations.

        2. Schema Adherence:
        - 5: Perfectly uses terminology, tables, column names, and domain concepts logically present in the provided schema.
        - 4: Mostly adheres to the schema but uses slight, acceptable variations in naming that do not change the business meaning.
        - 3: References a generic domain concept not strictly in the schema, but logically related enough to pass.
        - 2: References fictional tables/columns, or asks about data dimensions unsupported by the schema.
        - 1: Completely ignores the schema, hallucinating entirely new business entities or metrics.

        3. Groundedness (Result Shape):
        - 5: Explicitly asks for the exact output shape described in the Result Summary (e.g., "Which 3 items...").
        - 4: Implies the correct shape but could be slightly more explicit (e.g., "What are the top items..." when the limit is 5).
        - 3: Asks a generally correct question but completely misses a strict row limit or specific shape constraint.
        - 2: Asks an open-ended question when the result is heavily constrained (e.g., SQL limits to 1, but question is 'List all...').
        - 1: Asks for detailed rows when the output is a single aggregate number, or vice versa.

        4. Persona Alignment:
        - 5: Tone, terminology, and business focus perfectly match the provided persona description.
        - 4: Good alignment, but misses the opportunity to use nuanced vocabulary specific to the persona.
        - 3: Generic professional tone; lacks the specific analytical depth, operational focus, or strategic view of the persona.
        - 2: Disregards the persona description (e.g., uses highly technical database jargon for an Executive).
        - 1: Completely inappropriate tone or actively contradicts the persona's assumed knowledge level.

        5. Conciseness Clarity (Formatting & Filler):
        - 5: Direct, professional, and gets straight to the point without redundant conversational filler or meta-text.
        - 4: Generally concise but includes a minor conversational pleasantry that could be trimmed.
        - 3: Somewhat wordy or slightly repetitive, but the core intent is still clear.
        - 2: Excessively wordy, repetitious, or includes unnecessary meta-text (e.g., "Here is a query to find the...").
        - 1: So bloated with conversational filler or meta-text that the actual question is buried or confusing.

        6. Information Density and Clarity (IDC):
        - 5: Highly descriptive, concise, and sounds like a direct, unambiguous, and natural human inquiry.
        - 4: Clear and natural, but slightly less descriptive than a native domain expert might phrase it.
        - 3: Understandable, but leans a bit vague or feels slightly mechanical in its phrasing.
        - 2: Reads like a mechanically robotic, clause-by-clause translation of the SQL code (e.g., "Select the count of users where status is active").
        - 1: Overly vague, relies on metaphorical expressions lacking context, or reads entirely like machine/pseudo-code.

        7. Fluency (Grammar & Syntax):
        - 5: The generated NL question is grammatically perfect, well-organized, and syntactically correct.
        - 4: Contains one minor typo or slight awkwardness that does not impede reading.
        - 3: Contains noticeable grammatical mistakes or somewhat awkward phrasing that requires a slight re-read.
        - 2: Contains multiple spelling errors, broken English phrasing, or poor sentence structure.
        - 1: Completely unintelligible, structurally broken, or not written in coherent English.

        [BATCH DATA TO EVALUATE]
        {records_text}

        You MUST process exactly {len(batch_records)} records.
        Return a JSON object that strictly conforms to the following JSON structure. Do NOT output the schema itself. You MUST return exactly {len(batch_records)} evaluations.

        {{
            "evaluations": [
                {{
                    "self_debate": "Briefly argue one reason why this specific question might be flawed, vague, or inaccurate before scoring.",
                    "technical_accuracy": {{ "passed": true, "score": 5, "details": "string explanation" }},
                    "schema_adherence": {{ "passed": true, "score": 5, "details": "string explanation" }},
                    "groundedness": {{ "passed": true, "score": 5, "details": "string explanation" }},
                    "persona_alignment": {{ "passed": true, "score": 5, "details": "string explanation" }},
                    "conciseness_clarity": {{ "passed": true, "score": 5, "details": "string explanation" }},
                    "information_density_clarity": {{ "passed": true, "score": 5, "details": "string explanation" }},
                    "fluency": {{ "passed": true, "score": 5, "details": "string explanation" }},
                    "genai_total_score": 35
                }}
            ]
        }}
        """

        config = GenerationConfig(
            temperature=0.1, 
            response_mime_type="application/json"
        )
        max_retries = 6
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content([eval_prompt], generation_config=config)
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
    """
    Chunks a list into segments of size n.

    Args:
        lst (list): The list to chunk.
        n (int): The chunk size.

    Yields:
        list: A sub-list of size up to n.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_system_prompt(row, include_schema: bool, include_results_summary: bool) -> str:
    """
    Generates the system prompt for evaluation based on configurable parameters.

    Args:
        row (dict): The data record (must contain 'persona', 'sql', 'schema', etc.).
        include_schema (bool): Whether to include schema information in context.
        include_results_summary (bool): Whether to include the expected result summary.

    Returns:
        str: The generated prompt text.
    """
    persona_desc = get_persona_description(row.get('persona', ''))
    
    prompt = f"""You are an expert natural language generation agent.
        Your persona for this task is: {row.get('persona', '')}.
        Persona Description: {persona_desc}

        Your goal is Reverse Translation: Given the inputs below, generate the precise, pure Natural Language question that would have produced each exact SQL query.
        """
            
            if include_results_summary:
                prompt += "Condition your generation heavily on the ACTUAL result shape to avoid vague or hallucinated intents. (e.g., if the query limits to 5, the question must say 'Which 5...').\n"
                
            prompt += "Importantly, craft each question to match your exact persona"
            if include_schema:
                prompt += " and refer to the specific Database Schema corresponding to each query.\n"
            else:
                prompt += ".\n"
                
            if include_schema:
                prompt += f"""
        [CONTEXT]
        - Schema Used: {row.get('schema', '')}
        """
                
            prompt += f"""
        - Query to Translate:
        --- Query ---
        SQL: {row.get('sql', '')}
        """
    
    if include_results_summary:
        prompt += f"Result Summary: {row.get('result_summary', '')}\n"
        
    return prompt


if __name__ == "__main__":
    # CONFIGURATION FLAGS
    INCLUDE_SCHEMA = True
    INCLUDE_RESULTS_SUMMARY = True

    
    INPUT_FILE_PATH = "./results/results-910_records/merged_stage2_results_mt.json"
    OUTPUT_FILE_PATH = "./results/results-910_records/merged_stage3_eval_results_pro_v2.json"
    

    print(f"Configuration:")
    print(f"  INCLUDE_SCHEMA: {INCLUDE_SCHEMA}")
    print(f"  INCLUDE_RESULTS_SUMMARY: {INCLUDE_RESULTS_SUMMARY}")
    print(f"  INPUT_FILE_PATH: {INPUT_FILE_PATH}")
    print(f"  OUTPUT_FILE_PATH: {OUTPUT_FILE_PATH}")

    # Initialize Vertex AI
    vertexai.init() 
    
    # Instantiate the Generative Model 
    eval_model = GenerativeModel("gemini-2.5-pro")

    print(f"\nLoading data from {INPUT_FILE_PATH}")
    try:
        with open(INPUT_FILE_PATH, "r") as f:
            stage2_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {INPUT_FILE_PATH}")
        stage2_data = []
    
    # Inject system_prompt dynamically
    for row in stage2_data:
        row['system_prompt'] = generate_system_prompt(row, INCLUDE_SCHEMA, INCLUDE_RESULTS_SUMMARY)

    if stage2_data:
        # Remove items that do not have necessary prompts
        valid_data = [r for r in stage2_data if r.get("system_prompt") and r.get("nl_question")]
        print(f"Starting evaluation of {len(valid_data)} valid items via batched LLM calls...")
        
        BATCH_SIZE = 20 # Process 20 records per LLM call
        MAX_WORKERS = 5 # Number of parallel threads
        
        batches = list(chunk_list(valid_data, BATCH_SIZE))
        
        def process_batch_thread(batch_idx, batch):
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
            futures = [executor.submit(process_batch_thread, i, b) for i, b in enumerate(batches)]
            for future in as_completed(futures):
                pass # wait for all to complete
                
        # Calculate scores after all threads finish
        complexity_scores = {"Simple": [], "Medium": [], "Complex": []}
        categories = ["technical_accuracy", "persona_alignment", "schema_adherence", "groundedness", "conciseness_clarity", "information_density_clarity", "fluency"]
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
                print(f"{comp}: {avg:.2f} / 35 ({len(scores)} records)")
            else:
                print(f"{comp}: N/A")
                
        print("\n--- Average Scores by Category ---")
        for cat, scores in metric_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"{cat}: {avg:.2f} / 5")
            else:
                print(f"{cat}: N/A")

        # Save to output file
        print(f"\nSaving results to {OUTPUT_FILE_PATH}...")
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        with open(OUTPUT_FILE_PATH, "w") as f:
            json.dump(stage2_data, f, indent=4)
            
        print("Done!")
