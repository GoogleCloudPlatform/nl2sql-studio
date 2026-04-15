import os
import vertexai
import json
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage3_unified_eval import evaluate_batch, get_persona_description, process_batch_thread, chunk_list

if __name__ == "__main__":
    # CONFIGURATION FLAGS
    INPUT_FILE_PATH = "./results/stage2/s2_flash_synthetic_data_0_49_20260407_221624.json"
    OUTPUT_FILE_PATH = "./results/stage3/s3_flash_synthetic_data_0_49_20260407_221624.json"
    MODEL_NAME = "gemini-2.5-pro"
    

    print(f"Configuration:")
    print(f"  INPUT_FILE_PATH: {INPUT_FILE_PATH}")
    print(f"  OUTPUT_FILE_PATH: {OUTPUT_FILE_PATH}")

    # Initialize Vertex AI
    vertexai.init() 
    
    # Instantiate the Generative Model 
    eval_model = GenerativeModel(MODEL_NAME)

    print(f"\nLoading data from {INPUT_FILE_PATH}")
    try:
        with open(INPUT_FILE_PATH, "r") as f:
            stage2_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {INPUT_FILE_PATH}")
        stage2_data = []
    
    # Inject system_prompt dynamically
    for row in stage2_data:
        row['golden_context'] = (
             f"--- GOLDEN CONTEXT (GROUND TRUTH) ---\n"
            f"Target Persona: {row.get('persona', '')}\n"
            f"Persona Description: {get_persona_description(row.get('persona', ''))}\n\n"
            f"Database Schema:\n{row.get('schema', row.get('schema', ''))}\n\n"
            f"Original SQL Query:\n{row.get('sql', '')}\n\n"
            f"Result Summary (Expected Output Shape):\n{row.get('result_summary', '')}\n"
        )
        

    if stage2_data:
        # Remove items that do not have necessary prompts
        valid_data = [r for r in stage2_data if r.get("golden_context") and r.get("nl_question")]
        print(f"Starting evaluation of {len(valid_data)} valid items via batched LLM calls...")
        
        BATCH_SIZE = 5 # Process 5 records per LLM call
        MAX_WORKERS = 5 # Number of parallel threads
        
        batches = list(chunk_list(valid_data, BATCH_SIZE))
        

        print(f"Processing {len(batches)} batches using {MAX_WORKERS} threads...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_batch_thread, i, b, eval_model, len(batches)) for i, b in enumerate(batches)]
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