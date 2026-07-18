import os
import json
import sys
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from pydantic import BaseModel, Field

# from metrics.schema_coverage import calculate_schema_coverage
# from metrics.sql_uniqueness_rate import calculate_sur_masked
from stage3_unified_eval import evaluate_batch, get_persona_description, process_batch_thread, chunk_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Stage 3 Evaluation (Batched)')
    parser.add_argument('--input', type=str, required=True, help='Path to Stage 2 output JSON file')
    parser.add_argument('--output', type=str, help='Path to save evaluated output JSON file')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro', help='Generative model name')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of records per LLM call')
    parser.add_argument('--max-workers', type=int, default=5, help='Number of parallel threads')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not args.output:
        base_name = os.path.basename(args.input)
        name_part, _ = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.abspath(os.path.join(script_dir, f"../results/stage3/{name_part}_eval_{timestamp}.json"))

    print(f"Configuration:")
    print(f"  INPUT: {args.input}")
    print(f"  OUTPUT: {args.output}")
    print(f"  MODEL: {args.model}")
    print(f"  BATCH SIZE: {args.batch_size}")
    print(f"  MAX WORKERS: {args.max_workers}")

    # Initialize Vertex AI
    vertexai.init() 
    
    # Instantiate the Generative Model 
    eval_model = GenerativeModel(args.model)

    print(f"\nLoading data from {args.input}")
    try:
        with open(args.input, "r") as f:
            stage2_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {args.input}")
        sys.exit(1)
    
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
        
        batches = list(chunk_list(valid_data, args.batch_size))

        print(f"Processing {len(batches)} batches using {args.max_workers} threads...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
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

        # Calculate and print Dataset Quality Metrics
        # print("\n--- Dataset Quality Metrics ---")
        
        # # 1. Schema Coverage
        # try:
        #     TABLES_JSON_PATH = os.path.abspath(os.path.join(script_dir, "../tables-all.json"))
        #     df = pd.DataFrame(valid_data)
        #     if not df.empty:
        #         coverage_report, avg_sc = calculate_schema_coverage(df, TABLES_JSON_PATH)
        #         print(f"Average Schema Coverage: {avg_sc:.2%}")
        #         print("\nSchema Coverage Report per Database:")
        #         print(coverage_report[['db_id', 'tables_used', 'total_tables', 'columns_used', 'total_columns', 'schema_coverage']].to_string(index=False))
        #     else:
        #         print("Schema Coverage: N/A (no valid data)")
        # except Exception as e:
        #     print(f"Error calculating schema coverage: {e}")

        # # 2. SQL Uniqueness Rate
        # try:
        #     df = pd.DataFrame(valid_data)
        #     if not df.empty:
        #         sur_report, avg_sur = calculate_sur_masked(df)
        #         print(f"\nAverage SQL Uniqueness Rate (SUR): {avg_sur:.2%}")
        #         print("\nSQL Uniqueness Rate Report per Database:")
        #         print(sur_report[['db_id', 'total_queries', 'unique_queries', 'sur_masked']].to_string(index=False))
        #     else:
        #         print("SQL Uniqueness Rate: N/A (no valid data)")
        # except Exception as e:
        #     print(f"Error calculating SQL uniqueness rate: {e}")

        # # Save to output file
        # print(f"\nSaving results to {args.output}...")
        # os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # with open(args.output, "w") as f:
        #     json.dump(stage2_data, f, indent=4)
            
        # print("Done!")