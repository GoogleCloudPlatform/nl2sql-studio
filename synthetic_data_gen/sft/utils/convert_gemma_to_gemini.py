import json
import os
import sys
from tqdm import tqdm

def convert_gemma_to_gemini(input_path: str, output_path: str):
    """
    Converts a dataset file from Gemma fine-tuning format to Gemini fine-tuning format.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return False

    print(f"Converting: {os.path.basename(input_path)}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Count lines for the progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    converted_count = 0
    errors_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="Converting records"):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Extract messages from Gemma format
                if "messages" not in record:
                    errors_count += 1
                    continue
                
                messages = record["messages"]
                
                # Find system, user, and assistant messages
                system_prompt = ""
                user_content = ""
                assistant_content = ""
                
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        user_content = content
                    elif role == "assistant" or role == "model":
                        assistant_content = content
                
                # Format into Gemini style
                # user_content = f"{system_prompt}\n\nDATABASE SCHEMA:\njson\n{schema_json_string}\n\n\nQuestion: {question}"
                # Since gemma's user_content is already `DATABASE SCHEMA:...`, we just prepend system_prompt + \n\n
                if system_prompt:
                    gemini_user_text = f"{system_prompt}\n\n{user_content}"
                else:
                    gemini_user_text = user_content

                gemini_record = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": gemini_user_text}]
                        },
                        {
                            "role": "model",
                            "parts": [{"text": assistant_content}]
                        }
                    ]
                }
                
                outfile.write(json.dumps(gemini_record, ensure_ascii=False) + "\n")
                converted_count += 1
                
            except Exception as e:
                errors_count += 1
                print(f"\nError parsing line: {e}")

    print(f"Success: Converted {converted_count} records.")
    if errors_count > 0:
        print(f"Warning: Encountered {errors_count} errors/skipped records.")
    return True

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(current_dir, "../results/stage2"))
    
    # Default files to convert
    files_to_convert = [
        "results_sft-input_training_data_9k_records_stage2_merged_9k_combined_gemma_COT_train.jsonl",
        "results_sft-input_training_data_9k_records_stage2_merged_9k_combined_gemma_COT_val.jsonl"
    ]
    
    for filename in files_to_convert:
        input_file = os.path.join(results_dir, filename)
        
        # Output file name changes 'gemma' to 'gemini'
        output_filename = filename.replace("_gemma_", "_gemini_")
        if output_filename == filename:
            output_filename = filename.replace(".jsonl", "_gemini.jsonl")
            
        output_file = os.path.join(results_dir, output_filename)
        
        if os.path.exists(input_file):
            convert_gemma_to_gemini(input_file, output_file)
            print("-" * 50)
        else:
            print(f"File not found: {input_file}")

if __name__ == "__main__":
    # Allow specifying a custom input and output from command line arguments
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
        if len(sys.argv) > 2:
            out_path = sys.argv[2]
        else:
            out_path = in_path.replace("_gemma_", "_gemini_")
            if out_path == in_path:
                out_path = in_path.replace(".jsonl", "_gemini.jsonl")
        convert_gemma_to_gemini(in_path, out_path)
    else:
        main()
