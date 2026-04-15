import os
import sys
import asyncio

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_training_data import main

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration
    INPUT_FILE_PATH = os.path.abspath(os.path.join(script_dir, '../../results/stage2/s2_flash_synthetic_data_0_49_20260407_221624.json'))
    
    MODEL_TYPE = "qwen" # possible values: "llama", "gemini", "qwen"
    GENERATE_COT = True
    CONCURRENT_BATCH_SIZE = 10

    OUTPUT_FILE_PATH = INPUT_FILE_PATH[:-5]+f"_{MODEL_TYPE}_{'COT' if GENERATE_COT else 'no_COT'}.jsonl"

    print(f"Running create_training_data with input: {INPUT_FILE_PATH}")
    
    asyncio.run(
        main(
            INPUT_FILE_PATH,
            OUTPUT_FILE_PATH,
            model_type=MODEL_TYPE,
            generate_cot=GENERATE_COT,
            batch_size=CONCURRENT_BATCH_SIZE
        )
    )
