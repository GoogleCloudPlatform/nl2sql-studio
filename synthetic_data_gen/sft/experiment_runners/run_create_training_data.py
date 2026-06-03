import os
import sys
import asyncio
import argparse

# Add the sft directory to sys.path to import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_training_data import main

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create SFT training data from Stage 2 output.')
    parser.add_argument('--input', type=str, required=True, help='Path to Stage 2 output JSON file')
    parser.add_argument('--prompt', type=str, required=True, help='Path to COT prompt template file')
    parser.add_argument('--model-type', type=str, default='gemma', choices=['llama', 'gemini', 'qwen', 'gemma'], help='Target model formatting')
    parser.add_argument('--generate-cot', action=argparse.BooleanOptionalAction, default=False, help='Generate Chain of Thought reasoning steps')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of parallel calls to Gemini')
    args = parser.parse_args()

    # Derive output file name based on model type and CoT setting using the original old logic
    args.output = args.input[:-5] + f"_{args.model_type}_{'new_COT' if args.generate_cot else 'no_COT'}.jsonl"

    print(f"Configuration:")
    print(f"  INPUT: {args.input}")
    print(f"  OUTPUT: {args.output}")
    print(f"  PROMPT TEMPLATE: {args.prompt}")
    print(f"  MODEL TYPE: {args.model_type}")
    print(f"  GENERATE COT: {args.generate_cot}")
    print(f"  BATCH SIZE: {args.batch_size}")

    # Run the async main function
    asyncio.run(
        main(
            args.input,
            args.output,
            model_type=args.model_type,
            generate_cot=args.generate_cot,
            batch_size=args.batch_size,
            prompt_path=args.prompt
        )
    )

