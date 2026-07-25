#!/usr/bin/env python3
"""
Unified Self-Correction Runner (run_autonomous_loop.py)
Bridges failure analysis with error-driven data augmentation to synthesize SFT training queries.
"""
import argparse
import logging
import os
import sys

# Ensure module resolution for data_augmenter scripts across repository root and subdirectories
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, "synthetic_data_gen", "data_augmenter"))
sys.path.append(os.path.join(root_dir, "synthetic_data_gen"))

try:
    from autonomous_augmentor import AutonomousFailureAnalyzer
    from error_driven_augmentor import ErrorDrivenAugmentor
except ImportError as e:
    try:
        from synthetic_data_gen.data_augmenter.autonomous_augmentor import AutonomousFailureAnalyzer
        from synthetic_data_gen.data_augmenter.error_driven_augmentor import ErrorDrivenAugmentor
    except ImportError:
        raise ImportError(f"Could not import AutonomousFailureAnalyzer or ErrorDrivenAugmentor: {e}")


def main():
    parser = argparse.ArgumentParser(description="Unified Self-Correction Runner for NL2SQL Pipeline")
    parser.add_argument(
        "--failed-evals",
        type=str,
        required=True,
        help="Path to the input JSONL/JSON file containing Stage 3 failed evaluation results"
    )
    parser.add_argument(
        "--output-sft",
        type=str,
        required=True,
        help="Path where the newly synthesized, error-augmented SFT training queries should be saved"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Optional logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("run_autonomous_loop")

    logger.info("Starting Unified Self-Correction Runner...")
    logger.info(f"Failed Evals Input: {args.failed_evals}")
    logger.info(f"Output SFT Path: {args.output_sft}")

    try:
        logger.info("Instantiating AutonomousFailureAnalyzer...")
        analyzer = AutonomousFailureAnalyzer()

        logger.info("Instantiating ErrorDrivenAugmentor...")
        augmentor = ErrorDrivenAugmentor(analyzer=analyzer)

        logger.info("Invoking dataset augmentation...")
        results = augmentor.augment_dataset(failed_evals=args.failed_evals, output_sft=args.output_sft)

        logger.info(f"Successfully generated {len(results)} error-augmented queries and saved to {args.output_sft}.")
    except Exception as e:
        logger.error(f"Error during autonomous self-correction loop execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
