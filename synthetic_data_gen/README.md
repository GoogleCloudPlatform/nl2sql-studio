# NL2SQL Golden Dataset Generator

## Overview
This project implements an automated, multi-agent pipeline designed to generate high-quality, verified Natural Language to SQL (NL2SQL) query pairs. By validating generated queries against a live database and employing LLM-based evaluation, the system creates a diverse "Golden Truth Dataset." This curated data is optimized for fine-tuning models on complex text-to-SQL tasks and provides a rigorous foundation for benchmarking model performance.

## Architecture
The core architecture operates across three distinct phases to ensure that generated SQL is syntactically correct, contextually relevant, and accurately mapped to natural language questions.

### Phase 1: Verified Code Factory
The goal of this initial phase is to generate syntactically correct and executable SQL queries directly from a target database schema.
* **Smart Sampler:** Ingests the database JSON Schema and iterates through it table-by-table to ensure comprehensive domain coverage.
* **Agent A (Architect):** Generates valid SQL queries across varying levels of complexity (Simple, Medium, Complex).
* **Execution Engine:** The generated SQL is executed directly against the live database. Any queries that result in execution errors or return 0 rows are immediately discarded.
* **Result Summarizer:** For successful queries, an LLM analyzes the output and creates a summary describing the "shape" and context of the retrieved data.

### Phase 2: Context-Aware Translation
This phase bridges the gap between the verified technical code and human intent, effectively reverse-engineering the user prompt.
* **Agent B (Storyteller):** Takes the verified SQL query and the Result Summary as inputs to generate a natural, conversational Natural Language (NL) question that accurately reflects the underlying data extraction.

### Phase 3: The Quality Gate
The final phase acts as an automated evaluator and curator to ensure the resulting dataset is both accurate and diverse.
* **Agent C (The Judge):** Evaluates the resulting triplet `(NL Question, SQL Query, Result Summary)` and assigns a Confidence Score based on the logical alignment between the question and the SQL.
* **Diversity Binning:** High-scoring triplets are bucketed by topic or structural similarity to prevent the dataset from skewing toward repetitive query patterns.
* **Golden Truth Selection:** The system selects the Top *N* queries per bin, outputting the final, highly curated dataset ready for downstream model training.