import asyncio
import time
import random
import statistics

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

client = genai.Client(
    vertexai=True,
    project="proj-kous",
    location="us-central1",
)

generate_content_config = types.GenerateContentConfig(
    temperature = 0.1,
    top_p = 0.95,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=-1,
    ),
  )


def generate(query: str, model: str = MODEL):
    contents = [
        types.Content(
            role="user",
            parts=[
            types.Part.from_text(text=query)
            ]
        ),
    ]

    return client.models.generate_content(
        model = MODEL,
        contents = contents,
        config = generate_content_config,
    ).text


async def generate_async(query: str, model: str = MODEL):
    """Asynchronous wrapper for the generate function."""
    loop = asyncio.get_running_loop()
    # Run the synchronous `generate` function in a thread pool executor
    return await loop.run_in_executor(None, generate, query, model)


async def run_experiment(num_parallel_calls: int, baseline_time: float | None = None) -> tuple[float, float]:
    """
    Runs a number of `generate` calls in parallel and measures execution time.
    Returns a tuple of (total_time, effective_parallelism).
    """
    start_time = time.monotonic()

    tasks = [
        generate_async(f"hi! This is request {i+1}. Random number: {random.random()}")
        for i in range(num_parallel_calls)
    ]
    # Using return_exceptions=True to prevent one failed call from stopping the whole batch
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.monotonic()
    total_time = end_time - start_time

    effective_parallelism = 0.0

    # Filter out exceptions to count successful calls
    successful_calls = sum(1 for r in results if not isinstance(r, Exception))
    if successful_calls < num_parallel_calls:
        # This can happen under very high load due to rate limiting
        failed_calls = num_parallel_calls - successful_calls
        print(f"  (Note: {failed_calls}/{num_parallel_calls} calls failed in this run)")

    if baseline_time and successful_calls > 1:
        # The number of sequential "waves" it took to process all calls.
        # If truly parallel, this number should be close to 1.
        execution_waves = total_time / baseline_time
        # The effective number of calls that ran in a single "wave".
        # This estimates the true parallelism.
        effective_parallelism = successful_calls / execution_waves
    elif num_parallel_calls == 1:
        effective_parallelism = 1.0

    return total_time, effective_parallelism


async def main():
    """Main function to run experiments with increasing parallelism."""
    RUNS_PER_COUNT = 30
    # A list of different numbers of parallel calls to test
    parallel_counts = [2, 5]#, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print(f"--- Establishing baseline using {RUNS_PER_COUNT} individual runs ---")
    baseline_times = []
    for _ in range(RUNS_PER_COUNT):
        time_taken, _ = await run_experiment(1)
        baseline_times.append(time_taken)

    baseline_time = statistics.mean(baseline_times)
    print(f"Baseline established:")
    print(f"  Mean: {baseline_time:.2f}s")
    print(f"  Min:  {min(baseline_times):.2f}s")
    print(f"  Max:  {max(baseline_times):.2f}s")

    # Run experiments for all counts, including 1, for a complete table
    for count in parallel_counts:
        print(f"\n--- Running experiment for {count} parallel calls ({RUNS_PER_COUNT} times) ---")
        total_times = []
        parallelisms = []
        for i in range(RUNS_PER_COUNT):
            print(f"  Running batch {i+1}/{RUNS_PER_COUNT}...", end='\r')
            total_time, eff_parallelism = await run_experiment(count, baseline_time=baseline_time)
            total_times.append(total_time)
            if eff_parallelism > 0:
                parallelisms.append(eff_parallelism)
        print(" " * 30, end='\r') # Clear the line

        mean_time = statistics.mean(total_times)
        mean_parallelism = statistics.mean(parallelisms) if parallelisms else 0
        queued_calls = max(0, round(count - mean_parallelism))

        print(f"Results for {count} parallel calls:")
        print(f"  Total Time (s):  Min={min(total_times):.2f}, Max={max(total_times):.2f}, Mean={mean_time:.2f}")
        if count > 1:
            print(f"  True Parallelism:  Min={min(parallelisms):.1f}, Max={max(parallelisms):.1f}, Mean={mean_parallelism:.1f}")
            print(f"  Avg. Queued Calls: {queued_calls}")

if __name__ == "__main__":
    asyncio.run(main())
