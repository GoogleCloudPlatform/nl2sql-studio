import json
import numpy as np

def calculate_complexity(sql_obj):
    """
    Calculates a complexity score for a given SQL object.
    """
    if not isinstance(sql_obj, dict):
        return 0

    score = 0
    
    # Score for JOINs
    if 'from' in sql_obj and 'conds' in sql_obj['from']:
        # Each list inside 'conds' is a join condition
        score += 2 * len([c for c in sql_obj['from']['conds'] if isinstance(c, list)])

    # Score for GROUP BY and ORDER BY
    if sql_obj.get('groupBy'):
        score += 1
    if sql_obj.get('orderBy'):
        score += 1

    # Score for HAVING
    if sql_obj.get('having'):
        score += 2

    # Score for set operators and recursively score their sub-queries
    for op in ['intersect', 'union', 'except']:
        if sql_obj.get(op):
            score += 4
            score += calculate_complexity(sql_obj[op])

    # Score for subqueries in WHERE and HAVING
    for clause in ['where', 'having']:
        if sql_obj.get(clause):
            for cond in sql_obj[clause]:
                if isinstance(cond, list):
                    # The value part of a condition can be a subquery
                    if isinstance(cond[3], dict):
                        score += 3
                        score += calculate_complexity(cond[3])

    return score

def _partition_and_sample(data_with_scores, total_expected_filtered_count, distribution):
    """
    Partitions data into complexity strata and samples from them.
    """
    # Sort by complexity score
    data_with_scores.sort(key=lambda x: x['complexity_score'])
    n_total = len(data_with_scores)

    # Define strata based on percentiles (e.g., bottom 30% easy, next 40% medium, top 30% hard)
    easy_end_idx = int(n_total * 0.3)
    medium_end_idx = int(n_total * 0.7)

    easy_queries = data_with_scores[:easy_end_idx]
    medium_queries = data_with_scores[easy_end_idx:medium_end_idx]
    hard_queries = data_with_scores[medium_end_idx:]

    # Calculate how many to sample from each stratum
    num_hard = int(total_expected_filtered_count * distribution.get('hard', 0.0))
    num_medium = int(total_expected_filtered_count * distribution.get('medium', 0.0))
    num_easy = int(total_expected_filtered_count * distribution.get('easy', 0.0))

    # Print the distribution of complexity in the data
    print(f"Total queries: {n_total}, Easy: {len(easy_queries)}, Medium: {len(medium_queries)}, Hard: {len(hard_queries)}")

    # Print stats for each complexity level
    for name, dataset in [('Easy', easy_queries), ('Medium', medium_queries), ('Hard', hard_queries)]:
        if dataset:
            scores = [d['complexity_score'] for d in dataset]
            print(f"  - {name} complexity scores: Min: {np.min(scores)}, Median: {np.median(scores)}, Max: {np.max(scores)}")
        else:
            print(f"  - No queries in '{name}' level.")


    # To get a representative sample, we can take items evenly from each stratum
    # For simplicity, we'll take from the 'top' of each stratum for hard/medium, and 'bottom' for easy
    
    # Sample from hard (most complex of the hard group)
    hard_sample = hard_queries[-num_hard:] if num_hard > 0 else []
    # Sample from medium (most complex of the medium group)
    medium_sample = medium_queries[-num_medium:] if num_medium > 0 else []
    # Sample from easy (least complex of the easy group)
    easy_sample = easy_queries[:num_easy] if num_easy > 0 else []

    print(f"Sampling {len(hard_sample)} hard, {len(medium_sample)} medium, and {len(easy_sample)} easy queries.")

    return hard_sample + medium_sample + easy_sample

def filter_complex_queries(input_path, output_path, total_expected_filtered_count, distribution={'hard': 0.4, 'medium': 0.5, 'easy': 0.1}):
    """
    Filters a JSON file of SQL queries to get a specific count with a desired
    distribution of complexity (hard, medium, easy).

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to save the filtered JSON file.
        total_expected_filtered_count (int): The target number of queries in the output.
        distribution (dict): A dictionary with keys 'hard', 'medium', 'easy' and
                             values as their proportion (summing to 1.0).
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'")
        return

    if not isinstance(data, list):
        print("Error: Expected a list of objects in the JSON file.")
        return

    if not data:
        print("No data to process.")
        return

    # Calculate complexity and add it to each item
    data_with_scores = [{'complexity_score': calculate_complexity(item.get('sql', {})), **item} for item in data]
    
    filtered_data = _partition_and_sample(data_with_scores, total_expected_filtered_count, distribution)

    print(f"Original number of items: {len(data)}")
    print(f"Number of complex items kept: {len(filtered_data)}")
    print(f"Removed {len(data) - len(filtered_data)} simpler items.")

    # Save the filtered data
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"Filtered data saved to {output_path}")


if __name__ == '__main__':
    spider_path = 'spider_data/'
    input_file = spider_path + 'train_spider.json'
    output_file = spider_path + 'train_spider_filtered.json'

    # Define the target size and distribution for the filtered dataset
    TARGET_COUNT = 6000
    DISTRIBUTION = {'hard': 0.4, 'medium': 0.5, 'easy': 0.1}

    print(f"Filtering for a target of {TARGET_COUNT} queries with distribution: {DISTRIBUTION}")
    filter_complex_queries(input_file, output_file, total_expected_filtered_count=TARGET_COUNT, distribution=DISTRIBUTION)
