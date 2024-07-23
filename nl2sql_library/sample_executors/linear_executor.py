"""
    SQL Generation using Linear Executor
"""
import json

# import os
import sys
from os.path import dirname, abspath


# from nl2sql.datasets.base import Dataset
from nl2sql.executors.linear_executor.core import CoreLinearExecutor

sys.path.insert(1, dirname(dirname(abspath(__file__))))

dataset_name = "nl2sql_spider"  # @param {type:"string"}
f = open("../utils/spider_md_cache.json", encoding="utf-8")
spider_data = json.load(f)
data_dictionary_read = {
    "nl2sql_spider": {
        "description": "This dataset contains information about the concerts\
                      singers, country they belong to, stadiums where the  \
                      concerts happened",
        "tables": spider_data,
    },
}


# # Executor Setup Code

bigquery_connection_string = (
    "bigquery://sl-test-project-363109/nl2sql_spider"  # @param {type:"string"}
)

executor = CoreLinearExecutor.from_connection_string_map(
    {
        dataset_name: bigquery_connection_string,
    },
    data_dictionary=data_dictionary_read,
)

print("\n\n", "=" * 25, "Executor Created", "=" * 25, "\n\n")
print("Executor ID :", executor.executor_id)

# Now run the executor with a sample question
result = executor(
    db_name=dataset_name,
    question="What is the average, minimum, and maximum age\
             of all singers from France ?",  # @param {type:"string"}
)
print("\n\n", "=" * 50, "Generated SQL", "=" * 50, "\n\n")
print("Result ID:", result.result_id, "\n\n")
print(result.generated_query)
