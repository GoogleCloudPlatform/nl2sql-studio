"""
    SQL Generation using Linear Executor
"""
import json

# import os
import sys
from os.path import dirname, abspath

sys.path.insert(1, dirname(dirname(abspath(__file__))))

# from nl2sql.datasets.base import Dataset
from nl2sql.executors.linear_executor.core import CoreLinearExecutor

dataset_name = "zoominfo"  # @param {type:"string"}
f = open("../utils/zoominfo_tables.json", encoding="utf-8")
zi = json.load(f)

data_dictionary_read = {
    "zoominfo": {
        "description": "This dataset contains information of Zoominfo Data\
                  with details on headquarters, marketing professionaals and\
                    providng tuition services.",
        "tables": zi,
    },
}


# # Executor Setup Code

bigquery_connection_string = (
    "bigquery://sl-test-project-363109/zoominfo"  # @param {type:"string"}
)

executor = CoreLinearExecutor.from_connection_string_map(
    {
        dataset_name: bigquery_connection_string,
    },
    data_dictionary=data_dictionary_read,
)

print("\n\n", "=" * 25, "Executor Created", "=" * 25, "\n\n")
print("Executor ID :", executor.executor_id)

## Now run the executor with a sample question
result = executor(
    db_name=dataset_name,
    question="What is the revenue of construction industry?",  # @param {type:"string"}
)
print("\n\n", "=" * 50, "Generated SQL", "=" * 50, "\n\n")
print("Result ID:", result.result_id, "\n\n")
print(result.generated_query)
