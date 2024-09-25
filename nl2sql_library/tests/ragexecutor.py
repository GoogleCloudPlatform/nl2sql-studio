"""
    Rag Executor test file
"""
# import json
from nl2sql.llms.vertexai import text_bison_32k
from nl2sql.executors.linear_executor.core import CoreLinearExecutor
from nl2sql.tasks.sql_generation.rag import RagSqlGenerator

llm = text_bison_32k()

# dataset_name = "HHS_Program_Counts"
# bigquery_connection_string = \
# "bigquery://cdii-poc/HHS_Program_Counts_nl2sql_views"

dataset_name = "sl-test-project-363109.zoominfo"
bigquery_connection_string = "bigquery://sl-test-project-363109/zoominfo"

executor = CoreLinearExecutor.from_connection_string_map(
    {dataset_name: bigquery_connection_string},
    core_table_selector=None,
    core_column_selector=None,
    core_sql_generator=RagSqlGenerator(llm=llm),
)

print("\n\n", "=" * 25, "Executor Created", "=" * 25, "\n\n")
print("Executor ID :", executor.executor_id)

result2 = executor(
    db_name=dataset_name,
    question="What is the total revenue for constuction industry? ",
)
print("\n\n", "=" * 50, "Generated SQL", "=" * 50, "\n\n")
print("Result ID:", result2.result_id, "\n\n")
print(result2.generated_query)
