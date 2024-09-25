"""
    Rag Executor test file
"""

from nl2sql.llms.vertexai import text_bison_32k
from nl2sql.executors.linear_executor.core import CoreLinearExecutor

# from nl2sql.tasks.sql_generation.rag_pydantic import RagSqlGenerator
from nl2sql.tasks.sql_generation.rag_pydantic_1 import RagSqlGenerator

llm = text_bison_32k()

dataset_name = "sl-test-project-363109.zoominfo"
bigquery_connection_string = "bigquery://sl-test-project-363109/zoominfo"

PGPROJ = "sl-test-project-363109"
PGLOCATION = "us-central1"
PGINSTANCE = "nl2sql-test"
PGDB = "test-db"
PGUSER = "postgres"
PGPWD = "nl2sql-test"
project_id = "sl-test-project-363109"
dataset_id = "sl-test-project-363109.zoominfo"

rag_sql_generator = RagSqlGenerator(
    llm=llm,
    PGPROJ=PGPROJ,
    PGLOCATION=PGLOCATION,
    PGINSTANCE=PGINSTANCE,
    PGDB=PGDB,
    PGUSER=PGUSER,
    PGPWD=PGPWD,
    project_id=project_id,
    dataset_id=dataset_id,
)


executor = CoreLinearExecutor.from_connection_string_map(
    {dataset_name: bigquery_connection_string},
    # Unpack the inner dictionary here
    core_table_selector=None,
    core_column_selector=None,
    core_sql_generator=rag_sql_generator,
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
