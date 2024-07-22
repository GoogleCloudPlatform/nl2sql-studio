# """
#     RAG baased executor sample file
# """
# # import json
# from nl2sql.llms.vertexai import text_bison_32k
# from nl2sql.executors.linear_executor.core import CoreLinearExecutor

# # from nl2sql.tasks.sql_generation.rag_pydantic import RagSqlGenerator
# from nl2sql.tasks.sql_generation.rag_pydantic_1 import RagSqlGenerator

# llm = text_bison_32k()
# dataset_name = "sl-test-project-363109.zoominfo"
# bigquery_connection_string = "bigquery://sl-test-project-363109/zoominfo"


# class RAG_Executor:
#     """
#     Class to initialise and execute the RAG executor
#     """

#     def __init__(self):

#         self.PGPROJ = "sl-test-project-363109"
#         self.PGLOCATION = "us-central1"
#         self.PGINSTANCE = "nl2sql-test"
#         self.PGDB = "test-db"
#         self.PGUSER = "postgres"
#         self.PGPWD = "nl2sql-test"
#         self.project_id = "sl-test-project-363109"
#         self.dataset_id = "sl-test-project-363109.zoominfo"

#         self.rag_sql_generator = RagSqlGenerator(
#             llm=llm,
#             PGPROJ=self.PGPROJ,
#             PGLOCATION=self.PGLOCATION,
#             PGINSTANCE=self.PGINSTANCE,
#             PGDB=self.PGDB,
#             PGUSER=self.PGUSER,
#             PGPWD=self.PGPWD,
#             project_id=self.project_id,
#             dataset_id=self.dataset_id,
#         )

#         self.executor = CoreLinearExecutor.from_connection_string_map(
#             {
#                 dataset_name: bigquery_connection_string
#             },  # Unpack the inner dictionary here
#             core_table_selector=None,
#             core_column_selector=None,
#             core_sql_generator=self.rag_sql_generator,
#         )

#     def generate_sql(self, question=""):
#         """
#         Function to generate the SQL
#         """
#         print("\n\n", "=" * 25, "Executor Created", "=" * 25, "\n\n")
#         print("Executor ID :", self.executor.executor_id)

#         result2 = self.executor(db_name=dataset_name, question=question)
#         print("\n\n", "=" * 50, "Generated SQL", "=" * 50, "\n\n")
#         print("Result ID:", result2.result_id, "\n\n")
#         print(result2.generated_query)

#         return result2.result_id, result2.generated_query


# if __name__ == "__main__":

#     print("Inside the main body")
#     ragexec = RAG_Executor()
#     res_id, sql = ragexec.generate_sql(
#         "What is the total revenue for constuction industry? "
#     )
#     print("Exeuctor Id = ", res_id)
#     print("Gen SQLL = ", sql)
