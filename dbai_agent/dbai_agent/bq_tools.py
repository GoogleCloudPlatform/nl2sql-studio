# nl2sql_agent/bq_tools.py
import json
from google.cloud import bigquery
from google.adk.tools.tool_context import ToolContext
from .db_metadata import BigQueryMetadata
from . import config

# --- Initialize clients and metadata ---
PROJECT_ID = config.GOOGLE_CLOUD_PROJECT
DATASET_ID = config.DATASET_ID
TABLES_LIST = config.TABLES_LIST

# Instantiate the metadata handler once
metadata_handler = BigQueryMetadata(PROJECT_ID, DATASET_ID, TABLES_LIST)
bq_client = bigquery.Client(project=PROJECT_ID)

# --- Define ADK Function Tools ---

def list_tables(tool_context: ToolContext) -> dict:
    """
    Lists all available tables in the BigQuery dataset along with their descriptions.
    Use this tool first to understand which tables you can query.
    """
    print("--- Tool: list_tables executed ---")
    return metadata_handler.get_all_tables_info()

def get_table_metadata(table_id: str, tool_context: ToolContext) -> dict:
    """
    Gets detailed schema information for a specific table.
    Use this tool after list_tables to understand the columns of a table
    you want to query.
    Args:
        table_id: The name of the table (e.g., 'orders').
    """
    print(f"--- Tool: get_table_metadata executed for table: {table_id} ---")
    return metadata_handler.get_table_schema(table_id)

def execute_sql(query: str, tool_context: ToolContext) -> dict:
    """
    Executes a SQL query against the BigQuery database and returns the result.
    Only use this after you have the table schema. Ensure the query is syntactically correct.
    In the SQL query, always use the fully qualified table names, dont use the project_id.dataset_id.
    Args:
        query: The SQL query to execute.
    """
    print(f"--- Tool: execute_sql executed with query: {query} ---")
    try:
        job_config = bigquery.QueryJobConfig(
            default_dataset=f'{PROJECT_ID}.{DATASET_ID}'
            )
        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()
        result_list = [dict(row) for row in results]

        # Save result to session state for the visualization agent to use
        tool_context.state["sql_result"] = json.dumps(result_list, default=str)

        return {"status": "success", "result": result_list}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}