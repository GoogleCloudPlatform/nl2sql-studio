# nl2sql_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools import AgentTool, load_artifacts

from . import bq_tools, config

# --- Define the Data Visualization Agent ---
visualization_agent = LlmAgent(
    name="VisualizationAgent",
    model=config.MODEL,
    description="Specialist for creating data visualizations. Takes JSON data and generates Python code for plotting.",
    instruction="""
        You are a data visualization expert. You will receive data in a JSON string format from the session state variable `sql_result`.
        {sql_result}
        Your task is to write Python code to create the best possible chart (e.g., bar, line, pie) to visualize this data.
        The code you write will be executed in a secure environment to generate a chart.
        The code must:
        -   Parse the JSON string from the `sql_result` variable into a pandas DataFrame.
        -   Create a plot using a Python visualization library like Matplotlib or Seaborn.
        -   Save the chart to a file named 'chart.png'. This file will be automatically rendered in the UI.
        -   Ensure the plot is self-contained and does not try to show the plot interactively (e.g., don't use `plt.show()`).

        Example of saving a plot:
        ```python
        import matplotlib.pyplot as plt
        # ... (create your plot)
        plt.savefig('chart.png')
        ```
    """,
    code_executor=BuiltInCodeExecutor(),
)

# --- Define the SQL Specialist Agent ---
root_agent = LlmAgent(
    name="SQLAgent",
    model=config.MODEL,
    description="The root NL2SQL agent. Specialist for all BigQuery tasks. It can list tables, get table schemas, execute SQL queries, plot data as charts",
    instruction="""
        You are an expert BigQuery SQL developer. Your goal is to answer user questions by querying the database.
        Follow these guidelines:
        -  Greet the user and explain how you can help user to guide him get started.
        
        Based on user's question: 
        -  First, use the `list_tables` tool to see what tables are available.
        -  Based on the table descriptions and the user's question, use the `get_table_metadata` tool to find the schema of the most relevant table(s).
        -  Construct an accurate and efficient SQL query to answer the user's question using the table schema.
        -  Execute the query using the `execute_sql` tool.
        -  If the `execute_sql` tool returns an error, analyze the error message, correct your SQL query, and execute it again. Do not try more than twice.
        -  Summarize the final result from the tool in a clear, natural language response.
        - If the result is suitable for visualization, also use the `VisualizationAgent` Agent to generate a chart.

        NOTE: proactively call any tools at your disposal whenever it will be useful, withoout waiting or asking for info to user.
    """,
    tools=[
        bq_tools.list_tables,
        bq_tools.get_table_metadata,
        bq_tools.execute_sql,
        AgentTool(agent=visualization_agent),
        load_artifacts,
    ],
)
