import textwrap
from vertexai.generative_models import FunctionDeclaration


list_tables_func = FunctionDeclaration(
    name="list_tables",
    description="List tables in a dataset that will help answer the user's question by choosing the table first. This will ensure, we do not create query on wrong table.",
    parameters={
        "type": "object",
    },
)

get_table_metadata_func = FunctionDeclaration(
    name="get_table_metadata",
    description="Get information about a table, including the description, schema, and number of rows that will help answer the user's question. Always use the fully qualified dataset and table names.",
    parameters={
        "type": "object",
        "properties": {
            "table_id": {
                "type": "string",
                "description": "Fully qualified ID of the table to get information about",
            }
        },
        "required": [
            "table_id",
        ],
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Get information from data in BigQuery using SQL queries. Also generates SQL by observing the error output from previous SQL execution if SQL query fails to execute",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query on a single line that will help give quantitative answers to the user's question when run on a BigQuery dataset and table. In the SQL query, always use the fully qualified dataset and table names. Do not use hardcoded dates in query, always use date or time functiond instead. If SQL query failed to execute in past, then observe the past error and correct that and then create the SQL.",
            }
        },
        "required": [
            "query",
        ],
    },
)

self_debug_func = FunctionDeclaration(
    name="debug_sql_query",
    description="Generates SQL by observing the error output from previous SQL execution if SQL query fails to execute",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "If SQL query failed to execute in past, then observe the past error and correct that and then create the SQL. In the SQL query, always use the fully qualified dataset and table names. Do not use hardcoded dates in query, always use date or time functiond instead. .",
            }
        },
        "required": [
            "query",
        ],
    },
)

plot_chart_auto_func = FunctionDeclaration(
    name="plot_chart_auto",
    description="extract the data from the SQL query output and writes python code to plot charts to best visualize that data.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": textwrap.dedent('''
First extract the neseccary data, best suitable chart-type for that data, title, axis and any other required information to plot the chart successfully.
Get the data in such a way that it does not take too many repetitions or large number of characters.
Then write python code to create a plot using this information in plotly module.
Finally the code should store the plot in fig variable and do NOT do fig.show() .
Example:
```
import plotly.express as px
import pandas as pd

# Data extracted from the SQL output
data = {'Year': [2018, 2019, 2020, 2021, 2022],
        'Sales': [15000, 18000, 16500, 22000, 25000]}

df = pd.DataFrame(data)
# Create the Plot
fig = px.line(df, x='Year', y='Sales', title='Annual Sales Trend')
# fig.show() # DO NOT show the figure yet, as it will be shown by the UI code
```
                '''), #TODO handle backslash and \n
            },
        },
        "required": [
            "code",
        ]
    }
)


plot_chart_func = FunctionDeclaration(
    name="plot_chart",
    description="get the data to plot chart in python plotly. Also deteremines the best suit plot-type for that data.",
    parameters={
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": textwrap.dedent('''The data output from SQL query in a standard json format which must be directly convertible to a pandas dataframe. It must not have any extra keys like "content".
                example data: ```"{'month': ['Jan', 'Feb', 'Mar'],
                'count': [62, 64, 20]
                }"``` '''),
            },
            "plot_type": {
                "type": "string",
                "enum": ["bar", "line", "pie", "scatter",],
                "description": "The type of plot to be generated best fit for understanding the data. e.g. bar, line, pie, scatter",
            },
            "title": {
                "type": "string",
                "description": "The title of the plot.",
            },
            "x_axis": {
                "type": "string",
                "description": "The column to be used for the x-axis of the plot.",
            },
            "y_axis": {
                "type": "string",
                "description": "The column to be used for the y-axis of the plot.",
            },
            # "xlabel": {
            #     "type": "string",
            #     "description": "The label for the x-axis.",
            # },
            # "ylabel": {
            #     "type": "string",
            #     "description": "The label for the y-axis.",
            # },
            # "legend": {
            #     "type": "string",
            #     "description": "The legend for the plot.",
            # },
            # "color": {
            #     "type": "string",
            #     "description": "The color of the plot.",
            # },
            # "size": {
            #     "type": "string",
            #     "description": "The size of the plot.",
            # },
            # "alpha": {
            #     "type": "string",
            #     "description": "The alpha of the plot.",
            # },
            # "marker": {
            #     "type": "string",
            #     "description": "The marker of the plot.",
            # },
            # "linestyle": {
            #     "type": "string",
            #     "description": "The linestyle of the plot.",
            # },
            # "linewidth": {
            #     "type": "string",
            #     "description": "The linewidth of the plot.",
            # },
            # "grid": {
            #     "type": "string",
            #     "description": "The grid of the plot.",
            # },
            # "xticks": {
            #     "type": "string",
            #     "description": "The xticks of the plot.",
            # },
            # "yticks": {
            #     "type": "string",
            #     "description": "The yticks of the plot.",
            # },
            # "xlim": {
            #     "type": "string",
            #     "description": "The xlim of the plot.",
            # },
            # "ylim": {
            #     "type": ".",
            # }
        },
        "required": [
            "data",
            "plot_type",
        ]
    }
)
