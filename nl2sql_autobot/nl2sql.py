import time
from google.cloud import bigquery
import os, json
import vertexai
import pandas as pd
import plotly.express as px
from vertexai import generative_models
from vertexai.generative_models import (
    Content,
    GenerativeModel,
    Part,
    Tool,
    # ToolConfig
)
import streamlit as st
from bot_functions import *


PROJECT_ID = "proj-kous"
# DATASET_ID = "nl2sql_fiserv"
# TABLES_LIST = [] 
DATASET_ID = "Albertsons"
TABLES_LIST = ['camain_oracle_hcm', 'camain_ps']
METDATA_CACHE_PATH = f"./metadata_cache_{DATASET_ID}.json"

vertexai.init(project=PROJECT_ID)
client = bigquery.Client(project=PROJECT_ID)

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

gemini = GenerativeModel("gemini-1.5-pro-001",
                        generation_config={"temperature": 0.05},
                        safety_settings=safety_settings,
                        )

sql_query_tool = Tool(
    function_declarations=[
        list_tables_func,
        get_table_metadata_func,
        sql_query_func,
        plot_chart_auto_func,
        # plot_chart_func,
    ],
)

model = GenerativeModel("gemini-1.5-pro-001",
                        generation_config={"temperature": 0.05},
                        safety_settings=safety_settings,
                        tools=[sql_query_tool],
                        )

SYSTEM_PROMPT = """ You are a fluent person who efficiently communicates with the user over
 different Database queries. Please always call the functions at your disposal whenever you need to
 know something, and do not reply unless you feel you have all information to answer the question satisfactorily.
Only use information that you learn from BigQuery, do not make up information.
"""
# SYSTEM_PROMPT = ''

class Response:
    def __init__(self, text, interim_steps) -> None:
        self.text = text
        self.interim_steps = interim_steps


def create_metadata_cache(dataset_id, tables_list):
    gen_description_prompt = """Based on the columns information of this table.
    Generate a very brief description for this table.
    TABLE: {table_id}
    columns_info: {columns_info}"""

    if tables_list == []:
        api_response = client.list_tables(dataset_id)
        tables_list = [table.table_id for table in api_response]

    metadata = {}
    for table_id in tables_list:
        columns_info = client.get_table(f'{dataset_id}.{table_id}').to_api_repr()['schema']
        ## remove unwanted details like 'mode'
        for field in columns_info.get('fields', []):
            field.pop('mode', None)

        metadata[table_id] = {}
        metadata[table_id]["table_name"] = table_id
        metadata[table_id]["columns_info"] = columns_info
        prompt = gen_description_prompt.format(table_id=table_id, columns_info=columns_info)
        metadata[table_id]["table_description"] = gemini.generate_content(prompt).text

    return metadata

if not os.path.exists(METDATA_CACHE_PATH):
    metadata = create_metadata_cache(DATASET_ID, TABLES_LIST)
    with open(METDATA_CACHE_PATH, 'w') as f:
        f.write(json.dumps(metadata))
else:
    with open(METDATA_CACHE_PATH, 'r') as f:
        metadata = json.load(f)


def api_list_tables(DATASET_ID):
    # api_response = client.list_tables(DATASET_ID)
    # api_response = str([table.table_id for table in api_response])
    try:
        api_response = metadata
    except:
        api_response = TABLES_LIST
    return api_response

def api_get_table_metadata(table_id):
    try:
        table_metadata = str(metadata[table_id])
    except:
        ## if table_id is in form of dataset_id.table_id then remove dataset_id
        table_metadata = str(metadata[table_id.split('.')[-1]])
    return table_metadata

def execute_sql_query(query):
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=100000000,
        default_dataset=f'{PROJECT_ID}.{DATASET_ID}'
        )
    try:
        cleaned_query = query.replace("\\n", " ").replace("\n", "").replace("\\", "")
        query_job = client.query(cleaned_query, job_config=job_config)
        api_response = query_job.result()
        api_response = str([dict(row) for row in api_response])
        api_response = api_response.replace("\\", "").replace("\n", "")
    except Exception as e:
        api_response = f"{str(e)}"

    return api_response

def api_plot_chart(plot_params):
    data = plot_params['data']
    if isinstance(data, list):
        data = data[0]
    print('_'*100, data, '_'*100)
    data = data.replace('None', '-1')
    if 'content' in str(data):
        df = pd.DataFrame(json.loads(data['content'][0]))
    elif type(data) == str:
        df = pd.DataFrame(eval(data))
    else:
        df = pd.DataFrame(json.loads(str(data)))
    fig = px.bar(df, x=plot_params['x_axis'], y=plot_params['y_axis'], title=plot_params['title'])
    return fig

# def api_plot_chart_auto(code):
#     fig = eval(code)
#     return fig

def format_interim_steps(interim_steps):
    detailed_log = ""
    for i in interim_steps:
        detailed_log += f'''### Function call:\n
##### Function name:
```
{str(i['function_name'])}
```
\n\n
##### Function parameters:
```
{str(i['function_params'])}
```
\n\n
##### API response:
```
{str(i['API_response'])}
```
\n\n'''
    return detailed_log


def ask(question, chat):
    prompt = question + f"\n The dataset_id is {DATASET_ID}" + SYSTEM_PROMPT

    response = chat.send_message(prompt)
    response = response.candidates[0].content.parts[0]
    intermediate_steps = []

    function_calling_in_process = True
    while function_calling_in_process:
        try:
            function_name, params = response.function_call.name, {}
            for key, value in response.function_call.args.items():
                params[key] = value

            if function_name == "list_tables":
                api_response = api_list_tables(DATASET_ID)
                
            if function_name == "get_table_metadata":
                api_response = api_get_table_metadata(params["table_id"])

            if function_name == "sql_query":
                api_response = execute_sql_query(params["query"])

            if function_name == "plot_chart":
                fig = api_plot_chart(params)
                st.plotly_chart(fig)#, use_container_width=True)
                api_response = "here is the plot of the data."

            if function_name == "plot_chart_auto":
                print(type(params['code']), params['code'])
                local_namespace = {}
                # Execute the code string in the local namespace
                exec(params['code'].replace('\n', '\n'), globals(), local_namespace)
                # Access the 'fig' variable from the local namespace
                fig = local_namespace['fig']

                st.plotly_chart(fig)#, use_container_width=True)
                api_response = "here is the plot of the data shown below in separate tab."

            response = chat.send_message(
                Part.from_function_response(
                    name=function_name,
                    response={
                        "content": api_response,
                    },
                ),
            )
            response = response.candidates[0].content.parts[0]
            intermediate_steps.append({
                'function_name': function_name,
                'function_params': params,
                'API_response': api_response,
                'response': response
            })

        except AttributeError:
            function_calling_in_process = False

    return Response(text=response.text, interim_steps=intermediate_steps)



