"""The main module for the NL2SQL Autobot."""

from google.cloud import bigquery
import os
import json
import vertexai
import plotly.express as px
from pydantic import BaseModel
from vertexai import generative_models
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Tool,
    # ToolConfig
)
import streamlit as st
from dbai_src.bot_functions import (
    list_tables_func,
    get_table_metadata_func,
    sql_query_func,
    plot_chart_auto_func
)

ROOT_PATH = '/Users/koushikchak/_work/nl2sql-studio'
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

class Response:
    """ """
    def __init__(self, text, interim_steps) -> None:
        self.text = text
        self.interim_steps = interim_steps

class DBAI:
    """ """
    def __init__(
            self,
            proj_id="proj-kous",
            dataset_id="Albertsons",
            tables_list=['camain_oracle_hcm', 'camain_ps']
            ):
            
        self.proj_id = proj_id
        self.dataset_id = dataset_id
        self.tables_list = tables_list

        self.sql_query_tool = Tool(
            function_declarations=[
                list_tables_func,
                get_table_metadata_func,
                sql_query_func,
                plot_chart_auto_func,
                # plot_chart_func,
            ],
        )

        self.agent = GenerativeModel("gemini-1.5-pro-001",
                            generation_config={"temperature": 0.05},
                            safety_settings=safety_settings,
                            tools=[self.sql_query_tool],
                            )

        self.bq_client = bigquery.Client(project=self.proj_id)
        self.SYSTEM_PROMPT = """ You are a fluent person who efficiently communicates with the user over different Database queries. Please always call the functions at your disposal whenever you need to know something, and do not reply unless you feel you have all information to answer the question satisfactorily. 
        Only use information that you learn from BigQuery, do not make up information. Always use date or time functions instead of hard-coded values in SQL to reflect true current value.
        """
        self.load_metadata()

        vertexai.init(project=self.proj_id)

    def load_metadata(self):
        METDATA_CACHE_PATH = f"./metadata_cache_{self.dataset_id}.json"
        if not os.path.exists(METDATA_CACHE_PATH):
            self.metadata = self.create_metadata_cache()
            with open(METDATA_CACHE_PATH, 'w') as f:  # pylint-ignore: unspecified-encoding
                f.write(json.dumps(self.metadata))
        else:
            with open(METDATA_CACHE_PATH, 'r') as f:
                self.metadata = json.load(f)

    def create_metadata_cache(self):
        """ """
        gen_description_prompt = """Based on the columns information of this table.
        Generate a very brief description for this table.
        TABLE: {table_id}
        columns_info: {columns_info}"""

        if self.tables_list in [ [], [''], '' ]:
            api_response = self.bq_client.list_tables(self.dataset_id)
            self.tables_list = [table.table_id for table in api_response]

        metadata = {}
        for table_id in self.tables_list:
            columns_info = self.bq_client.get_table(f'{self.dataset_id}.{table_id}').to_api_repr()['schema']
            ## remove unwanted details like 'mode'
            for field in columns_info.get('fields', []):
                field.pop('mode', None)

            metadata[table_id] = {}
            metadata[table_id]["table_name"] = table_id
            metadata[table_id]["columns_info"] = columns_info
            prompt = gen_description_prompt.format(table_id=table_id, columns_info=columns_info)
            metadata[table_id]["table_description"] = gemini.generate_content(prompt).text

        return metadata


    def api_list_tables(self):
        """ """
        # api_response = client.list_tables(DATASET_ID)
        # api_response = str([table.table_id for table in api_response])
        try:
            api_response = self.metadata
        except Exception:
            api_response = self.tables_list
        return api_response

    def api_get_table_metadata(self, table_id):
        """ """
        try:
            table_metadata = str(self.metadata[table_id])
        except Exception:
            ## if table_id is in form of dataset_id.table_id then remove dataset_id
            table_metadata = str(self.metadata[table_id.split('.')[-1]])
        return table_metadata

    def execute_sql_query(self, query):
        """ """
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=100000000,
            default_dataset=f'{self.proj_id}.{self.dataset_id}'
            )
        try:
            cleaned_query = query.replace("\\n", " ").replace("\n", "").replace("\\", "")
            query_job = self.bq_client.query(cleaned_query, job_config=job_config)
            api_response = query_job.result()
            api_response = str([dict(row) for row in api_response])
            api_response = api_response.replace("\\", "").replace("\n", "")
        except Exception as e:
            api_response = f"{str(e)}"

        return api_response

    # def api_plot_chart(self, plot_params):
    #     data = plot_params['data']
    #     if isinstance(data, list):
    #         data = data[0]
    #     print('_'*100, data, '_'*100)
    #     data = data.replace('None', '-1')
    #     if 'content' in str(data):
    #         df = pd.DataFrame(json.loads(data['content'][0]))
    #     elif type(data) == str:
    #         df = pd.DataFrame(eval(data))
    #     else:
    #         df = pd.DataFrame(json.loads(str(data)))
    #     fig = px.bar(df, x=plot_params['x_axis'], y=plot_params['y_axis'], title=plot_params['title'])
    #     return fig

    # def api_plot_chart_auto(code):
    #     fig = eval(code)
    #     return fig

    def format_interim_steps(self, interim_steps):
        """ """
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


    def ask(self, question, chat):
        """ """
        prompt = question + f"\n The dataset_id is {self.dataset_id}" + self.SYSTEM_PROMPT

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
                    api_response = self.api_list_tables()
                    
                if function_name == "get_table_metadata":
                    api_response = self.api_get_table_metadata(params["table_id"])

                if function_name == "sql_query":
                    api_response = self.execute_sql_query(params["query"])

                # if function_name == "plot_chart":
                #     fig = api_plot_chart(params)
                #     st.plotly_chart(fig)#, use_container_width=True)
                #     api_response = "here is the plot of the data."

                if function_name == "plot_chart_auto":
                    print(type(params['code']), params['code'])
                    local_namespace = {}
                    # Execute the code string in the local namespace
                    exec(params['code'].replace('\r\n', '\n'), globals(), local_namespace)
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



class NL2SQL_resp:
    """ """
    def __init__(self, nl_output, generated_sql, sql_output) -> None:
        self.nl_output = nl_output
        self.generated_sql = generated_sql
        self.sql_output = sql_output
    
    def __str__(self) -> str:
        return f" NL_OUTPUT: {self.nl_output}\n\n GENERATED_SQL: {self.generated_sql}\n\n SQL_OUTPUT: {self.sql_output}"
        

class DBAI_nl2sql(DBAI):
    """ """
    def __init__(
            self,
            proj_id="proj-kous",
            dataset_id="Albertsons",
            tables_list=['camain_oracle_hcm', 'camain_ps']
            ):
        super().__init__(proj_id, dataset_id, tables_list)

        self.nl2sql_tool = Tool(
            function_declarations=[
                list_tables_func,
                get_table_metadata_func,
                sql_query_func,
            ],
        )

        self.agent = GenerativeModel("gemini-1.5-pro-001",
                            generation_config={"temperature": 0.05},
                            safety_settings=safety_settings,
                            tools=[self.nl2sql_tool],
                            )

    
    def get_sql(self, question):
        """ """
        chat = self.agent.start_chat()
        prompt = question + f"\n The dataset_id is {self.dataset_id}" + self.SYSTEM_PROMPT

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
                    api_response = self.api_list_tables()
                    
                if function_name == "get_table_metadata":
                    api_response = self.api_get_table_metadata(params["table_id"])

                if function_name == "sql_query":
                    api_response = self.execute_sql_query(params["query"])

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

        generated_sql, sql_output = '', ''
        for i in intermediate_steps[::-1]:
            if i['function_name'] == 'sql_query':
                generated_sql = i['function_params']['query']
                sql_output = i['API_response']

        return NL2SQL_resp(response.text, generated_sql, sql_output)