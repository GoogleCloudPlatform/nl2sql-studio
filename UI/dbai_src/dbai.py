"""The main module for the NL2SQL chat Agent which is multi-turn."""

import ast
import os
import json
import re
import pandas as pd
import plotly.express as px
import vertexai
from google.cloud import bigquery
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

_SQL_TOKEN_RE = re.compile(
    r"('(?:''|\\'|[^'])*'|\"(?:\"\"|\\\"|[^\"])*\"|`[^`]*`)"
    r"|(--[^\r\n]*|/\*[\s\S]*?\*/)"
)

_DISALLOWED_SQL_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE(?!\s*\()"
    r"|MERGE|GRANT|REVOKE|CALL|EXEC|EXECUTE|EXPORT)\b",
    re.IGNORECASE,
)

_ALLOWED_PX_CHARTS = frozenset({
    "bar",
    "line",
    "pie",
    "scatter",
    "histogram",
    "box",
    "violin",
    "area",
    "funnel",
    "sunburst",
    "treemap",
    "icicle",
    "density_heatmap",
    "strip",
})

_ALLOWED_FIG_METHODS = frozenset({
    "update_layout",
    "update_traces",
    "update_xaxes",
    "update_yaxes",
})


def validate_readonly_sql(query: str) -> str:
    """Validate that a SQL query is a single read-only SELECT/WITH statement."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Empty SQL query")
    cleaned = re.sub(
        r"^\s*```(?:sql)?\s*|\s*```\s*$", "", query.strip(), flags=re.IGNORECASE
    )
    cleaned = _SQL_TOKEN_RE.sub(
        lambda m: m.group(1) if m.group(1) is not None else " ", cleaned
    ).strip()
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        raise ValueError("Empty SQL query")
    masked = _SQL_TOKEN_RE.sub(
        lambda m: "''" if m.group(1) is not None else " ", cleaned
    )
    if ";" in masked:
        raise ValueError("Multiple SQL statements are not allowed")
    if not re.match(r"^\s*\(*\s*(SELECT|WITH)\b", masked, re.IGNORECASE):
        raise ValueError("Only read-only SELECT queries are allowed")
    if _DISALLOWED_SQL_KEYWORDS.search(masked):
        raise ValueError("Disallowed DDL/DML keyword in SQL query")
    return cleaned


def _eval_safe_ast_node(node: ast.AST, env: dict):
    """Safely evaluate an AST node restricted to literals, env vars, pd.DataFrame, and px charts."""
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"Disallowed identifier: {node.id}")
    if isinstance(node, ast.List):
        return [_eval_safe_ast_node(elt, env) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_safe_ast_node(elt, env) for elt in node.elts)
    if isinstance(node, ast.Set):
        return {_eval_safe_ast_node(elt, env) for elt in node.elts}
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("Dict unpacking is not allowed")
        return {
            _eval_safe_ast_node(k, env): _eval_safe_ast_node(v, env)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Call):
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise ValueError("*args unpacking is not allowed")
        args = [_eval_safe_ast_node(arg, env) for arg in node.args]
        kwargs = {
            kw.arg: _eval_safe_ast_node(kw.value, env)
            for kw in node.keywords
            if kw.arg is not None
        }
        if len(kwargs) != len(node.keywords):
            raise ValueError("**kwargs unpacking is not allowed")
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only method or module attribute calls are allowed")
        attr_name = node.func.attr
        if isinstance(node.func.value, ast.Name):
            mod_name = node.func.value.id
            if mod_name in ("pd", "pandas") and attr_name == "DataFrame":
                return pd.DataFrame(*args, **kwargs)
            if mod_name == "px" and attr_name in _ALLOWED_PX_CHARTS:
                return getattr(px, attr_name)(*args, **kwargs)
            if mod_name in env and attr_name in _ALLOWED_FIG_METHODS:
                receiver = env[mod_name]
                if hasattr(receiver, "to_plotly_json"):
                    return getattr(receiver, attr_name)(*args, **kwargs)
            raise ValueError(f"Disallowed call: {mod_name}.{attr_name}")
        if (
            isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "plotly"
            and node.func.value.attr == "express"
            and attr_name in _ALLOWED_PX_CHARTS
        ):
            return getattr(px, attr_name)(*args, **kwargs)
        if (
            isinstance(node.func.value, ast.Call)
            and attr_name in _ALLOWED_FIG_METHODS
        ):
            receiver = _eval_safe_ast_node(node.func.value, env)
            if hasattr(receiver, "to_plotly_json"):
                return getattr(receiver, attr_name)(*args, **kwargs)
        raise ValueError("Only direct calls on pd, px, or figure objects are allowed")
    return ast.literal_eval(node)


def _safe_build_plotly_figure(code_str: str):
    """Parse LLM-generated chart code via AST without exec()/eval() and return fig."""
    cleaned = re.sub(
        r"^```(?:python|py)?|```$",
        "",
        code_str.strip(),
        flags=re.MULTILINE | re.IGNORECASE,
    )
    tree = ast.parse(cleaned.replace("\r\n", "\n"), mode="exec")
    env = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name not in ("pandas", "plotly.express", "plotly"):
                    raise ValueError(f"Disallowed import: {alias.name}")
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module == "plotly" and all(
                alias.name == "express" for alias in stmt.names
            ):
                continue
            raise ValueError(f"Disallowed import from: {stmt.module}")
        elif isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(
                stmt.targets[0], ast.Name
            ):
                raise ValueError("Only simple variable assignments are allowed")
            target_name = stmt.targets[0].id
            if target_name.startswith("_"):
                raise ValueError("Private variable names are not allowed")
            env[target_name] = _eval_safe_ast_node(stmt.value, env)
        elif isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt.value, ast.Call) and isinstance(
                stmt.value.func, ast.Attribute
            ):
                if (
                    isinstance(stmt.value.func.value, ast.Name)
                    and stmt.value.func.value.id in env
                    and stmt.value.func.attr == "show"
                    and not stmt.value.args
                    and not stmt.value.keywords
                ):
                    continue
                if stmt.value.func.attr in _ALLOWED_FIG_METHODS:
                    _eval_safe_ast_node(stmt.value, env)
                    continue
            raise ValueError("Disallowed expression in chart code")
        else:
            raise ValueError(
                f"Disallowed statement in chart code: {type(stmt).__name__}"
            )
    if "fig" not in env:
        raise ValueError("Chart code did not produce a 'fig' object")
    return env["fig"]


safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
        generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
        generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
        generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
        generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

gemini = GenerativeModel(
    "gemini-1.5-pro-002",
    generation_config={"temperature": 0.05},
    safety_settings=safety_settings,
    )


class Response:
    """The base response template class for DBAI output"""
    def __init__(self, text, interim_steps) -> None:
        self.text = text
        self.interim_steps = interim_steps


class DBAI:
    """
    The base class for DBAI agent which is the multi-turn chat
    and can plot graphs
    """
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

        self.agent = GenerativeModel("gemini-1.5-pro-002",
                                     generation_config={"temperature": 0.05},
                                     safety_settings=safety_settings,
                                     tools=[self.sql_query_tool],
                                     )

        self.bq_client = bigquery.Client(project=self.proj_id)
        self.system_prompt = """
        You are a fluent person who efficiently communicates with the user
        over different Database queries. Please always call the functions
        at your disposal whenever you need to know something,
        and do not reply unless you feel you have all information to answer
        the question satisfactorily.
        Only use information that you learn from BigQuery,
        do not make up information.
        Always use date or time functions instead of hard-coded values in SQL
        to reflect true current value.
        """
        self.load_metadata()

        vertexai.init(project=self.proj_id)

    def load_metadata(self):
        """
        Load the metadata cache file from the defined path if exists
        else creates
        """
        safe_dataset_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.dataset_id))
        metdata_cache_path = f"./metadata_cache_{safe_dataset_id}.json"
        if not os.path.exists(metdata_cache_path):
            self.metadata = self.create_metadata_cache()
            with open(metdata_cache_path, 'w') as f:
                # pylint: disable=unspecified-encoding
                f.write(json.dumps(self.metadata))
        else:
            with open(metdata_cache_path, 'r') as f:
                # pylint: disable=unspecified-encoding
                self.metadata = json.load(f)

    def create_metadata_cache(self):
        """
        create the metadata cache file for the specified Tables in DB
        for all columns
        """
        gen_description_prompt = """
        Based on the columns information of this table.
        Generate a very brief description for this table.
        TABLE: {table_id}
        columns_info: {columns_info}
        """

        if self.tables_list in [[], [''], '']:
            api_response = self.bq_client.list_tables(self.dataset_id)
            self.tables_list = [table.table_id for table in api_response]

        metadata = {}
        for table_id in self.tables_list:
            columns_info = self.bq_client.get_table(
                    f'{self.dataset_id}.{table_id}'
                ).to_api_repr()['schema']
            # remove unwanted details like 'mode'
            for field in columns_info.get('fields', []):
                field.pop('mode', None)

            metadata[table_id] = {}
            metadata[table_id]["table_name"] = table_id
            metadata[table_id]["columns_info"] = columns_info
            prompt = gen_description_prompt.format(table_id=table_id,
                                                   columns_info=columns_info
                                                   )
            metadata[table_id]["table_description"] = gemini.generate_content(
                prompt
                ).text

        return metadata

    def api_list_tables(self):
        """Gemini Tool for listing all tables info. """
        # api_response = client.list_tables(DATASET_ID)
        # api_response = str([table.table_id for table in api_response])
        try:
            api_response = self.metadata
        except Exception:  # pylint: disable=broad-except
            api_response = self.tables_list
        return api_response

    def api_get_table_metadata(self, table_id):
        """Gemini Tool to fetch metadata for the given Table"""
        try:
            table_metadata = str(self.metadata[table_id])
        except Exception:  # pylint: disable=broad-except
            # if table_id is in form of dataset_id.table_id
            # then remove dataset_id
            table_metadata = str(self.metadata[table_id.split('.')[-1]])
        return table_metadata

    def execute_sql_query(self, query):
        """Gemini Tool to execute given SQL and return execution result. """
        job_config = bigquery.QueryJobConfig(
            default_dataset=f'{self.proj_id}.{self.dataset_id}'
            )
        try:
            normalized_query = query.replace("\\n", "\n")
            validated_query = validate_readonly_sql(normalized_query)
            validated_query = (
                validated_query.replace("\n", " ").replace("\\", "").strip()
            )
            query_job = self.bq_client.query(validated_query,
                                             job_config=job_config
                                             )
            api_response = query_job.result()
            api_response = str([dict(row) for row in api_response])
            api_response = api_response.replace("\\", "").replace("\n", "")
        except Exception as e:  # pylint: disable=broad-except
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
    #     fig = px.bar(df, x=plot_params['x_axis'], y=plot_params['y_axis']
    #                   , title=plot_params['title'])
    #     return fig

    # def api_plot_chart_auto(code):
    #     fig = eval(code)
    #     return fig

    def format_interim_steps(self, interim_steps):
        """Format all the intermediate steps for showing them in UI."""
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
        """main interface for interacting in multi-turn chat mode. """
        prompt = question\
            + f"\n The dataset_id is {self.dataset_id}"\
            + self.system_prompt

        response = chat.send_message(prompt)
        response = response.candidates[0].content.parts[0]
        intermediate_steps = []

        function_calling_in_process = True
        while function_calling_in_process:
            try:
                function_name, params = response.function_call.name, {}
                for key, value in response.function_call.args.items():
                    params[key] = value

                api_response = ''
                if function_name == "list_tables":
                    api_response = self.api_list_tables()

                if function_name == "get_table_metadata":
                    api_response = self.api_get_table_metadata(
                        params["table_id"]
                        )

                if function_name == "sql_query":
                    api_response = self.execute_sql_query(params["query"])

                # if function_name == "plot_chart":
                #     fig = api_plot_chart(params)
                #     st.plotly_chart(fig)#, use_container_width=True)
                #     api_response = "here is the plot of the data."

                if function_name == "plot_chart_auto":
                    print(type(params['code']), params['code'])
                    fig = _safe_build_plotly_figure(params['code'])

                    st.plotly_chart(fig)  # use_container_width=True)
                    api_response = "here is the plot of the data shown\
                          below in separate tab."

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


class NL2SQLResp:
    """NL2SQL output format class"""
    def __init__(self, nl_output, generated_sql, sql_output) -> None:
        self.nl_output = nl_output
        self.generated_sql = generated_sql
        self.sql_output = sql_output

    def __str__(self) -> str:
        return f''' NL_OUTPUT: {self.nl_output}\n
          GENERATED_SQL: {self.generated_sql}\n
          SQL_OUTPUT: {self.sql_output}'''


class DBAI_nl2sql(DBAI):  # pylint: disable=invalid-name
    """
    DBAI child class for generating NL2SQL response, instead of
    multi-turn chat-agent
    """
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

        self.agent = GenerativeModel("gemini-1.5-pro-002",
                                     generation_config={"temperature": 0.05},
                                     safety_settings=safety_settings,
                                     tools=[self.nl2sql_tool],
                                     )

    def get_sql(self, question):
        """
        For given question, returns the genrated SQL, result and description
        """
        chat = self.agent.start_chat()
        prompt = question +\
            f"\n The dataset_id is {self.dataset_id}" +\
            self.system_prompt

        response = chat.send_message(prompt)
        response = response.candidates[0].content.parts[0]
        intermediate_steps = []

        function_calling_in_process = True
        while function_calling_in_process:
            try:
                function_name, params = response.function_call.name, {}
                for key, value in response.function_call.args.items():
                    params[key] = value

                api_response = ''
                if function_name == "list_tables":
                    api_response = self.api_list_tables()

                if function_name == "get_table_metadata":
                    api_response = self.api_get_table_metadata(
                        params["table_id"]
                        )

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

        return NL2SQLResp(response.text, generated_sql, sql_output)
