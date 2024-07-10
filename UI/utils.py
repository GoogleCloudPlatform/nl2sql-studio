# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Utility functions for the NL2SQL Studio User Interface written in Streamlit
    This module contains the functions required to invoke the APIs, track user
    actions and other support tasks
"""

import os
import re
import time
import json
import configparser
import streamlit as st
from streamlit_feedback import streamlit_feedback
from streamlit.components.v1 import html
from dotenv import load_dotenv
from loguru import logger
# from google.auth.transport import requests
import requests
from jose import jwt
import plotly.express as px
from fuzzywuzzy import process

import vertexai
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

load_dotenv()

# API Calls
LITE_API_PART = 'lite'
FEW_SHOT_GENERATION = "Few Shot"
GEN_BY_CORE = "CORE_EXECUTORS"
GEN_BY_LITE = "LITE_EXECUTORS"


def default_func(prompt) -> str:
    """
        Test function that returns a reversed question output
        instead of executor
    """
    time.sleep(3)
    sql = prompt[::-1]
    st.session_state.messages[-1]['content'] = sql
    st.session_state.new_question = False
    st.rerun()
    st.session_state.refresh = True
    return sql


def call_generate_sql_api(question, endpoint) -> tuple[str, str]:
    """
        Common SQL generation function
    """
    # api_url = os.getenv('CORE_EXECUTORS')

    if LITE_API_PART in endpoint:
        api_url = os.getenv('LITE_EXECUTORS')
        few_shot_gen = False
        if st.session_state.lite_model == FEW_SHOT_GENERATION:
            few_shot_gen = True
        data = {"question": question,
                "execute_sql": st.session_state.execution,
                "few_shot": few_shot_gen}
        st.session_state.sql_generated_by = GEN_BY_LITE
    else:
        api_url = os.getenv('CORE_EXECUTORS')
        data = {"question": question,
                "execute_sql": st.session_state.execution}
        st.session_state.sql_generated_by = GEN_BY_CORE

    headers = {"Content-type": "application/json",
               "Authorization": f"Bearer {st.session_state.access_token}"}
    api_endpoint = f"{api_url}/{endpoint}"

    logger.info(f"Invoking API : {api_endpoint}")
    logger.info(f"Provided parameters are : Data = {data}")
    api_response = requests.post(api_endpoint,
                                 data=json.dumps(data),
                                 headers=headers,
                                 timeout=None)
    # logger.info('-'*100, api_response.text)
    exec_result = ""
    try:
        resp = api_response.json()
        logger.info(f"API resonse : {resp}")
        sql = resp['generated_query']
        st.session_state.result_id = resp['result_id']
        exec_result = resp['sql_result']
        try:
            exec_result_df = run_query(sql)
        except Exception as e:
            logger.info(f"SQL query execution failed due to {str(e)}")
            exec_result_df = None
    except RuntimeError:
        sql = "Execution Failed ! Error encountered in RAG Executor"

    logger.info(f"Generated SQL = {sql}")
    logger.info(f"Generation ID = {st.session_state.result_id}")
    return sql, exec_result, exec_result_df


def rag_gen_sql(question) -> str:
    """
        SQL Generation using the RAG Executor
    """
    logger.info("Invoking the RAG Executor")
    sql, exec_result, exec_result_df = \
        call_generate_sql_api(question, 'api/executor/rag')
    if exec_result_df is not None and st.session_state.execution:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql
    else:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql


def cot_gen_sql(question) -> str:
    """
        SQL Generation using the Chain of Thought executor
    """
    logger.info("Invoking the Chain of Thought Executor")
    sql, exec_result, exec_result_df = \
        call_generate_sql_api(question, 'api/executor/cot')

    if exec_result_df is not None and st.session_state.execution:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql
    else:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql


def linear_gen_sql(question) -> str:
    """
        SQL Generation using the Linear executor
    """
    logger.info("Invoking the Linear Executor")
    sql, exec_result, exec_result_df = \
        call_generate_sql_api(question, 'api/executor/linear')

    if exec_result_df is not None and st.session_state.execution:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql
    else:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql


def lite_gen_sql(question) -> str:
    """
        SQL Generation using the NL2SQLStudio Lite
    """
    logger.info("Invoking the NL2SQLStudio Lite Generator")
    sql, exec_result, exec_result_df = \
        call_generate_sql_api(question, '/api/lite/generate')

    if exec_result_df is not None and st.session_state.execution:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql
    else:
        st.session_state.messages[-1]['content'] = \
            format_response(sql, exec_result)
        st.session_state.messages[-1]['dataframe'] = exec_result_df
        st.session_state.new_question = False
        st.rerun()
        return sql


def format_response(sql, exec_result):
    """
    Formats the response string to append the message queue
    """
    md_style_start1 = '<span style="font-size: 1rem;">'
    md_style_start2 = '<span style="font-size: 1.1rem;color:blueviolet;">'
    md_style_end = '</span>'
    exec_result = exec_result.replace('\n', " ")
    response_string = md_style_start1 + \
        sql + md_style_end + "<br><br>" + md_style_start2 + \
        exec_result + md_style_end if st.session_state.execution else \
        sql + md_style_end
    return response_string

# Utility functions


def submit_feedback(user_response) -> bool:
    """
        Function to capture the score of Feedback widget click
    """
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }
    logger.info(
        f"Score Mapping = {score_mappings['thumbs'][user_response['score']]}"
        )
    st.session_state.user_response = \
        score_mappings["thumbs"][user_response['score']]
    st.session_state.user_responded = True
    logger.info(f"User Response state = {st.session_state.user_responded}")
    return user_response


def message_queue(question) -> None:
    """
        Append user queries and system responses to the message queue
    """
    base_url = "https://cdn3.emoji.gg/emojis/7048-loading.gif"
    emoj_url = "https://emoji.gg/emoji/7048-loading"
    st.session_state.messages.append({"role": "user",
                                      "content": question, "dataframe": None})
    st.session_state.messages.append({"role": "assistant",
                                      "content": f"""Fetching results..
                                      [![Loading]({base_url})]({emoj_url})""",
                                      "dataframe": None})


def get_feedback() -> None:
    """
        Position the Thumbs Up/Down User feedback widget
    """
    i = 0
    num_msgs = len(st.session_state.messages)
    with st.session_state.fc:
        for i in range(1, num_msgs):
            fb_cont = "c"+str(i)
            fb_cont = st.container(height=70, border=False)
            with fb_cont:
                st.write('')
                if "User feedback captured" in \
                        st.session_state.messages[i]['content']:

                    fb_cont2 = "c"+str(i)+"2"
                    fb_cont2 = st.container(height=70, border=False)
                    with fb_cont2:
                        st.write('')
            i += 1

        if feedback := streamlit_feedback(feedback_type="thumbs",
                                          on_submit=submit_feedback,
                                          key='fbkey' +
                                          str(st.session_state.fb_count)):
            print(feedback)
            del st.session_state['fbkey'+str(st.session_state.fb_count)]
            st.session_state.fb_count += 1
            st.session_state.refresh = True


def add_question_to_db(sample_question,
                       sample_sql,
                       framework='CORE_EXECUTORS') -> None:
    """
        Add Sample questions and corresponding SQLs to the
        PostgreSQL DB
    """
    url = os.getenv(framework)
    url = f"{url}/api/record/create"
    logger.info(f"Adding {sample_question} and {sample_sql} to DB")
    logger.info(f"Adding Sample question and SQL to {framework} via {url}")

    data = {"question": sample_question, "sql": sample_sql}
    headers = {'Content-type': 'application/json',
               'Accept': 'text/plain',
               "Authorization": f"Bearer {st.session_state.access_token}"}
    _ = requests.post(url=url, data=json.dumps(data),
                      headers=headers,
                      timeout=None)
    st.session_state.add_question_status = True


def back_to_login_page() -> None:
    """
        Open the given URL
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    url = config['DEFAULT']['GOOGLE_REDIRECT_URI']
    open_script = """
        <script type="text/javascript">
            window.open('%s', '_self').focus();
        </script>
    """ % (url)

    st.session_state.token = None
    st.session_state.login_status = False
    st.query_params.clear()

    html(open_script)
    # st.sidebar.markdown(url)
    # AUTH_REQUESTS.Request().get(url)


def init_auth() -> None:
    """
        Authentication Initialisation function
    """
    logger.info("Initialising Authentication process")
    # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    config = configparser.ConfigParser()
    config.read('config.ini')
    # GOOGLE_CLIENT_ID = config['DEFAULT']['GOOGLE_CLIENT_ID']
    # GOOGLE_CLIENT_SECRET = config['DEFAULT']['GOOGLE_CLIENT_SECRET']
    google_redirect_uri = config['DEFAULT']['GOOGLE_REDIRECT_URI']

    logger.info(f"Redirect URI = {google_redirect_uri} ")


def login_user() -> None:
    """
        Trigger Logging in
    """
    init_auth()
    logger.info("Authenticating...")
    view_login_google()


def view_login_google() -> str:
    """
        Navigating to Authentication URL
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    google_client_id = config['DEFAULT']['GOOGLE_CLIENT_ID']
    # GOOGLE_CLIENT_SECRET = config['DEFAULT']['GOOGLE_CLIENT_SECRET']
    google_redirect_uri = config['DEFAULT']['GOOGLE_REDIRECT_URI']

    auth_url = f"""https://accounts.google.com/o/oauth2/auth?response_type=\
code&client_id={google_client_id}&redirect_uri={google_redirect_uri}\
&scope=openid%20profile%20email&access_type=offline"""
    logger.info(f"URL to authenticate = {auth_url}")
    return auth_url


def view_auth_google(code) -> tuple[str, str]:
    """
        Retrieve the Code and Tokens
    """

    logger.info("Extracting the Code and Generating the Tokens")
    logger.info(f"Query Parameters - {st.query_params}")
    config = configparser.ConfigParser()
    config.read('config.ini')

    google_client_id = config['DEFAULT']['GOOGLE_CLIENT_ID']
    google_client_secret = config['DEFAULT']['GOOGLE_CLIENT_SECRET']
    google_redirect_uri = config['DEFAULT']['GOOGLE_REDIRECT_URI']

    token_url = "https://accounts.google.com/o/oauth2/token"
    data = {
        "code": code,
        "client_id": google_client_id,
        "client_secret": google_client_secret,
        "redirect_uri": google_redirect_uri,
        "grant_type": "authorization_code",
    }
    logger.info(f"Auth info =, {data}")

    try:
        logger.info("Using requests library itself")
        response = requests.post(token_url, data=data, timeout=None)
        logger.info(f"Auth response = {response.json()}")
        access_token = response.json().get("access_token")
        id_token = response.json().get("id_token")
        logger.info(f"Access token = {access_token}")
        logger.info(f"ID Token = {id_token}")

    except Exception:
        logger.error("Authentication via Requests library  failed")

    user_info = requests\
        .get("https://www.googleapis.com/oauth2/v1/userinfo",
             headers={"Authorization": f"Bearer {access_token}"},
             timeout=None)
    logger.info(f"Decoded User info : {user_info.json()}")
    response_data = {"token": id_token, "access_token": access_token}
    logger.info(f"Response data = {response_data}")
    return id_token, access_token


def view_get_token(token) -> None:
    """
        Retrieve the token
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    google_client_secret = config['DEFAULT']['GOOGLE_CLIENT_SECRET']

    logger.info("Retrieving token")
    algorithm = jwt.get_unverified_header(token).get('alg')
    logger.info("Algorithms to use : {algorithm}")
    try:
        response = jwt.decode(token,
                              google_client_secret,
                              algorithms=algorithm)
        logger.info("Decoded token=", response)
        return response
    except Exception:
        logger.error("Something went wrong while decooding")
        return "Decode error due to Algorithmm mismatch"
    # return jwt.decode(token, GOOGLE_CLIENT_SECRET, algorithms=["RS256"])


def run_query(sql_query):
    """Runs a BigQuery SQL query and returns results as a Pandas DataFrame."""

    client = bigquery.Client()

    # Execute the query
    job = client.query(sql_query)

    # Convert to Pandas DataFrame
    df = job.to_dataframe()

    return df


def generate_result(query):
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95
        }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
            generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    vertexai.init(project="my-project-1-361607", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-001")
    responses = model.generate_content(
        [query],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
        )

    final_response = """"""
    for response in responses:
        final_response = final_response + response.text
    return final_response


def extract_substring(input_string):
    match = re.search(r'{([^}]*)}', input_string)
    if match:
        match_result = match.group(1)
        if match_result.startswith('{'):
            return match_result
        else:
            return "{" + match_result + "}"
    else:
        return {}


def get_best_fuzzy_match(input_string, string_list):
    best_match = process.extractOne(input_string, string_list)
    return best_match[0] if best_match else None


# Plotting Functions
def plot_line_chart(df, x_cols, y_cols):
    for y_col in y_cols:
        st.plotly_chart(px.line(df, x=x_cols[0], y=y_col,
                                title=f'Line Chart of {y_col} vs {x_cols[0]}'))


def plot_bar_chart(df, x_cols, y_cols):
    for y_col in y_cols:
        st.plotly_chart(px.bar(df, x=x_cols[0], y=y_col,
                               title=f'Bar Chart of {y_col} vs {x_cols[0]}'))


def plot_histogram(df, x_cols):
    for x_col in x_cols:
        st.plotly_chart(px.histogram(df, x=x_col,
                                     title=f'Histogram of {x_col}'))


def plot_scatter_plot(df, x_cols, y_cols):
    st.plotly_chart(px.scatter(df, x=x_cols[0], y=y_cols[0],
                               title=f'Scatter Plot of {y_cols[0]} \
                                   vs {x_cols[0]}'))


def plot_pie_chart(df, x_cols):
    st.plotly_chart(px.pie(df, names=x_cols[0],
                           title=f'Pie Chart of {x_cols[0]}'))


def plot_area_chart(df, x_cols, y_cols):
    for y_col in y_cols:
        st.plotly_chart(px.area(df, x=x_cols[0],
                                y=y_col, title=f'Area Chart of \
                                    {y_col} vs {x_cols[0]}'))


def plot_box_plot(df, x_cols, y_cols):
    for y_col in y_cols:
        st.plotly_chart(px.box(df, x=x_cols[0], y=y_col,
                               title=f'Box Plot of {y_col} vs {x_cols[0]}'))


def plot_heatmap(df, x_cols):
    st.plotly_chart(px.imshow(df[x_cols].corr(), title='Heatmap'))


def plot_bubble_chart(df, x_cols, y_cols):
    st.plotly_chart(px.scatter(df, x=x_cols[0], y=y_cols[0],
                               size=y_cols[0],
                               title=f'Bubble Chart of {y_cols[0]} \
                                   vs {x_cols[0]}'))


def plot_map(df, lat_col='lat', lon_col='lon'):
    st.map(df)


def plot_data(df, plot_type, x_column_list, y_column_list):
    # Plot based on plot_type
    if plot_type == 'Line Chart' and x_column_list and y_column_list:
        plot_line_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Bar Chart' and x_column_list and y_column_list:
        plot_bar_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Histogram' and x_column_list:
        plot_histogram(df, x_column_list)
    elif plot_type == 'Scatter Plot' and x_column_list and y_column_list:
        plot_scatter_plot(df, x_column_list, y_column_list)
    elif plot_type == 'Pie Chart' and x_column_list:
        plot_pie_chart(df, x_column_list)
    elif plot_type == 'Area Chart' and x_column_list and y_column_list:
        plot_area_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Box Plot' and x_column_list and y_column_list:
        plot_box_plot(df, x_column_list, y_column_list)
    elif plot_type == 'Heatmap' and x_column_list:
        plot_heatmap(df, x_column_list)
    elif plot_type == 'Bubble Chart' and x_column_list and y_column_list:
        plot_bubble_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Map':
        plot_map(df)
    else:
        st.write('Please select appropriate columns for the chosen plot type.')


def run_visualization(df, custom_flag, key_counter):
    columns = df.columns.tolist()
    data_types = [str(dtype) for dtype in df.dtypes]
    metadata = dict(zip(columns, data_types))
    plot_chart_list = ['Line Chart', 'Bar Chart',
                       'Histogram', 'Scatter Plot',
                       'Pie Chart', 'Area Chart',
                       'Box Plot', 'Heatmap', 'Bubble Chart', 'Map']

    # 3. "plot_chooser" Prompt Template
    plot_chooser_template = f"""
    You are an expert in data visualization.
    Given the following DataFrame metadata:

    dataframe columns: {columns}
    Data Types: {metadata}

    You are tasked to suggest:
    1. The best plot type out of {plot_chart_list}
    2. The dataframe columns to use for plotting.
    Consider the data types and potential relationships between columns.

    Output should be strictly in json format as described below:
    ```{{
    "plot_type": "...",
    "x_column": "[...]",
    "y_column": "[...]" // Optional, depending on plot type
    }}
    ```
    """

    result_json_string = generate_result(plot_chooser_template)
    result_json = json.loads(extract_substring(result_json_string))

    if custom_flag:
        plot_type = result_json.get("plot_type", "line")
        x_column_list = result_json.get("x_column", columns[0])
        logger.info(f"Plot type is : {plot_type}")

        if len(columns) > 1:
            y_column_list = result_json.get("y_column", columns[1])
            plot_data(df, plot_type, x_column_list, y_column_list)
        else:
            plot_data(df, plot_type, x_column_list, x_column_list)
    else:
        with st.container():
            plot_type = st.selectbox('Select Plot Type',
                                     ['Line Chart', 'Bar Chart',
                                      'Histogram', 'Scatter Plot',
                                      'Pie Chart', 'Area Chart',
                                      'Box Plot', 'Heatmap',
                                      'Bubble Chart', 'Map'],
                                     key=f'plot_type_{key_counter}')
            x_column_list = st.multiselect('Select X Column(s)',
                                           df.columns,
                                           key=f'x_column_{key_counter}')
            y_column_list = (
                st.multiselect(
                    'Select Y Column(s)',
                    df.columns,
                    key=f'y_column_{key_counter}'
                )
                if plot_type not in ['Pie Chart', 'Heatmap', 'Map'] else None
            )

            if y_column_list is not None:
                plot_data(df, plot_type, x_column_list,
                          y_column_list)
            else:
                plot_data(df, plot_type, x_column_list,
                          x_column_list)
