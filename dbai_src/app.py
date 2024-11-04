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
Main file serving the DBAI module exposing the APIs
"""
import json
import uuid
from flask_cors import CORS
from flask import Flask, request
from loguru import logger
from vertexai.generative_models import Content

from dbai import DBAI_nl2sql, DBAI


PROJECT_ID = 'sl-test-project-363109'
DATASET_ID = 'nl2sql_spider'
TABLES_LIST = []

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/")
def spec():
    """
    Default API Route
    """
    logger.info("Welcome to Database AI API")
    return json.dumps({"response": "DBAI NL2SQL Generation library"})


@app.route("/", methods=["POST"])
def dbai_nl2sql_api():
    """
    Invokes the DBAI SQL Generator

    sample request_body: 
    {
        "question": "what is the average rate of concerts over the years?",
    }

    response from the API:
    {
        "result_id": unique_id,
        "generated_query": sql as a string,
        "sql_exec_output": results from SQL execution,
        "sql_result": Natural Language description of the SQL result,
        "error_msg": error details if any
    }
    """
    question = request.json["question"]
    logger.info(f"DBAI nl2sql for question : [{question}]")

    try:
        dbai_nl2sql = DBAI_nl2sql(
            proj_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            tables_list=TABLES_LIST
        )
        dbai_response = dbai_nl2sql.get_sql(question)
        sql = dbai_response.generated_sql
        res_id = str(uuid.uuid4())

        logger.info(f"{res_id}\n DBAI response = {dbai_response}")

        results = dbai_response.sql_output
        sql_result = dbai_response.nl_output
        response_string = {
            "result_id": res_id,
            "generated_query": sql,
            "sql_exec_output": results,
            "sql_result": sql_result,
            "error_msg": ""
        }

    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            f"DBAI SQL Generation unsuccessful: [{question}] {e}"
        )
        response_string = {
            "result_id": 0,
            "generated_query": "",
            "sql_exec_output": "",
            "sql_result": str(e),
            "error_msg": "Error in DBAI SQL generation",
        }

    return json.dumps(response_string)

@app.route("/chat", methods=["POST"])
def dbai_chat_api():
    """
    sample request_body to start a new chat: 
    {
        "question": ,
        "chat_history": "[]"
    }

    response from the API:
    {
        "result_id": unique_id,
        "response_text": model response in text,
        "interim_steps": intermediate_steps as a list,
        "chat_history": chat_history as a string of list of dicts,
        "error_msg": ""
    }

    Use the returned chat_history in subsequent questions for multi-turn chat:
    request_body for subsequent chat:
    {
        "question": ,
        "chat_history": chat_history as string of list from API response before.
    }

    sample Curl command to test the API:
    >>> curl -i -X POST \
    -H 'Content-Type: application/json' \
    -d '{"question": "what can you do?", "chat_history": "[]" }' \ 
    http://127.0.0.1:8000/chat

    """
    question = request.json["question"]
    chat_history = request.json["chat_history"]
    logger.info(f"DBAI chat for question : [{question}]")

    try:
        dbai = DBAI(
                proj_id=PROJECT_ID,
                dataset_id=DATASET_ID,
                tables_list=TABLES_LIST
            )
        
        history = eval(chat_history) # convert string to list
        if len(history) < 1:
            session = dbai.agent.start_chat()
        else:
            history = [Content.from_dict(i) for i in history]
            session = dbai.agent.start_chat(history=history)

        response = dbai.ask(question, session)
        res_id = str(uuid.uuid4())

        response_string = {
            "result_id": res_id,
            "response_text": response.text,
            "interim_steps": str(response.interim_steps),
            "chat_history": str([i.to_dict() for i in session.history]),
            "error_msg": ""
        }

    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            f"DBAI chat unsuccessful: [{question}] {e}"
        )
        response_string = {
            "result_id": 0,
            "response_text": "",
            "interim_steps": "",
            "chat_history": str([i.to_dict() for i in chat_history]),
            "error_msg": str(e)
        }

        
    return json.dumps(response_string)



if __name__ == "__main__":
    app.run(debug=True, port=5000)
