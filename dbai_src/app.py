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


from dbai import DBAI_nl2sql


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
def nl2sql_lite_generate():
    """
    Invokes the DBAI SQL Generator
    """
    question = request.json["question"]
    logger.info(f"DBAI engine for question : [{question}]")

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
