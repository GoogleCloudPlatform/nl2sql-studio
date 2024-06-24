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
    Main file serving the Executors modules exposing the APIs
    for Linear Executor, Chain of Thought executor, RAG Executor
    Updating user feedback etc
"""
# from nl2sql_lib_executors import NL2SQL_Executors
import json
import os
from flask_cors import CORS
from flask import Flask, request
from dotenv import load_dotenv
from loguru import logger

from nl2sql_generic import Nl2sqlBq

PROJECT_ID = 'sl-test-project-363109'
LOCATION = 'us-central1'
DATASET_ID = 'zoominfo'

load_dotenv()

app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

dataset_name = 'zoominfo'


@app.route("/")
def spec():
    """
    Default API Route
    """
    logger.info("Welcome to NL2SQL Lite")
    return json.dumps({"response": "NL2SQL Lite Generation library"})


@app.route("/api/lite/generate", methods=["POST"])
def nl2sql_lite_generate():
    """
    Invokes the NL2SQL Lite SQL Generator
    """
    question = request.json["question"]
    execute_sql = request.json["execute_sql"]
    few_shot = request.json["few_shot"]

    logger.info(f"NL2SQL Lite engine for question : [{question}]")

    try:
        curdir = os.getcwd()
        metadata_json_path = f"{curdir}/cache_metadata/metadata_cache.json"
        logger.info(f"path  {metadata_json_path}")

        nl2sqlbq_client_base = Nl2sqlBq(project_id=PROJECT_ID,
                                        dataset_id=DATASET_ID,
                                        metadata_json_path=metadata_json_path,
                                        model_name="text-bison@002",
                                        tuned_model=False)
        if few_shot:
            logger.info("NL2SQL Studio Lite - Few shot SQL generation")
            sql = nl2sqlbq_client_base.generate_sql_few_shot(question)
        else:
            logger.info("NL2SQL Studio Lite - SQL generation")
            sql = nl2sqlbq_client_base.generate_sql(question)

        logger.info(f"NL2SQL Studio Lite generated SQL = {sql}")
        sql_result = ""
        res_id = "lite"
        response_string = {
            "result_id": res_id,
            "generated_query": sql,
            "sql_result": sql_result,
            "error_msg": "",
        }
        # log_sql(res_id, question, sql, "Linear Executor", execute_sql)
        if execute_sql:
            try:
                results = nl2sqlbq_client_base.execute_query(sql)
                sql_result = nl2sqlbq_client_base.result2nl(result=results,
                                                            question=question)
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "error_msg": "",
                }
            except Exception:
                logger.error("Error executing the query on BigQuery")
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "error_msg": "Error in NL2SQL Studio Lite Query generation",
                }
    except Exception:
        logger.error(f"NL2SQL Lite SQL Generation uncussessful: [{question}]")
        response_string = {
            "result_id": 0,
            "generated_query": "",
            "sql_result": "",
            "error_msg": "Error in NL2SQL Studio Lite Query generation",
        }

    return json.dumps(response_string)


# @app.route("/projconfig", methods=["POST"])
# def project_config():
#     """
#     Updates the Project Configuration details
#     """
#     logger.info("Updating project configuration")
#     project = request.json["proj_name"]
#     dataset = request.json["bq_dataset"]
#     metadata_file = request.json["metadata_file"]
#     config_project(project, dataset, metadata_file)

#     return json.dumps({"status": "success"})


# @app.route("/uploadfile", methods=["POST"])
# def upload_file():
#     """
#     Saves the data dictionary / metadata cache data
#      received over HTTP request into a file
#     """
#     logger.info("File received")
#     try:
#         file = request.files["file"]
#         data = file.read()
#         my_json = data.decode("utf8")
#         data2 = json.loads(my_json)
#         data_to_save = json.dumps(data2, indent=4)
#         target_file = get_project_config()["config"]["metadata_file"]
#         with open(f"utils/{target_file}", "w", encoding="utf-8") as outfile:
#             outfile.write(data_to_save)
#         return json.dumps({"status": "Successfully uploaded file"})
#     except RuntimeError:
#         return json.dumps({"status": "Failed to upload file"})


# @app.route("/userfb", methods=["POST"])
# def user_feedback():
#     """
#     Updates the User feedback sent from UI
#     """
#     logger.info("Updating user feedback")
#     result_id = request.json["result_id"]
#     feedback = request.json["user_feedback"]
#     try:
#         log_update_feedback(result_id, feedback)
#         return json.dumps({"response": "successfully updated user feedback"})
#     except RuntimeError:
#         return json.dumps({"response": "failed to update user feedback"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
