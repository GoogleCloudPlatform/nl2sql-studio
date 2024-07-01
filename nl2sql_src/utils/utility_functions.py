"""
    Utilities for working with BQ, Saving Project configuration
    Saving User feedback, etc
"""
import os
import json

# import pandas as pd

from google.cloud import bigquery
from dotenv import load_dotenv
from loguru import logger

# from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
import vertexai

load_dotenv()
# bigquery_connection_string = "bigquery://sl-test-project-363109/zoominfo"
PROJ_CONFIG_FILE = "utils/proj_config.json"
SQL_LOG_FILE = "utils/sqlgen_log.json"

proj_config_dict = {
    "default": {
        "proj_name": "sl-test-project-363109",
        "dataset": "zoominfo",
        "metadata_file": "metadata_cache.json",
    },
    "config": {"proj_name": "", "dataset": "", "metadata_file": ""},
}


def initialize_db(proj="sl-test-project-363109", dataset="zoominfo"):
    """
    Initialize the BQ connection string
    """
    print("Init init DB ", proj, dataset)
    bigquery_connection_string = f"bigquery://{proj}/{dataset}"
    print(bigquery_connection_string)
    logger.info(f"BQ connection string: {bigquery_connection_string}")
    return bigquery_connection_string


def config_project(
    proj_name="sl-test-project-363109",
    dataset="zoominfo",
    metadata_file="zoominfo_tables.json",
):
    """
    Save the project configuration details
    """
    try:
        with open(PROJ_CONFIG_FILE, "r", encoding="utf-8") as infile:
            proj_config = json.load(infile)
    # except:
    except FileNotFoundError:

        logger.error("File not found, using default configuration")
        proj_config = proj_config_dict

    logger.info(
        f"Saving following data : {proj_name}, {dataset}, {metadata_file}"
        )
    proj_config["config"]["proj_name"] = proj_name
    proj_config["config"]["dataset"] = dataset
    proj_config["config"]["metadata_file"] = metadata_file
    json_obj = json.dumps(proj_config, indent=4)
    with open(PROJ_CONFIG_FILE, "w", encoding="utf-8") as outfile:
        outfile.write(json_obj)
        # json.dump(proj_config, outfile)
    with open(PROJ_CONFIG_FILE, "r", encoding="utf-8") as infile:
        proj_config = json.load(infile)
        logger.info(
            f"New file name : {proj_config['config']['metadata_file']}"
            )

    initialize_db(proj=proj_name, dataset=dataset)


def get_project_config():
    """
    Return the Project configuration details
    """
    try:
        with open(PROJ_CONFIG_FILE, "r", encoding="utf-8") as infile:
            proj_config = json.load(infile)
    # except:
    except FileNotFoundError:
        logger.error("File not found, using default configuration")
        proj_config = proj_config_dict
        proj_config["config"] = proj_config["default"]

    json_obj = json.dumps(proj_config, indent=4)
    with open(PROJ_CONFIG_FILE, "w", encoding="utf-8") as outfile:
        outfile.write(json_obj)
        # json.dump(proj_config, outfile)

    return proj_config


def execute_bq_query(sql_query=""):
    """
    Execute the given query on BigQuery
    """
    project = get_project_config()["config"]["proj_name"]
    client = bigquery.Client(project=project)
    logger.info(f"Execute bq query : {sql_query}")
    # query_job = client.query(
    # "select * from `q_and_a_db.questions_and_gensqls` \
    #                          where created_by = 'CoT executor' " )
    try:
        query_job = client.query(sql_query)
        results = query_job.result()
        results = query_job.to_dataframe()
        logger.info("Execution result = ", results)
        return results
    except Exception:
        return "Execution failed"


# Below function for future upgrades enabling logging within BQ
# def bq_insert_data(result_id="dummy",
#                     question="dummy",
#                     sql="dummy", executor="dummy",
#                     feedback="False"):
#     """
#         Adds a new row to BigQuery table
#     """
#     project = get_project_config()['config']['proj_name']
#     dataset = "q_and_a_db"
#     client = bigquery.Client(project=project)
#     table_ref = f"{project}.{dataset}.questions_and_gensqls"

#     insert_query = f'INSERT INTO {table_ref} VALUES\
#           ("{result_id}", "{question}", "{sql}", "{executor}", {feedback})'
#     logger.info("Query to insert=", insert_query)
#     query_job = client.query(insert_query)
#     query_job.result()


def log_sql(result_id="dummy",
            question="dummy",
            sql="dummy",
            executor="dummy",
            feedback="False"
            ):
    """
    Saves the generated SQL in a log file locally
    """
    print("Logging the data")
    # project = get_project_config()['config']['proj_name']
    try:
        with open(SQL_LOG_FILE, "r", encoding="utf-8") as inpfile:
            logdata = json.load(inpfile)
    # except:
    except FileNotFoundError:
        logdata = {}

    logdata[result_id] = {}
    logdata[result_id]["question"] = question
    logdata[result_id]["sql"] = sql
    logdata[result_id]["executor"] = executor
    logdata[result_id]["feedback"] = feedback
    with open(SQL_LOG_FILE, "w", encoding="utf-8") as outfile:
        json.dump(logdata, outfile)


def log_update_feedback(result_id, user_feedback):
    """
    Updates the saved log with user feedback
    """
    # project = get_project_config()['config']['proj_name']
    try:
        with open(SQL_LOG_FILE, "r", encoding="utf-8") as inpfile:
            logdata = json.load(inpfile)
    # except:
    except RuntimeError:
        logdata = {}
    logdata[result_id]["feedback"] = user_feedback
    with open(SQL_LOG_FILE, "w", encoding="utf-8") as outfile:
        json.dump(logdata, outfile)


# Below for future upgrade to manage usr feedback in BQ
# def bq_update_userfeedback(result_id, user_feedback):
#     """
#         Updates BQ row with user feedback
#     """
#     project = get_project_config()['config']['proj_name']
#     dataset = "q_and_a_db"
#     client = bigquery.Client(project=project)
#     # table_ref = "sl-test-project-363109.q_and_a_db.questions_and_gensqls"
#     table_ref = f"{project}.{dataset}.questions_and_gensqls"
#     update_query = f"UPDATE {table_ref} SET user_feedback={user_feedback}\
#           WHERE result_id = '{result_id}'"
#     logger.info("Query to update = ", update_query)
#     query_job = client.query(update_query )
#     query_job.result()


def result2nl(question, result):
    """
    Converts the BQ query result to Natural language response
    """
    vertexai.init(project="sl-test-project-363109", location="us-central1")
    # question="What are the top 5industries in terms of revenue for companies\
    #       having headquarters in California ?"

    model = TextGenerationModel.from_pretrained("text-bison@001")
    Result2nl_prompt = """
You are an expert Data Analyst. Given a report of SQL query and the question
in natural language, provide a very crisp, short, intuitive and
easy-to-understand summary of the result. If the result does not have any
data, then just mention that briefly in the summary.

question: {question}
result: {result}
"""
    prompt = Result2nl_prompt.format(question=question, result=str(result))
    response = model.predict(prompt, temperature=0.2, max_output_tokens=1024)

    logger.info("Natural language result: ", response.text)
    return response.text


if __name__ == "__main__":
    # print(get_project_config())
    print(os.getcwd())
    log_sql(
        "dummy2",
        "what is revenue for construction industry",
        "select revenue from table1 where indusry='consttruction'",
        "COT",
        "False",
    )

    log_update_feedback("dummy2", True)
