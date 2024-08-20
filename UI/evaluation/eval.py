"""
Evaluation main code for different NL2SQL frameworks
"""
import os
import json
import time
import requests
import pandas as pd
# from langchain_google_vertexai import VertexAI
from google.cloud import bigquery
import streamlit as st
from loguru import logger

from dbai_src.dbai import DBAI_nl2sql

LITE_API_PART = 'lite'
FEW_SHOT_GENERATION = "Few Shot"
GEN_BY_CORE = "CORE_EXECUTORS"
GEN_BY_LITE = "LITE_EXECUTORS"

os.environ[GEN_BY_LITE] = "https://nl2sqlstudio-lite-prod-dot-sl-test-project-363109.uc.r.appspot.com"
os.environ[GEN_BY_CORE] = "https://nl2sqlexecutors-prod-dot-sl-test-project-363109.uc.r.appspot.com"

ENDPOINTS = {
    "Few Shot": "/api/lite/generate",
    "Linear Executor": "/api/executor/linear",
    "Rag Executor": "/api/executor/rag",
    "COT": "/api/executor/cot"
}

params = dict(
    execution=False,
    lite_model=FEW_SHOT_GENERATION,
    access_token=""
)

# llm = VertexAI(temperature=0, model_name="gemini-1.5-pro-001", max_output_tokens=1024)

# def auto_verify(nl_description, ground_truth, llm_amswer):
#     """
#     This function verifies the accuracy of SQL query based on a natural language description
#     and a ground truth query, using text-bison model.

#     Parameters:
#     - nl_description (str): The natural language description of the SQL query.
#     - ground_truth (str): The ground truth SQL query.
#     - llm_amswer (str): The student's generated SQL query for validation.
#     Returns:
#     str: "Yes" if the student's answer matches the ground truth and fits the NL description correctly,
#          "No" otherwise.
#     """
#     prompt = f'''You are an expert at validating SQL queries. Given the Natrual language description
#       and the SQL query corresponding to that description, please check if the students answer is correct.
#       There can be different ways to achieve the same result by forming the query differently.
#       If the students SQL query matches the ground truth and fits the NL description correctly, then return yes
#       else return no.
#       Natural language description: {nl_description}
#       Ground truth: {ground_truth}
#       students answer: {llm_amswer}
#     '''
#     return llm(prompt)


def execute_sql_query(query, client, job_config):
    """Execute given SQL query to fetch result"""
    try:
        cleaned_query = query.replace("\\n", " ").replace("\n", " ").replace("\\", "")
        query_job = client.query(cleaned_query, job_config=job_config)
        response = query_job.result().to_dataframe()
    except Exception as e:  # pylint: disable=broad-except
        response = f"{str(e)}"

    return response


def dbai_framework(question, bq_project_id, bq_dataset_id, tables_list=[]):
    """call DBAI to get nl2sql response"""
    dbai_nl2sql = DBAI_nl2sql(
            proj_id=bq_project_id,
            dataset_id=bq_dataset_id,
            tables_list=tables_list
        )
    return dbai_nl2sql.get_sql(question).generated_sql


def call_generate_sql_api(question, endpoint) -> tuple[str, str]:
    """
        Common SQL generation function
    """
    if LITE_API_PART in endpoint:
        api_url = os.getenv('LITE_EXECUTORS')
        few_shot_gen = False
        if params["lite_model"] == FEW_SHOT_GENERATION:
            few_shot_gen = True
        data = {"question": question,
                "execute_sql": params["execution"],
                "few_shot": few_shot_gen}
    else:
        api_url = os.getenv('CORE_EXECUTORS')
        data = {"question": question,
                "execute_sql": params["execution"]}

    headers = {"Content-type": "application/json",
               "Authorization": f"Bearer {params['access_token']}"}
    api_endpoint = f"{api_url}/{endpoint}"

    logger.info(f"Invoking API : {api_endpoint}")
    logger.info(f"Provided parameters are : Data = {data}")
    api_response = requests.post(api_endpoint,
                                 data=json.dumps(data),
                                 headers=headers,
                                 timeout=None)

    exec_result = ""
    try:
        resp = api_response.json()
        logger.info(f"API resonse : {resp}")
        sql = resp['generated_query']
        exec_result = resp['sql_result']
    except RuntimeError:
        sql = "Execution Failed ! Error encountered in RAG Executor"

    logger.info(f"Generated SQL = {sql}")
    return sql, exec_result


def db_setup(project_id, dataset_id, metadata_path, method):
    """ """
    token = "Bearer "
    body = {
        "proj_name": project_id,
        "bq_dataset": dataset_id,
    }
    headers = {"Content-type": "application/json",
               "Authorization": token}
    if method == FEW_SHOT_GENERATION:
        url = os.getenv(GEN_BY_LITE)
    else:
        url = os.getenv(GEN_BY_CORE)

    if metadata_path:
        with open(metadata_path, "r") as f:
            string_data = f.read()
        files = {"file": (metadata_path.split("/")[-1], string_data)}
        body["metadata_file"] = metadata_path.split("/")[-1]

    _ = requests.post(
            url=url+"/projconfig",
            data=json.dumps(body),
            headers=headers,
            timeout=None)

    if metadata_path:
        _ = requests.post(
                url=url+"/uploadfile",
                headers={"Authorization": token},
                files=files,
                timeout=None
            )


def bq_evaluator(
        bq_project_id,
        bq_dataset_id,
        ground_truth_path,
        method,
        metadata_path=None,
        pb=None,
        render_result=True,
        ):
    """ """
    ts = time.strftime("%y%m%d%H%M")
    client = bigquery.Client(project=bq_project_id)
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=100000000,
        default_dataset=f'{bq_project_id}.{bq_dataset_id}'
        )
    if method != "DBAI":
        db_setup(bq_project_id, bq_dataset_id, metadata_path, method)
    df = pd.read_csv(ground_truth_path)

    all_results = []
    for idx, (question, ground_truth_sql) in df.iterrows():
        match method:
            case "DBAI":
                generated_query = dbai_framework(
                    question, bq_project_id, bq_dataset_id)
            case _:
                generated_query, _ = call_generate_sql_api(
                    question=question, endpoint=ENDPOINTS[method])

        generated_query_result = execute_sql_query(generated_query, client, job_config)
        actual_query_result = execute_sql_query(ground_truth_sql, client, job_config)

        # llm_rating = auto_verify(question, ground_truth_sql, generated_query)
        llm_rating = 'No'
        result_eval = 0
        try:
            if generated_query_result.equals(actual_query_result):
                result_eval = 1
            else:
                result_eval = 0
        except RuntimeError:
            result_eval = 0

        out = [(
            question, ground_truth_sql, actual_query_result, generated_query,
            generated_query_result, llm_rating, result_eval
            )]
        all_results.extend(out)

        out_df = pd.DataFrame(
            out,
            columns=[
                'question', 'ground_truth_sql', 'actual_query_result',
                'generated_query', 'generated_query_result', 'query_eval', 'result_eval'
                ])
        out_df.to_csv(f'evaluation/eval_output/eval_result_{ts}.csv', index=False, mode='a')

        if pb:
            pb.progress(
                (idx+1)/len(df),
                text=f"Evaluation in progress. Please wait... {idx+1}/{len(df)}"
                )
        if render_result:
            if idx == 0:
                redndered_df = st.dataframe(out_df)
            else:
                redndered_df.add_rows(out_df)

    pb.empty()
    accuracy = out_df.result_eval.sum()/len(df)
    all_results_df = pd.DataFrame(
        all_results,
        columns=[
            'question', 'ground_truth_sql', 'actual_query_result',
            'generated_query', 'generated_query_result', 'query_eval', 'result_eval'
            ])
    print(f'Accuracy: {accuracy}')
    return {
        "accuracy": accuracy,
        "output": all_results_df
    }


if __name__ == '__main__':
    BQ_PROJECT_ID = 'proj-kous'
    BQ_DATASET_ID = 'nl2sql_fiserv'
    GROUND_TRUTH_PATH = 'evaluation/fiserv_ground_truth.csv'
    METADATA_PATH = "./nl2sql_src/cache_metadata/fiserv.json"
    METHOD = "lite"

    bq_evaluator(BQ_PROJECT_ID, BQ_DATASET_ID, GROUND_TRUTH_PATH, METHOD, METADATA_PATH)
