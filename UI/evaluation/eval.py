"""
Evaluation main code for different NL2SQL frameworks.
"""
import json
import os
import time
from google.cloud import bigquery
from loguru import logger
import pandas as pd
import requests
import streamlit as st

from dbai_src.dbai import DBAI_nl2sql, validate_readonly_sql
from utils import validate_gcp_project_and_dataset

LITE_API_PART = "lite"
FEW_SHOT_GENERATION = "Few Shot"
GEN_BY_CORE = "CORE_EXECUTORS"
GEN_BY_LITE = "LITE_EXECUTORS"

os.environ.setdefault(
    GEN_BY_LITE,
    "https://nl2sqlstudio-lite-prod-dot-sl-test-project-363109"
    ".uc.r.appspot.com",
)
os.environ.setdefault(
    GEN_BY_CORE,
    "https://nl2sqlexecutors-prod-dot-sl-test-project-363109.uc.r.appspot.com",
)

ENDPOINTS = {
    "Few Shot": "/api/lite/generate",
    "Zero Shot": "/api/lite/generate",
    "Linear Executor": "/api/executor/linear",
    "Rag Executor": "/api/executor/rag",
    "Chain of Thought": "/api/executor/cot",
    "COT": "/api/executor/cot",
}

params = dict(
    execution=False,
    lite_model=FEW_SHOT_GENERATION,
    access_token="",
)


def _get_bearer_token() -> str:
    """Return active OIDC token from Streamlit session or module params."""
    try:
        session_tok = (
            st.session_state.get("token")
            or st.session_state.get("access_token")
        )
        if session_tok:
            return str(session_tok)
    except Exception:
        pass
    return str(params.get("access_token") or "")


def execute_sql_query(query, client, job_config):
    """Validate single-statement read-only SQL before executing on BigQuery."""
    normalized_query = str(query or "").replace("\\n", "\n")
    validated_query = validate_readonly_sql(normalized_query)
    cleaned_query = " ".join(validated_query.splitlines()).strip()
    try:
        query_job = client.query(cleaned_query, job_config=job_config)
        response = query_job.result().to_dataframe()
    except Exception as exc:  # pylint: disable=broad-except
        response = f"{str(exc)}"

    return response


def dbai_framework(question, bq_project_id, bq_dataset_id, tables_list=None):
    """Call DBAI to get NL2SQL response."""
    dbai_nl2sql = DBAI_nl2sql(
        proj_id=bq_project_id,
        dataset_id=bq_dataset_id,
        tables_list=tables_list or [],
    )
    return dbai_nl2sql.get_sql(question).generated_sql


def call_generate_sql_api(question, endpoint) -> tuple[str, str]:
    """Common SQL generation API client."""
    if LITE_API_PART in endpoint:
        api_url = os.getenv(GEN_BY_LITE)
        few_shot_gen = params["lite_model"] == FEW_SHOT_GENERATION
        data = {
            "question": question,
            "execute_sql": params["execution"],
            "few_shot": few_shot_gen,
        }
    else:
        api_url = os.getenv(GEN_BY_CORE)
        data = {
            "question": question,
            "execute_sql": params["execution"],
        }

    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {_get_bearer_token()}",
    }
    api_endpoint = f"{api_url}/{endpoint}"

    logger.info(f"Invoking API : {api_endpoint}")
    api_response = requests.post(
        api_endpoint,
        data=json.dumps(data),
        headers=headers,
        timeout=60,
    )

    exec_result = ""
    try:
        resp = api_response.json()
        sql = resp["generated_query"]
        exec_result = resp["sql_result"]
    except (RuntimeError, ValueError, KeyError):
        sql = "Execution Failed ! Error encountered in Executor"

    logger.info(f"Generated SQL = {sql}")
    return sql, exec_result


def db_setup(project_id, dataset_id, metadata_path, method):
    """Configure backend project and upload metadata with auth."""
    valid_proj, valid_ds = validate_gcp_project_and_dataset(
        project_id, dataset_id
    )
    token = f"Bearer {_get_bearer_token()}"
    body = {
        "proj_name": valid_proj,
        "bq_dataset": valid_ds,
    }
    headers = {
        "Content-type": "application/json",
        "Authorization": token,
    }
    if method == FEW_SHOT_GENERATION:
        url = os.getenv(GEN_BY_LITE)
    else:
        url = os.getenv(GEN_BY_CORE)

    files = None
    if metadata_path:
        meta_name = os.path.basename(str(metadata_path))
        with open(metadata_path, "r", encoding="utf-8") as meta_f:
            string_data = meta_f.read()
        files = {"file": (meta_name, string_data)}
        body["metadata_file"] = meta_name

    _ = requests.post(
        url=url + "/projconfig",
        data=json.dumps(body),
        headers=headers,
        timeout=30,
    )

    if metadata_path and files is not None:
        _ = requests.post(
            url=url + "/uploadfile",
            headers={"Authorization": token},
            files=files,
            timeout=30,
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
    """Evaluate NL2SQL framework against read-only ground-truth SQL CSV."""
    valid_proj, valid_ds = validate_gcp_project_and_dataset(
        bq_project_id, bq_dataset_id
    )
    if not ground_truth_path or not os.path.isfile(ground_truth_path):
        raise ValueError("Ground-truth CSV file path does not exist")

    df = pd.read_csv(ground_truth_path)
    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain at least 2 columns: Question and golden_sql"
        )

    validated_rows = []
    for _, row in df.iterrows():
        question = str(row.iloc[0])
        raw_gt_sql = str(row.iloc[1])
        safe_gt_sql = validate_readonly_sql(raw_gt_sql)
        validated_rows.append((question, safe_gt_sql))

    ts = time.strftime("%y%m%d%H%M")
    client = bigquery.Client(project=valid_proj)
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=100000000,
        default_dataset=f"{valid_proj}.{valid_ds}",
    )
    if method != "DBAI":
        db_setup(valid_proj, valid_ds, metadata_path, method)

    eval_out_dir = os.path.join("evaluation", "eval_output")
    os.makedirs(eval_out_dir, exist_ok=True)

    all_results = []
    rendered_df = None
    for idx, (question, ground_truth_sql) in enumerate(validated_rows):
        if method == "DBAI":
            generated_query = dbai_framework(question, valid_proj, valid_ds)
        else:
            generated_query, _ = call_generate_sql_api(
                question=question,
                endpoint=ENDPOINTS[method],
            )

        try:
            generated_query_result = execute_sql_query(
                generated_query,
                client,
                job_config,
            )
        except ValueError as exc:
            generated_query_result = f"Rejected unsafe SQL: {exc}"

        actual_query_result = execute_sql_query(
            ground_truth_sql,
            client,
            job_config,
        )

        llm_rating = "No"
        result_eval = 0
        try:
            if (
                hasattr(generated_query_result, "equals")
                and generated_query_result.equals(actual_query_result)
            ):
                result_eval = 1
        except (RuntimeError, TypeError, ValueError):
            result_eval = 0

        out = [(
            question,
            ground_truth_sql,
            actual_query_result,
            generated_query,
            generated_query_result,
            llm_rating,
            result_eval,
        )]
        all_results.extend(out)

        out_df = pd.DataFrame(
            out,
            columns=[
                "question",
                "ground_truth_sql",
                "actual_query_result",
                "generated_query",
                "generated_query_result",
                "query_eval",
                "result_eval",
            ],
        )
        out_df.to_csv(
            os.path.join(eval_out_dir, f"eval_result_{ts}.csv"),
            index=False,
            mode="a",
        )

        if pb:
            pb.progress(
                (idx + 1) / max(len(validated_rows), 1),
                text=(
                    "Evaluation in progress. "
                    f"Please wait...{idx + 1}/{len(validated_rows)}"
                ),
            )
        if render_result:
            if idx == 0:
                rendered_df = st.dataframe(out_df)
            elif rendered_df is not None:
                rendered_df.add_rows(out_df)

    if pb:
        pb.empty()
    all_results_df = pd.DataFrame(
        all_results,
        columns=[
            "question",
            "ground_truth_sql",
            "actual_query_result",
            "generated_query",
            "generated_query_result",
            "query_eval",
            "result_eval",
        ],
    )
    accuracy = (
        all_results_df["result_eval"].sum() / len(all_results_df)
        if len(all_results_df) > 0
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "output": all_results_df,
    }
