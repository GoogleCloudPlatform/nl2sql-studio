"""
    Main file serving the Executors modules exposing the APIs
    for Linear Executor, Chain of Thought executor, RAG Executor
    Updating user feedback etc
"""
# from nl2sql_lib_executors import NL2SQL_Executors
import sys
import inspect
import json
import os
from flask_cors import CORS
from flask import Flask, g, request
from dotenv import load_dotenv
from loguru import logger

from utils.utility_functions import (
    initialize_db,
    config_project,
    execute_bq_query,
    log_update_feedback,
    result2nl,
    get_project_config,
    log_sql,
    sanitize_metadata_filename,
    validate_project_and_dataset,
    validate_readonly_sql,
    verify_bearer_token,
)
from nl2sql_query_embeddings import Nl2Sql_embed

load_dotenv()

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type, Authorization"


def _load_data_dictionary(
    metadata_file: str, dataset_key: str | None = None
) -> dict:
    """Load metadata JSON safely from nl2sql_library/utils."""
    safe_file = sanitize_metadata_filename(metadata_file)
    candidates = [
        os.path.join(currentdir, "utils", safe_file),
        os.path.join("utils", safe_file),
    ]
    spider_rows = {}
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as handle:
                spider_rows = json.load(handle)
            break
    entry = {
        "description": (
            "This dataset contains information about the concerts "
            "singers, country they belong to, stadiums where the "
            "concerts happened"
        ),
        "tables": spider_rows,
    }
    result = {"nl2sql_spider": entry}
    if dataset_key and dataset_key != "nl2sql_spider":
        result[dataset_key] = entry
    return result


dataset_name = get_project_config()["config"]["dataset"]
bigquery_connection_string = initialize_db(
    get_project_config()["config"]["proj_name"],
    get_project_config()["config"]["dataset"],
)
data_file_name = sanitize_metadata_filename(
    get_project_config()["config"]["metadata_file"]
)
data_dictionary_read = _load_data_dictionary(data_file_name)


@app.before_request
def _enforce_bearer_auth():
    """Require a verified Google OIDC Bearer token on API endpoints."""
    if request.method == "OPTIONS":
        return None
    if request.method == "GET" and request.path == "/":
        return None
    try:
        g.auth_claims = verify_bearer_token(
            request.headers.get("Authorization")
        )
    except Exception:
        return json.dumps({"error": "Unauthorized"}), 401
    return None


@app.route("/")
def spec():
    """
    Default API Route
    """
    return json.dumps({"response": "Multi mode NL2SQL Generation library"})


@app.route("/api/executor/linear", methods=["POST"])
def linear_executor():
    """
    Invokes the Linear Executor
    """
    question = request.json["question"]
    execute_sql = bool(request.json.get("execute_sql", False))

    logger.info(f"Linear Execution engine for question : [{question}]")
    from nl2sql_lib_executors import NL2SQL_Executors

    try:
        active_cfg = get_project_config()["config"]
        nle = NL2SQL_Executors()
        res_id, sql, df = nle.linear_executor(
            question=question,
            data_dict=_load_data_dictionary(
                active_cfg["metadata_file"], active_cfg["dataset"]
            ),
            execute_sql=execute_sql,
        )
        sql_result = ""
        response_string = {
            "result_id": res_id,
            "generated_query": sql,
            "sql_result": sql_result,
            "error_msg": "",
        }
        log_sql(res_id, question, sql, "Linear Executor", execute_sql)
        if execute_sql:
            try:
                validated_sql = validate_readonly_sql(sql)
                result = execute_bq_query(validated_sql)
                sql_result = result2nl(question, result)
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "df": df.to_json() if df is not None else "",
                    "error_msg": "",
                }
            except (RuntimeError, ValueError):
                print("internal try catch")
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "df": df.to_json() if df is not None else "",
                    "error_msg": "",
                }
    except (RuntimeError, ValueError):
        logger.debug(f"Linear SQL Generation uncussessful : [{question}]")
        response_string = {
            "result_id": 0,
            "generated_query": "",
            "sql_result": "",
            "error_msg": "Error encountered in Linear executor",
        }

    return response_string


@app.route("/api/executor/cot", methods=["POST"])
def cot_executor():
    """
    Invokes the Chain of Thought executor
    """
    question = request.json["question"]
    execute_sql = bool(request.json.get("execute_sql", False))

    logger.info("CoT SQL Generation engine for question : [{question}]")
    from nl2sql_lib_executors import NL2SQL_Executors

    try:
        logger.info("CoT initialising the class")
        active_cfg = get_project_config()["config"]
        nle = NL2SQL_Executors()
        res_id, sql, df = nle.cot_executor(
            question=question,
            data_dict=_load_data_dictionary(
                active_cfg["metadata_file"], active_cfg["dataset"]
            ),
            execute_sql=execute_sql,
        )
        sql_result = ""
        response_string = {
            "result_id": res_id,
            "generated_query": sql,
            "sql_result": sql_result,
            "error_msg": "",
        }
        sql2 = "\t".join([line.strip() for line in sql])
        log_sql(res_id, question, str(sql2), "CoT Executor", execute_sql)
        if execute_sql:
            try:
                validated_sql = validate_readonly_sql(sql)
                result = execute_bq_query(validated_sql)
                sql_result = result2nl(question, result)
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "df": df.to_json() if df is not None else "",
                    "error_msg": "",
                }
            except (RuntimeError, ValueError):
                print("internal try catch")
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "df": df.to_json() if df is not None else "",
                    "error_msg": "",
                }
    except (RuntimeError, ValueError):
        logger.debug(f"CoT SQL generation unsuccessful : [{question}]")
        response_string = {
            "result_id": 0,
            "generated_query": "",
            "sql_result": "",
            "error_msg": "Error encountered in CoT executor",
        }

    return response_string


@app.route("/api/executor/rag", methods=["POST"])
def rag_executor():
    """
    Invokes the RAG Executor
    """
    question = request.json["question"]
    execute_sql = request.json["execute_sql"]

    logger.info("RAG SQL Generation engine for question : [{question}]")
    from nl2sql_lib_executors import NL2SQL_Executors

    try:
        nle = NL2SQL_Executors()
        res_id, sql = nle.rag_executor(question=question)
        # res_id, sql = nle.generate_query(question)
        sql_result = ""
        response_string = {
            "result_id": res_id,
            "generated_query": sql,
            "sql_result": sql_result,
            "error_msg": "",
        }
        log_sql(res_id, question, sql, "Rag Executor", execute_sql)
        if execute_sql:
            try:
                validated_sql = validate_readonly_sql(sql)
                result = execute_bq_query(validated_sql)
                sql_result = result2nl(question, result)
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "error_msg": "",
                }
            except (RuntimeError, ValueError):
                print("internal try catch")
                response_string = {
                    "result_id": res_id,
                    "generated_query": sql,
                    "sql_result": sql_result,
                    "error_msg": "",
                }
    except (RuntimeError, ValueError):
        logger.debug(f"RAG SQL generation unsuccessful : [{question}]")
        response_string = {
            "result_id": 0,
            "generated_query": "",
            "sql_result": "",
            "error_msg": "Error encountered in RAG executor",
        }

    return json.dumps(response_string)


@app.route("/projconfig", methods=["POST"])
def project_config():
    """
    Updates the Project Configuration details
    """
    logger.info("Updating project configuration")
    try:
        payload = request.get_json(force=True) or {}
        project, dataset = validate_project_and_dataset(
            payload["proj_name"],
            payload["bq_dataset"],
        )
        raw_meta = (
            payload.get("metadata_file")
            or get_project_config()["config"]["metadata_file"]
        )
        metadata_file = sanitize_metadata_filename(raw_meta)
        logger.info(f"Received info - {project}, {dataset}, {metadata_file}")

        config_project(project, dataset, metadata_file)

        return json.dumps({"status": "success"})
    except (RuntimeError, ValueError, KeyError, TypeError):
        return json.dumps({"status": "Failed to update project config"}), 400


@app.route("/uploadfile", methods=["POST"])
def upload_file():
    """
    Saves the data dictionary / metadata cache data
     received over HTTP request into a file
    """
    logger.info("File received")
    try:
        file = request.files["file"]
        data = file.read()
        my_json = data.decode("utf8")
        data2 = json.loads(my_json)
        data_to_save = json.dumps(data2, indent=4)
        target_file = sanitize_metadata_filename(
            get_project_config()["config"]["metadata_file"]
        )
        logger.info(f"Saving file as : {target_file}")

        utils_dir = os.path.realpath(os.path.join(currentdir, "utils"))
        target_path = os.path.realpath(os.path.join(utils_dir, target_file))
        if not target_path.startswith(utils_dir + os.sep):
            raise ValueError("Path traversal detected")

        with open(target_path, "w", encoding="utf-8") as outfile:
            outfile.write(data_to_save)

        logger.info(f"List of files - {os.listdir(utils_dir)}")

        return json.dumps({"status": "Successfully uploaded file"})
    except (RuntimeError, ValueError, KeyError):
        return json.dumps({"status": "Failed to upload file"}), 400


@app.route("/userfb", methods=["POST"])
def user_feedback():
    """
    Updates the User feedback sent from UI
    """
    logger.info("Updating user feedback")
    result_id = request.json["result_id"]
    feedback = request.json["user_feedback"]
    try:
        log_update_feedback(result_id, feedback)
        return json.dumps({"response": "successfully updated user feedback"})
    except RuntimeError:
        return json.dumps({"response": "failed to update user feedback"})


@app.route("/execsql", methods=["POST"])
def execute_sql_query():
    """
    Executes the query on BQ
    """
    try:
        sql = validate_readonly_sql(request.json["sql"])
        result = execute_bq_query(sql)
        print("result = ", result)
        sql_result = result.to_dict()  # orient="records")
        res_id = ""
        result_text = result2nl("", sql_result)
        response_string = {
            "result_id": res_id,
            "generated_query": sql,
            "sql_result": result_text,
            "error_msg": "",
        }
        return json.dumps(response_string)
    except (RuntimeError, ValueError, KeyError) as exc:
        return (
            json.dumps({
                "result_id": "",
                "generated_query": "",
                "sql_result": "",
                "error_msg": str(exc),
            }),
            400,
        )


@app.route('/api/record/create', methods=['POST'])
def create_record():
    """
        Insert record with Question and MappedSQL in the Table or Local file
    """
    question = request.json['question']
    mappedsql = request.json['sql']
    logger.info(f"Inserting data. Input : {question} and {mappedsql}")
    try:
        # pge = PgSqlEmb(PGPROJ, PGLOCATION, PGINSTANCE, PGDB, PGUSER, PGPWD)
        # pge.insert_row(question, mappedsql)
        embed = Nl2Sql_embed()
        embed.insert_data(question=question, sql=mappedsql)
        return json.dumps({"response": "Successfully inserted record"})
    except RuntimeError:
        return json.dumps({"response": "Unable to insert record"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
