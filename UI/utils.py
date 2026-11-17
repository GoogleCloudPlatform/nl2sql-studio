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
import html as html_lib
import tempfile
import uuid
import configparser
import random
import string
from pathlib import Path
import streamlit as st
from streamlit_feedback import streamlit_feedback
from streamlit.components.v1 import html
from dotenv import load_dotenv
from loguru import logger
import requests
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token as google_id_token
from werkzeug.utils import secure_filename
import plotly.express as px
from fuzzywuzzy import process
import pandas as pd

import vertexai
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

import looker_sdk
from looker_sdk.sdk.api40 import models as ml

load_dotenv()

# API Calls
LITE_API_PART = 'lite'
FEW_SHOT_GENERATION = "Few Shot"
GEN_BY_CORE = "CORE_EXECUTORS"
GEN_BY_LITE = "LITE_EXECUTORS"

# Looker sdk variable
sdk = None

UI_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = UI_DIR / "config.ini"

_SQL_TOKEN_RE = re.compile(
    r"("
    r"(?:\b(?:[rR][bB]?|[bB][rR]))(?:"
    r"'''(?:\\\\|\\[^'\\]|(?!''')[^\\])*?'''"
    r"|\"\"\"(?:\\\\|\\[^\"\\]|(?!\"\"\")[^\\])*?\"\"\""
    r"|'(?:\\\\|\\[^'\\\r\n]|[^'\\\r\n])*'"
    r"|\"(?:\\\\|\\[^\"\\\r\n]|[^\"\\\r\n])*\""
    r")"
    r"|(?<![A-Za-z0-9_])(?:[bB])?(?:"
    r"'''(?:\\[\s\S]|(?!''')[\s\S])*?'''"
    r"|\"\"\"(?:\\[\s\S]|(?!\"\"\")[\s\S])*?\"\"\""
    r"|'(?:\\[\s\S]|[^'\\\r\n])*'"
    r"|\"(?:\\[\s\S]|[^\"\\\r\n])*\""
    r")"
    r"|`(?:\\[\s\S]|[^`\\\r\n])*`"
    r")"
    r"|(--[^\r\n]*|#[^\r\n]*|/\*[\s\S]*?\*/)"
)
_DISALLOWED_SQL_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE(?!\s*\()"
    r"|MERGE|GRANT|REVOKE|CALL|EXEC|EXECUTE|EXPORT|LOAD|UNDROP|INTO)\b",
    re.IGNORECASE,
)
_UNMASKED_SQL_DELIMS_RE = re.compile(r"['\"`\\#]|--|/\*|\*/")
_GCP_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_BQ_DATASET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,1023}$")
_MARKDOWN_SPECIAL_MAP = str.maketrans({
    "[": "&#91;",
    "]": "&#93;",
    "(": "&#40;",
    ")": "&#41;",
    "!": "&#33;",
    "*": "&#42;",
    "_": "&#95;",
    "`": "&#96;",
})


def _clean_config_val(raw_val: str | None) -> str:
    """Strip whitespace and surrounding literal quotes from config values."""
    if raw_val is None:
        return ""
    return str(raw_val).strip().strip('"').strip("'").strip()


def load_ui_config() -> configparser.ConfigParser:
    """Load UI config.ini reliably regardless of current working directory."""
    config = configparser.ConfigParser()
    if CONFIG_FILE_PATH.exists():
        config.read(str(CONFIG_FILE_PATH), encoding="utf-8")
    else:
        config.read("config.ini", encoding="utf-8")
    return config


def validate_readonly_sql(sql_query: str) -> str:
    """Ensure the SQL query is a single read-only SELECT/WITH statement."""
    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("Empty SQL query")
    cleaned = re.sub(
        r"^\s*```(?:sql)?\s*|\s*```\s*$",
        "",
        sql_query.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = _SQL_TOKEN_RE.sub(
        lambda m: m.group(1) if m.group(1) is not None else " ", cleaned
    ).strip()
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        raise ValueError("Empty SQL query")
    tight_masked = _SQL_TOKEN_RE.sub(
        lambda m: "__LIT__" if m.group(1) is not None else " ", cleaned
    )
    if "__LIT____LIT__" in tight_masked:
        raise ValueError("Adjacent string/identifier literals are not allowed")
    masked = _SQL_TOKEN_RE.sub(
        lambda m: " __LIT__ " if m.group(1) is not None else " ", cleaned
    )
    if _UNMASKED_SQL_DELIMS_RE.search(masked):
        raise ValueError("Unclosed or malformed literal/comment in SQL query")
    if ";" in masked:
        raise ValueError("Multiple SQL statements are not allowed")
    if not re.match(r"^\s*\(*\s*(SELECT|WITH)\b", masked, re.IGNORECASE):
        raise ValueError("Only read-only SELECT queries are allowed")
    if _DISALLOWED_SQL_KEYWORDS.search(masked):
        raise ValueError("Disallowed DDL/DML keyword in SQL query")
    return cleaned


def validate_gcp_project_and_dataset(
    project_id: str,
    dataset_id: str,
) -> tuple[str, str]:
    """Validate GCP project and BigQuery dataset identifiers and allowlists."""
    proj = (project_id or "").strip()
    ds = (dataset_id or "").strip()
    if not _GCP_PROJECT_RE.match(proj):
        raise ValueError(f"Invalid GCP project identifier: {project_id!r}")
    if not _BQ_DATASET_RE.match(ds):
        raise ValueError(
            f"Invalid BigQuery dataset identifier: {dataset_id!r}"
        )

    allowed_projs_env = os.getenv("ALLOWED_GCP_PROJECTS", "").strip()
    if allowed_projs_env:
        allowed_projs = {
            p.strip() for p in allowed_projs_env.split(",") if p.strip()
        }
        if proj not in allowed_projs:
            raise ValueError(f"Project {proj!r} is not in allowlist")

    allowed_ds_env = os.getenv("ALLOWED_BQ_DATASETS", "").strip()
    if allowed_ds_env:
        allowed_ds = {
            d.strip() for d in allowed_ds_env.split(",") if d.strip()
        }
        if ds not in allowed_ds:
            raise ValueError(f"Dataset {ds!r} is not in allowlist")
    return proj, ds


def escape_untrusted_text(value: object) -> str:
    """HTML-escape and neutralize Markdown link/image/style injection."""
    if value is None:
        return ""
    escaped = html_lib.escape(str(value), quote=True)
    return escaped.translate(_MARKDOWN_SPECIAL_MAP)


def save_uploaded_csv_safely(
    uploaded_file,
    upload_root: str | None = None,
) -> str:
    """Sanitize uploaded CSV filename and write inside isolated temp dir."""
    raw_name = str(getattr(uploaded_file, "name", "") or "").strip()
    if (
        not raw_name
        or "/" in raw_name
        or "\\" in raw_name
        or ".." in raw_name
        or raw_name.startswith(".")
    ):
        raise ValueError(f"Unsafe upload filename: {raw_name!r}")
    safe_name = secure_filename(raw_name)
    if (
        not safe_name
        or safe_name.startswith(".")
        or ".." in safe_name
        or not safe_name.lower().endswith(".csv")
    ):
        raise ValueError(f"Unsafe or non-CSV upload filename: {raw_name!r}")

    payload = bytes(uploaded_file.getbuffer())
    if not payload or len(payload) > 10 * 1024 * 1024:
        raise ValueError("Uploaded CSV file is empty or exceeds 10 MB limit")

    if upload_root is None:
        target_dir = tempfile.mkdtemp(prefix="nl2sql_eval_upload_")
    else:
        target_dir = os.path.realpath(upload_root)
        os.makedirs(target_dir, mode=0o700, exist_ok=True)

    real_dir = os.path.realpath(target_dir)
    dest_name = f"{uuid.uuid4().hex}_{safe_name}"
    dest_path = os.path.realpath(os.path.join(real_dir, dest_name))
    if not dest_path.startswith(real_dir + os.sep):
        raise ValueError("Path traversal detected in uploaded file path")

    with open(dest_path, "wb") as out_f:
        out_f.write(payload)
    return dest_path


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
    selected_framework = None
    if LITE_API_PART in endpoint:
        selected_framework = "LITE"
        api_url = os.getenv('LITE_EXECUTORS')
        few_shot_gen = False
        if st.session_state.lite_model == FEW_SHOT_GENERATION:
            few_shot_gen = True
        data = {"question": question,
                "execute_sql": st.session_state.execution,
                "few_shot": few_shot_gen}
        st.session_state.sql_generated_by = GEN_BY_LITE
    else:
        selected_framework = "CORE"
        api_url = os.getenv('CORE_EXECUTORS')
        data = {"question": question,
                "execute_sql": st.session_state.execution}
        st.session_state.sql_generated_by = GEN_BY_CORE

    bearer = (
        st.session_state.get("token")
        or st.session_state.get("access_token")
        or ""
    )
    headers = {"Content-type": "application/json",
               "Authorization": f"Bearer {bearer}"}
    api_endpoint = f"{api_url}/{endpoint}"

    logger.info(f"Invoking API : {api_endpoint}")
    logger.info(f"Provided parameters are : Data = {data}")
    api_response = requests.post(api_endpoint,
                                 data=json.dumps(data),
                                 headers=headers,
                                 timeout=None)

    logger.info(f"{str(api_response.text)}")
    # logger.info('-'*100, api_response.text)
    exec_result = ""
    try:
        resp = api_response.json()

        sql = resp['generated_query']
        st.session_state.result_id = resp['result_id']

        if selected_framework == 'CORE' and data["execute_sql"]:
            try:
                # Parse the JSON string to a dictionary
                df_dict = json.loads(resp['df'])
                exec_result_df = pd.DataFrame(df_dict)
                exec_result = ''
            except Exception as e:
                logger.info(f"SQL query execution failed due to {str(e)}")
                exec_result_df = None
                exec_result = ''

        elif selected_framework == 'LITE' and data["execute_sql"]:
            exec_result = resp['sql_result']
            try:
                exec_result_df = run_query(sql)
            except Exception as e:
                logger.info(f"SQL query execution failed due to {str(e)}")
                exec_result_df = None
                exec_result = None
        else:
            exec_result_df = None
            exec_result = ''

    except RuntimeError:
        sql = "Execution Failed ! Error encountered"
        st.session_state.result_id = None

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
    Formats the response string safely with HTML and Markdown escaping.
    """
    safe_sql = escape_untrusted_text(sql)
    normalized_exec = str(exec_result or "").replace("\n", " ")
    safe_exec = escape_untrusted_text(normalized_exec)
    if bool(st.session_state.get("execution")) and safe_exec:
        return f"Generated SQL: {safe_sql}\n\nResult: {safe_exec}"
    return f"Generated SQL: {safe_sql}"

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
    safe_question = escape_untrusted_text(question)
    st.session_state.messages.append({"role": "user",
                                      "content": safe_question,
                                      "dataframe": None})
    st.session_state.messages.append({"role": "assistant",
                                      "content": "Fetching results...",
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

    bearer = (
        st.session_state.get("token")
        or st.session_state.get("access_token")
        or ""
    )
    data = {"question": sample_question, "sql": sample_sql}
    headers = {'Content-type': 'application/json',
               'Accept': 'text/plain',
               "Authorization": f"Bearer {bearer}"}
    _ = requests.post(url=url, data=json.dumps(data),
                      headers=headers,
                      timeout=30)
    st.session_state.add_question_status = True


def back_to_login_page() -> None:
    """
        Open the given URL
    """
    config = load_ui_config()
    url = (
        _clean_config_val(os.getenv("GOOGLE_REDIRECT_URI"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_REDIRECT_URI"))
    )
    st.session_state.token = None
    st.session_state.access_token = None
    st.session_state.login_status = False
    st.query_params.clear()

    if url and re.match(r"^https?://", url, flags=re.IGNORECASE):
        open_script = (
            "<script type=\"text/javascript\">"
            f"window.open({json.dumps(url)}, '_self').focus();"
            "</script>"
        )
        html(open_script)


def init_auth() -> None:
    """
        Authentication Initialisation function
    """
    logger.info("Initialising Authentication process")
    config = load_ui_config()
    google_redirect_uri = (
        _clean_config_val(os.getenv("GOOGLE_REDIRECT_URI"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_REDIRECT_URI"))
    )
    logger.info(f"Redirect URI configured: {bool(google_redirect_uri)}")


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
    config = load_ui_config()
    google_client_id = (
        _clean_config_val(os.getenv("GOOGLE_CLIENT_ID"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_CLIENT_ID"))
    )
    google_redirect_uri = (
        _clean_config_val(os.getenv("GOOGLE_REDIRECT_URI"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_REDIRECT_URI"))
    )
    from urllib.parse import urlencode
    query = urlencode({
        "response_type": "code",
        "client_id": google_client_id,
        "redirect_uri": google_redirect_uri,
        "scope": "openid profile email",
        "access_type": "offline",
    })
    return f"https://accounts.google.com/o/oauth2/auth?{query}"


def verify_google_id_token(
    token: str,
    expected_client_id: str | None = None,
) -> dict:
    """Verify Google OIDC ID token signature, issuer, and audience."""
    if (
        not isinstance(token, str)
        or not token.strip()
        or token.strip() == "dummy token"
    ):
        raise ValueError("Missing or invalid OIDC ID token")
    client_id = _clean_config_val(expected_client_id) or None
    claims = google_id_token.verify_oauth2_token(
        token.strip(),
        google_auth_requests.Request(),
        audience=client_id,
    )
    if claims.get("iss") not in (
        "accounts.google.com",
        "https://accounts.google.com",
    ):
        raise ValueError("Invalid OIDC token issuer")
    if not claims.get("sub"):
        raise ValueError("Missing subject claim in OIDC token")
    return claims


def view_auth_google(code) -> tuple[str | None, str | None]:
    """
        Exchange authorization code for OAuth tokens without logging secrets.
    """
    logger.info("Exchanging OAuth authorization code for tokens")
    if not isinstance(code, str) or not code.strip():
        logger.warning("Empty authorization code received")
        return None, None

    config = load_ui_config()
    google_client_id = (
        _clean_config_val(os.getenv("GOOGLE_CLIENT_ID"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_CLIENT_ID"))
    )
    google_client_secret = (
        _clean_config_val(os.getenv("GOOGLE_CLIENT_SECRET"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_CLIENT_SECRET"))
    )
    google_redirect_uri = (
        _clean_config_val(os.getenv("GOOGLE_REDIRECT_URI"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_REDIRECT_URI"))
    )

    if not google_client_id or not google_client_secret:
        logger.error("Google OAuth client credentials are not configured")
        return None, None

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code.strip(),
        "client_id": google_client_id,
        "client_secret": google_client_secret,
        "redirect_uri": google_redirect_uri,
        "grant_type": "authorization_code",
    }

    try:
        response = requests.post(token_url, data=data, timeout=15)
        response.raise_for_status()
        token_payload = response.json()
        access_token = token_payload.get("access_token")
        id_token = token_payload.get("id_token")
        if not id_token or not access_token:
            logger.error("OAuth token response missing required token fields")
            return None, None
        verify_google_id_token(id_token, google_client_id)
        logger.info("Google OAuth token exchange and verification succeeded")
        return id_token, access_token
    except Exception as exc:
        logger.error(
            f"Google OAuth token exchange failed: {type(exc).__name__}"
        )
        return None, None


def view_get_token(token):
    """
        Verify a Google OIDC ID token using Google's public JWKS keys.
    """
    config = load_ui_config()
    google_client_id = (
        _clean_config_val(os.getenv("GOOGLE_CLIENT_ID"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_CLIENT_ID"))
    )
    if not google_client_id:
        logger.error("GOOGLE_CLIENT_ID is not configured")
        return None
    try:
        return verify_google_id_token(token, google_client_id)
    except Exception as exc:
        logger.error(f"OIDC token verification failed: {type(exc).__name__}")
        return None


def authenticate_session() -> bool:
    """
    Enforce Google OAuth session state across main and multipage routes.
    Fails closed whenever OAuth is enabled or credentials/tokens are invalid.
    """
    config = load_ui_config()
    oauth_mode = (
        _clean_config_val(os.getenv("GOOGLE_OAUTH"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_OAUTH", "ENABLE"))
        or "ENABLE"
    ).upper()
    allow_unauth_dev = (
        _clean_config_val(
            os.getenv("NL2SQL_ALLOW_UNAUTHENTICATED_DEV")
        ).lower()
        == "true"
    )
    if oauth_mode == "DISABLE" and allow_unauth_dev:
        st.session_state.login_status = True
        st.session_state.token = "dev-local-session"
        st.session_state.access_token = "dev-local-session"
        return True

    google_client_id = (
        _clean_config_val(os.getenv("GOOGLE_CLIENT_ID"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_CLIENT_ID"))
    )
    google_client_secret = (
        _clean_config_val(os.getenv("GOOGLE_CLIENT_SECRET"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_CLIENT_SECRET"))
    )
    if not google_client_id or not google_client_secret:
        st.session_state.token = None
        st.session_state.access_token = None
        st.session_state.login_status = False
        return False

    existing_token = st.session_state.get("token")
    if st.session_state.get("login_status") is True and existing_token:
        try:
            verify_google_id_token(existing_token, google_client_id)
            return True
        except Exception:
            st.session_state.token = None
            st.session_state.access_token = None
            st.session_state.login_status = False

    code = None
    try:
        code = st.query_params.get("code")
    except Exception:
        code = None

    if code:
        id_token, _ = view_auth_google(code)
        if id_token:
            st.session_state.token = id_token
            st.session_state.access_token = id_token
            st.session_state.login_status = True
            try:
                st.query_params.clear()
            except Exception:
                pass
            return True

    st.session_state.token = None
    st.session_state.access_token = None
    st.session_state.login_status = False
    return False


def require_authentication() -> bool:
    """Gate a Streamlit page behind Google OAuth and stop if unauthed."""
    if authenticate_session():
        return True
    st.warning("Please sign in with Google OAuth to access NL2SQL Studio.")
    config = load_ui_config()
    client_id = (
        _clean_config_val(os.getenv("GOOGLE_CLIENT_ID"))
        or _clean_config_val(config["DEFAULT"].get("GOOGLE_CLIENT_ID"))
    )
    if client_id:
        auth_url = view_login_google()
        st.markdown(f"[Login with Google]({auth_url})")
    else:
        st.error(
            "Authentication is enabled, but GOOGLE_CLIENT_ID / "
            "GOOGLE_CLIENT_SECRET are not configured."
        )
    st.stop()
    return False


def run_query(sql_query):
    """Run a validated read-only BigQuery SQL query and return a DataFrame."""
    validated_query = validate_readonly_sql(sql_query)
    client = bigquery.Client()
    job = client.query(validated_query)
    df = job.to_dataframe()
    return df


def generate_result(query):

    config = configparser.ConfigParser()
    config.read('config.ini')
    project_id = config['DEFAULT']['PROJECT_ID']
    location = config['DEFAULT']['LOCATION']

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

    vertexai.init(project=project_id, location=location)
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


def plot_histogram(df, x_cols, y_cols=None):
    for x_col in x_cols:
        st.plotly_chart(px.histogram(df, x=x_col, y=y_cols[0],
                                     title=f'Histogram of {x_col},\
                                     {y_cols[0]}'))


def plot_scatter_plot(df, x_cols, y_cols):
    st.plotly_chart(px.scatter(df, x=x_cols[0], y=y_cols[0],
                               title=f'Scatter Plot of {y_cols[0]} \
                                   vs {x_cols[0]}'))


def plot_pie_chart(df, x_cols, y_cols=None):
    if y_cols is not None:
        st.plotly_chart(px.pie(df, names=x_cols[0], values=y_cols[0],
                               title=f'Pie Chart of {x_cols[0]} \
                                   with values  {y_cols[0]}'))
    else:
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


def plot_heatmap(df, x_cols, y_cols=None, z_cols=None):
    """Plot a heatmap of the correlation
    matrix of specified columns."""

    y_cols = y_cols or []
    z_cols = z_cols or []
    columns = x_cols + y_cols + z_cols

    if len(columns) < 2:
        st.write("Please select at least two columns.")
        return

    # Compute the correlation matrix
    correlation_matrix = df[columns].corr()

    # Plot the heatmap
    fig = px.imshow(correlation_matrix, text_auto=True, title="Heatmap")
    st.plotly_chart(fig)


def plot_bubble_chart(df, x_cols, y_cols, z_cols):
    """Plot a bubble chart with specified x, y, and bubble size columns."""
    # Ensure z_cols is not empty
    if not z_cols:
        st.write('Please provide a z_column for bubble size.')
        return

    # Plot the bubble chart
    st.plotly_chart(
        px.scatter(
            df, x=x_cols[0], y=y_cols[0], size=z_cols[0],
            title=f'Bubble Chart of {y_cols[0]} vs {x_cols[0]}\
                with bubble size {z_cols[0]}'
        )
    )


def plot_data(df, plot_type, x_column_list,
              y_column_list=None, z_column_list=None):
    # Plot based on plot_type
    if plot_type == 'Line Chart' and x_column_list and y_column_list:
        plot_line_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Bar Chart' and x_column_list and y_column_list:
        plot_bar_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Histogram' and x_column_list:
        plot_histogram(df, x_column_list, y_column_list)
    elif plot_type == 'Scatter Plot' and x_column_list and y_column_list:
        plot_scatter_plot(df, x_column_list, y_column_list)
    elif plot_type == 'Pie Chart' and x_column_list:
        plot_pie_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Area Chart' and x_column_list and y_column_list:
        plot_area_chart(df, x_column_list, y_column_list)
    elif plot_type == 'Box Plot' and x_column_list and y_column_list:
        plot_box_plot(df, x_column_list, y_column_list)
    elif plot_type == 'Heatmap' and x_column_list:
        plot_heatmap(df, x_column_list, y_column_list, z_column_list)
    elif (plot_type == 'Bubble Chart' and x_column_list and
          y_column_list and z_column_list):
        plot_bubble_chart(df, x_column_list, y_column_list, z_column_list)
    else:
        st.write('Please select appropriate columns for the chosen plot type.')


def run_visualization(df, custom_flag, key_counter):
    columns = df.columns.tolist()
    data_types = [str(dtype) for dtype in df.dtypes]
    metadata = dict(zip(columns, data_types))
    plot_chart_list = ['Line Chart', 'Bar Chart',
                       'Histogram', 'Scatter Plot',
                       'Pie Chart', 'Area Chart',
                       'Box Plot', 'Heatmap', 'Bubble Chart']

    # 3. "plot_chooser" Prompt Template
    plot_chooser_template = f"""
    You are an expert in data visualization.
    Given the following DataFrame metadata:
    dataframe columns: {columns}
    Data Types: {metadata}
    number of columns: {len(columns)}
    You are tasked to suggest:
    1. The best plot type out of {plot_chart_list}
    2. The dataframe columns to use for plotting.
    Consider the data types and potential relationships between columns.
    Remember to suggest below plots according to number of dataframe coumns
    ** 1 Column: Histogram, Pie Chart
    ** 2 Columns: Histogram, Pie Chart, Line Chart, Bar Chart,\
        Scatter Plot, Area Chart, Box Plot
    ** More Than 2 Columns: Heatmap, Bubble Chart
    Output should be strictly in json format as described below:
    ```{{
    "plot_type": "...",
    "x_column": "[...]",
    "y_column": "[...]", // Optional, depending on plot type
    "z_column": "[...]"  // Optional, depending on plot type
    }}
    ```
    """

    result_json_string = generate_result(plot_chooser_template)
    result_json = json.loads(extract_substring(result_json_string))

    if not custom_flag:
        plot_type = result_json.get("plot_type", "line")
        x_column_list = result_json.get("x_column", columns[0])
        logger.info(f"Plot type is : {plot_type}")

        if len(columns) > 2:
            y_column_list = result_json.get("y_column", columns[1])
            z_column_list = result_json.get("y_column", columns[2])
            plot_data(df, plot_type, x_column_list,
                      y_column_list, z_column_list)
        elif len(columns) == 2:
            y_column_list = result_json.get("y_column", columns[1])
            plot_data(df, plot_type, x_column_list, y_column_list)
        else:
            plot_data(df, plot_type, x_column_list)
    else:
        with st.container():
            plot_type = st.selectbox('Select Plot Type',
                                     ['Line Chart', 'Bar Chart',
                                      'Histogram', 'Scatter Plot',
                                      'Pie Chart', 'Area Chart',
                                      'Box Plot', 'Heatmap',
                                      'Bubble Chart'],
                                     key=f'plot_type_{key_counter}')
            x_column_list = st.multiselect('Select Primary Column(s)',
                                           df.columns,
                                           key=f'x_column_{key_counter}')
            y_column_list = (
                st.multiselect(
                    'Select Secondary Column(s) if applicable',
                    df.columns,
                    key=f'y_column_{key_counter}'
                )
            )

            z_column_list = (
                st.multiselect(
                    'Select Other Column(s) if applicable',
                    df.columns,
                    key=f'z_column_{key_counter}'
                )
                if plot_type not in ['Histogram', 'Pie Chart', 'Line Chart',
                                     'Bar Chart', 'Scatter Plot',
                                     'Area Chart', 'Box Plot'] else None
            )

            if z_column_list is not None:
                plot_data(df, plot_type, x_column_list,
                          y_column_list, z_column_list)
            elif y_column_list is not None:
                plot_data(df, plot_type, x_column_list,
                          y_column_list)
            else:
                plot_data(df, plot_type, x_column_list)


def field_selector(df):
    columns = df.columns.tolist()
    data_types = [str(dtype) for dtype in df.dtypes]
    metadata = dict(zip(columns, data_types))
    plot_chart_list = ['looker_column', 'looker_bar']

    plot_chooser_template = f"""
    You are an expert in data visualization.
    Given the following DataFrame metadata:

    dataframe columns: {columns}
    Data Types: {metadata}

    You are tasked to suggest:
    1. The best plot type out of {plot_chart_list}
    2. The dataframe column(s) list to use for plotting.
    Consider the data types and potential relationships between columns.

    Output should be strictly in json format as described below:
    ```{{
    "plot_type": "...",
    "x_column": "[...]",
    }}
    ```
    """
    result_json_string = generate_result(plot_chooser_template)
    result_json = json.loads(extract_substring(result_json_string))
    print(result_json)


def create_random_string(length=10,
                         characters=string.ascii_letters + string.digits):
    return ''.join(random.choice(characters) for _ in range(length))


def create_look_ui():
    try:
        # Retrieve Looker Configs
        config = configparser.ConfigParser()
        config.read('looker.ini')
        base_url = config['Looker']['base_url']
        print(base_url)

        # Initialise Looker SDK
        global sdk

        if sdk is None:
            sdk = looker_sdk.init40("looker.ini")

        query = ml.WriteQuery(
            model="intermediate_ecomm",
            view="intermediate_example_ecommerce",
            fields=["products.cost", "products.brand"],
            limit=5,
            vis_config={
                "type": "looker_column",
                "label": "My Column Chart with Predefined Settings",
                "x_axis_gridlines": True,
                "y_axis_gridlines": True,
            }
        )
        query_id = sdk.create_query(query).id

        look = sdk.create_look(
                ml.WriteLookWithQuery(
                    title="new-title-"+create_random_string(),
                    description="new_desc-"+create_random_string(),
                    deleted=False,
                    is_run_on_load=True,
                    public=True,
                    query_id=query_id,
                    folder_id="6",
                    )
                )
        return look, base_url

    except looker_sdk.error.SDKError as e:
        error_messages = []
        for error_detail in e.errors:
            error_messages.append(f"Field: {error_detail.field},\
                Message: {error_detail.message}")
        print(f"Error: {', '.join(error_messages)}")
        return None, None
