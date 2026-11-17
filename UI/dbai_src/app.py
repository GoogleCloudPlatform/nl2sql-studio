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
import os
import uuid
from flask_cors import CORS
from flask import Flask, g, request
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token as google_id_token
from loguru import logger


from dbai import DBAI_nl2sql


PROJECT_ID = 'sl-test-project-363109'
DATASET_ID = 'nl2sql_spider'
TABLES_LIST = []

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type, Authorization"


def _verify_dbai_bearer_token(auth_header: str | None) -> dict:
    """Verify Authorization: Bearer <Google OIDC ID token>."""
    oauth_mode = os.getenv("GOOGLE_OAUTH", "ENABLE").strip().upper()
    allow_dev = (
        os.getenv("NL2SQL_ALLOW_UNAUTHENTICATED_DEV", "").strip().lower()
        == "true"
    )
    if oauth_mode == "DISABLE" and allow_dev:
        return {"sub": "dev-local-user"}
    if not isinstance(auth_header, str) or not auth_header.startswith(
        "Bearer "
    ):
        raise ValueError("Missing Authorization Bearer header")
    token = auth_header[len("Bearer "):].strip()
    if not token or token == "dummy token":
        raise ValueError("Empty or dummy Bearer token")
    expected_aud = (
        os.getenv("GOOGLE_CLIENT_ID", "").strip().strip('"').strip("'")
    )
    if not expected_aud:
        raise ValueError("GOOGLE_CLIENT_ID is not configured")
    claims = google_id_token.verify_oauth2_token(
        token,
        google_auth_requests.Request(),
        audience=expected_aud,
    )
    if claims.get("iss") not in (
        "accounts.google.com",
        "https://accounts.google.com",
    ):
        raise ValueError("Invalid OIDC token issuer")
    if not claims.get("sub"):
        raise ValueError("Missing OIDC subject claim")
    allowed_domains_raw = os.getenv("ALLOWED_AUTH_DOMAINS", "").strip()
    if allowed_domains_raw:
        allowed_domains = {
            d.strip().lower()
            for d in allowed_domains_raw.split(",")
            if d.strip()
        }
        email_dom = str(claims.get("email", "")).partition("@")[2].lower()
        hd_dom = str(claims.get("hd", "")).lower()
        if email_dom not in allowed_domains and hd_dom not in allowed_domains:
            raise ValueError("Authenticated user domain is not permitted")
    return claims


@app.before_request
def _enforce_bearer_auth():
    """Require a verified Google OIDC Bearer token on API endpoints."""
    if request.method == "OPTIONS":
        return None
    if request.method == "GET" and request.path == "/":
        return None
    try:
        g.auth_claims = _verify_dbai_bearer_token(
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
            "sql_result": "",
            "error_msg": "Error in DBAI SQL generation",
        }

    return json.dumps(response_string)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
