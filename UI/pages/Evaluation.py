"""Page for Evaluation framework."""
import os
from pathlib import Path
import shutil
import pandas as pd
from loguru import logger
import streamlit as st

from evaluation.eval import bq_evaluator
from utils import (
    require_authentication,
    save_uploaded_csv_safely,
    validate_gcp_project_and_dataset,
)

CORE = "NL2SQL Studio Core"
LITE = "NL2SQL Studio Lite"

LINEAR = "Linear Executor"
RAG = "Rag Executor"
COT = "Chain of Thought"
ZERO_SHOT = "Zero Shot"
FEW_SHOT = "Few Shot"
DBAI = "DBAI"

UI_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV_PATH = UI_ROOT / "concert_singer_spider.csv"


def _cleanup_temp_upload(file_path: str | None) -> None:
    """Remove isolated temporary CSV upload file and directory."""
    if not file_path:
        return
    try:
        parent_dir = os.path.dirname(os.path.realpath(file_path))
        if os.path.basename(parent_dir).startswith("nl2sql_eval_upload_"):
            shutil.rmtree(parent_dir, ignore_errors=True)
        elif os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


def render_evaluation_page() -> None:
    """Render the authenticated NL2SQL Evaluation Studio page."""
    st.set_page_config(layout="centered")
    if not require_authentication():
        return

    st.sidebar.title("Evaluation Settings")

    gen_engine = st.sidebar.selectbox(
        "Choose NL2SQL framework",
        (LITE, CORE, DBAI),
    )
    logger.info(f"Generation using : {gen_engine}")
    if gen_engine == CORE:
        st.session_state.generation_engine = CORE
        with st.sidebar.container(height=140):
            st.session_state.model = st.radio(
                "Select Prompting Technique",
                [LINEAR, RAG, COT],
            )
    elif gen_engine == LITE:
        st.session_state.generation_engine = LITE
        with st.sidebar.container(height=115):
            st.session_state.model = st.radio(
                "Select Prompting Technique",
                [FEW_SHOT, ZERO_SHOT],
            )
    elif gen_engine == DBAI:
        st.session_state.model = DBAI
    else:
        st.session_state.generation_engine = None

    st.title("NL2SQL Evaluation Studio")

    project = st.text_input(
        "Mention the GCP project name",
        value="sl-test-project-363109",
    )
    dataset = st.text_input(
        "Specify the BigQuery dataset name",
        value="nl2sql_spider",
    )
    _ = st.text_input("Specify the list of tables names (optional)")

    uploaded_file = st.file_uploader(
        "Choose the test dataset file (csv format) "
        "(must have columns: Question, golden_sql)",
        type="csv",
    )

    if SAMPLE_CSV_PATH.exists():
        with open(SAMPLE_CSV_PATH, "rb") as sample_f:
            st.download_button(
                "Download Spider Sample file",
                sample_f,
                file_name="concert_singer_spider.csv",
            )

    start_eval = st.button("Start Evaluation")

    if start_eval:
        if uploaded_file is None:
            st.error("Please upload a valid CSV test dataset file first.")
            return
        try:
            valid_proj, valid_ds = validate_gcp_project_and_dataset(
                project, dataset
            )
            safe_csv_path = save_uploaded_csv_safely(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.session_state.uploaded_file_path = safe_csv_path
        try:
            pb = st.progress(0, text="Evaluation in progress. Please wait...")
            eval_results = bq_evaluator(
                valid_proj,
                valid_ds,
                safe_csv_path,
                st.session_state.model,
                None,
                pb,
                render_result=True,
            )
        except ValueError as exc:
            st.error(f"Evaluation rejected unsafe input: {exc}")
            return
        finally:
            _cleanup_temp_upload(safe_csv_path)
            st.session_state.uploaded_file_path = None

        st.markdown(f"Accuracy is {eval_results['accuracy']}")

        @st.cache_data
        def convert_df(result_df: pd.DataFrame) -> bytes:
            return result_df.to_csv(index=False).encode("utf-8")

        csv_bytes = convert_df(eval_results["output"])
        st.download_button(
            "Download",
            csv_bytes,
            "results.csv",
            "text/csv",
            key="download-csv",
        )


render_evaluation_page()
