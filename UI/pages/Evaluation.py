""" """
import streamlit as st
from io import StringIO
from loguru import logger

CORE = "NL2SQL Studio Core"
LITE = "NL2SQL Studio Lite"

LINEAR = "Linear Executor"
RAG = "Rag Executor"
COT = "Chain of Thought"
ZERO_SHOT = "Zero Shot"
FEW_SHOT = "Few Shot"

st.set_page_config(layout='centered')

st.sidebar.title("Evaluation Settings")

gen_engine = st.sidebar.selectbox(
    "Choose NL2SQL framework",
    (LITE, CORE, "DBAI")
    )
logger.info(f"Generation using : {gen_engine}")
if gen_engine == CORE:
    st.session_state.generation_engine = CORE
    with st.sidebar.container(height=140):
        st.session_state.model = st.radio('Select Prompting Technique',
                                            [LINEAR, RAG, COT])
elif gen_engine == LITE:
    st.session_state.generation_engine = LITE
    with st.sidebar.container(height=115):
        st.session_state.lite_model = st.radio(
            'Select Prompting Technique',
            [ZERO_SHOT, FEW_SHOT])
else:
    st.session_state.generation_engine = None


st.title("NL2SQL Evaluation Studio")


project = st.text_input('Mention the GCP project name')
dataset = st.text_input(
    'Specify the BigQuery dataset name'
    )

tables_list = st.text_input(
    'Specify the list of tables names (optional)'
    )

uploaded_file = st.file_uploader(
            "Choose the test dataset file (csv format) (must have columns: Question, golden_sql)",
            type="csv"
            )

proj_conf = st.button(" Start Evaluation")


if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(
        uploaded_file.getvalue().decode("utf-8")
        )

    logger.info(
        f"Uploading file : {uploaded_file.name}"
        )
    # To read file as string:
    string_data = stringio.read()


# st.session_state.generation_engine