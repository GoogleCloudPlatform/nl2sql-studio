"""Page for Evaluation framework """
import streamlit as st
from loguru import logger
import pandas as pd
import os
from evaluation.eval import bq_evaluator

UPLOAD_DIRECTORY = "./"

CORE = "NL2SQL Studio Core"
LITE = "NL2SQL Studio Lite"

LINEAR = "Linear Executor"
RAG = "Rag Executor"
COT = "Chain of Thought"
ZERO_SHOT = "Zero Shot"
FEW_SHOT = "Few Shot"
DBAI = "DBAI"

st.set_page_config(layout='centered')

st.sidebar.title("Evaluation Settings")

gen_engine = st.sidebar.selectbox(
    "Choose NL2SQL framework",
    (LITE, CORE, DBAI)
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
        st.session_state.model = st.radio(
            'Select Prompting Technique',
            [FEW_SHOT, ZERO_SHOT])
elif gen_engine == DBAI:
    st.session_state.model = DBAI
else:
    st.session_state.generation_engine = None

st.title("NL2SQL Evaluation Studio")

# st.markdown("Latest NL2SQL Benchmark Report")

df = pd.DataFrame(
        [(86, 75, 82), (91, 80, 86), (93, 83, 90)],
        index=['Fiserv', 'EY', 'Spider'],
        columns=['NL2SQL core', 'NL2SQL fiserv', 'DBAI']
    )

df = df.map(lambda x: str(x)+'%')
# st.dataframe(df)

project = st.text_input('Mention the GCP project name',
                        value="sl-test-project-363109")
dataset = st.text_input(
    'Specify the BigQuery dataset name',
    value="nl2sql_spider"
    )

tables_list = st.text_input(
    'Specify the list of tables names (optional)'
    )

uploaded_file = st.file_uploader(
            "Choose the test dataset file (csv format) \
                (must have columns: Question, golden_sql)",
            type="csv"
            )

with open('concert_singer_spider.csv', 'rb') as f:
    st.download_button('Download Spider Sample file',
                       f,
                       file_name='concert_singer_spider.csv'
                       )

start_eval = st.button("Start Evaluation")

st.session_state.uploaded_file_path = None
if uploaded_file is not None:
    # Get the file details
    file_details = {
        "filename": uploaded_file.name,
        "filetype": uploaded_file.type,
        "filesize": uploaded_file.size
    }

    # Save the uploaded file to the server
    uploaded_file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.uploaded_file_path = uploaded_file_path


# st.session_state.generation_engine

if start_eval:
    pb = st.progress(0, text="Evaluation in progress. Please wait...")
    eval_results = bq_evaluator(
        project,
        dataset,
        st.session_state.uploaded_file_path,
        st.session_state.model, None, pb, render_result=True)

    st.markdown(f'Accuracy is {eval_results["accuracy"]}')

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(eval_results["output"])
    st.download_button(
        "Download",
        csv,
        "results.csv",
        "text/csv",
        key='download-csv'
    )
