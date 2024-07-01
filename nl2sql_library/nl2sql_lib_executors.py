"""
    Wrapper file for invoking the Executors from the app.py file
"""
import json
import sys
import vertexai
from loguru import logger

from nl2sql.executors.linear_executor.core import CoreLinearExecutor
from utils.utility_functions import get_project_config, initialize_db

from nl2sql.llms.vertexai import text_bison_32k
from nl2sql.tasks.table_selection.core import CoreTableSelector
from nl2sql.tasks.table_selection.core import prompts as cts_prompts
from nl2sql.tasks.column_selection.core import (
    CoreColumnSelector,
    prompts as ccs_prompts,
)
from nl2sql.tasks.sql_generation.core import CoreSqlGenerator
from nl2sql.tasks.sql_generation.core import prompts as csg_prompts
from sample_executors.rag_executor import RAG_Executor

vertexai.init(
    project=get_project_config()["config"]["proj_name"], location="us-central1"
)

dataset_name = get_project_config()["config"]["dataset"]  # "zoominfo"
bigquery_connection_string = initialize_db(
    get_project_config()["config"]["proj_name"],
    get_project_config()["config"]["dataset"],
)
data_file_name = get_project_config()["config"]["metadata_file"]
logger.info(
    f"Data = {bigquery_connection_string}, {dataset_name}, {data_file_name}"
    )

question_to_gen = "What is the revenue for construction industry?"


class NL2SQL_Executors:
    """
    Class with wrapper functions for all Executors
    """

    def __init__(self):
        self.executor = ""

    def linear_executor(
        self,
        question=question_to_gen,
        bq_conn_string=bigquery_connection_string,
        data_dict=None,
    ):
        """
        SQL Generation using Linear Executor
        """
        executor_linear = CoreLinearExecutor.from_connection_string_map(
            {
                dataset_name: bq_conn_string,
            },
            data_dictionary=data_dict,
        )
        logger.info("Linear Executor Executing for ", question)
        result = executor_linear(db_name=dataset_name, question=question)
        logger.info(f"Linear executor output: [{result.generated_query}]")
        return result.result_id, result.generated_query

    def cot_executor(
        self,
        question=question_to_gen,
        bq_conn_string=bigquery_connection_string,
        data_dict=None,
    ):
        """
        SQL Generation using Chain of Thought Executor
        """
        logger.info("Chain of Thought executor LLM initialised")
        llm = text_bison_32k()
        # Disabling logs because these steps generate a LOT of logs.
        logger.disable("nl2sql.datasets.base")
        core_table_selector = CoreTableSelector(
            llm=llm, prompt=cts_prompts.CURATED_FEW_SHOT_COT_PROMPT
        )
        core_column_selector = CoreColumnSelector(
            llm=llm, prompt=ccs_prompts.CURATED_FEW_SHOT_COT_PROMPT
        )
        core_sql_generator = CoreSqlGenerator(
            llm=llm, prompt=csg_prompts.CURATED_FEW_SHOT_COT_PROMPT
        )
        logger.enable("nl2sql.datasets.base")
        executor_cot = CoreLinearExecutor.from_connection_string_map(
            {dataset_name: bq_conn_string},
            core_table_selector=core_table_selector,
            core_column_selector=core_column_selector,
            core_sql_generator=core_sql_generator,
            data_dictionary=data_dict,
        )
        logger.info("Chain of Thought Executor executing for ", question)
        result = executor_cot(db_name=dataset_name, question=question)
        logger.info(f"Chain of Thought output: [{result.generated_query}]")
        return result.result_id, result.generated_query

    def rag_executor(self, question=question_to_gen):
        """
        SQL Generation using RAG Executor
        """
        ragexec = RAG_Executor()
        res_id, sql = ragexec.generate_sql(question)

        return res_id, sql


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage : python nl2sql_lib_executors.py <executor name>")
        print("For ex: python nl2sql_lib_executors.py rag ")
        print("Types of Executors : linear, cot, rag")
        print("Default is Linear executor")
        executor = "linear"
    elif sys.argv[1] == "linear":
        executor = "linear"
    elif sys.argv[1] == "cot":
        executor = "cot"
    elif sys.argv[1] == "rag":
        executor = "rag"
    else:
        print("Invalid executor type")
        print("Defaulting to Linear exeutor")
        executor = "linear"

    f = open(f"utils/{data_file_name}", encoding="utf-8")
    zi = json.load(f)
    data_dictionary_read = {
        "zoominfo": {
            "description": "This dataset contains information of Zoominfo Data\
                        with details on headquarters, marketing professionaals\
                            and providng tuition services.",
            "tables": zi,
        },
    }

    nle = NL2SQL_Executors()
    if executor == "linear":
        result_id, gen_sql = nle.linear_executor(
            data_dict=data_dictionary_read
            )
        print("linear")
    elif executor == "cot":
        result_id, gen_sql = nle.cot_executor(data_dict=data_dictionary_read)
        print("cot")
    elif executor == "rag":
        result_id, gen_sql = nle.rag_executor(question=question_to_gen)
        print("rag")

    print("Generated SQL = ", gen_sql)
