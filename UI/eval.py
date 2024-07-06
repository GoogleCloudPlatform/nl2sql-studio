import json
import time
import sqlite3
import pandas as pd
from langchain_google_vertexai import VertexAI
from google.cloud import bigquery
from UI.dbai import datasets
from nl2sql.executors.linear_executor.core import CoreLinearExecutor
from nl2sql.llms.vertexai import text_bison_latest
from nl2sql.datasets import fetch_dataset
from dbai import DBAI_nl2sql

llm = VertexAI(temperature=0, model_name="gemini-pro", max_output_tokens=1024)

def auto_verify(nl_description, ground_truth, llm_amswer):
    """
    This function verifies the accuracy of SQL query based on a natural language description
    and a ground truth query, using text-bison model.

    Parameters:
    - nl_description (str): The natural language description of the SQL query.
    - ground_truth (str): The ground truth SQL query.
    - llm_amswer (str): The student's generated SQL query for validation.

    Returns:
    str: "Yes" if the student's answer matches the ground truth and fits the NL description correctly,
         "No" otherwise.
    """
    
    prompt = f'''You are an expert at validating SQL queries. Given the Natrual language description
      and the SQL query corresponding to that description, please check if the students answer is correct.
      There can be different ways to achieve the same result by forming the query differently.
      If the students SQL query matches the ground truth and fits the NL description correctly, then return yes
      else return no.
      Natural language description: {nl_description}
      Ground truth: {ground_truth}
      students answer: {llm_amswer}
    '''
    return llm(prompt)


def execute_sql_query(query, client, job_config):
    try:
        cleaned_query = query.replace("\\n", " ").replace("\n", "").replace("\\", "")
        query_job = client.query(cleaned_query, job_config=job_config)
        response = query_job.result()
    except Exception as e:
        response = f"{str(e)}"

    return response



def nl2sql_generate_sql(question, sql_generator, bq_project_id, bq_dataset_id):
    if 'DBAI' in str(sql_generator):
        return dbai_framework(question, bq_project_id, bq_dataset_id)
    else:
        executor = sql_generator.from_connection_string_map(
            {
                bq_dataset_id: f"bigquery://{bq_project_id}/{bq_dataset_id}"
            }
        )
        return executor(
                db_name= bq_dataset_id,
                question = question
            ).generated_query

def dbai_framework(question, bq_project_id, bq_dataset_id, tables_list=[]):
    dbai_nl2sql = DBAI_nl2sql(
            proj_id=bq_project_id,
            dataset_id=bq_dataset_id,
            tables_list=tables_list
        )
    return dbai_nl2sql.get_sql(question).generated_sql

def bq_evaluator(sql_generator, bq_project_id, bq_dataset_id, ground_truth_path):
    client = bigquery.Client(project=bq_project_id)
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=100000000,
        default_dataset=f'{bq_project_id}.{bq_dataset_id}'
        )

    df = pd.read_csv(ground_truth_path)
    out = []
    for _, (_, question, ground_truth_sql) in df.iterrows():
        generated_query = nl2sql_generate_sql(question, sql_generator,
                                               bq_project_id, bq_dataset_id)

        generated_query_result = execute_sql_query(generated_query, client, job_config)
        actual_query_result = execute_sql_query(ground_truth_sql, client, job_config)

        llm_rating = auto_verify(question, ground_truth_sql, generated_query)
        result_eval = 0
        if generated_query_result.equals(actual_query_result):
            result_eval = 1

        out.append((question, ground_truth_sql, actual_query_result, generated_query,
                    generated_query_result, llm_rating, result_eval))

    df = pd.DataFrame(
        out,
        columns=[
            'question', 'ground_truth_sql', 'actual_query_result',
            'generated_query', 'generated_query_result', 'query_eval', 'result_eval'
            ])
    ts = time.strftime("%y%m%d%H%M")
    df.to_csv(f'eval_output/eval_result_{ts}.csv', index=False)
    print(f'Accuracy: {df.result_eval.sum()/len(df)}')
    return df


# Spider Evaluator Module
def spider_evaluator(spider_db_path, spider_eval_json, ExecutorType,
                      eval_output_file_name = "eval_results.csv",
                     eval_limit = 100, continuous_failure_limit=1):

    with open(spider_eval_json, 'r') as f:
        eval_data = json.load(f)

    total_no_of_questions = 0
    total_no_of_successfully_executed_questions = 0
    total_no_of_questions_for_which_result_is_correct = 0
    no_of_continuous_failures = 0
    eval_output = []

    for data in eval_data:
        print("\n")
        if total_no_of_questions >= eval_limit or no_of_continuous_failures>=continuous_failure_limit:
            break
        total_no_of_questions +=1
        try:
            db_id = data["db_id"]
            query = data["query"]
            question = data["question"]
            print(question)
            executor = ExecutorType.from_connection_string_map(
                {
                "spider": f"sqlite:///{spider_db_path}/{db_id}/{db_id}.sqlite"
                }
            )

            result = executor(
                db_name= "spider",
                question = question
            )
            generated_query = result.generated_query

            actual_query = query
            print("Actual Query:", actual_query)
            print("Generated Query:", generated_query)
            connection = sqlite3.connect(f"{spider_db_path}/{db_id}/{db_id}.sqlite")
            cursor = connection.cursor()
            generated_query_result = cursor.execute(actual_query).fetchall()
            print("generated_query_result:",generated_query_result)
            actual_query_result = cursor.execute(actual_query).fetchall()
            
            print("actual_query_result:",actual_query_result)
            correct_result = False
            if generated_query_result == actual_query_result:
                total_no_of_questions_for_which_result_is_correct +=1
                correct_result = True
            eval_output.append((db_id,question,actual_query,generated_query,correct_result))
            total_no_of_successfully_executed_questions +=1
            no_of_continuous_failures = 0
        except Exception as e:
            print(e)
            no_of_continuous_failures +=1
        print("\n")
    eval_output_df = pd.DataFrame(eval_output, columns=['db_name', 'question', 'actual_query', 'generated_query',
                                         'correct_generated_result ?'])
    eval_output_df.to_csv(eval_output_file_name, index=False)
    success_ratio_for_execution = total_no_of_successfully_executed_questions/total_no_of_questions
    success_ratio_for_result = total_no_of_questions_for_which_result_is_correct/total_no_of_questions
    print("Total_no_of_questions:",total_no_of_questions)
    print("Total_no_of_successfully_executed_questions:",total_no_of_successfully_executed_questions)
    print("Total_no_of_questions_for_which_result_is_correct:",total_no_of_questions_for_which_result_is_correct)
    print("Success ratio for execution:",success_ratio_for_execution )
    print("Success ratio for result:",success_ratio_for_result )



if __name__ == '__main__':
    BQ_PROJECT_ID = 'vertexai-pgt'
    BQ_DATASET_ID = 'nl2sql_fiserv'
    GROUND_TRUTH_PATH = './fiserv_ground_truth.csv'

    ## BQ data
    bq_evaluator(CoreLinearExecutor, BQ_PROJECT_ID, BQ_DATASET_ID, GROUND_TRUTH_PATH)

    ## spider run
    # It will download dataset in /var/tmp/NL2SQL_SPIDER_DATASET/extracted/spider/ folder
    spider_dataset_path = "/var/tmp/NL2SQL_SPIDER_DATASET/extracted/spider/"
    #For evaluating on test databases:
    spider_db_path = spider_dataset_path + "test_database"
    spider_eval_json = spider_dataset_path + "test_data/dev.json"

    # run
    spider_evaluator(spider_db_path, spider_eval_json, ExecutorType = CoreLinearExecutor, eval_limit = 100)



