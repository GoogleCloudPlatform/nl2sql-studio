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

import re
import json
import traceback
import sqlalchemy
import pandas as pd
# import langchain
import sqlglot

# from prompts import *
from prompts import Table_filtering_prompt, Table_filtering_prompt_promptonly
from prompts import Auto_verify_sql_prompt, Sql_Generation_prompt
from prompts import Sql_Generation_prompt_few_shot, additional_context_prompt
from prompts import Table_info_template, join_prompt_template
from prompts import join_prompt_template_one_shot, multi_table_prompt
from prompts import follow_up_prompt, Sql_Generation_prompt_few_shot_multiturn
from prompts import Result2nl_insight_prompt, Result2nl_prompt

from langchain_google_vertexai import VertexAI
from google.cloud import bigquery
from nl2sql_query_embeddings import PgSqlEmb, Nl2Sql_embed
import os

from loguru import logger

# from google.cloud import aiplatform

from vertexai.preview.generative_models import GenerativeModel
# from vertexai.preview.generative_models import GenerationConfig
# from json import loads, dumps
# from vertexai.language_models import TextGenerationModel

from vertexai.language_models import CodeChatSession
from vertexai.language_models import CodeChatModel


client = bigquery.Client()


class Nl2sqlBq:
    """
        NL2SQL Lite SQL Generator class
    """

    def __init__(self,
                 project_id,
                 dataset_id,
                 metadata_json_path=None,
                 model_name="gemini-pro",
                 tuned_model=True):
        """
            Init function
        """
        self.dataset_id = f"{project_id}.{dataset_id}"
        self.metadata_json = None
        self.model_name = model_name

        if model_name == 'text-bison@002' and tuned_model:
            # self.llm = VertexAI(temperature=0,
            #                     model_name=self.model_name,
            #                     tuned_model_name='projects/862253555914/\
            #                     locations/us-central1/\
            #                     models/7566417909400993792',
            #                     max_output_tokens=1024)
            tuned_model_name = 'projects/174482663155/locations/' + \
                                'us-central1/models/6975883408262037504'
            self.llm = VertexAI(temperature=0,
                                model_name=self.model_name,
                                tuned_model_name=tuned_model_name,
                                max_output_tokens=1024)

        else:
            self.llm = VertexAI(temperature=0,
                                model_name=self.model_name,
                                max_output_tokens=1024)

        logger.info(f"Current LLM model : {self.model_name}")

        # self.llm = VertexAI(temperature=0,
        #                     model_name=self.model_name,
        #                     tuned_model_name='projects/862253555914/\
        #                     locations/us-central1/models/\
        #                     7566417909400993792',
        #                     max_output_tokens=1024)

        self.engine = sqlalchemy.engine.create_engine(
            f"bigquery://{self.dataset_id.replace('.','/')}")
        if metadata_json_path:
            f = open(metadata_json_path, encoding="utf-8")
            self.metadata_json = json.loads(f.read())

    def init_pgdb(self,
                  proj_id,
                  loc,
                  pg_inst,
                  pg_db,
                  pg_uname,
                  pg_pwd,
                  pg_table,
                  index_file='saved_index_pgdata'):
        """
            Initialising the PG DB
        """
        self.pge = PgSqlEmb(proj_id,
                            loc,
                            pg_inst,
                            pg_db,
                            pg_uname,
                            pg_pwd,
                            pg_table)

    def get_all_table_names(self):
        """
            Provides list of table names in dataset
        """
        tables = client.list_tables(self.dataset_id)
        all_table_names = [table.table_id for table in tables]
        return all_table_names

    def get_column_value_examples(self, tname, column_name, enum_option_limit):
        """
        Provide example values for string columns
        """
        examples_str = ""
        if pd.read_sql(
                sql=f"SELECT COUNT(DISTINCT {column_name}) <=\
                    {enum_option_limit} FROM {tname}",
                con=self.engine).values[0][0]:

            sql_string = f"SELECT DISTINCT {column_name} AS vals FROM {tname}"
            examples_str = "It contains values : \"" + ("\", \"".join(
                            filter(
                                lambda x: x is not None,
                                pd.read_sql(
                                    sql=sql_string,
                                    con=self.engine
                                    )["vals"].to_list()
                            )
                            )
                        ) + "\"."
        return examples_str

    def create_metadata_json(self,
                             metadata_json_dest_path,
                             data_dict_path=None,
                             col_values_distribution=False,
                             enum_option_limit=10):
        """
        Creates metadata json file
        """
        try:
            data_dict = dict()
            if data_dict_path:
                f = open(data_dict_path, encoding="utf-8")
                data_dict = json.loads(f.read())
            table_ls = self.get_all_table_names()
            metadata_json = dict()
            for table_name in table_ls:
                table = client.get_table(f"{self.dataset_id}.{table_name}")
                table_description = ""
                if table_name in data_dict and data_dict[table_name].strip():
                    table_description = data_dict[table_name]
                elif table.description:
                    table_description = table.description
                columns_info = dict()

                for schema in table.schema:
                    schema_description = ""
                    if f"{table_name}.{schema.name}" in data_dict and \
                       data_dict[f"{table_name}.{schema.name}"].strip():
                        schema_description = data_dict[
                            f"{table_name}.{schema.name}"]
                    elif schema.description:
                        schema_description = schema.description

                    columns_info[schema.name] = {
                        "Name": schema.name,
                        "Type": schema.field_type,
                        "Description": schema_description,
                        "Examples": ""
                        }
                    if col_values_distribution and \
                       schema.field_type == "STRING":

                        all_examples = self.get_column_value_examples(
                            table_name, schema.name, enum_option_limit)
                        columns_info[schema.name]["Examples"] = all_examples
                metadata_json[table_name] = {"Name": table_name,
                                             "Description": table_description,
                                             "Columns": columns_info}
            with open(metadata_json_dest_path, 'w', encoding="utf-8") as f:
                json.dump(metadata_json, f)
            self.metadata_json = metadata_json
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def table_filter(self, question):
        """
        This function selects the relevant table(s) to the provided question
        based on their description-keywords.
        It assists in selecting a table from a list of tables based on
        their description-keywords.
        It presents a prompt containing a list of table names along
        with their corresponding description-keywords.
        The function uses a text-based model (text_bison) to analyze
        the prompt and extract the selected table name(s).

        Parameters:
        - question (str): The question for which the relevant table
          need to be identified.

        Returns:
        list: A list of table names most likely relevant to the provided
        question.
        """

        only_tables_info = ""

        for table in self.metadata_json:
            only_tables_info = only_tables_info + f"{table} | \
                {self.metadata_json[table]['Description']}\n"

        prompt = Table_filtering_prompt.format(
            only_tables_info=only_tables_info,
            question=question
            )

        result = self.llm.invoke(prompt)

        segments = result.split(',')
        tables_list = []

        for segment in segments:
            segment = segment.strip()
            if ':' in segment:
                value = segment.split(':')[-1].strip()
                tables_list.append(value.strip())
            elif '\n' in segment:
                value = segment.split('\n')[-1].strip()
                tables_list.append(value.strip())
            else:
                tables_list.append(segment)

        return tables_list

    def table_filter_promptonly(self, question):
        """
        This function returns the prompt for the base question
        in the Multi-turn execution

        Parameters:
        - question (str): The question for which the relevant table
          need to be identified.

        Returns:
        list: A list of table names most likely relevant to the
        provided question.
        """

        only_tables_info = ""

        for table in self.metadata_json:
            only_tables_info = only_tables_info + f"{table} | \
                {self.metadata_json[table]['Description']}\n"

        prompt = Table_filtering_prompt_promptonly.format(
            only_tables_info=only_tables_info
            )

        return prompt

    def case_handler_transform(self, sql_query: str) -> str:
        """
        This function implements case-handling mechanism transformation
        for a SQL query.

        Parameters:
        - sql_query (str): The original SQL query.

        Returns:
        str: The transformed SQL query with case-handling mechanism applied,
            or the original query if no transformation is needed.
        """
        # print("Case handller transform", sql_query)
        node = sqlglot.parse_one(sql_query)

        if (
          isinstance(node, sqlglot.expressions.EQ) and
          node.find_ancestor(sqlglot.expressions.Where) and
          len(operands := list(node.unnest_operands())) == 2 and
          isinstance(
              literal := operands.pop(), sqlglot.expressions.Literal
              ) and
          isinstance(predicate := operands.pop(), sqlglot.expressions.Column)
        ):
            transformed_query =\
                sqlglot.parse_one(f"LOWER({predicate}) =\
                '{literal.this.lower()}'")
            return str(transformed_query)
        else:
            return sql_query

    def add_dataset_to_query(self, sql_query):
        """
        This function adds the specified dataset prefix to the tables
        in the FROM clause of a SQL query.

        Parameters:
        - dataset (str): The dataset name to be added as a prefix.
        - sql_query (str): The original SQL query.

        Returns:
        str: Modified SQL query with the specified dataset prefix
        added to the tables in the FROM clause.
        """
        logger.info(f"Original query : {sql_query}")
        dataset = self.dataset_id
        if sql_query:
            sql_query = sql_query.replace('`', '')
            # Define a regular expression pattern to match the FROM clause
            pattern = re.compile(r'\bFROM\b\s+(\w+)', re.IGNORECASE)

            # Find all matches of the pattern in the SQL query
            matches = pattern.findall(sql_query)

            # Iterate through matches and replace the table name
            for match in matches:
                # check text following the match if it is a complete table name
                next_text = sql_query.split(match)[1].split('\n')[0]
                next_text = next_text.split(' ')[0]

                # Check if the previous word is not DAY, YEAR, or MONTH
                if re.search(r'\b(?:DAY|YEAR|MONTH)\b',
                             sql_query[:sql_query.find(match)],
                             re.IGNORECASE) is None:

                    # Replace the next word after FROM with dataset.table
                    if match == dataset.split('.')[0]:
                        # checking if in generated SQL, table
                        # includes the project-id and dataset or not
                        replacement = f'`{match}'
                    else:
                        sql_query = sql_query.replace(next_text, '')
                        replacement = f'{dataset}.`{match}{next_text}`'

                    # replacement = f'{dataset}.{match}'
                    sql_query = re.sub(r'\bFROM\b\s+' + re.escape(match),
                                       f'FROM {replacement}',
                                       sql_query,
                                       flags=re.IGNORECASE
                                       )
                    if match == dataset.split('.')[0]:
                        sql_query = sql_query.replace(f'{match}{next_text}',
                                                      f'{match}{next_text}`'
                                                      )

            sql_query = sql_query.replace('CAST', 'SAFE_CAST')
            sql_query = sql_query.replace('SAFE_SAFE_CAST', 'SAFE_CAST')
            return sql_query
        else:
            return ""

    def generate_sql(self, question, table_name=None, logger_file="log.txt"):
        """
        Main function which converts NL to SQL
        """

        # step-1 table selection
        try:
            if not table_name:
                if len(self.metadata_json.keys()) > 1:
                    table_list = self.table_filter(question)
                    table_name = table_list[0]
                else:
                    table_name = list(self.metadata_json.keys())[0]

            table_json = self.metadata_json[table_name]
            columns_json = table_json["Columns"]
            columns_info = ""
            for column_name in columns_json:
                column = columns_json[column_name]
                column_info = f"""{column["Name"]} \
                    ({column["Type"]}) : {column["Description"]}.\
                    {column["Examples"]}
                    \n"""
                columns_info = columns_info + column_info
            sql_prompt = Sql_Generation_prompt.format(
                    table_name=table_json["Name"],
                    table_description=table_json["Description"],
                    columns_info=columns_info,
                    question=question
                    )
            response = self.llm.invoke(sql_prompt)
            sql_query = response.replace('sql', '').replace('```', '')
            # sql_query = self.case_handler_transform(sql_query)
            sql_query = self.add_dataset_to_query(sql_query)
            with open(logger_file, 'a', encoding="utf-8") as f:
                f.write(f">>\nModel:{self.model_name} \n\nQuestion: {question}\
                         \n\nPrompt:{sql_prompt} \nSql_query:{sql_query}<<\n")
            if sql_query.strip().startswith("Response:"):
                sql_query = sql_query.split(":")[1].strip()
            return sql_query
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def generate_sql_few_shot(self,
                              question,
                              table_name=None,
                              logger_file="log.txt"):
        """
        Main function which converts NL to SQL using few shot prompting
        """

        # step-1 table selection
        try:
            if not table_name:
                if len(self.metadata_json.keys()) > 1:
                    table_list = self.table_filter(question)
                    table_name = table_list[0]
                else:
                    table_name = list(self.metadata_json.keys())[0]
            table_json = self.metadata_json[table_name]
            columns_json = table_json["Columns"]
            columns_info = ""
            for column_name in columns_json:
                column = columns_json[column_name]
                column_info = f"""{column["Name"]} \
                    ({column["Type"]}) : {column["Description"]}.\
                    {column["Examples"]}\n"""
                columns_info = columns_info + column_info

            # few_shot_json = self.pge.search_matching_queries(question)
            embed = Nl2Sql_embed()
            few_shot_json = embed.search_matching_queries(question)
            logger.info(f"Few sjot examples : {few_shot_json}")
            few_shot_examples = ""

            for item in few_shot_json:
                example_string = f"Question: {item['question']}"
                few_shot_examples += example_string + "\n"
                example_string = f"SQL : {item['sql']} "
                few_shot_examples += example_string + "\n\n"

            sql_prompt = Sql_Generation_prompt_few_shot.format(
                table_name=table_json["Name"],
                table_description=table_json["Description"],
                columns_info=columns_info,
                few_shot_examples=few_shot_examples,
                question=question)
            response = self.llm.invoke(sql_prompt)
            sql_query = response.replace('sql', '').replace('```', '')

            # sql_query = self.case_handler_transform(sql_query)

            sql_query = self.add_dataset_to_query(sql_query)
            with open(logger_file, 'a', encoding="utf-8") as f:
                f.write(f">>\nModel:{self.model_name} \n\nQuestion: {question}\
                         \n\nPrompt:{sql_prompt} \n\nSql_query:{sql_query}<\n")

            if sql_query.strip().startswith("Response:"):
                sql_query = sql_query.split(":")[1].strip()
            return sql_query
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def generate_sql_few_shot_promptonly(self,
                                         question,
                                         table_name=None,
                                         prev_sql="",
                                         logger_file="log.txt"):
        """
        Returns only the few shot prompt
        """

        # step-1 table selection
        try:
            if not table_name:
                if len(self.metadata_json.keys()) > 1:
                    table_list = self.table_filter(question)
                    table_name = table_list[0]
                else:
                    table_name = list(self.metadata_json.keys())[0]
            table_json = self.metadata_json[table_name]
            columns_json = table_json["Columns"]
            columns_info = ""
            for column_name in columns_json:
                column = columns_json[column_name]
                column_info = f"""{column["Name"]} \
                    ({column["Type"]}) : {column["Description"]}.\
                    {column["Examples"]}\n"""
                columns_info = columns_info + column_info

            few_shot_json = self.pge.search_matching_queries(question)
            few_shot_examples = ""
            for item in few_shot_json:
                example_string = f"Question: {item['question']}"
                few_shot_examples += example_string + "\n"
                example_string = f"SQL : {item['sql']} "
                few_shot_examples += example_string + "\n\n"

            if prev_sql:
                additional_context = additional_context_prompt.format(
                    prev_sql=prev_sql
                    )
            else:
                additional_context = ""

            sql_prompt = Sql_Generation_prompt_few_shot_multiturn.format(
                                table_name=table_json["Name"],
                                table_description=table_json["Description"],
                                columns_info=columns_info,
                                few_shot_examples=few_shot_examples,
                                question=question,
                                additional_context=additional_context
                                )

            return sql_prompt
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def execute_query_old(self, query):
        """
        This function executes an SQL query using the configured
        BigQuery client.

        Parameters:
        - query (str): The SQL query to be executed.

        Returns:
        pandas.DataFrame: The result of the executed query as a DataFrame.
        """
        try:
            # Run the SQL query
            query_job = client.query(query)

            # Wait for the job to complete
            query_job.result()

            # Fetch the result if needed
            results = query_job.to_dataframe()

            return results
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def execute_query(self, query, dry_run=False):
        """
        This function executes an SQL query using the configured
        BigQuery client.

        Parameters:
        - query (str): The SQL query to be executed.

        Returns:
        pandas.DataFrame: The result of the executed query as a DataFrame.
        """
        if dry_run:
            job_config = bigquery.QueryJobConfig(dry_run=True,
                                                 use_query_cache=False)
            query_job = client.query(query, job_config=job_config)

            if query_job.total_bytes_processed > 0:
                logger.info("Query is valid")
                return True, 'Query is valid'
            else:
                return False, 'Invalid query. Regenerate'
        else:
            try:
                # Run the SQL query
                query_job = client.query(query)

                # Wait for the job to complete
                query_job.result()

                # Fetch the result if needed
                results = query_job.to_dataframe()

                return results
            except Exception as exc:
                raise Exception(traceback.print_exc()) from exc

    def self_reflection(self, question, query, max_tries=5):
        """
        Retries the query generation process in case of failure
        for the specified number of times
        """
        status, _ = self.execute_query(query, dry_run=True)
        good_sql = False
        if not status:
            # Repeat generation of the sql
            iter = 0
            while iter < max_tries or good_sql:
                prompt = self.generate_sql_few_shot_promptonly(question,
                                                               table_name="",
                                                               prev_sql=query)
                query = self.invoke_llm(prompt)
                good_sql, msg = self.execute_query(query, dry_run=True)
                iter += 1
        return good_sql, query

    def text_to_sql_execute(self,
                            question,
                            table_name=None,
                            logger_file="log.txt"):
        """
        Converts text to sql and also executes sql query
        """
        try:
            # query = self.text_to_sql(question,
            #                          table_name,logger_file = logger_file)
            query = self.generate_sql(question,
                                      table_name,
                                      logger_file=logger_file
                                      )
            results = self.execute_query(query)
            return results
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def text_to_sql_execute_few_shot(self,
                                     question,
                                     table_name=None,
                                     logger_file="log.txt"):
        """
        Converts text to sql and also executes sql query
        """
        try:
            query = self.generate_sql_few_shot(question,
                                               table_name,
                                               logger_file=logger_file
                                               )
            logger.info(f"Executing query : {query}")
            results = self.execute_query(query)
            return results, query
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def result2nl(self, result, question, insight=True):
        """
        The function converts an SQL query result into an insightful
        and well-explained natural language summary, using text-bison model.

        Parameters:
        - result (str): The result of the SQL query.
        - question (str): The natural language question corresponding
          to the SQL query.

        Returns:
        str: A natural language summary of the SQL query result.
        """
        try:
            if insight:
                prompt = Result2nl_insight_prompt.format(question=question,
                                                         result=str(result)
                                                         )
            else:
                prompt = Result2nl_prompt.format(question=question,
                                                 result=str(result)
                                                 )

            return self.llm.invoke(prompt)
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def auto_verify(self, nl_description, ground_truth, llm_amswer):
        """
        This function verifies the accuracy of SQL query based on a natural
        language description and a ground truth query, using text-bison model.

        Parameters:
        - nl_description (str): The natural language description of the
          SQL query.
        - ground_truth (str): The ground truth SQL query.
        - llm_amswer (str): The student's generated SQL query for validation.

        Returns:
        str: "Yes" if the student's answer matches the ground truth
        and fits the NL description correctly,"No" otherwise.
        """

        prompt = Auto_verify_sql_prompt.format(nl_description=nl_description,
                                               ground_truth=ground_truth,
                                               llm_amswer=llm_amswer
                                               )
        return self.llm.invoke(prompt)

    def batch_run(self,
                  test_file_name,
                  output_file_name,
                  execute_query=False,
                  result2nl=False,
                  insight=True,
                  logger_file="log.txt"):
        """
        This function procesess a batch of questions from a test file,
        generate SQL queries, and evaluate their accuracy.
        It reads questions from a CSV file, generates SQL queries using the
        `gen_sql` function,
        evaluates the accuracy of the generated queries using the `auto_verify`
        function,
        and optionally converts SQL queries to natural language
        using the `sql2result` and `result2nl` functions.
        The results are stored in a DataFrame and saved to a CSV file in the
        'output' directory,
        with a timestamped filename.

        Parameters:
        - test_file_name (str):
        The name of the CSV file containing test questions and ground truth
        SQL queries.

        - sql2nl (bool, optional):
        Flag to convert SQL queries to natural language. Defaults to False.

        Returns:
        pandas.DataFrame: A DataFrame containing question, ground truth SQL,
        LLM-generated SQL, LLM rating, SQL execution result, and NL response.
        """
        try:
            questions = pd.read_csv(test_file_name)

            out = []
            columns = ['question',
                       'ground_truth',
                       'llm_response',
                       'llm_rating'
                       ]
            if execute_query:
                columns.append('sql_result')
            if result2nl:
                columns.append('nl_response')
            for _, row in questions.iterrows():
                table_name = None
                if row["table"].strip():
                    table_name = row["table"]
                question = row["question"]
                # print(question)
                sql_gen = self.generate_sql(question,
                                            table_name=table_name,
                                            logger_file=logger_file
                                            )
                # print(sql_gen)
                rating = self.auto_verify(question,
                                          row["ground_truth_sql"], sql_gen
                                          )
                row_result = [question,
                              row["ground_truth_sql"], sql_gen, rating]
                if execute_query:
                    result = self.execute_query(sql_gen)
                    # print(result)
                    row_result.append(result)
                if execute_query and result2nl:
                    nl = self.result2nl(result, question, insight=insight)
                    row_result.append(nl)
                out.append(row_result)
                # print("\n\n")

            df = pd.DataFrame(out, columns=columns)
            df.to_csv(output_file_name, index=False)
            return df
        except Exception as exc:
            raise Exception(traceback.print_exc()) from exc

    def table_details(self, table_name):
        """
        Cretes the Table details required for Joins
        """
        f = open(self.metadata_json, encoding="utf-8")
        metadata_json = json.loads(f.read())

        table_json = metadata_json[table_name]
        columns_json = table_json["Columns"]
        columns_info = ""
        for column_name in columns_json:
            column = columns_json[column_name]
            column_info = f"""
                {column["Name"]} \
                ({column["Type"]}) : {column["Description"]}.\
                {column["Examples"]}\n"""
            columns_info = columns_info + column_info

        prompt = Table_info_template.format(
            table_name=table_name,
            table_description=metadata_json[table_name]['Description'],
            columns_info=columns_info
        )
        return prompt

    def get_join_prompt(self,
                        dataset,
                        table_1_name,
                        table_2_name,
                        question,
                        sample_question=None,
                        sample_sql=None,
                        one_shot=False):
        """
            Crete the prompt for Joins
        """
        prompt = ""
        table_1 = self.table_details(table_1_name)
        table_2 = self.table_details(table_2_name)

        if one_shot:
            prompt = join_prompt_template_one_shot.format(
                data_set=dataset,
                table_1=table_1,
                table_2=table_2,
                sample_question=sample_question,
                sample_sql=sample_sql,
                question=question
            )
        else:
            prompt = join_prompt_template.format(data_set=dataset,
                                                 table_1=table_1,
                                                 table_2=table_2,
                                                 question=question)

        return prompt

    def invoke_llm(self, prompt):
        """
            Invoke the LLM
        """
        response = self.llm.invoke(prompt)
        sql_query = response.replace('sql', '').replace('```', '')

        # sql_query = self.case_handler_transform(sql_query)

        sql_query = self.add_dataset_to_query(sql_query)

        # with open(logger_file, 'a',encoding="utf-8") as f:
        #     f.write(f">>>>\nModel:{self.model_name} \n\nQuestion: {question}\
        #      \n\nPrompt:{join_prompt} \n\nSql_query:{sql_query}<<<<\n\n\n")
        return sql_query

    def multi_turn_table_filter(self,
                                table_1_name,
                                table_2_name,
                                sample_question,
                                sample_sql,
                                question):
        """
            Table filter for multi-turn prompting
        """
        table_info = self.table_filter_promptonly(question)
        prompt = multi_table_prompt.format(table_info=table_info,
                                           example_question=sample_question,
                                           example_sql=sample_sql,
                                           question=question,
                                           table_1_name=table_1_name,
                                           table_2_name=table_2_name)
        model = GenerativeModel("gemini-1.0-pro")
        multi_chat = model.start_chat()
        _ = multi_chat.send_message(prompt)  # response1
        response2 = multi_chat.send_message(follow_up_prompt)
        try:
            identified_tables = response2.candidates[0].content.parts[0].text
        except Exception:
            identified_tables = ""
        return identified_tables

    def gen_and_exec_and_self_correct_sql(self,
                                          prompt,
                                          genai_model_name="GeminiPro",
                                          max_tries=5,
                                          return_all=False):
        """
            Wrapper function for Standard, Multi-turn and Self Correct
            approach of SQL generation
        """
        tries = 0
        error_messages = []
        prompts = [prompt]
        successful_queries = []
        TEMPERATURE = 0.3
        MAX_OUTPUT_TOKENS = 8192

        MODEL_NAME = 'codechat-bison-32k'
        code_gen_model = CodeChatModel.from_pretrained(MODEL_NAME)

        model = GenerativeModel("gemini-1.0-pro")

        if genai_model_name == "GeminiPro":
            chat_session = model.start_chat()
        else:
            chat_session = CodeChatSession(model=code_gen_model,
                                           temperature=TEMPERATURE,
                                           max_output_tokens=MAX_OUTPUT_TOKENS
                                           )

        while tries < max_tries:
            try:
                if genai_model_name == "GeminiPro":
                    response = chat_session.send_message(prompt)
                else:
                    response = chat_session.send_message(
                        prompt,
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS
                    )

                generated_sql_query = response.text
                generated_sql_query = '\n'.join(
                    generated_sql_query.split('\n')[1:-1]
                )

                generated_sql_query = self.case_handler_transform(
                    generated_sql_query
                )
                generated_sql_query = self.add_dataset_to_query(
                    generated_sql_query
                )
                df = client.query(generated_sql_query).to_dataframe()
                successful_queries.append({
                    "query": generated_sql_query,
                    "dataframe": df
                })
                if len(successful_queries) > 1:
                    prompt = f"""Modify the last successful SQL query by
                        making changes to it and optimizing it for latency.
                        ENSURE that the NEW QUERY is DIFFERENT from the
                        previous one while prioritizing faster execution.
                        Reference the tables only from the above given
                        project and dataset
                        The last successful query was:
                        {successful_queries[-1]["query"]}"""

            except Exception as e:
                msg = str(e)
                error_messages.append(msg)
                prompt = f"""Encountered an error: {msg}.
                    To address this, please generate an alternative SQL
                    query response that avoids this specific error.
                    Follow the instructions mentioned above to
                    remediate the error.

                    Modify the below SQL query to resolve the issue and
                    ensure it is not a repetition of all previously
                    generated queries.
                    {generated_sql_query}

                    Ensure the revised SQL query aligns precisely with the
                    requirements outlined in the initial question.
                    Keep the table names as it is. Do not change hyphen
                    to underscore character
                    Additionally, please optimize the query for latency
                    while maintaining correctness and efficiency."""
                prompts.append(prompt)

            tries += 1

        if len(successful_queries) == 0:
            return {
                "error": "All attempts exhausted.",
                "prompts": prompts,
                "errors": error_messages
            }
        else:
            df = pd.DataFrame(
                [(q["query"], q["dataframe"])
                 for q in successful_queries], columns=["Query", "Result"]
            )
            return {
                "dataframe": df
            }

    def generate_sql_with_join(self,
                               dataset,
                               table_1_name,
                               table_2_name,
                               question,
                               example_table1,
                               example_table2,
                               sample_question=None,
                               sample_sql=None,
                               one_shot=False,
                               join_gen="STANDARD"):
        gen_join_sql = ""
        match join_gen:
            case 'STANDARD':
                if not one_shot:
                    # Zero-shot Join query generation
                    join_prompt = self.get_join_prompt(dataset,
                                                       table_1_name,
                                                       table_2_name,
                                                       question)
                    gen_join_sql = self.invoke_llm(join_prompt)
                else:
                    # One-shot Join query generation
                    join_prompt_one_shot = self.get_join_prompt(
                        dataset,
                        table_1_name,
                        table_2_name,
                        question,
                        sample_question,
                        sample_sql,
                        one_shot=True
                    )
                    gen_join_sql = self.invoke_llm(
                        join_prompt_one_shot
                    )

            case 'MULTI_TURN':
                table_1_name, table_2_name = \
                    self.multi_turn_table_filter(
                        table_1_name=example_table1,
                        table_2_name=example_table2,
                        sample_question=sample_question,
                        sample_sql=sample_sql,
                        question=question
                        )
                # One-shot Join query generation
                join_prompt_one_shot = self.get_join_prompt(
                    data_set,
                    table_1_name,
                    table_2_name,
                    question,
                    sample_question,
                    sample_sql,
                    one_shot=True
                )
                gen_join_sql = self.invoke_llm(
                    join_prompt_one_shot
                )

            case 'SELF_CORRECT':
                join_prompt_one_shot = self.get_join_prompt(
                    data_set,
                    table_1_name,
                    table_2_name,
                    question,
                    sample_question,
                    sample_sql,
                    one_shot=True
                )
                # Self-Correction Approach
                responses = self.gen_and_exec_and_self_correct_sql(
                    join_prompt_one_shot
                )
                gen_join_sql = responses[0]['query']

        return gen_join_sql


if __name__ == '__main__':

    project_id = os.environ['PROJECT_ID']
    dataset_id = os.environ['DATASET_ID']
    print("Info =", project_id, dataset_id)
    meta_data_json_path = "cache_metadata/metadata_cache.json"
    nl2sqlbq_client = Nl2sqlBq(
        project_id=project_id,
        dataset_id=dataset_id,
        metadata_json_path=meta_data_json_path,
        # "../cache_metadata/metadata_cache.json",
        model_name="text-bison"
        # model_name="code-bison"
        )
    questions = ["How have these race and ethnicity trends changed over time?",
                 "What about three or more additional programs?",
                 "Which five counties have the lowest number of WIC\
                     authorized vendors compared to WIC participants?",
                 "How do infant mortality rates, low birthweight rates,\
                     and preterm and very preterm rates \
                     compare to WIC enrollment rates by county?",
                 "How many Black individuals are served across\
                     CalHHS programs?",
                 "What is the breakdown by program?",
                 "Has this changed over time?",
                 "What is the ratio of non-suspended doctors \
                     to Medi-Cal members by County?",
                 ]
    question = questions[0]
    table_identified = nl2sqlbq_client.table_filter(question)

    PGPROJ = os.environ['PROJECT_ID']  # "sl-test-project-363109"
    PGLOCATION = os.environ['REGION']  # 'us-central1'
    PGINSTANCE = os.environ['PG_INSTANCE']  # "nl2sql-test"
    PGDB = os.environ['PG_DB']  # "test-db"
    PGUSER = os.environ['PG_USER']  # "postgres"
    PGPWD = os.environ['PG_PWD']  # "nl2sql-test"

    nl2sqlbq_client.init_pgdb(PGPROJ,
                              PGLOCATION,
                              PGINSTANCE,
                              PGDB,
                              PGUSER,
                              PGPWD)
    sql_query, _ = nl2sqlbq_client.text_to_sql_execute_few_shot(
        question,
        'medi-cal-and-calfresh-enrollment'
    )
    print("Generated query == ", sql_query)

    nl_resp = nl2sqlbq_client.result2nl(sql_query, question)
    print("Response in NL = ", nl_resp)

    table_1_name = ""
    table_2_name = ""
    sample_question = ""
    sample_sql = ""
    data_set = ""
    example_table_1 = ""
    example_table_2 = ""

    # Zero-shot Join query generation
    join_prompt = nl2sqlbq_client.get_join_prompt(data_set,
                                                  table_1_name,
                                                  table_2_name,
                                                  question)
    gen_join_sql = nl2sqlbq_client.invoke_llm(join_prompt)
    print("SQL query wiith Join - ", gen_join_sql)

    # One-shot Join query generation
    join_prompt_one_shot = nl2sqlbq_client.get_join_prompt(data_set,
                                                           table_1_name,
                                                           table_2_name,
                                                           question,
                                                           sample_question,
                                                           sample_sql,
                                                           one_shot=True)
    gen_join_sql = nl2sqlbq_client.invoke_llm(join_prompt_one_shot)
    print("SQL query wiith Join - ", gen_join_sql)

    # Table Identification with Multi-turn approach
    example_table_1 = ""
    example_table_2 = ""
    table_1_name, table_2_name = nl2sqlbq_client.multi_turn_table_filter(
        table_1_name=example_table_1,
        table_2_name=example_table_2,
        sample_question=sample_question,
        sample_sql=sample_sql,
        question=question
    )
    # One-shot Join query generation
    join_prompt_one_shot = nl2sqlbq_client.get_join_prompt(data_set,
                                                           table_1_name,
                                                           table_2_name,
                                                           question,
                                                           sample_question,
                                                           sample_sql,
                                                           one_shot=True)
    gen_join_sql = nl2sqlbq_client.invoke_llm(join_prompt_one_shot)
    print("SQL query wiith Join - ", gen_join_sql)

    # Self-Correction Approach
    responses = nl2sqlbq_client.gen_and_exec_and_self_correct_sql(
        join_prompt_one_shot
    )
    print(responses)

    # Common function to perform either of the operations
    gen_join_sql = nl2sqlbq_client.generate_sql_with_join(
        data_set,
        table_1_name,
        table_2_name,
        question,
        example_table_1,
        example_table_2,
        sample_question,
        sample_sql,
        True,
        "STANDARD"  # STANDARD or  MULTI_TURN or SELF_CORRECT
        )
