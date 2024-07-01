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


Table_filtering_prompt = """
You are a database expert at selecting a table from a list of tables based
on their description.
For the provided question choose what is the table_name most likely to be
relevant.
Only mention the table name from the following list and their description.
Do not mention any information more than the table name.
Output should be only 1 table that is the most likely table to contain the
relevant data
Do not include any special characters in the generated output

Table name | description
{only_tables_info}

Question: {question}
"""

# Table_filtering_prompt_promptonly = """
# You are a database expert at selecting a table from a list of tables based
# on their description.
# For the provided question choose what is the table_name most likely to be
# relevant.
# Only mention the table name from the following list and their description.
# Do not mention any information more than the table name.
# Output should be only 1 table that is the most likely table to contain the
# relevant data
# Do not include any special characters in the generated output

# Table name | description
# {only_tables_info}

# """

Table_filtering_prompt_promptonly = """
You are a database expert at selecting a table from a list of tables based
on their description.
For the provided question choose what is the table_name most likely to be
relevant.
Only mention the table name from the following list and their description.
Do not mention any information more than the table name.
Do not include any special characters in the generated output

Table name | description
{only_tables_info}

"""


Result2nl_insight_prompt = '''
You are an expert Data Analyst. Given a report of SQL query and the question
in natural language, provide a very insightful, intuitive and a not too long
well-explained summary of the result which would help the user understand the
result better and take informed decisions.  If the result does not have any
data, then just mention that briefly in the summary.

question: {question}
result: {result}'''

Result2nl_prompt = '''
You are an expert Data Analyst. Given a report of SQL query and the question
in natural language, provide a very crisp, short, intuitive and
easy-to-understand summary of the result.  If the result does not have any
data, then just mention that briefly in the summary.

question: {question}
result: {result}
'''

Auto_verify_sql_prompt = '''
You are an expert at validating SQL queries. Given the Natrual language
description and the SQL query corresponding to that description, please check
if the students answer is correct.  There can be different ways to achieve the
same result by forming the query differently. If the students SQL query
matches the ground truth and fits the NL description correctly, then return
yes else return no.

Natural language description: {nl_description}
Ground truth: {ground_truth}
students answer: {llm_amswer}
'''

Sql_Generation_prompt = '''
Only use the following table's meta-data:

```
Table Name : {table_name}

Description: {table_description}

This table has the following columns :
{columns_info}
\n
```

You are an SQL expert at generating SQL queries from a natural language
question. Given the input question, create a syntactically correct
Biguery query to run.

Only use the few relevant columns for the given question.
Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table. Do not use more than 10
columns in the query. Focus on the keywords indicating calculation.
Please think step by step and always validate the reponse.
Generate SQL with Join only when 2 tables are involved
Rectify each column names by referencing them from the meta-data.

For this question what would be the most accurate SQL query?
Question: {question}
'''

Sql_Generation_prompt_few_shot = '''
Only use the following tables meta-data:

```
Table Name : {table_name}

Description: {table_description}

This table has the following columns :
{columns_info}
\n
```

You are an SQL expert at generating SQL queries from a natural language
question. Given the input question, create a syntactically correct
Biguery query to run.

Only use the few relevant columns given in the question.
Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table. Do not use more than 10
columns in the query. Focus on the keywords indicating calculation.
Please think step by step and always validate the reponse.
Rectify each column names by referencing them from the meta-data.
Generate SQL with Join only when 2 tables are involved
Use the following examples as guidelines to generate the new BigQuery SQL
accordingly

{few_shot_examples}

For this question what would be the most accurate SQL query?
Question: {question}
'''

Sql_Generation_prompt_few_shot_multiturn = '''
Only use the following tables meta-data:

```
Table Name : {table_name}

Description: {table_description}

This table has the following columns :
{columns_info}
\n
```

You are an SQL expert at generating SQL queries from a natural language
question. Given the input question, create a syntactically correct Biguery
query to run.

Only use the few relevant columns given in the question.
Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table. Do not use more than 10
columns in the query. Focus on the keywords indicating calculation.
Please think step by step and always validate the reponse.
rectify each column names by referencing them from the meta-data.
Use the following examples as guidelines to generate the new BigQuery SQL
accordingly

{few_shot_examples}

{additional_context}

For this question what would be the most accurate SQL query?
Question: {question}
'''

additional_context_prompt = """As you are database expert who can generate
SQL query statements from natural language statements you need to modify a
given SQL to suit the current question.
An SQL query that is generated for another related question is given below

SQL Query : {prev_sql}

The question given below is a follow-up question for an already answered
question. The SQL query statement that needs to be generated will be a
modification or enhancement to the SQL Query statement given above
Enhance the above given SQL query to fulfil the requirements of the question
given below

Consider only the table given in the above SQL to generate the query.
"""

Table_info_template = """

Table Name : {table_name}

Description: {table_description}

This table has the following columns :
{columns_info}
\n
"""

join_prompt_template = """You are an SQL expert at generating SQL queries
from a natural language question.

Please craft a SQL query for BigQuery that is valid for the QUESTION
provided below.
Ensure you reference the appropriate BigQuery tables and column names provided
in the SCHEMA below.
Break down the question meaningfully into sub questions before making a
decision to generate SQL
Understand the Business Intelligence given below to craft the SQL query
When joining tables, employ type coercion to guarantee data type consistency
for the join columns.
Additionally, the output column names should specify units where applicable.\n

Only use the few relevant columns required based on the question.
Pay attention to use only the column names that you can see in the schema
description.
Be careful to not query for columns that do not exist. Also, pay attention to
which column is in which table.
Do not use more than 10 columns in the query.
Please think step by step and always validate the reponse.
Rectify each column names by referencing them from the SCHEMA.
Ensure you do not alter the table names in the SQL query

SCHEMA: \n
Project and Dataset : {data_set}
Table 1: {table_1}
Table 2: {table_2}

Business Intelligence:
For this question, the two tables are to joined on Number and County columns
in the respective tables where the RecordType having value as Provider and
Level column is having value as County to calculate the ratio of number of
ObjectId and total number of persons

For this question what would be the most accurate SQL query?
QUESTION: {question}
"""

join_prompt_template_one_shot = """You are an SQL expert at generating SQL
queries from a natural language question.

Please craft a SQL query for BigQuery that addresses the QUESTION provided
below.
Ensure you reference the appropriate BigQuery tables and column names provided
in the SCHEMA below.
Break down the question meaningfully into sub questions before making a
decision to generate SQL
Understand the Business Intelligence given below to craft the SQL query
When joining tables, employ type coercion to guarantee data type
consistency for the join columns.
Additionally, the output column names should specify units where applicable.\n

Only use the few relevant columns required based on the question.
Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.
Do not use more than 10 columns in the query.
Please think step by step and always validate the reponse.
Rectify each column names by referencing them from the SCHEMA.
Ensure you do not alter the table names in the SQL query

SCHEMA: \n
Project and Dataset : {data_set}
Table 1: {table_1}
Table 2: {table_2}

Business Intelligence:
For this question, the two tables are to joined on Number and County columns
in the respective tables where the RecordType having value as Provider and
Level column is having value as County to calculate the ratio of number of
ObjectId and total number of persons

For reference, one example SQL query with JOIN between two tables is given
below

Question: {sample_question}
SQL : {sample_sql}

For this question what would be the most accurate SQL query?
QUESTION: {question}
"""

multi_table_prompt = """
Tables context:
{table_info}

Example Question, SQL and tables containing the required info are given below
You are required to identify more than 1 table that probably contains the
information requested in the question given below
Return the list of tables that may contain the information

Question : {example_question} :
SQL : {example_SQL}
Tables: {table_name_1} and {table_name_2}

Question: {question}
Tables:
"""

follow_up_prompt = """Review the question given in above context along with
the table and column description and determine whether one table contains
all the required information or you need to get data from another table
If two tables's information are required, then identify those tables from
the tables info
What are the two tables that should be joined in the SQL query
Only mention the table name from the tables context.
"""
