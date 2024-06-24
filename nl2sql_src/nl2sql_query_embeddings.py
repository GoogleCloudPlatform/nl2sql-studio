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

# import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# from google.cloud import aiplatform
# from google.cloud import storage

from vertexai.preview.language_models import TextEmbeddingModel
# from io import StringIO
# import csv

# from vertexai.language_models import CodeGenerationModel
# import pickle
import json
import faiss
from faiss import write_index, read_index

# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer

import os
from pandas import DataFrame

from google.cloud.sql.connector import Connector, IPTypes
import pg8000

import sqlalchemy
from loguru import logger

ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC


class Nl2Sql_embed():
    """
        Local Embeddings and Local Vector DB class
    """
    def __init__(self):
        # Init function
        self.EMBEDDING_FILE = "dataset/embeddings.json"
        self.INDEX_FILE = 'dataset/saved_index_localdata'
        self.embedding_model =\
            TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    def generate_embedding(self, query, sql='blank sql'):
        """
            Generates text embeddings
        """

        # Replace this with your actual embedding generation
        # using text-gecko003 or another model
        q_embeddings = self.embedding_model.get_embeddings([query])[0].values
        sql_embeddings = self.embedding_model.get_embeddings([sql])[0].values

        return q_embeddings, sql_embeddings

    # def generate_bert_embeddings(self, documents):
    #     # Load pre-trained BERT model
    #     model = SentenceTransformer('bert-base-nli-mean-tokens')
    #     # Generate BERT embeddings for documents
    #     embeddings = model.encode(documents)

    # return embeddings

    def insert_data(self, question, sql):
        """
            Inserts data to Embeddings file
        """

        logger.info(f"Inserting data. Question : {question}, SQL : {sql}")
        try:
            with open(self.EMBEDDING_FILE, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []

        q_emb, sql_emb = self.generate_embedding(question, sql)
        data.append({
            "question": question,
            "sql": sql,
            "question_embedding": q_emb,
            "sql_embedding": sql_emb
        })

        with open(self.EMBEDDING_FILE, "w") as f:
            json.dump(data, f)

    def load_embeddings(self):
        """
            Read the Embeddigs.json file to memory
        """

        with open(self.EMBEDDING_FILE, "r") as f:
            data = json.load(f)
        return data

    def distance(self, embedding1, embedding2):
        """Calculates negative cosine similarity"""
        return -cosine_similarity([embedding1], [embedding2])[0][0]

    def find_closest_questions(self, new_question, data, n=3):
        """
            Return 3 most similar queeries and SQLs
        """

        new_embedding, _ = self.generate_embedding(new_question)

        distances = [
            self.distance(
                new_embedding,
                item["question_embedding"]) for item in data]

        closest_indices = np.argsort(distances)[:n]

        return [(data[i]['question'], data[i]['sql']) for i in closest_indices]

    def create_vectordb_index(self):
        """
            Recreate VectorDB indes file
        """

        embeddings_data = self.load_embeddings()

        query_embeddings = [
            item['question_embedding'] for item in embeddings_data
            ]

        # query_array_updated = [[item['question'],
        #                         item['sql']] for item in embeddings_data]
        embeddings_data_array = np.asarray(query_embeddings, dtype=np.float32)

        index = faiss.IndexFlatIP(len(query_embeddings[0]))
        index.add(embeddings_data_array)
        write_index(index, self.INDEX_FILE)

        # return index, query_array_updated
        return

    def search_matching_queries(self, new_query):
        """
            Return 3 most similar queeries and SQLs
        """

        embeddings_data = self.load_embeddings()
        query_array_updated = [[item['question'],
                                item['sql']] for item in embeddings_data]

        nq_emb = self.embedding_model.get_embeddings([new_query])[0].values
        nq_emb_array = np.asarray([nq_emb], dtype=np.float32)

        index = read_index(self.INDEX_FILE)

        scores, id = index.search(nq_emb_array, k=3)

        output_json = []
        for i in range(len(scores[0])):
            res = {}
            res['question'] = query_array_updated[id[0][i]][0]
            res['sql'] = query_array_updated[id[0][i]][1]
            output_json.append(res)

        return output_json


class PgSqlEmb():
    """
        PostgreSQL DB interface class
    """

    def __init__(self,
                 proj_id,
                 loc,
                 pg_inst,
                 pg_db,
                 pg_uname,
                 pg_pwd,
                 pg_table='documents',
                 index_file='saved_index_pgdata'):
        # Init function
        # self.EMBEDDING_FILE = "embeddings.json"

        self.PGPROJ = proj_id
        self.PGLOCATION = loc
        self.PGINSTANCE = pg_inst
        self.CONNSTRING = f"{self.PGPROJ}:{self.PGLOCATION}:{self.PGINSTANCE}"
        self.USER = pg_uname
        self.PWD = pg_pwd
        self.PGDB = pg_db
        self.PGTABLE = pg_table

        # self.INDEX_FILE = 'saved_index_pgdata'
        self.INDEX_FILE =\
            f"../../nl2sql-generic/nl2sql_src/cache_metadata/{index_file}"
        self.embedding_model =\
            TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        self.pool = self.getpool()

    def getconn(self) -> pg8000.dbapi.Connection:
        """
        Get DB connection
        """
        connector = Connector()

        conn: pg8000.dbapi.Connection = connector.connect(
            self.CONNSTRING,
            "pg8000",
            user=self.USER,
            password=self.PWD,
            db=self.PGDB,
            ip_type=ip_type,
        )
        return conn

    def getpool(self):
        """
        return connection pool
        """
        pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=self.getconn,
            # ...
            )
        return pool

    def create_table(self):
        """
        Create table in PostgreSQL Db
        """
        sql_create = f"""CREATE TABLE IF NOT EXISTS {self.PGTABLE} (
             q_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
             question TEXT,
             sql TEXT,
             query_embedding TEXT
            );"""
        with self.pool.connect() as conn:
            conn.execute(sql_create)

    def empty_table(self, remove_index=True):
        """
        Delete all rows in the PostgreSQL DB
        """
        sql_clear = f'DELETE from {self.PGTABLE}'
        with self.pool.connect() as conn:
            conn.execute(sql_clear)
        if remove_index:
            try:
                os.remove(self.INDEX_FILE)
            except Exception:
                pass

    def insert_row(self, query, sql):
        """
            Insert question and embeddings to PostgreSQL DB
        """
        sql = sql.replace("'", "<sq>")
        sql = sql.replace('"', '<dq>')
        emb = self.embedding_model.get_embeddings([query])[0].values

        sql_ins = f"INSERT INTO {self.PGTABLE}\
            (question, sql, query_embedding) values\
            ('{query}', '{sql}', '{emb}')"
        with self.pool.connect() as conn:
            conn.execute(sql_ins)

        self.update_vectordb_index(query)

    def extract_data(self):
        """
            REturn all data from DB
        """
        sql_data = f'SELECT * FROM {self.PGTABLE}'
        with self.pool.connect() as conn:
            data = conn.execute(sql_data)
        return data

    def extract_pg_embeddings(self):
        """
            Extract embeddings data fro PG database
        """
        tmp = self.extract_data()
        df = DataFrame(tmp.fetchall())

        q_embed = df['query_embedding']
        len(q_embed)
        query_embeddings = [item.split(' ') for item in q_embed]
        new_array = []
        for elem in query_embeddings:
            new_row = []

            # Need to make some computations to convert the
            # embeddings stored as string to array of floats

            first_elem = elem[0].split('[')[1].split(',')[0]
            new_row.append(float(first_elem))
            for i in range(1, len(elem)-1):
                t_elem = elem[i].split(',')[0]
                new_row.append(float(t_elem))
            last_elem = elem[len(elem)-1].split(']')[0]
            new_row.append(float(last_elem))

            new_array.append(new_row)

        return df['question'], df['sql'], new_array

    def recreate_vectordb_index(self):
        """
            Regenerate VectorDB file from PG Table data
        """
        tmp = self.extract_data()
        df = DataFrame(tmp.fetchall())

        q_embed = df['query_embedding']
        query_embeddings = [item.split(' ') for item in q_embed]
        new_array = []
        for elem in query_embeddings:
            new_row = []

            # Need to make some computations to convert the
            # embeddings stored as string to array of floats

            first_elem = elem[0].split('[')[1].split(',')[0]
            new_row.append(float(first_elem))
            for i in range(1, len(elem)-1):
                t_elem = elem[i].split(',')[0]
                new_row.append(float(t_elem))
            last_elem = elem[len(elem)-1].split(']')[0]
            new_row.append(float(last_elem))

            new_array.append(new_row)

        embeddings_data_array = np.asarray(new_array, dtype=np.float32)
        index = faiss.IndexFlatIP(len(query_embeddings[0]))
        index.add(embeddings_data_array)
        write_index(index, self.INDEX_FILE)
        return

    def update_vectordb_index(self, query):
        """
            Update VectorDB on every query insert
        """
        emb = self.embedding_model.get_embeddings([query])[0].values
        new_array = [emb]

        embeddings_data_array = np.asarray(new_array, dtype=np.float32)

        # Read the index from stored index file
        try:
            index = read_index(self.INDEX_FILE)
        except Exception:
            index = faiss.IndexFlatIP(len(new_array[0]))

        index.add(embeddings_data_array)
        write_index(index, self.INDEX_FILE)

        return

    def search_matching_queries(self, new_query):
        """
            Return 3 most similar queeries and SQLs
        """
        tmp = self.extract_data()
        df = DataFrame(tmp.fetchall())

        # q_embed = df['query_embedding']
        # query_embeddings = [item.split(' ') for item in q_embed]

        queries_array = df['question']
        sql_array = df['sql']

        nq_emb = self.embedding_model.get_embeddings([new_query])[0].values
        nq_emb_array = np.asarray([nq_emb], dtype=np.float32)

        try:
            logger.info(f"Trying to read the index file : {self.INDEX_FILE}")
            index = read_index(self.INDEX_FILE)
        except Exception:
            self.recreate_vectordb_index()
            index = read_index(self.INDEX_FILE)

        scores, id = index.search(nq_emb_array, k=3)

        output_json = []
        for i in range(len(scores[0])):
            res = {}
            tmp_sql = ''
            res['question'] = queries_array[id[0][i]]

            tmp_sql = sql_array[id[0][i]]
            tmp_sql = tmp_sql.replace('<dq>', '"')
            tmp_sql = tmp_sql.replace("<sq>", "'")
            res['sql'] = tmp_sql
            output_json.append(res)

        return output_json
