# nl2sql_agent/db_metadata.py
import json
import os
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel

from . import config

class BigQueryMetadata:
    """Handles caching of BigQuery table schemas and descriptions."""

    def __init__(self, project_id: str, dataset_id: str, tables_list: list[str] | None = None):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.tables_list = tables_list or []
        self.bq_client = bigquery.Client(project=self.project_id)
        self.metadata_cache_path = f"./metadata_cache_{self.dataset_id}.json"
        self.metadata = self._load_or_create_metadata()

    def _load_or_create_metadata(self):
        """Loads metadata from a local cache or creates it if it doesn't exist."""
        if os.path.exists(self.metadata_cache_path):
            print(f"Loading metadata from cache: {self.metadata_cache_path}")
            with open(self.metadata_cache_path, 'r') as f:
                return json.load(f)
        print("Metadata cache not found. Creating new cache...")
        return self._create_metadata_cache()

    def _create_metadata_cache(self):
        """Creates metadata cache by fetching schemas and generating descriptions."""
        model = GenerativeModel(config.MODEL)

        gen_description_prompt = """
        Based on the table name and column information, generate a concise,
        brief, description for this table.
        TABLE NAME: {table_id}
        COLUMNS_INFO: {columns_info}
        DESCRIPTION:
        """

        if not self.tables_list:
            api_response = self.bq_client.list_tables(self.dataset_id)
            self.tables_list = [table.table_id for table in api_response]

        metadata = {}
        for table_id in self.tables_list:
            table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"
            columns_info = self.bq_client.get_table(table_ref).to_api_repr()['schema']

            for field in columns_info.get('fields', []):
                field.pop('mode', None)

            # Enrich columns with stats and distinct values
            numeric_columns = []
            string_columns = []
            for field in columns_info.get('fields', []):
                if field['type'] in ('INTEGER', 'FLOAT', 'NUMERIC', 'BIGNUMERIC'):
                    numeric_columns.append(field['name'])
                elif field['type'] == 'STRING':
                    string_columns.append(field['name'])

            if numeric_columns or string_columns:
                select_clauses = []
                for col in numeric_columns:
                    select_clauses.append(f"MIN({col}) as min_{col}")
                    select_clauses.append(f"MAX({col}) as max_{col}")
                    select_clauses.append(f"AVG({col}) as avg_{col}")
                for col in string_columns:
                    # Using a subquery to get the top 5 most frequent values for a string column.
                    select_clauses.append(f"(SELECT ARRAY_AGG({col}) FROM (SELECT {col} FROM `{table_ref}` WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY COUNT(*) DESC LIMIT 5)) as distinct_{col}")

                if select_clauses:
                    query = f"SELECT {', '.join(select_clauses)} FROM `{table_ref}`"
                    print(f"Executing stats query for {table_id}...")
                    try:
                        query_job = self.bq_client.query(query)
                        stats_results = list(query_job.result())
                        if stats_results:
                            stats_row = stats_results[0]
                            col_map = {field['name']: field for field in columns_info.get('fields', [])}
                            for col in numeric_columns:
                                col_map[col]['stats'] = {'min': stats_row[f'min_{col}'], 'max': stats_row[f'max_{col}'], 'avg': stats_row[f'avg_{col}']}
                            for col in string_columns:
                                col_map[col]['distinct_values'] = stats_row[f'distinct_{col}']
                    except Exception as e:
                        print(f"Could not fetch stats for table {table_id}: {e}")

            metadata[table_id] = {
                "table_name": table_id,
                "columns_info": columns_info,
            }
            prompt = gen_description_prompt.format(
                table_id=table_id,
                columns_info=json.dumps(columns_info)
            )
            response = model.generate_content(prompt)
            metadata[table_id]["table_description"] = response.text.strip()
            print(f"Generated description for table: {table_id}")

        with open(self.metadata_cache_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata cache created at: {self.metadata_cache_path}")
        return metadata

    def get_all_tables_info(self) -> dict:
        """Returns a summary of all cached tables."""
        return {
            name: {"description": info["table_description"]}
            for name, info in self.metadata.items()
        }

    def get_table_schema(self, table_id: str) -> dict:
        """Returns the schema for a specific table from the cache."""
        # Handle cases where table_id might include the dataset
        clean_table_id = table_id.split('.')[-1]
        return self.metadata.get(clean_table_id, {})
