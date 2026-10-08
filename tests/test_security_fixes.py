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
"""Unit tests for security fixes across API, UI, and executor modules."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.dont_write_bytecode = True


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _install_stub_modules():
    """Install lightweight stubs for optional cloud/LLM dependencies if absent."""
    stub_names = [
        "vertexai",
        "vertexai.generative_models",
        "vertexai.language_models",
        "vertexai.preview",
        "vertexai.preview.generative_models",
        "dotenv",
        "loguru",
        "sqlglot",
        "langchain",
        "langchain.llms",
        "langchain.llms.base",
        "langchain.output_parsers",
        "langchain.prompts",
        "langchain.prompts.prompt",
        "langchain.prompts.few_shot",
        "langchain.schema",
        "langchain.sql_database",
        "langchain_google_vertexai",
        "nl2sql_query_embeddings",
        "nl2sql.assets.prompts",
        "sqlalchemy",
        "sqlalchemy.exc",
        "sqlalchemy.engine",
        "sqlalchemy.engine.base",
        "sqlalchemy.sql",
        "sqlalchemy.sql.ddl",
        "sqlalchemy.sql.functions",
        "sqlalchemy.sql.schema",
        "sqlalchemy.sql.sqltypes",
        "sqlalchemy.schema",
    ]
    for name in stub_names:
        if name not in sys.modules:
            mock_mod = MagicMock()
            mock_mod.__path__ = []
            sys.modules[name] = mock_mod


_install_stub_modules()


def _load_module_from_path(module_name: str, rel_path: str):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    parent_dir = os.path.dirname(abs_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, os.path.join(REPO_ROOT, "UI"))
sys.path.insert(0, os.path.join(REPO_ROOT, "nl2sql_library"))
ui_dbai = _load_module_from_path("ui_dbai_mod", "UI/dbai_src/dbai.py")
root_dbai = _load_module_from_path("root_dbai_mod", "dbai_src/dbai.py")
lib_utils = _load_module_from_path(
    "lib_utils_mod", "nl2sql_library/utils/utility_functions.py"
)
src_utils = _load_module_from_path(
    "src_utils_mod", "nl2sql_src/utils/utility_functions.py"
)
with unittest.mock.patch(
    "google.cloud.bigquery.Client", return_value=MagicMock()
):
    nl2sql_generic = _load_module_from_path(
        "nl2sql_generic_mod", "nl2sql_src/nl2sql_generic.py"
    )
datasets_base = _load_module_from_path(
    "datasets_base_mod", "nl2sql_library/nl2sql/datasets/base.py"
)


class TestSafeBuildPlotlyFigure(unittest.TestCase):
    """Tests for AST-based Plotly figure builder replacing exec()."""

    def setUp(self):
        self.builders = [
            ui_dbai._safe_build_plotly_figure,
            root_dbai._safe_build_plotly_figure,
        ]

    def test_valid_standard_chart_with_fences(self):
        code = """```python
import plotly.express as px
import pandas as pd

data = {'Year': [2020, 2021, 2022], 'Sales': [100, 150, 200]}
df = pd.DataFrame(data)
fig = px.line(df, x='Year', y='Sales', title='Annual Sales')
```"""
        for build in self.builders:
            fig = build(code)
            self.assertTrue(hasattr(fig, "to_plotly_json"))
            self.assertEqual(fig.layout.title.text, "Annual Sales")

    def test_container_literals_with_variables_and_update_layout(self):
        code = """
from plotly import express as px
import pandas as pd

years = [2020, 2021]
sales = [10, 20]
data = {'Year': years, 'Sales': sales}
df = pd.DataFrame(data)
fig = px.bar(df, x='Year', y='Sales')
fig.update_layout(title='Updated Title')
fig.show()
"""
        for build in self.builders:
            fig = build(code)
            self.assertEqual(fig.layout.title.text, "Updated Title")

    def test_chained_update_layout_and_plotly_express_attr(self):
        code = """
import plotly.express
import pandas as pd

df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
fig = plotly.express.scatter(df, x='a', y='b').update_layout(title='Chained')
"""
        for build in self.builders:
            fig = build(code)
            self.assertEqual(fig.layout.title.text, "Chained")

    def test_rejects_rce_payloads(self):
        malicious_snippets = [
            "import os\nos.system('echo pwned')",
            "from os import system\nsystem('echo pwned')",
            "fig = __import__('os').system('echo pwned')",
            "fig = open('/etc/passwd').read()",
            "_private = 1\nfig = px.bar()",
            "df = pd.read_csv('/etc/passwd')\nfig = px.bar(df)",
            "data = {'a': [1]}\ndf = pd.DataFrame(**data)\nfig = px.bar(df)",
            "args = [{'a': [1]}]\ndf = pd.DataFrame(*args)\nfig = px.bar(df)",
            "df = pd.DataFrame({'a': [1]})\n",  # no fig produced
        ]
        for build in self.builders:
            for snippet in malicious_snippets:
                with self.assertRaises(
                    ValueError, msg=f"Failed to reject: {snippet}"
                ):
                    build(snippet)


class TestValidateReadonlySql(unittest.TestCase):
    """Tests for read-only SQL query validation across modules."""

    def setUp(self):
        self.validators = [
            ui_dbai.validate_readonly_sql,
            root_dbai.validate_readonly_sql,
            lib_utils.validate_readonly_sql,
            src_utils.validate_readonly_sql,
            nl2sql_generic.Nl2sqlBq._validate_readonly_query,
            datasets_base.Database.validate_readonly_query,
        ]

    def test_allows_valid_select_and_with_queries(self):
        valid_queries = [
            "SELECT * FROM `proj.dataset.table` WHERE id = 1;",
            "```sql\nSELECT count(*) FROM singers;\n```",
            "WITH cte AS (SELECT id FROM t) SELECT * FROM cte",
            "(SELECT 1) UNION ALL (SELECT 2)",
            "SELECT TRUNCATE(3.14159, 2) AS val",
        ]
        for validate in self.validators:
            for q in valid_queries:
                cleaned = validate(q)
                self.assertTrue(len(cleaned) > 0)

    def test_preserves_string_literals_with_comments_semicolons_and_keywords(self):
        query = (
            "SELECT * FROM `my-proj.nl2sql_spider.update` "
            "WHERE name = 'Grant' "
            "AND status = 'UPDATE' "
            "AND note LIKE '%--not-a-comment%' "
            "AND path LIKE '/*not-a-block*/' "
            "AND sep = ';';"
        )
        for validate in self.validators:
            cleaned = validate(query)
            self.assertIn("'%--not-a-comment%'", cleaned)
            self.assertIn("'/*not-a-block*/'", cleaned)
            self.assertIn("';'", cleaned)
            self.assertFalse(cleaned.endswith(";"))

    def test_rejects_destructive_and_multi_statement_queries(self):
        invalid_queries = [
            "",
            "   ",
            "-- just a comment",
            "DROP TABLE singers",
            "DELETE FROM singers WHERE 1=1",
            "INSERT INTO singers VALUES (1, 'a')",
            "UPDATE singers SET name = 'b'",
            "ALTER TABLE singers ADD COLUMN c INT64",
            "CREATE TABLE hacked AS SELECT 1",
            "TRUNCATE TABLE singers",
            "MERGE INTO target USING src ON true WHEN MATCHED THEN DELETE",
            "GRANT ALL ON dataset TO user",
            "REVOKE ALL ON dataset FROM user",
            "CALL my_dataset.stored_proc()",
            "EXECUTE IMMEDIATE 'DROP TABLE x'",
            "EXPORT DATA OPTIONS(uri='gs://bucket/*') AS SELECT * FROM t",
            "WITH cte AS (SELECT 1) DELETE FROM singers WHERE id IN (SELECT * FROM cte)",
            "SELECT 1; DROP TABLE singers",
            "SELECT * FROM singers -- comment\n; DROP TABLE singers",
            "SELECT * FROM singers /* comment */; SELECT 2",
        ]
        for validate in self.validators:
            for q in invalid_queries:
                with self.assertRaises(ValueError, msg=f"Failed to reject: {q}"):
                    validate(q)


class TestSanitizeMetadataFilename(unittest.TestCase):
    """Tests for path traversal and reserved file protection in metadata filenames."""

    def setUp(self):
        self.sanitizers = [
            lib_utils.sanitize_metadata_filename,
            src_utils.sanitize_metadata_filename,
        ]

    def test_allows_safe_json_filenames(self):
        for sanitize in self.sanitizers:
            self.assertEqual(
                sanitize("spider_md_cache.json"), "spider_md_cache.json"
            )
            self.assertEqual(
                sanitize("zoominfo-metadata_v2.json"),
                "zoominfo-metadata_v2.json",
            )

    def test_rejects_path_traversal_and_non_json(self):
        unsafe_names = [
            "",
            None,
            "../proj_config.json",
            "../../etc/passwd",
            "sub/metadata.json",
            "\\..\\metadata.json",
            "metadata..json",
            "metadata.py",
            "metadata.json.bak",
            ".hidden.json",
        ]
        for sanitize in self.sanitizers:
            for name in unsafe_names:
                with self.assertRaises(ValueError, msg=f"Failed to reject: {name}"):
                    sanitize(name)

    def test_rejects_reserved_internal_utils_files(self):
        reserved = [
            "proj_config.json",
            "PROJ_CONFIG.JSON",
            "sqlgen_log.json",
            "embeddings.json",
            "embeddings_zi.json",
        ]
        for sanitize in self.sanitizers:
            for name in reserved:
                with self.assertRaises(ValueError, msg=f"Failed to reject reserved file: {name}"):
                    sanitize(name)


class TestDbaiAndGenericExecution(unittest.TestCase):
    """Tests for DBAI.execute_sql_query and Nl2sqlBq.execute_query dry_run handling."""

    def test_dbai_execute_sql_query_handles_line_comments_with_newlines(self):
        for dbai_mod in (ui_dbai, root_dbai):
            agent = dbai_mod.DBAI.__new__(dbai_mod.DBAI)
            agent.proj_id = "test-proj"
            agent.dataset_id = "test_ds"
            agent.bq_client = MagicMock()
            mock_job = MagicMock()
            mock_job.result.return_value = [{"cnt": 5}]
            agent.bq_client.query.return_value = mock_job

            query_with_comment = "-- initial comment\\nSELECT COUNT(*) AS cnt\\nFROM table1"
            resp = agent.execute_sql_query(query_with_comment)
            self.assertIn("'cnt': 5", resp)
            executed_sql = agent.bq_client.query.call_args[0][0]
            self.assertEqual(executed_sql, "SELECT COUNT(*) AS cnt FROM table1")

    def test_dbai_execute_sql_query_blocks_destructive_sql(self):
        for dbai_mod in (ui_dbai, root_dbai):
            agent = dbai_mod.DBAI.__new__(dbai_mod.DBAI)
            agent.proj_id = "test-proj"
            agent.dataset_id = "test_ds"
            agent.bq_client = MagicMock()

            resp = agent.execute_sql_query("DROP TABLE table1")
            self.assertIn("Only read-only SELECT queries are allowed", resp)
            agent.bq_client.query.assert_not_called()

    def test_nl2sql_generic_dry_run_returns_false_tuple_on_invalid_sql(self):
        bq = nl2sql_generic.Nl2sqlBq.__new__(nl2sql_generic.Nl2sqlBq)
        valid, msg = bq.execute_query("DROP TABLE singers", dry_run=True)
        self.assertFalse(valid)
        self.assertEqual(msg, "Invalid query. Regenerate")

        with self.assertRaises(ValueError):
            bq.execute_query("DROP TABLE singers", dry_run=False)

    def test_dbai_load_metadata_sanitizes_dataset_id(self):
        for dbai_mod in (ui_dbai, root_dbai):
            agent = dbai_mod.DBAI.__new__(dbai_mod.DBAI)
            agent.dataset_id = "../../etc/passwd"
            agent.create_metadata_cache = MagicMock(return_value={"t": {}})
            with unittest.mock.patch(
                "os.path.exists", return_value=True
            ) as mock_exists, unittest.mock.patch(
                "builtins.open",
                unittest.mock.mock_open(read_data='{"table": {}}'),
            ):
                agent.load_metadata()
                checked_path = mock_exists.call_args[0][0]
                self.assertEqual(
                    checked_path, "./metadata_cache_______etc_passwd.json"
                )
                self.assertNotIn("..", checked_path)

    def test_markdown_fence_cleanup_preserves_sql_substring(self):
        fenced_query = (
            "```sql\nSELECT * FROM nl2sql_spider.singer "
            "WHERE skill = 'SQL'\n```"
        )
        cleaned = datasets_base.Database.validate_readonly_query(fenced_query)
        self.assertIn("nl2sql_spider.singer", cleaned)
        self.assertIn("'SQL'", cleaned)


if __name__ == "__main__":
    unittest.main()
