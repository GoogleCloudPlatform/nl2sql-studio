import unittest
from unittest.mock import patch

from metadata_update_bq import generate_metadata, TextGenerationModel


class TestGenerateMetadata(unittest.TestCase):

    @patch.object(TextGenerationModel, 'predict')  # Patch LLM predictions
    def test_metadata_generation(self, mock_predict):
        # Define expected LLM predictions
        expected_table_description = """The `question_answers` table contains
            information about questions and their answers. Each row in the
            table represents a single question and its answer. The table has
            the following columns:\n\n* `id`: The unique identifier for the
            question and answer.\n* `question`: The text of the question.\n*
            `answer`: The text of the answer.\n* `question_type`: The type of
            question. This can be one of the following values:\n    *
            `multiple_choice`\n    * `true_false`\n    * `open_ended`\n*
            `difficulty`: The difficulty level of the question."""

        expected_column_descriptions = [
            """**unique_id** (STRING)\nThis column represents a unique
            identifier for each row in the 'question_answers' table. It
            is likely a combination of characters and numbers generated
            by the database or application to ensure uniqueness.""",
            """**result_data** (STRING)\nThis column stores the actual text
            or data that represents the answer(s) to the question. It can be
            plain text, structured data, or any other relevant information
            that provides the response to the question.""",
            """**result_sql** (STRING)\nThe 'result_sql' column contains the
            SQL query or statement that was executed to retrieve the answer(s)
            from the database. This column is useful for understanding the
            source of the answer and the logic behind retrieving it. It
            allows for traceability and reproducibility of the results.""",
            """**question** (STRING)\nThis column holds the actual question or
            prompt that was posed by the user or system. It represents the
            input that led to the retrieval of the answers.""",
            """**created_date** (DATETIME)\nThe 'created_date' column stores
            the date and time when the row was created in the
            'question_answers' table. It indicates when the question was asked
            and the answer was retrieved.""",
            """**status** (BOOLEAN)\nThe 'status' column likely represents a
            flag that indicates whether the answer(s) are valid, accurate, or
            completed. It can be used to track the state of the answer
            retrieval process. True may indicate a successful retrieval,
            while False could indicate an error or incompleteness.""",
        ]
        print("expected_column_descriptions-", expected_column_descriptions)
        metadata = generate_metadata(
            project_id="sl-test-project-363109",  # Replace with your proj ID
            location="us-central1",    # Replace with your Vertex AI location
            dataset_name="sl-test-project-363109.qnadb"
        )

        print(f"Generated Metadata:\n{metadata}")
        # Assert the generated metadata structure
        self.assertIn("Dataset Name", metadata)
        self.assertIn("Tables", metadata)

        # Assert table information - table name
        table_info = metadata["Tables"]["question_answers"]
        self.assertEqual(table_info["Table Name"],
                         "question_answers")

        # Assert table information - description
        self.assertEqual(table_info["Description"],
                         expected_table_description)

        # Assert some column information (assuming the first column)
        column_info = table_info["Columns"]["unique_id"]
        self.assertIn("Name", column_info)
        self.assertIn("Type", column_info)
        self.assertIn("Description", column_info)
        self.assertEqual(column_info["Description"],
                         expected_table_description.split('\n\n')[0])
