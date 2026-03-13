import json

from google import genai
from google.genai import types

from datetime import datetime

client = genai.Client(vertexai=True, project="proj-kous", location="us-central1")
MODEL = "gemini-2.5-flash"

generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=-1,
    ),
  )

def get_response(prompt: str) -> str:
    """
    Returns the LLM's response to the given prompt.
    """
    # print("-"*100, prompt)
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt)
        ]
        ),
    ]

    return client.models.generate_content(
        model = MODEL,
        contents = contents,
        config = generate_content_config,
    ).text.replace("json", "").replace("```", "")


class SQLEvaluator:
    """
    Evaluates a generated SQL query against a natural language query and DB schema
    using an LLM for probabilistic accuracy scoring.
    """
    def __init__(self, db_metadata: dict, weights: dict = None):
        self.db_metadata = db_metadata
        self.schema_string = self._get_schema_string()
        
        if weights is None:
            # Default weights prioritizing correct tables and columns
            self.weights = {
                'wT': 0.30,  # Table accuracy
                'wC': 0.25,  # Column accuracy
                'wJ': 0.20,  # Join accuracy
                'wF': 0.15,  # Filter accuracy
                'wO': 0.10,  # Operations accuracy
                'wP': 0.10   # Complexity penalty weight
            }
        else:
            self.weights = weights

    def _get_schema_string(self) -> str:
        """Formats the DB metadata into a string for the LLM prompt."""
        schema_parts = []
        for table_name, table_info in self.db_metadata.items():
            columns_str = ", ".join([f"{col['name']} ({col['type']})" for col in table_info['columns_info']['fields']])
            table_desc = table_info.get('table_description', 'No description available.')
            schema_parts.append(f"Table: {table_name}\nColumns: {columns_str}\nDescription: {table_desc}\n")
        return "\n".join(schema_parts)

    def _parse_sql(self, sql_query: str) -> dict:
        """Parses the SQL query to extract its components."""
        prompt = f"""
        You are an expert SQL parser. Your task is to analyze the following SQL query and extract its components into a structured JSON format.

        SQL Query:
        ---
        {sql_query}
        ---

        Please extract the following components from the above SQL query:
        - tables: A list of all tables mentioned in the FROM and JOIN clauses.
        - columns: A list of all columns selected or used in WHERE, GROUP BY, or ORDER BY clauses.
        - joins: A list of strings, where each string represents a full JOIN condition (e.g., "orders AS o ON u.id = o.user_id").
        - where_clause: A string containing the entire WHERE clause content (without the "WHERE" keyword).
        - group_by: A list of columns used in the GROUP BY clause.
        - order_by: A list of columns or expressions used in the ORDER BY clause.
        - limit: The integer value from the LIMIT clause, or null if not present.
        - aggregations: A list of aggregation functions used (e.g., "SUM(oi.sale_price)", "COUNT(*)").
        - is_subquery: A boolean indicating if the main query contains any subqueries.
        - subqueries_num: An integer count of the number of subqueries.

        Think step-by-step. Provide your response as a single JSON object with the keys listed above along with values.
        If a component is not present in the query, use an empty list.
        """
        
        response_str = get_response(prompt)

        try:
            return json.loads(response_str)
        except (json.JSONDecodeError, TypeError):
            # Return a default structure on failure to prevent crashes downstream
            return {
                "tables": [],
                "columns": [],
                "joins": [],
                "where_clause": None,
                "group_by": [],
                "order_by": [],
                "limit": None,
                "aggregations": [],
                "is_subquery": False,
                "subqueries_num": 0
            }

    def _evaluate_components_by_set_match(self, expected: list, actual: list) -> float:
        """Calculates a score based on the intersection and union of two lists (Jaccard Index)."""
        if not expected and not actual:
            return 1.0  # Correctly identified that nothing was needed.
        if not expected and actual:
            return 0.0 # Used components when none were expected.
        if expected and not actual:
            return 0.0 # Failed to use any of the expected components.

        expected_set = set(item.lower() for item in expected)
        actual_set = set(item.lower() for item in actual)

        intersection = len(expected_set.intersection(actual_set))
        union = len(expected_set.union(actual_set))
        return intersection / union if union > 0 else 0.0

    def _evaluate_tables(self, nl_query: str, parsed_sql: dict) -> float:
        """AT: Evaluates the accuracy of the tables used in the SQL."""
        if not parsed_sql['tables']:
            return 0.0

        prompt = f"""
        Given the database schema:
        ---
        {self.schema_string}
        ---
        And the user's question: "{nl_query}"
        
        The generated SQL uses the following tables: {parsed_sql['tables']}

        Please evaluate the relevance of the tables used.
        1. For each table, provide a relevance score from 0.0 to 1.0.
        2. Identify any crucial tables that are missing from the SQL.
        
        Provide your response as a JSON object with keys: "reasoning", "table_scores" (a dictionary of table names to scores), and "missing_tables" (a list of strings).
        """
        
        response_str = get_response(prompt)
        try:
            result = json.loads(response_str)
            if result.get("missing_tables"):
                return 0.0  # Heavy penalty for missing a crucial table
            
            scores = result.get("table_scores", {}).values()
            if not scores:
                return 0.0
            return sum(scores) / len(scores)
        except (json.JSONDecodeError, AttributeError):
            return 0.0 # Failed to parse LLM response

    def _evaluate_columns(self, nl_query: str, parsed_sql: dict) -> float:
        """AC: Evaluates the accuracy of the columns used."""
        if not parsed_sql['columns']:
            return 0.0

        prompt = f"""
        Given the database schema:
        ---
        {self.schema_string}
        ---
        Given the user's question: "{nl_query}"
        And the generated SQL which uses tables: {parsed_sql['tables']}
        The SQL references these columns: {parsed_sql['columns']}

        Please evaluate if the correct columns were used to answer the question.
        1. For each column, provide a relevance score from 0.0 to 1.0.
        2. Identify any crucial columns that are missing.

        Provide your response as a JSON object with keys: "reasoning", "column_scores" (a dictionary of column names to scores), and "missing_columns" (a list of strings).
        """
        response_str = get_response(prompt)
        try:
            result = json.loads(response_str)
            if result.get("missing_columns"):
                return 0.0 # Heavy penalty for missing a crucial column
                
            scores = result.get("column_scores", {}).values()
            if not scores:
                return 0.0
            return sum(scores) / len(scores)
        except (json.JSONDecodeError, AttributeError):
            return 0.0

    def _evaluate_joins(self, nl_query: str, parsed_sql: dict) -> float:
        """AJ: Evaluates the correctness of joins."""
        if len(parsed_sql['tables']) <= 1:
            return 1.0 # No joins needed or performed, so no error.
        
        prompt = f"""
        Given the database schema:
        ---
        {self.schema_string}
        ---
        And the user's question "{nl_query}", and the following JOIN conditions from the SQL:
        {parsed_sql['joins']}

        Please evaluate the correctness of the JOIN conditions. Are the tables joined on the correct foreign keys to logically link the data for this query?

        Provide your response as a JSON object with keys: "reasoning" and "score" (a float from 0.0 to 1.0).
        """
        response_str = get_response(prompt)
        try:
            return float(json.loads(response_str).get("score", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def _evaluate_filters(self, nl_query: str, parsed_sql: dict) -> float:
        """AF: Evaluates the correctness of the WHERE clause."""
        prompt = f"""
        Given the database schema:
        ---
        {self.schema_string}
        ---
        Given the user's question: "{nl_query}"
        The generated SQL contains the following WHERE clause:
        "{parsed_sql['where_clause']}"

        Does this WHERE clause accurately reflect the filtering conditions mentioned or implied in the user's question? Consider the columns, operators (=, >, <, LIKE), and values. If no filter was needed and none is present, the score should be 1.0.

        Provide your response as a JSON object with keys: "reasoning" and "score" (a float from 0.0 to 1.0).
        """
        response_str = get_response(prompt)
        try:
            return float(json.loads(response_str).get("score", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def _evaluate_operations(self, nl_query: str, parsed_sql: dict) -> float:
        """AO: Evaluates aggregations, grouping, ordering, and limiting."""
        operations = {
            "aggregations": parsed_sql['aggregations'],
            "group_by": parsed_sql['group_by'],
            "order_by": parsed_sql['order_by'],
            "limit": parsed_sql['limit']
        }
        
        prompt = f"""
        Given the database schema:
        ---
        {self.schema_string}
        ---
        Given the user's question: "{nl_query}"
        The generated SQL performs the following operations:
        {json.dumps(operations, indent=2)}

        Do these operations (aggregations, grouping, ordering, limit) correctly match the user's intent? For example, 'total' implies SUM, 'how many' implies COUNT, 'top 5' implies ORDER BY and LIMIT 5.

        Provide your response as a JSON object with keys: "reasoning" and "score" (a float from 0.0 to 1.0).
        """
        response_str = get_response(prompt)
        try:
            return float(json.loads(response_str).get("score", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def _calculate_complexity_penalty(self, sql_query: str, parsed_sql: dict) -> float:
        """Calculates a penalty based on query complexity."""
        num_subqueries = parsed_sql['subqueries_num']
        num_joins = sql_query.upper().count("JOIN")
        query_length = len(sql_query)
        
        penalty = (num_subqueries * 0.1) + (num_joins * 0.05) + (query_length * 0.001)
        return penalty

    def evaluate(self, nl_query: str, sql_query: str) -> dict:
        """
        Performs the full evaluation of the SQL query.
        """
        parsed_sql = self._parse_sql(sql_query)
        # print(f"Parsed SQL Components:\n{json.dumps(parsed_sql, indent=2)}\n")

        # Calculate individual accuracy scores
        accuracies = {
            'AT': self._evaluate_tables(nl_query, parsed_sql),
            'AC': self._evaluate_columns(nl_query, parsed_sql),
            'AJ': self._evaluate_joins(nl_query, parsed_sql),
            'AF': self._evaluate_filters(nl_query, parsed_sql),
            'AO': self._evaluate_operations(nl_query, parsed_sql),
        }
        
        # Calculate final score using the formula
        weighted_sum = sum(self.weights[f'w{k[1:]}'] * v for k, v in accuracies.items() if f'w{k[1:]}' in self.weights)
        total_weight = sum(self.weights[f'w{k[1:]}'] for k in accuracies if f'w{k[1:]}' in self.weights)

        sql_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate penalty
        # penalty = self._calculate_complexity_penalty(sql_query, parsed_sql)
        # Optional: Re-introduce penalty if desired
        # sql_score -= self.weights['wP'] * penalty
        # sql_score = max(0, min(1, sql_score)) # Clamp score between 0 and 1

        # Calculate confidence score
        confidence_scores = [s for s in accuracies.values() if s is not None]
        score_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Determine need for human review
        needs_human_review = score_confidence < 0.8 or sql_score < 0.75
        
        return {
            "sql_score": sql_score,
            "score_confidence": score_confidence,
            "needs_human_review": needs_human_review,
            "breakdown": {
                "table_accuracy (AT)": accuracies['AT'],
                "column_accuracy (AC)": accuracies['AC'],
                "join_accuracy (AJ)": accuracies['AJ'],
                "filter_accuracy (AF)": accuracies['AF'],
                "operations_accuracy (AO)": accuracies['AO'],
                # "complexity_penalty (P)": penalty
            },
            "parsed_sql": parsed_sql
        }


# --- Main Execution ---
if __name__ == "__main__":
    # Generate a timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"sql_evaluation_report_{timestamp}.md"

    # Load test data from JSON file
    with open('sql_eval_test_data.json', 'r') as f:
        test_data = json.load(f)

    db_metadata = test_data['db_metadata']

    # Instantiate the evaluator
    evaluator = SQLEvaluator(db_metadata)

    # Open the report file for writing
    with open(report_filename, 'w') as report_file:
        report_file.write("# SQL Fitness Evaluation Report\n")
        report_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # --- Test Cases ---
        for i, test_group in enumerate(test_data['test_cases']):
            nl_query = test_group['nl_query']
            
            # Write to console
            print(f"\n\n{'='*20} Evaluating NL Query #{i+1}: '{nl_query}' {'='*20}")
            # Write to file
            report_file.write(f"## Evaluation for NL Query #{i+1}: `{nl_query}`\n\n")

            for test in test_group['queries']:
                print(f"\n----- Evaluating Test Case: {test['name']} -----")
                print(f"Generated SQL:\n{test['sql']}\n")

                # Get the evaluation result
                result = evaluator.evaluate(nl_query, test['sql'])

                # Write to file
                report_file.write(f"### Test Case: {test['name']}\n\n")
                report_file.write(f"**Generated SQL:**\n```sql\n{test['sql']}\n```\n\n")
                report_file.write("**Evaluation Report:**\n")
                report_file.write(f"- **Final SQL Score:** {result['sql_score']:.3f}\n")
                report_file.write(f"- **Confidence Score:** {result['score_confidence']:.3f}\n")
                report_file.write(f"- **Needs Human Review:** {'Yes' if result['needs_human_review'] else 'No'}\n\n")
                report_file.write("**Score Breakdown:**\n")
                for key, value in result['breakdown'].items():
                    report_file.write(f"- `{key}`: {value:.3f}\n")
                report_file.write("\n---\n\n")

                # Print a pretty report to console
                print("--- EVALUATION REPORT ---")
                print(f"Final SQL Score: {result['sql_score']:.3f}")
                print(f"Confidence Score: {result['score_confidence']:.3f}")
                print(f"Needs Human Review: {'Yes' if result['needs_human_review'] else 'No'}")
                print("\n--- Score Breakdown ---")
                for key, value in result['breakdown'].items():
                    print(f"{key:<25}: {value:.3f}")
                print("-------------------------")

    print(f"\n\nEvaluation complete. Report saved to '{report_filename}'")