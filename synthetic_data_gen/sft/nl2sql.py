from google import genai
from google.genai import types
import config

MODEL = "gemini-2.5-flash"

client = genai.Client(
    vertexai=config.VERTEXAI,
    project=config.PROJECT,
    location=config.LOCATION,
)

generate_content_config = types.GenerateContentConfig(
    temperature = 0.1,
    top_p = 0.95,
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


import time
import random

def generate(query: str, model: str = MODEL):
    contents = [
        types.Content(
            role="user",
            parts=[
            types.Part.from_text(text=query)
            ]
        ),
    ]

    max_retries = 10
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model = model,
                contents = contents,
                config = generate_content_config,
            ).text
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # For 429 rate limits, use a more aggressive backoff with jitter
                sleep_time = (2 ** attempt) * 5 + random.uniform(1, 5)
            else:
                sleep_time = 2 ** attempt
            
            time.sleep(sleep_time)

def get_ai_sql(schema_details: str, query: str, db_dialect: str = 'sqlite', model: str = MODEL) -> str:
    prompt = f"""Given the database schema details below and a natural language query, generate the corresponding SQL query.
    Make sure the SQL query is compatible with {db_dialect}.
    Double check all the table names are matching with schema and all the column names are mathcing for the corresponding table in the schema.
    Think step by step and ensure the SQL query is syntactically correct and executable.
    
Schema Details:
{schema_details}

Natural Language Query: {query}
SQL Query:"""
    generated_sql = generate(prompt, model)
    generated_sql = generated_sql.\
        replace('```', '').\
        replace(db_dialect, '').\
        replace("sql", '').\
        strip()
    return generated_sql


if __name__ == "__main__":
    print(get_ai_sql('''''', "List top 5 singers"))  # Example usage