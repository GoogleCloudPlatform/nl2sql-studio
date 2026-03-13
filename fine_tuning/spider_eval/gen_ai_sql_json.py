import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

from get_schema_details import get_schema_details
from nl2sql import get_ai_sql

def add_ai_sql_to_json(file_path: str, model: str = "gemini-2.5-flash"):
    """
    Reads a JSON file, adds AI-generated SQL to entries containing a 'query' key,
    and writes the updated data back to the file.

    Args:
        file_path: The path to the JSON file to be updated.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    # Determine output filename
    model_name = model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    out_filename = file_path[:-5] + f'_ai_{model_name}.json'

    output_file = None
    first_entry = True
    try:
        with open(file_path, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)

        # Ensure data is a list of dictionaries
        if not isinstance(data, list):
            print(f"Error: Expected a list of objects in '{file_path}', but found {type(data)}.")
            return

        # Open output file and start JSON list
        output_file = open(out_filename, 'w')
        output_file.write('[\n')

        for item in tqdm(data):
            if isinstance(item, dict) and 'db_id' in item and 'question' in item:
                db_id = item['db_id']
                question = item['question']

                # Get schema details for the database
                schema = get_schema_details(db_id)

                # Get the AI-generated SQL
                ai_sql = get_ai_sql(schema, question, model=model)

                # Prepare the output entry
                keys_to_keep = ["db_id", "question", "query"]
                output_entry = {k: item[k] for k in keys_to_keep if k in item}
                output_entry['ai_generated_sql'] = ai_sql

                # Write entry to file
                if not first_entry:
                    output_file.write(',\n')
                json.dump(output_entry, output_file, indent=4)
                first_entry = False

        print(f"Successfully created '{out_filename}' with AI-generated SQL.")
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}' with AI-generated SQL.")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Please check its format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if output_file and not output_file.closed:
            output_file.write('\n]')
            output_file.close()



if __name__ == "__main__":
    # The name of the JSON file to be processed.
    # This script assumes 'dev.json' is in the same directory.
    spider_path = 'spider_data/'
    json_file = spider_path + 'dev_filtered_balanced.json'
    # MODEL = "gemini-2.5-pro"
    # MODEL = "gemini-2.5-flash"
    MODEL = "projects/862253555914/locations/us-central1/endpoints/1267968903779188736"

    add_ai_sql_to_json(json_file, model=MODEL)
