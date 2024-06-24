import json
import os
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextGenerationModel


def generate_metadata(project_id,
                      location,
                      dataset_name,
                      model_name="text-bison-32k",
                      model_parameters=None):

    vertexai.init(project=project_id, location=location)
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.9,
        "top_p": 1
        }

    model = TextGenerationModel.from_pretrained(model_name)
    client = bigquery.Client(project=project_id)

    data = {}

    for table_ref in client.list_tables(dataset_name):
        try:
            table = client.get_table(table_ref)
        except Exception as e:
            print(f"Error getting table {table_ref}: {e}")
            continue

        table_name = table.reference.table_id

        table_description = table.description if table.description else ""
        if not table_description:
            prompt = f"""Describe the data in table '{table_name}'.
                    Do not include any markdown symmbols or special characters
                    while generating the description"""
            response = model.predict(prompt=prompt)

            table_description = response.text.strip()
            table_description = table_description.replace('**', '')

        columns = {}
        schema = table.schema

        prompt = f"Describe the following columns in table '{table_name}'"
        for field in schema:
            prompt += f"- Column Name: {field.name}, Type: {field.field_type}"
        prompt += "\nPlease generate response in 'column name : Description'"
        prompt += " format. Do not include any markdown symmbols or special "
        prompt += "characters while generating the description"

        response = model.predict(prompt=prompt, **parameters)
        print(response)
        response.text = response.text.replace('**', '')
        column_descriptions = [
            line for line in response.text.strip().split("\n") if line.strip()
            ]

        for i, field in enumerate(schema):
            if i < len(column_descriptions):
                column_description = column_descriptions[i]
            else:
                column_description = ""  # Handle missing descriptions

            column_dict = {
                'Name': field.name,
                'Type': field.field_type,
                'Description': column_description,
                'Examples': f"Sample value for {field.name}"
            }
            columns[field.name] = column_dict

        data[table_name] = {
            "Name": table_name,
            "Description": table_description,
            "Columns": columns
        }

    return data


if __name__ == "__main__":
    PROJECT_ID = os.environ.get('PROJECT_ID')  # 'sl-test--project'
    DATASET = os.environ.get('DATASET_NAME')  # "sl-test--project.zoominfo"
    OUTPUTFILE = "./cache_metadata/metadata_cache.json"

    if PROJECT_ID is None or DATASET is None:
        print("Ensure you set the PROJECT_ID and DATASET_NAME env variables ")
    else:
        metadata = generate_metadata(
            project_id=PROJECT_ID,
            location="us-central1",
            dataset_name=DATASET
        )

        with open(OUTPUTFILE, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f'JSON file created successfully: {OUTPUTFILE}')
