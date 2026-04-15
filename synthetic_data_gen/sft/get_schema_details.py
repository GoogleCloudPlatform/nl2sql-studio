import os
import sqlite3

def get_schema_details(db_id):
    """
    Generates a detailed schema string for a given database ID.

    This function connects to the corresponding .sqlite database to extract
    table and column metadata. The resulting string contains a detailed
    breakdown of each table, including column names, data types, and
    constraints. This output is formatted to be easily parsable by a
    Large Language Model (LLM).

    Args:
        db_id (str): The ID of the database, which corresponds to the
                     subfolder name under the 'database' directory.

    Returns:
        str: A string containing the complete schema details, ready for use
             with an LLM. Returns an empty string if the database files
             are not found.
    """
    # Construct path relative to this script's location for robustness.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # db_path = os.path.join(os.path.dirname(script_dir), 'spider_data', 'database', db_id)
    db_path = os.path.join(os.path.dirname(script_dir), "database")
    sqlite_file = os.path.join(db_path, db_id + '.sqlite')

    if not os.path.exists(sqlite_file):
        print(f"Database file for '{db_id}' not found at {sqlite_file}.")
        return ""

    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()

    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_details = ""
    for table in tables:
        # Ignore sqlite_sequence table which is internal to sqlite
        if table[0] == 'sqlite_sequence':
            continue

        table_name = table[0]
        schema_details += f"Table: {table_name}\n"

        # Get column details
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for column in columns:
            col_name = column[1]
            col_type = column[2]
            col_notnull = "NOT NULL" if column[3] else ""
            col_pk = "PRIMARY KEY" if column[5] else ""
            schema_details += f"  {col_name} {col_type} {col_notnull} {col_pk}".strip() + "\n"
        schema_details += "\n"

    conn.close()

    return schema_details.strip()

if __name__ == '__main__':
    # Example usage:
    # Replace 'academic' with any database ID from the 'database' directory
    db_id = 'concert_singer'
    schema_info = get_schema_details(db_id)
    if schema_info:
        print(f"Schema details for database: {db_id}")
        print(schema_info)
    else:
        print(f"Database with ID '{db_id}' not found.")
