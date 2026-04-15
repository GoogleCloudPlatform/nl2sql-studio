"""
Database Schema Extractor
-------------------------
This script extracts table and column details from a SQLite database and
formats them into a readable string suitable for inclusion in LLM prompts.
"""

import os
import sqlite3

def get_schema_details(db_id, base_db_path):
    """
    Generates a detailed schema string for a given database ID.

    This function connects to the corresponding .sqlite database to extract
    table and column metadata. The resulting string contains a detailed
    breakdown of each table, including column names, data types, and
    constraints.

    Args:
        db_id (str): The ID of the database (usually matching the folder name).
        base_db_path (str): The base directory where the .sqlite file is located.

    Returns:
        str: A string containing the complete schema details, ready for use
             with an LLM. Returns an empty string if the database file is not found.
    """
    # Construct path to the SQLite file
    sqlite_file = os.path.join(base_db_path, db_id + '.sqlite')

    if not os.path.exists(sqlite_file):
        print(f"Database file for '{db_id}' not found at {sqlite_file}.")
        return ""

    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()

    # Query all table names from the SQLite master table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_details = ""
    for table in tables:
        # Ignore sqlite_sequence table which is internal to SQLite for autoincrement
        if table[0] == 'sqlite_sequence':
            continue

        table_name = table[0]
        schema_details += f"Table: {table_name}\n"

        # Get column details using PRAGMA table_info
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()
        
        for column in columns:
            # column tuple format: (cid, name, type, notnull, dflt_value, pk)
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
    db_id = 'concert_singer'
    # Assuming the database is in the relative path '../database'
    base_path = '../database'
    
    print(f"Extracting schema for database: {db_id}")
    # Fixed the example usage to pass both required arguments
    schema_info = get_schema_details(db_id, base_path)
    
    if schema_info:
        print(f"Schema details for database: {db_id}")
        print(schema_info)
    else:
        print(f"Database with ID '{db_id}' not found at '{base_path}'.")