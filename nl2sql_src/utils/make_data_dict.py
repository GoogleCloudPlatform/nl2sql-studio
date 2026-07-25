import argparse
import json
import os
import pandas as pd
from typing import Dict, Any, Optional


def generate_data_dictionary(
    data_dict_file: str,
    table_file: Optional[str] = None,
    sheet_name: Any = 0,
    table_name: Optional[str] = None,
    output_file: Optional[str] = None,
    base_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Generates a JSON data dictionary mapping table.column names to descriptions from Excel documentation files.
    """
    full_data_dict_path = os.path.join(base_dir, data_dict_file) if base_dir else data_dict_file
    if not os.path.exists(full_data_dict_path):
        raise FileNotFoundError(f"Data dictionary file not found: {full_data_dict_path}")

    # Read Excel data dictionary sheet
    df_data_dict = pd.read_excel(full_data_dict_path, sheet_name=sheet_name)

    # Standardize column extraction
    if "Name" in df_data_dict.columns and "ID" in df_data_dict.columns and "Description" in df_data_dict.columns:
        data_dict = {i: (j, k) for i, j, k in zip(df_data_dict["Name"], df_data_dict["ID"], df_data_dict["Description"])}
    else:
        # Fallback for alternative sheet layouts: map first column to description
        cols = list(df_data_dict.columns)
        if len(cols) >= 2:
            data_dict = {row[cols[0]]: (str(row[cols[0]]), str(row[cols[1]])) for _, row in df_data_dict.iterrows()}
        else:
            data_dict = {}

    final_dict = {}

    if table_file:
        full_table_path = os.path.join(base_dir, table_file) if base_dir else table_file
        if not os.path.exists(full_table_path):
            raise FileNotFoundError(f"Table file not found: {full_table_path}")
        df_table = pd.read_excel(full_table_path)
        prefix = f"{table_name}." if table_name else ""
        for col in df_table.columns:
            if col in data_dict:
                key = f"{prefix}{data_dict[col][0]}"
                value = str(data_dict[col][1])
                final_dict[key] = value
            else:
                print(f"Unmapped column: {col}")
    else:
        # If no table_file is provided, map all rows in data_dict
        prefix = f"{table_name}." if table_name else ""
        for col_name, (col_id, col_desc) in data_dict.items():
            key = f"{prefix}{col_id}" if prefix else str(col_id)
            final_dict[key] = str(col_desc)

    if output_file:
        full_output_path = os.path.join(base_dir, output_file) if base_dir else output_file
        os.makedirs(os.path.dirname(os.path.abspath(full_output_path)), exist_ok=True)
        with open(full_output_path, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, indent=4)
        print(f"Data dictionary saved to {full_output_path}")

    return final_dict


# Alias for compatibility
make_data_dict = generate_data_dictionary


def main():
    parser = argparse.ArgumentParser(description="Generate JSON Data Dictionary from Excel Documentation")
    parser.add_argument("--data-dict-file", type=str, required=True, help="Path to Table List Documentation Excel file")
    parser.add_argument("--table-file", type=str, default=None, help="Path to Sample Report Data Excel file")
    parser.add_argument("--sheet-name", type=str, default="0", help="Excel sheet name or index for data dictionary")
    parser.add_argument("--table-name", type=str, default=None, help="Target table name prefix")
    parser.add_argument("--output-file", type=str, default=None, help="Path to save output JSON dictionary")
    parser.add_argument("--base-dir", type=str, default=None, help="Optional base directory prefix for file paths")

    args = parser.parse_args()

    # Convert sheet_name to int if numeric index
    sheet = int(args.sheet_name) if args.sheet_name.isdigit() else args.sheet_name

    generate_data_dictionary(
        data_dict_file=args.data_dict_file,
        table_file=args.table_file,
        sheet_name=sheet,
        table_name=args.table_name,
        output_file=args.output_file,
        base_dir=args.base_dir
    )


if __name__ == "__main__":
    main()
