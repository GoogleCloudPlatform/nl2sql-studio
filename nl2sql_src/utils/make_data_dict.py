import json
import pandas as pd

base_dir = '/Users/koushikchak/Downloads/fiserv/fiservsamplequeries/'
data_dict_file = 'Table List Documentation (1).xlsx'

table_file = 'Sample Report Data (1)/Funding_search.xlsx'
sheet_name = 'Funding'
table_name = 'Funding_search'


df_table = pd.read_excel(base_dir+table_file)
df_data_dict = pd.read_excel(base_dir+data_dict_file, sheet_name=sheet_name)

data_dict = {i: (j, k) for i, j, k in zip(df_data_dict.Name,
                                          df_data_dict.ID,
                                          df_data_dict.Description)}

final_dict = {}
for col in df_table.columns:
    if col in data_dict:
        key = table_name + '.' + data_dict[col][0]
        value = data_dict[col][1]
        final_dict[key] = value
    else:
        print(col)


with open(base_dir+table_name+'.json', 'w') as f:
    json.dump(final_dict, f, indent=4)
