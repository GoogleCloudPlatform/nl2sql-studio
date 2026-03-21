# NL2SQL Evaluation 
NL2SQL multi-agent solution with automated SQL Eval score


# spider eval
download spider data to spider_data folder and run
click to download [spider_data](https://yale-lily.github.io/spider)

```bash
python spider_eval/gen_ai_sql_json.py --file_path spider_data/dev.json --model gemini-2.5-flash
```
This will create a new file spider_data/dev_with_ai_sql.json with AI generated SQL queries.

Then run the evaluation script
```bash
python spider_eval/run_eval.py