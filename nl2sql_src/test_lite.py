"""
    NL2SQL Executors Test case file
"""
import json
import requests
from dotenv import load_dotenv

load_dotenv()

URL = "http://127.0.0.1:8000/"
# URL = "https://nl2sql-lib-executors\
# -p2r-dot-sl-test-project-363109.uc.r.appspot.com/"


def linear_exec(question, execute_sql):
    """
    Insert record test case
    """
    end_point = URL + "api/executor/linear"
    data = {"question": question, "execute_sql": execute_sql}
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    resp = requests.post(
        url=end_point, data=json.dumps(data), headers=headers, timeout=None
    )
    # r = resp.json()
    print("Linear executor API response =", resp.json())


def cot_exec(question, execute_sql):
    """
    Insert record test case
    """
    end_point = URL + "api/executor/cot"
    data = {"question": question, "execute_sql": execute_sql}
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    resp = requests.post(
        url=end_point, data=json.dumps(data), headers=headers, timeout=None
    )
    # r = resp.json()
    print("Cot executor API response =", resp.json())


def rag_exec(question, execute_sql):
    """
    Insert record test case
    """
    end_point = URL + "api/executor/rag"
    data = {"question": question, "execute_sql": execute_sql}
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    resp = requests.post(
        url=end_point, data=json.dumps(data), headers=headers, timeout=None
    )
    # r = resp.json()
    print("Rag executor API response =", resp.json())


def proj_config(proj_name, dataset_name, metadata_filename):
    """
    Insert record test case
    """
    end_point = URL + "/projconfig"
    data = {
        "proj_name": proj_name,
        "bq_dataset": dataset_name,
        "metadata_file": metadata_filename,
    }
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    resp = requests.post(
        url=end_point, data=json.dumps(data), headers=headers, timeout=None
    )
    # r = resp.json()
    print("Proj config =", resp.json())


def userfb(result_id, feedback):
    """
    Insert record test case
    """
    end_point = URL + "/userfb"
    data = {"result_id": result_id, "user_feedback": feedback}
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    resp = requests.post(
        url=end_point, data=json.dumps(data), headers=headers, timeout=None
    )
    # r = resp.json()
    print("Update FB =", resp.json())


def getapi():
    """
    Sample test
    """
    resp = requests.get(URL, timeout=None)
    r = resp.json()
    print(resp, r)


def lite_exec(question, execute_sql, few_shot_gen):
    end_point = URL + "api/lite/generate"
    data = {"question": question,
            "execute_sql": execute_sql,
            "few_shot": few_shot_gen}
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    resp = requests.post(
        url=end_point, data=json.dumps(data), headers=headers, timeout=None
    )
    # r = resp.json()
    print("Lite API response =", resp.json())


def add_question(question, sql):
    end_point = URL + "api/record/create"
    data = {"question": question,
            "sql": sql,
            }
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    resp = requests.post(
        url=end_point, data=json.dumps(data), headers=headers, timeout=None
    )
    # r = resp.json()
    print("Lite API response =", resp.json())


if __name__ == "__main__":
    # QUESTION = "What are the top 5 industries in terms of revenue?"
    # QUESTION = "What is the average, minimum,
    #             and maximum age of all singers from France?"
    QUESTION = "Find the number of concerts happened \
          in the stadium with the highest capacity"
    print("*" * 20, " Testing NL2SQL Executors APIs ", "*" * 20)
    print("*" * 20, " Default API ", "*" * 20)
    getapi()

    print("\n", "*" * 20, " Generating SQLs only ", "*" * 20)
    print("\n", "*" * 20, " NL@SQL Lite  API call ", "*" * 20)
    lite_exec(QUESTION, False, False)

    print("\n", "*" * 20, " Generating and Executing SQL ", "*" * 20)

    print("\n", "*" * 20, " Lite Genrator API call ", "*" * 20)
    lite_exec(QUESTION, True, False)

    print("\n", "*" * 20, " Generating SQLs with FewShot only ", "*" * 20)
    print("\n", "*" * 20, " NL@SQL Lite  API call ", "*" * 20)
    lite_exec(QUESTION, False, True)

    print("\n", "*" * 20, " Generating and Executing SQL Few shot ", "*" * 20)

    print("\n", "*" * 20, " Lite Genrator API call ", "*" * 20)
    lite_exec(QUESTION, True, True)

    # data = [["What is the average, minimum,
    #           and maximum age of all singers from Netherlands ?",
    #          "select avg(age) , min(age) , max(age)
    #           from singer where country = 'Netherlands'"],
    #         ["List the countries and the number of singers
    #           from each country",
    #          "select country, count(*) from singer group by country"],
    #         ["List all song names by singers who are below the average age",
    #          "select song_name from singer where age <
    #           (select avg(age) from singer)"],
    #         ["Show the stadium name and the number of concerts
    #            in each stadium",
    #          "select t2.name, count(*) from concert as t1 join stadium
    #            as t2 on t1.stadium_id = t2.stadium_id group
    #            by t1.stadium_id"],
    #         ["Find the number of concerts happened in the stadium
    #            with the lowest capacity ",
    #          "select count(*) from concert where stadium_id =
    #            (select stadium_id from stadium order by capacity
    #              asc limit 1)"],
    #         ["How many stadiums have capacity more than 10000? ",
    #          "select count(*) from stadium where capacity > 10000"],
    #         [" Show countries where a singer above age 30
    #            and a singer below 50 are from",
    #          "select country from singer where age > 30
    #            intersect select country from singer where age < 50"],
    #        ]
    # for elem in data:
    #     add_question(question=elem[0], sql=elem[1])
