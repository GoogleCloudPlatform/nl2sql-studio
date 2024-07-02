# import unittest
# from unittest.mock import patch
import csv
import langchain
from nl2sql import table_filter

llm = langchain.llms.VertexAI(temperature=0,
                              model_name="text-bison@latest",
                              max_output_tokens=2048)


def run_tests_generate_csv():
    test_cases = [
        {
            'question': "What were my sales for yesterday?",
            'expected_table': ['authorizations_search']
        },
        {
            'question': "What is my approval rate for last 2 weeks?",
            'expected_table': ['authorizations_search']
        },
        {
            'question': "How many transactions went to Visa and Mastercard\
                for the last month?",
            'expected_table': ['authorizations_search']
        },
        {
            'question': "List all online declined discover transactions\
                for today.",
            'expected_table': ['authorizations_search']
        },
        {
            'question': "What settled past week?",
            'expected_table': ['settlement_search']
        },
        {
            'question': "What was the total sale amount for transactions\
                settled via debit networks?",
            'expected_table': ['settlement_search']
        },
        {
            'question': "Show me settlement summary by plan code for last\
                week for swiped transactions?",
            'expected_table': ['settlement_search']
        },
        {
            'question': "What are my total deposits broken down by deposit\
                type last week?",
            'expected_table': ['funding_search']
        },
        {
            'question': "Which day within the last month was my highest\
                funding?",
            'expected_table': ['funding_search']
        },
        {
            'question': "What is the Fees to Deposit ratio for\
                yesterdays bank deposit?",
            'expected_table': ['funding_search']
        },
        {
            'question': "What is the Win to Loss ratio on disputes\
                last month?",
            'expected_table': ['chargebacks_search']
        },
        {
            'question': "How many disputes over $100 do I have?",
            'expected_table': ['chargebacks_search']
        },
        {
            'question': "What are my top 5 stores from\
                where I am getting the most disputes?",
            'expected_table': ['chargebacks_search']
        },
        {
            'question': "What is my Sales to Dispute ratio for last month?",
            'expected_table': ['authorizations_search', 'chargebacks_search']
        }
    ]

    output_file = 'table_filter_test_results.csv'
    fieldnames = ['Question', 'Expected_Table', 'LLM_Selected_Table', 'Result']
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for test_case in test_cases:
            question = test_case['question']
            expected_table = test_case['expected_table']

            llm_selected_table = table_filter(question)

            if expected_table == llm_selected_table:
                result = 'Pass'
            else:
                result = 'Fail'

            print(result)

            writer.writerow({
                'Question': question,
                'Expected_Table': ', '.join(expected_table),
                'LLM_Selected_Table': ', '.join(llm_selected_table),
                'Result': result
            })

    print(f"Test results written to {output_file}")


if __name__ == '__main__':
    # unittest.main()
    run_tests_generate_csv()
