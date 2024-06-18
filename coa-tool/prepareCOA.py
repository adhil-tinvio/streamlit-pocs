import streamlit as st
import pandas as pd
import requests
import concurrent.futures
from collections import defaultdict

API_KEY = st.secrets["API_KEY"]

prompt = f"""
You are a skilled accountant. 
Determine the closest matching account type for each account below. 
If there is no matching type, name it 'Not an Account type'. 

The possible types are:
1. Asset - Bank Accounts
2. Asset - Cash
3. Asset - Current Asset
4. Asset - Fixed Asset
5. Asset - Inventory
6. Asset - Non-current Asset
7. Equity - Shareholders Equity
8. Expense - Direct Costs
9. Expense - Operating Expense
10. Expense - Other Expense
11. Liability - Current Liability
12. Liability - Non-current Liability
13. Revenue - Operating Revenue
14. Revenue - Other Revenue

For each bank account, return only mapped account type.
An example of the format is within the three backticks:
```
Expense - Operating Expense
Expense - Other Expense
Expense - Direct Costs
# Add more if there are more than 3 bank accounts given. 
```
Do not add additional words. If there are 15 bank accounts given, you must return exactly 15 mapped account types. 
"""

"""
sga_chart_of_accounts = [
    "Accounts Payable", "Accounts Receivable",
    "Business Bank Account", "Cash on Hand", "Cost of Goods Sold", "Creditable Withholding Tax",
    "Deferred Input VAT Receivable", "Deferred Output VAT Payable", "Depreciation Expense",
    "FX Bank Revaluation (Gains)/Loss", "FX Realized Currency (Gains)/Loss",
    "FX Rounding (Gains)/Loss", "FX Unrealized Currency (Gains)/Loss", "Income Tax Expense",
    "Input VAT Receivable", "Inventory", "Loans Payable", "Output VAT Payable",
    "Property, Plant, Equipment", "Rent Expense", "Repair & Maintenance Expense",
    "Retained Earnings", "Salary & Payroll Expense", "Sales Revenue",
    "Selling, General & Administrative Expense", "Shipping Expense", "Shipping Revenue",
    "Transaction Fees & Charges", "Utility Expense", "Withholding Tax Payable"
]
"""


def sga_prompt_generator(chart_of_accounts):
    sga_prompt = """
    You are a skilled financial analyst.
    Match the given account names with the most suitable account from the Chart of Accounts given to you.
    Below is the list of available Chart of Accounts:
    """ + "\n- " + "\n- ".join(chart_of_accounts) + "\n"

    sga_prompt += """
    1) If there is no close match, name it 'No Suitable Match'.
    2) If there are 15 bank accounts given in a batch, you must return exactly 15 mapped account types (meaning 15 values returned, even if all 15 are no suitable match), Including No Suitable Match. This is so that the format will not get messed up. 
    3) Please do not give them index numbers at all.
    4) Make sure the return list length is exactly the same as the input size length (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
    5) Please do not have empty lines in your return. The results should all be in the next line IMPORTANT
    """

    return sga_prompt


def classify_account_types(account_names, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(account_names)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(account_names))
        batch = account_names[start_index:end_index]
        messages = [{
            'role': 'system',
            'content': prompt
        }]
        for name in batch:
            messages.append({
                'role': 'user',
                'content': f"Account name: {name}"
            })

        data = {
            "model": "gpt-4-turbo",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1000
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            content = content.strip("```").strip()
            batch_results = [line.strip() for line in content.split('\n')]

            if len(batch_results) != len(batch):
                print("batch: ", batch)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(batch)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(batch)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(account_names), batch_size)
        executor.map(process_batch, indices)

    return results


def recommend_sga_match(coa_names, account_names, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(account_names)
    sga_prompt = sga_prompt_generator(coa_names)
    st.write("COa nems",coa_names)
    st.write("SGA prompt",sga_prompt)
    st.write("account names",account_names)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(account_names))
        batch = account_names[start_index:end_index]
        messages = [{'role': 'system', 'content': sga_prompt}]
        for name in batch:
            messages.append({'role': 'user', 'content': f"Account Name: {name}"})

        data = {
            "model": "gpt-4-turbo",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1000
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            content = content.strip("```").strip()
            batch_results = [line.strip() for line in content.split('\n') if line.strip()]

            if len(batch_results) != len(batch):
                print("batch: ", batch)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(batch)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(batch)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(account_names), batch_size)
        executor.map(process_batch, indices)

    return results


def match_coa_using_gpt(external_coa, jaz_coa, chart_of_accounts_map):
    st.write(external_coa, "COA")
    st.write(external_coa.columns, "COLS")
    external_coa_account_names = external_coa['*Name'].tolist()
    st.write(external_coa_account_names, "ACNAMES")
    st.write("jaz template", jaz_coa)
    jazz_an = []
    st.write("jaz_ans",jaz_coa['Name*'].tolist())
    for name in jaz_coa['Name*'].tolist():
        if chart_of_accounts_map[name]['Match']:
            continue
        else:
            jazz_an.append(name)
    st.write("jax an ",jazz_an)
    combined_accounts = [f"{name} " for name in external_coa_account_names if name not in chart_of_accounts_map]
    jaz_account_names = [f"{name} " for name in jazz_an]
    st.write("JAN", jaz_account_names)
    sga_matches = recommend_sga_match(jaz_account_names, combined_accounts)
    st.write("SGA_matches",sga_matches)
    ###############
    external_coa_data['SGA Match Recommendation'] = sga_matches

    external_coa_data['Status'] = 'Active'
    external_coa_data['Unique ID'] = ''

    final_data = external_coa_data[
        ['*Type', '*Name', '*Code', 'Status', 'Unique ID', 'SGA Match Recommendation']]
    return final_data


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


st.title('Jaz COA Import Mapping Tool (SG-EN)')
external_coa_file = st.file_uploader("Choose the external coa file", type=["csv"])
jaz_coa_file = st.file_uploader("Choose the jaz coa import file", type=["xlsx"])
if external_coa_file is not None and jaz_coa_file is not None:
    #st.write("print_sga prompt", sga_prompt)
    external_coa_data = pd.read_csv(external_coa_file)
    jaz_coa_data = pd.read_excel(jaz_coa_file, sheet_name=1)
    coa_map = defaultdict(dict)
    for j in range(len(jaz_coa_data)):
        row = jaz_coa_data.iloc[j]
        account_name = row['Name*']
        account_type = row['Account Type*']
        code = row['Code']
        description = row['Description']
        lock_date = row['Lock Date']
        unique_id = row['Unique ID (do not edit)']
        coa_map[account_name] = {
            "Account Type*": account_type,
            "Name*": account_name,
            "Code": code,
            "Description": description,
            "Lock Date": lock_date,
            "Unique ID (do not edit)": unique_id,
            "Match": False
        }

    for i in range(len(external_coa_data)):
        row = external_coa_data.iloc[i]
        if row['jaz_sga_name'] == '' or pd.isnull(row['jaz_sga_name']):
            continue
        else:
            if row['jaz_sga_name'] in coa_map:
                coa_map[row['jaz_sga_name']]['Code'] = row['*Code']
                coa_map[row['jaz_sga_name']]['Description'] = row['Description']
                coa_map[row['jaz_sga_name']]['Match'] = True
                coa_map[row['jaz_sga_name']]['Status'] = 'ACTIVE'
    st.write("GOAT", coa_map)
    processed_data = match_coa_using_gpt(external_coa_data, jaz_coa_data, coa_map)
    st.write("Processed Data", processed_data)
    csv = convert_df_to_csv(processed_data)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='mapped_coa.csv',
        mime='text/csv',
    )
