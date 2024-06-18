import streamlit as st
import pandas as pd
import requests
import concurrent.futures

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

sga_prompt = """
You are a skilled financial analyst.
Match the given trial balance account with the most suitable account from the SG Chart of Accounts.
Below is the list of available SG Chart of Accounts:
""" + "\n- " + "\n- ".join(sga_chart_of_accounts) + "\n"

sga_prompt += """
1) If there is no close match, name it 'No Suitable Match'.
2) If there are 15 bank accounts given in a batch, you must return exactly 15 mapped account types (meaning 15 values returned, even if all 15 are no suitable match), Including No Suitable Match. This is so that the format will not get messed up. 
3) Please do not give them index numbers at all.
4) Make sure the return list length is exactly the same as the input size length (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
5) Please do not have empty lines in your return. The results should all be in the next line IMPORTANT
"""


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


def recommend_sga_match(account_names, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(account_names)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(account_names))
        batch = account_names[start_index:end_index]
        messages = [{'role': 'system', 'content': sga_prompt}]
        for name in batch:
            messages.append({'role': 'user', 'content': f"TB Account: {name}"})

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


def process_trial_balance(file):
    external_coa_data = pd.read_csv(file)
    st.write(external_coa_data,"COA")
    st.write(external_coa_data.columns,"COLS")
    account_names = external_coa_data['*Name'].tolist()
    st.write(account_names,"ACNAMES")

    #account_types = classify_account_types(account_names)
    #trial_balance_cleaned['Account Type'] = account_types
    #combined_accounts = [f"{name} - {type}" for name, type in zip(account_names, account_types)]  ############
    combined_accounts = [f"{name} " for name in account_names]

    sga_matches = recommend_sga_match(combined_accounts)  ###############
    external_coa_data['SGA Match Recommendation'] = sga_matches

    external_coa_data['Status'] = 'Active'
    external_coa_data['Unique ID'] = ''

    final_data = external_coa_data[
        ['*Type', '*Name', '*Code', 'Status', 'Unique ID', 'SGA Match Recommendation']]
    return final_data


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


st.title('Jaz COA Import Mapping Tool (SG-EN)')

external_file = st.file_uploader("Choose the external coa file", type=["csv"])
jaz_coa_file = st.file_uploader("Choose the jaz coa import file", type=["xlsx"])
if external_file is not None and jaz_coa_file is not None:
    st.write("print_sga prompt",sga_prompt)
    processed_data = process_trial_balance(external_file)
    st.write("Processed Data", processed_data)
    csv = convert_df_to_csv(processed_data)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='mapped_coa.csv',
        mime='text/csv',
    )
