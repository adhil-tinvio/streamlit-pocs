import streamlit as st
import pandas as pd
import requests
import concurrent.futures
from collections import defaultdict

API_KEY = st.secrets["API_KEY"]
ACTIVE_ONLY_ACCOUNTS = ['FX Realized Currency (Gains)/Losses',
                        'Input GST Receivable',
                        'Accounts Payable'
                        'FX Unrealized Currency (Gains)/Losses'
                        'FX Bank Revaluation (Gains)/Losses'
                        'Business Bank Account'
                        'FX Rounding (Gains)/Losses'
                        'Accounts Receivable'
                        'Retained Earnings'
                        'Output VAT Payable']

COLUMNS_WITH_CURRENCY = ['Account Type*',
                         'Currency*',
                         'Name*',
                         'Code',
                         'Description',
                         'Lock Date',
                         'Status',
                         'Unique ID (do not edit)']

COLUMNS_WITHOUT_CURRENCY = ['Account Type*',
                            'Name*',
                            'Code',
                            'Description',
                            'Lock Date',
                            'Status',
                            'Unique ID (do not edit)']

account_type_prompt = """
You are a skilled accountant. Determine the closest matching Accepted Account Type for each Account Type or Report Code combination given as input below. 
If there is no matching type, return an empty value

The accepted account types are:
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

Only if the account type was not sufficient to determine the correct accepted account type, then use the report code to 
determine the accepted account type. Here is a help abbreviation glossary that maps to the accepted account types:
EXP = Expense - Operating Expense
EQU = Equity - Shareholders Equity

Criteria:
1) Return only Accepted Account Type from the list above as response. If there is no close match,return ''.
2) Please do not give them index numbers at all.
3) Make sure the return list length is exactly the same as the input size length (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
4) Please do not have empty lines in your return. The results should all be in the next line IMPORTANT5
"""


def classify_account_types(account_types,report_codes, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(account_types)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(account_types))
        account_type_batch = account_types[start_index:end_index]
        report_code_batch = report_codes[start_index:end_index]
        messages = [{
            'role': 'system',
            'content': account_type_prompt
        }]
        for i in range(len(account_type_batch)):
            messages.append({
                'role': 'user',
                'content': f"Account Type: {account_type_batch[i]}  Report Code: {report_code_batch}"
            })

        data = {
            "model": "gpt-4-turbo",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            content = content.strip("```").strip()
            batch_results = [line.strip() for line in content.split('\n')]

            if len(batch_results) != len(account_type_batch):
                print("batch: ", account_type_batch)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(account_type_batch)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(account_type_batch)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(account_types), batch_size)
        executor.map(process_batch, indices)

    return results


def sga_prompt_generator(chart_of_accounts):
    sga_prompt = """
    You are a skilled financial analyst.
    Match the given account names with the most suitable account from the Chart of Accounts given to you.
    Below is the list of available Chart of Accounts in the format Account Name - Account Type:
    the input will also be similar format Account Name - Account Type
    """ + "\n- " + "\n- ".join(chart_of_accounts) + "\n"

    sga_prompt += """
    1) Return only Account Name as response,If there is no close match, name it 'No Suitable Match'.
    2) The matches must be 1:1, meaning each account name from the list must be paired uniquely with one account from the COA and vice versa. This is very important.
    3) If there are 15 chart of accounts details given in a batch, you must return exactly 15 mapped account names (meaning 15 values returned, even if all 15 are no suitable match), Including No Suitable Match. This is so that the format will not get messed up. 
    4) Please do not give them index numbers at all.
    5) Make sure the return list length is exactly the same as the input size length (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
    6) Please do not have empty lines in your return. The results should all be in the next line IMPORTANT
    7) the account types when matched should not be conflicting
    """

    return sga_prompt


def recommend_sga_match(jaz_account_details, ext_coa_account_names, ext_coa_account_types, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(ext_coa_account_names)
    sga_prompt = sga_prompt_generator(jaz_account_details)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(ext_coa_account_names))
        c_an = ext_coa_account_names[start_index:end_index]
        c_at = ext_coa_account_types[start_index:end_index]
        messages = [{'role': 'system', 'content': sga_prompt}]
        for t in range(len(c_an)):
            messages.append({'role': 'user', 'content': f"Account Name: {c_an[t]} - Account Type:{c_at[t]}"})

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

            if len(batch_results) != len(c_an):
                print("batch: ", c_an)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(c_an)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(c_an)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(ext_coa_account_names), batch_size)
        executor.map(process_batch, indices)

    return results


def match_coa_using_gpt(external_coa_df, jaz_coa_df, jaz_coa_map, mapped_coa_names):
    unmapped_external_coa = external_coa_df[~(external_coa_df['*Name'].isin(mapped_coa_names))]
    jaz_account_names = []
    jaz_account_types = []
    for i in range(len(jaz_coa_df)):
        account_name = jaz_coa_df.iloc[i]['Name*']
        account_type = jaz_coa_df.iloc[i]['Account Type*']
        if jaz_coa_map[account_name]['Match']:
            continue
        else:
            jaz_account_names.append(account_name)
            jaz_account_types.append(account_type)
    ext_coa_account_names = unmapped_external_coa['*Name'].tolist()
    ext_coa_account_types = unmapped_external_coa['*Type'].tolist()
    jaz_account_details = [f"{account_name} - {account_type}" for account_name, account_type in
                           zip(jaz_account_names, jaz_account_types)]
    sga_matches = recommend_sga_match(jaz_account_details, ext_coa_account_names, ext_coa_account_types, 15)
    for i in range(len(sga_matches)):
        if sga_matches[i] != 'No Suitable Match':
            jaz_coa_name = sga_matches[i]
            ext_coa_name = ext_coa_account_names[i]
            mapped_coa_names.add(ext_coa_name)
            filtered_df = external_coa_df[external_coa_df['*Name'] == ext_coa_name]
            if len(filtered_df) > 0:
                filtered_row = filtered_df.iloc[0]
                jaz_coa_map[jaz_coa_name]['Code'] = filtered_row['*Code']
                jaz_coa_map[jaz_coa_name]['Description'] = filtered_row['Description']
                jaz_coa_map[jaz_coa_name]['Match'] = True
                jaz_coa_map[jaz_coa_name]['Status'] = 'ACTIVE'
                jaz_coa_map[jaz_coa_name]['Match Type'] = 'GPT'
    return jaz_coa_map, mapped_coa_names


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


def run_process():
    st.markdown(
        """
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
            text-align: center;
            font-size: 45px;
            font-weight: bold;
            white-space: nowrap;
        ">
            Jaz COA Import Mapping Tool (SG-EN)
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")
    st.markdown("""
            <div style='text-align: center; margin-bottom: 5px;'>
                <h1 style='font-size: 20px; font-weight: bold;'>Please upload External COA File</h1>
            </div>
        """, unsafe_allow_html=True)
    st.write("")
    external_coa_file = st.file_uploader("", type=["csv"])
    st.write("")
    st.write("")
    st.markdown("""
            <div style='text-align: center; margin-bottom: 5px;'>
                <h1 style='font-size: 20px; font-weight: bold;'>Please upload JAZ COA Import File</h1>
            </div>
        """, unsafe_allow_html=True)
    jaz_coa_file = st.file_uploader("", type=["xlsx"])
    if external_coa_file is not None and jaz_coa_file is not None:
        external_coa_data = pd.read_csv(external_coa_file)
        jaz_coa_data = pd.read_excel(jaz_coa_file, sheet_name=1)
        jaz_coa_df_columns = jaz_coa_data.columns
        currency_flag = False
        if 'Currency*' in jaz_coa_df_columns:
            currency_flag = True
        column_order = []
        for col in jaz_coa_df_columns:
            if currency_flag:
                if col in COLUMNS_WITH_CURRENCY:
                    column_order.append(col)
            else:
                if col in COLUMNS_WITHOUT_CURRENCY:
                    column_order.append(col)

        external_coa_account_types = external_coa_data['*Type'].tolist()
        external_coa_report_codes =external_coa_data['Report Code'].tolist()
        output = classify_account_types(external_coa_account_types,external_coa_report_codes,15)
        st.write(external_coa_account_types, output)
        jaz_coa_map = defaultdict(dict)
        mapped_external_coa_names = set()
        for j in range(len(jaz_coa_data)):
            row = jaz_coa_data.iloc[j]
            account_name = row['Name*']
            account_type = row['Account Type*']
            code = row['Code']
            description = row['Description']
            lock_date = row['Lock Date']
            unique_id = row['Unique ID (do not edit)']
            jaz_coa_map[account_name] = {
                "Account Type*": account_type,
                "Name*": account_name,
                "Code": code,
                "Description": description,
                "Lock Date": lock_date,
                "Unique ID (do not edit)": unique_id,
                "Match": False,
                "Status": "INACTIVE"
            }
            if currency_flag:
                jaz_coa_map[account_name]['Currency*'] = row['Currency*']

        for i in range(len(external_coa_data)):
            row = external_coa_data.iloc[i]
            if row['jaz_sga_name'] == '' or pd.isnull(row['jaz_sga_name']):
                continue
            else:
                if row['jaz_sga_name'] in jaz_coa_map:
                    jaz_coa_map[row['jaz_sga_name']]['Code'] = row['*Code']
                    jaz_coa_map[row['jaz_sga_name']]['Description'] = row['Description']
                    jaz_coa_map[row['jaz_sga_name']]['Match'] = True
                    jaz_coa_map[row['jaz_sga_name']]['Status'] = 'ACTIVE'
                    jaz_coa_map[row['jaz_sga_name']]['Match Type'] = 'SGA NAME'
                    mapped_external_coa_names.add(row['*Name'])
        jaz_coa_map, mapped_external_coa_names = match_coa_using_gpt(external_coa_data, jaz_coa_data, jaz_coa_map,
                                                                     mapped_external_coa_names)
        for p in range(len(external_coa_data)):
            row = external_coa_data.iloc[p]
            if row['*Name'] not in mapped_external_coa_names:
                account_name = row['*Name']
                account_type = row['*Type']
                code = row['*Code']
                description = row['Description']
                lock_date = ""
                unique_id = ""
                jaz_coa_map[account_name] = {
                    "Account Type*": account_type,
                    "Name*": account_name,
                    "Code": code,
                    "Description": description,
                    "Lock Date": lock_date,
                    "Unique ID (do not edit)": unique_id,
                    "Match": False,
                    "Status": "ACTIVE"
                }
                if currency_flag:
                    jaz_coa_map[account_name]['Currency*'] = ""

        for key, value in jaz_coa_map.items():
            if key in ACTIVE_ONLY_ACCOUNTS and jaz_coa_map[key]['Status'] == 'INACTIVE':
                jaz_coa_map[key]['Status'] = 'ACTIVE'

        final_df = pd.DataFrame.from_dict(jaz_coa_map, orient='index')
        # Reset the index to move the outer dictionary keys to a column
        final_df.reset_index(drop=True, inplace=True)
        final_df = final_df[column_order]
        final_output_csv = convert_df_to_csv(final_df)
        instructions = """
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; font-size: 18px; text-align: left;">
            <div>
                <h3 style="text-align: center;">Instructions:</h3>
                <p style="margin: 4px 0;"><strong>1. Copy the spreadsheet data and replace the import file entry sheet.</strong></p>
                <p style="margin: 4px 0;"><strong>2. Upload the file to the correct Jaz organization account.</strong></p>
            </div>
        </div>
        """

        # Display the instructions using Streamlit with HTML for center alignment
        st.markdown(instructions, unsafe_allow_html=True)
        st.write("")
        st.write("")
        col1, col2, col3 = st.columns([17, 10, 17])
        with col2:
            st.download_button(
                label="Download File",
                data=final_output_csv,
                file_name='COA_mapped_to_jaz_import.csv',
                mime='text/csv',
            )


if __name__ == '__main__':
    run_process()
