import streamlit as st
import pandas as pd
import requests
import concurrent.futures
from collections import defaultdict
from fuzzywuzzy import fuzz

API_KEY = st.secrets["API_KEY"]
ACTIVE_ONLY_ACCOUNTS = ['FX Realized Currency (Gains)/Loss',
                        'Input GST Receivable',
                        'Accounts Payable',
                        'FX Unrealized Currency (Gains)/Loss',
                        'FX Bank Revaluation (Gains)/Loss',
                        'Business Bank Account',
                        'FX Rounding (Gains)/Loss',
                        'Accounts Receivable',
                        'Retained Earnings',
                        'Output VAT Payable']

COLUMNS_WITH_CURRENCY = ['Account Type*',
                         'Currency*',
                         'Name*',
                         'Code',
                         'Description',
                         'Lock Date',
                         'Status',
                         'Controlled Account (do not edit)',
                         'Unique ID (do not edit)']

COLUMNS_WITHOUT_CURRENCY = ['Account Type*',
                            'Name*',
                            'Code',
                            'Description',
                            'Lock Date',
                            'Status',
                            'Controlled Account (do not edit)',
                            'Unique ID (do not edit)']

jaz_account_type_mappings = {
    "Current Liability": "Liability - Current Liability",
    "Bank Accounts": "Asset - Bank Accounts",
    "Operating Expense": "Expense - Operating Expense",
    "Current Asset": "Asset - Current Asset",
    "Direct Costs": "Expense - Direct Costs",
    "Shareholders Equity": "Equity - Shareholders Equity",
    "Fixed Asset": "Asset - Fixed Asset",
    "Non-current Asset": "Asset - Non-current Asset",
    "Non-current Liability": "Liability - Non-current Liability",
    "Other Revenue": "Revenue - Other Revenue",
    "Operating Revenue": "Revenue - Operating Revenue",
    "Inventory": "Asset - Inventory",
    "Cash": "Asset - Cash"
}

external_account_type_mappings = {
    "Accounts Payable": "Liability - Current Liability",
    "Accounts Receivable": "Asset - Current Asset",
    "Bank": "Asset - Bank Accounts",
    "Bank Revaluations": "Expense - Operating Expense",
    "Current Asset": "Asset - Current Asset",
    "Current Liability": "Liability - Current Liability",
    "Direct Costs": "Expense - Direct Costs",
    "Equity": "Equity - Shareholders Equity",
    "Expense": "Expense - Operating Expense",
    "Fixed Asset": "Asset - Fixed Asset",
    "Historical Adjustment": "Liability - Current Liability",
    "Non-Current Asset": "Asset - Non-current Asset",
    "Non-current Liability": "Liability - Non-current Liability",
    "Other Income": "Revenue - Other Revenue",
    "Realized Currency Gains": "Expense - Operating Expense",
    "Retained Earnings": "Equity - Shareholders Equity",
    "Revenue": "Revenue - Operating Revenue",
    "Rounding": "Liability - Current Liability",
    "Sales Tax": "Liability - Current Liability",
    "Tracking": "Liability - Current Liability",
    "Unpaid Expense Claims": "Liability - Current Liability",
    "Unrealized Currency Gains": "Expense - Operating Expense",
    "Wages Payable": "Liability - Current Liability",
    "Inventory": "Asset - Inventory",
    "Cash": "Asset - Cash"
}


def validate_sga_match_response(value):
    if value in ['No Suitable Match', 'Error in classification', '']:
        return False
    if pd.isna(value):
        return False
    return True


# Function to return the right column value
def get_account_type_mapping(external_account_type):
    for key, value in external_account_type_mappings.items():
        if fuzz.ratio(key, external_account_type) > 95:
            return value

    for key, value in jaz_account_type_mappings.items():
        if fuzz.ratio(key, external_account_type) > 95:
            return value

    return f'Not Mapped: {external_account_type}'


def sga_prompt_generator(chart_of_accounts):
    sga_prompt = """
    You are a skilled financial analyst.
    Match the given account names with the most suitable account from the Chart of Accounts given to you.
    Below is the list of available Chart of Accounts in the format:  {'Account Name','Account Type'}
    The input will also be similar format {'Account Name','Account Type'}
    """ + "\n- " + "\n- ".join(chart_of_accounts) + "\n"

    sga_prompt += """
    1) Return only Account Name as response,If there is no close match,Return ONLY 'No Suitable Match' (VERY IMPORTANT).
    2) Make sure the return list length is exactly the same as the input size length (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
    3) The matches must be 1:1, meaning each Account Name from the list must be paired uniquely with 
    one Account from the Chart of Accounts given and vice versa (VERY IMPORTANT).
    4) Please do not give them index numbers at all.
    5) Please do not have empty lines in your return. The results should all be in the next line IMPORTANT
    6) the Account Type when matched should not be conflicting
    """

    return sga_prompt


def recommend_sga_match(jaz_account_details, ext_coa_account_details, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(ext_coa_account_details)
    sga_prompt = sga_prompt_generator(jaz_account_details)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(ext_coa_account_details))
        ext_ac = ext_coa_account_details[start_index:end_index]
        messages = [{'role': 'system', 'content': sga_prompt}]
        for t in range(len(ext_ac)):
            messages.append({'role': 'user', 'content': ext_ac[t]})

        data = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            content = content.strip("```").strip()
            batch_results = [line.strip() for line in content.split('\n') if line.strip()]

            if len(batch_results) != len(ext_ac):
                print("batch: ", ext_ac)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(ext_ac)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(ext_ac)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(ext_coa_account_details), batch_size)
        executor.map(process_batch, indices)

    return results


def match_coa_using_gpt(external_coa_df, jaz_coa_df, jaz_coa_map, mapped_coa_names):
    unmapped_external_coa = external_coa_df[~(external_coa_df['*Name'].isin(mapped_coa_names))]
    jaz_account_names = []
    jaz_account_types = []
    count = 0
    for i in range(len(jaz_coa_df)):
        account_name = jaz_coa_df.iloc[i]['Name*']
        account_type = jaz_coa_df.iloc[i]['Account Type*']
        if jaz_coa_map[account_name]['Match']:
            count += 1
            continue
        else:
            jaz_account_names.append(account_name)
            jaz_account_types.append(account_type)
    ext_coa_account_names = unmapped_external_coa['*Name'].tolist()
    ext_coa_account_types = unmapped_external_coa['*Type'].tolist()
    jaz_account_details = [f"{account_name} - {account_type}" for account_name, account_type in
                           zip(jaz_account_names, jaz_account_types)]
    sga_matches = recommend_sga_match(jaz_account_details, ext_coa_account_names, ext_coa_account_types, 15)
    if len(sga_matches) != len(ext_coa_account_names):
        return jaz_coa_map, mapped_coa_names
    sga_conflict_map = defaultdict(int)
    for i in range(len(sga_matches)):
        value = sga_matches[i]
        sga_conflict_map[value] += 1
    for i in range(len(sga_matches)):
        if validate_sga_match_response(sga_matches[i]) and sga_conflict_map[sga_matches[i]] == 1:
            jaz_coa_name = sga_matches[i]
            ext_coa_name = ext_coa_account_names[i]
            filtered_df = external_coa_df[external_coa_df['*Name'] == ext_coa_name]
            if len(filtered_df) > 0 and jaz_coa_name in jaz_coa_map:
                filtered_row = filtered_df.iloc[0]
                jaz_coa_map[jaz_coa_name]['Code'] = filtered_row['*Code']
                jaz_coa_map[jaz_coa_name]['Description'] = filtered_row['Description']
                jaz_coa_map[jaz_coa_name]['Match'] = True
                jaz_coa_map[jaz_coa_name]['Status'] = 'ACTIVE'
                jaz_coa_map[jaz_coa_name]['Match Type'] = 'GPT'
                mapped_coa_names.add(ext_coa_name)
    return jaz_coa_map, mapped_coa_names


def convert_df_to_csv(df):
    csv_df = df.to_csv(index=False, encoding='utf-8')
    return csv_df


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
        if 'jaz_sga_name' not in external_coa_data.columns:
            st.error("""
                Please add a new column “jaz_sga_name” to the external COA file.\t
                In this column, please enter a value from the list below to map to the correct controlled account.
                
                **Controlled Accounts**:
                1. Accounts Payable
                2. Accounts Receivable
                3. Business Bank Account
                4. FX Bank Revaluation (Gains)/Loss
                5. FX Realized Currency (Gains)/Loss
                6. FX Rounding (Gains)/Loss
                7. FX Unrealized Currency (Gains)/Loss
                8. Input GST Receivable
                9. Output VAT Payable
                10. Retained Earnings
                """)
            st.stop()
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

        for i in range(len(external_coa_data)):
            external_coa_data.at[i, '*Type'] = get_account_type_mapping(external_coa_data.iloc[i]['*Type'])
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
            if key in ACTIVE_ONLY_ACCOUNTS:
                jaz_coa_map[key]['Status'] = 'ACTIVE'

        final_df = pd.DataFrame.from_dict(jaz_coa_map, orient='index')
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
